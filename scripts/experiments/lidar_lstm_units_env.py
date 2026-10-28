"""Teste rapido: LSTM com N units (hp vencedores do Optuna, exceto units).

Nao sobrescreve nada: modelo -> pipeline/tmp/trained_8020_lstm_u{N}/lstm,
metricas -> data/results/lstm_u{N}_8020/summary.json.

    python scripts/experiments/lidar_lstm_units_test.py --units 224
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common

SEED = 42
import os
HPO_HP = common.resolve(os.environ.get("HP_JSON", "pipeline/tmp/trained_8020_lstm_hpo/lstm/model.json"))
CANONICAL = common.resolve("data/results/lidar_8020_features/all.json")
BASELINES = common.resolve("data/results/baselines/arima_rf/metrics.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    hyperparameters = json.load(open(HPO_HP))["hyperparameters"]
    hyperparameters["lstm_units"] = args.units
    print(f"hp (base HPO, units={args.units}): {hyperparameters}")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    val_end = len(train_df)

    from keras import backend as K

    K.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    wrapper = common.wrapper_factory("lstm")()
    wrapper.prepare(
        train_df,
        train_df.tail(400),
        input_steps=72,
        output_steps=36,
        target_col="ws100_wavelet",
        denoise=("ws100",),
        denoise_level=2,
    )
    batch_size = int(hyperparameters.get("batch_size", 64))
    if hasattr(wrapper, "schedule_total_steps"):
        steps = int((len(train_df) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps * args.epochs
    wrapper.build(common.FixedHyperParameters(hyperparameters))

    print(f"[lstm-{args.units}u] treino {args.epochs} epocas...")
    started = time.perf_counter()
    history = wrapper.fit(epochs=args.epochs, batch_size=batch_size,
                          verbose=0, callbacks=None, use_validation=False)
    elapsed = time.perf_counter() - started
    losses = history.history.get("loss", [])
    print(f"[lstm-{args.units}u] {len(losses)} epocas em {elapsed:.0f}s | "
          f"params={wrapper.model.count_params():,}")

    model_dir = common.resolve(f"pipeline/tmp/trained_8020_lstm_u{args.units}") / "lstm"
    model_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save(model_dir / "model.keras")
    common.write_json(model_dir / "model.json", {
        "wrapper_key": "lstm", "name": wrapper.name,
        "persistence_gate": False, "gate_mode": "static",
        "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
        "target_mode": getattr(wrapper, "target_mode", "absolute"),
        "normalize": bool(getattr(wrapper, "normalize", True)),
        "hyperparameters": hyperparameters,
        "params_source": f"optuna HPO com units={args.units}",
        "input_steps": 72, "output_steps": 36,
        "target_col": "ws100_wavelet", "denoise": ["ws100"], "denoise_level": 2,
        "loss": "mse",
        "training": {"epochs_run": len(losses), "epochs_requested": args.epochs,
                     "seed": args.seed, "patience": 0,
                     "final_loss": float(losses[-1]) if losses else None,
                     "training_time_sec": round(elapsed, 1)},
        "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                    "split": "80/20 sem validacao (teste units)"},
    })

    all_df = common.predict_all_horizons(wrapper, test_df)
    all_df["actual_raw"] = dataset["ws100"].reindex(
        all_df["timestamp"]).to_numpy()
    metrics = common.compute_metrics(all_df["actual_raw"], all_df["predicted"])
    persistence = common.compute_metrics(
        all_df["actual_raw"], all_df["persistence"])
    predictions, actuals = wrapper.rolling_forecast(dataset, test_start=val_end)
    roll = common.compute_metrics(actuals.ravel(), predictions.ravel())

    canonical = json.load(open(CANONICAL))
    hpo = json.load(open(common.resolve("data/results/lstm_hpo_8020/summary.json")))
    baselines = json.load(open(BASELINES))

    results_dir = common.resolve(f"data/results/lstm_u{args.units}_8020")
    results_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(results_dir / "summary.json", {
        f"lstm_{args.units}u": {**metrics, "rolling": roll,
                                "skill": 1 - metrics["mse"] / persistence["mse"],
                                "params": int(wrapper.model.count_params())},
        "lstm_hpo_192u": hpo["lstm_hpo"],
        "lstm_canonica_128u": {k: canonical["lstm"][k]
                               for k in ("mse", "mae", "r2", "skill")},
        "persistencia": persistence,
        "arima": baselines["arima"]["all_horizons"],
    })

    print("\n=== comparacao (todos os horizontes vs ws100 bruto) ===")
    print(f"lstm {args.units}u (teste)  MSE={metrics['mse']:.4f}  MAE={metrics['mae']:.4f}  "
          f"R2={metrics['r2']:+.3f}  roll={roll['mse']:.4f}")
    print(f"lstm 192u (HPO)       MSE={hpo['lstm_hpo']['mse']:.4f}  "
          f"MAE={hpo['lstm_hpo']['mae']:.4f}  R2={hpo['lstm_hpo']['r2']:+.3f}")
    print(f"lstm 128u (canonica)  MSE={canonical['lstm']['mse']:.4f}")
    print(f"persistencia          MSE={persistence['mse']:.4f}")
    print(f"\nResumo: {results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
