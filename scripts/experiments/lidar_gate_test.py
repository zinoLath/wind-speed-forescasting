"""Teste isolado: persistence gate na tcn_bi (80/20) vs ARIMA.

Nao sobrescreve relatorios nem os modelos canonicos:
- modelo -> pipeline/tmp/trained_8020_gate/tcn_bi
- metricas -> data/results/gate_test_8020/summary.json

    python scripts/lidar_gate_test.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import common

EPOCHS = 60
SEED = 42
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_gate")
RESULTS_DIR = common.resolve("data/results/gate_test_8020")
BASELINES = common.resolve("data/results/baselines/arima_rf/metrics.json")
CANONICAL = common.resolve("data/results/lidar_8020_features/all.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--gate-mode", default="static",
                        choices=("static", "dynamic"))
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | gate-mode={args.gate_mode}")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    val_end = len(train_df)
    print(f"Dataset: {len(dataset)} linhas | treino {len(train_df)} | "
          f"teste {len(test_df)}")

    wrapper_key = "tcn_bi"
    hyperparameters, params_source = common.resolve_hyperparameters(
        wrapper_key,
        {"params_source": "optuna", "hyperparameters": {}},
        {"paths": {"tmp_dir": "pipeline/tmp", "results_dir": "data/results"}},
    )
    from keras import backend as K

    K.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    wrapper = common.wrapper_factory(wrapper_key)()
    wrapper.gate_mode = args.gate_mode
    wrapper.prepare(
        train_df,
        train_df.tail(400),
        input_steps=72,
        output_steps=36,
        target_col="ws100_wavelet",
        denoise=("ws100",),
        denoise_level=2,
        persistence_gate=True,
    )
    batch_size = int(hyperparameters.get("batch_size", 32))
    if hasattr(wrapper, "schedule_total_steps"):
        steps = int((len(train_df) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps * args.epochs
    wrapper.build(common.FixedHyperParameters(hyperparameters))

    print(f"[tcn_bi+gate] treino {args.epochs} epocas...")
    started = time.perf_counter()
    history = wrapper.fit(
        epochs=args.epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=None,
        use_validation=False,
    )
    elapsed = time.perf_counter() - started
    losses = history.history.get("loss", [])
    print(f"[tcn_bi+gate] {len(losses)} epocas em {elapsed:.0f}s | "
          f"loss final={losses[-1]:.4f}")

    model_dir = OUTPUT_DIR / wrapper_key
    model_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save(model_dir / "model.keras")
    common.write_json(model_dir / "model.json", {
        "wrapper_key": wrapper_key,
        "name": wrapper.name,
        "persistence_gate": True,
        "gate_mode": args.gate_mode,
        "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
        "target_mode": getattr(wrapper, "target_mode", "absolute"),
        "normalize": bool(getattr(wrapper, "normalize", True)),
        "hyperparameters": hyperparameters,
        "params_source": params_source,
        "input_steps": 72,
        "output_steps": 36,
        "target_col": "ws100_wavelet",
        "denoise": ["ws100"],
        "denoise_level": 2,
        "training": {"epochs_run": len(losses), "seed": SEED},
        "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                    "split": "80/20 sem validacao (teste gate)"},
    })

    all_df = common.predict_all_horizons(wrapper, test_df)
    all_df["actual_raw"] = dataset["ws100"].reindex(
        all_df["timestamp"]).to_numpy()
    metrics = common.compute_metrics(all_df["actual_raw"], all_df["predicted"])
    persistence = common.compute_metrics(
        all_df["actual_raw"], all_df["persistence"])

    predictions, actuals = wrapper.rolling_forecast(dataset, test_start=val_end)
    roll = common.compute_metrics(actuals.ravel(), predictions.ravel())

    baselines = json.load(open(BASELINES))
    canonical = json.load(open(CANONICAL))["tcn_bi"]

    summary = {
        "tcn_bi_gate": {**metrics, "rolling": roll,
                        "skill": 1 - metrics["mse"] / persistence["mse"],
                        "gate_mode": args.gate_mode,
                        "epochs": len(losses),
                        "training_time_sec": round(elapsed, 1)},
        "tcn_bi_canonico": {k: canonical[k] for k in
                            ("mse", "mae", "r2", "skill")},
        "arima": {"all_horizons": baselines["arima"]["all_horizons"],
                  "rolling": baselines["arima"]["rolling"]},
        "rf": {"all_horizons": baselines["rf"]["all_horizons"]},
        "persistencia": {"all_horizons": persistence,
                         "rolling_mse": None},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json(RESULTS_DIR / "summary.json", summary)

    def row(label, all_m, roll_m):
        skill = (f"{1 - all_m['mse'] / persistence['mse']:+.1%}"
                 if persistence["mse"] else "—")
        print(f"{label:<18} MSE={all_m['mse']:.4f}  MAE={all_m['mae']:.4f}  "
              f"R2={all_m['r2']:+.3f}  skill={skill}  roll_MSE={roll_m:.4f}")

    print("\n=== comparacao (todos os horizontes vs ws100 bruto) ===")
    row("tcn_bi+gate", metrics, roll["mse"])
    row("tcn_bi canonico", canonical, canonical.get("rolling_mse", float("nan"))
        if isinstance(canonical.get("rolling_mse"), float) else float("nan"))
    row("ARIMA", baselines["arima"]["all_horizons"],
        baselines["arima"]["rolling"]["mse"])
    row("RF", baselines["rf"]["all_horizons"],
        baselines["rf"]["rolling"]["mse"])
    row("persistencia", persistence, float("nan"))
    print(f"\nResumo: {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
