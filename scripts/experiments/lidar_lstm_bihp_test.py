"""Teste isolado: LSTM unidirecional com os hp exatos da lstm_bi (80/20).

Isola a questao "arquitetura vs hiperparametros" no gap lstm 2.08 vs
lstm_bi 1.67: mesmo otimizador/lr/weight_decay/unidades/dropouts da lstm_bi,
mesma seed, mesmo orcamento de epocas, mesmo split. Nada sobrescrito:
- modelo -> pipeline/tmp/trained_8020_lstm_bihp/lstm
- metricas -> data/results/lstm_bihp_8020/summary.json

    python scripts/experiments/lidar_lstm_bihp_test.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import common

EPOCHS = 60
SEED = 42
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_8020_lstm_bihp")
RESULTS_DIR = common.resolve("data/results/lstm_bihp_8020")
BI_HP = common.resolve("pipeline/tmp/trained_8020/lstm_bi/model.json")
CANONICAL = common.resolve("data/results/lidar_8020_features/all.json")
BASELINES = common.resolve("data/results/baselines/arima_rf/metrics.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    hyperparameters = json.load(open(BI_HP))["hyperparameters"]
    print(f"hp da lstm_bi: {hyperparameters}")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.80, 0.0)
    val_end = len(train_df)
    print(f"Dataset: {len(dataset)} linhas | treino {len(train_df)} | "
          f"teste {len(test_df)}")

    from keras import backend as K

    K.clear_session()
    tf.keras.utils.set_random_seed(SEED)
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
    batch_size = int(hyperparameters.get("batch_size", 32))
    if hasattr(wrapper, "schedule_total_steps"):
        steps = int((len(train_df) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps * args.epochs
    wrapper.build(common.FixedHyperParameters(hyperparameters))

    print(f"[lstm+bihp] treino {args.epochs} epocas...")
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
    print(f"[lstm+bihp] {len(losses)} epocas em {elapsed:.0f}s | "
          f"loss final={losses[-1]:.4f}")

    model_dir = OUTPUT_DIR / "lstm"
    model_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save(model_dir / "model.keras")
    common.write_json(model_dir / "model.json", {
        "wrapper_key": "lstm",
        "name": wrapper.name,
        "persistence_gate": False,
        "gate_mode": "static",
        "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
        "target_mode": getattr(wrapper, "target_mode", "absolute"),
        "normalize": bool(getattr(wrapper, "normalize", True)),
        "hyperparameters": hyperparameters,
        "params_source": "lstm_bi (80/20 canonico)",
        "input_steps": 72,
        "output_steps": 36,
        "target_col": "ws100_wavelet",
        "denoise": ["ws100"],
        "denoise_level": 2,
        "training": {"epochs_run": len(losses), "seed": SEED},
        "dataset": {"train_ratio": 0.80, "val_ratio": 0.0,
                    "split": "80/20 sem validacao (teste hp lstm_bi)"},
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
    baselines = json.load(open(BASELINES))

    summary = {
        "lstm_bihp": {**metrics, "rolling": roll,
                      "skill": 1 - metrics["mse"] / persistence["mse"],
                      "hp_source": "lstm_bi canonico 80/20",
                      "epochs": len(losses),
                      "training_time_sec": round(elapsed, 1)},
        "lstm_canonico": {k: canonical["lstm"][k] for k in
                          ("mse", "mae", "r2", "skill")},
        "lstm_bi_canonico": {k: canonical["lstm_bi"][k] for k in
                             ("mse", "mae", "r2", "skill")},
        "arima": {"all_horizons": baselines["arima"]["all_horizons"],
                  "rolling": baselines["arima"]["rolling"]},
        "persistencia": persistence,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json(RESULTS_DIR / "summary.json", summary)

    print("\n=== comparacao (todos os horizontes vs ws100 bruto) ===")
    for label, m, roll_mse in (
            ("lstm + hp da bi", metrics, roll["mse"]),
            ("lstm canonico", canonical["lstm"], float("nan")),
            ("lstm_bi canonico", canonical["lstm_bi"], float("nan")),
            ("ARIMA", baselines["arima"]["all_horizons"],
             baselines["arima"]["rolling"]["mse"])):
        skill = 1 - m["mse"] / persistence["mse"]
        print(f"{label:<18} MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  "
              f"R2={m['r2']:+.3f}  skill={skill:+.1%}  "
              f"roll_MSE={roll_mse:.4f}")
    print(f"persistencia        MSE={persistence['mse']:.4f}")
    print(f"\nResumo: {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
