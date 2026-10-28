"""AB: LR constante + ReduceLROnPlateau (atual) vs warmup cosine.

Para cada um dos 4 modelos principais, retreina com os melhores
hiperparâmetros do round 3 (pipeline/tmp/optuna/<name>/best_trial.json) nas
duas condições de scheduling, seed 42, 150 épocas, patience 12, decoder
direct, com normalização. Métricas no teste contra o ws100 bruto (denso e
rolling h=36). Condições já concluídas são puladas (retomada).

Uso: python studies/ab_lrschedule.py [--wrappers lstm lstm_bi tcn tcn_bi]
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
from keras import backend as K

from src import common

WRAPPERS = ["lstm", "lstm_bi", "tcn", "tcn_bi"]
SEED = 42
EPOCHS = 150
BATCH = 32
PATIENCE = 12
OUT_BASE = PROJECT_ROOT / "pipeline/tmp/ab_lrschedule"

CONDITIONS = [
    {"slug": "constant", "label": "constante + ReduceLROnPlateau",
     "override": None},
    {"slug": "warmup_cosine", "label": "warmup 10% + cosine (min 1%)",
     "override": "warmup_cosine"},
]

# preenchidos em main()
RAW_WS100 = None
TEST_DF = None
FULL_DF = None
TEST_START = None


def set_seed(seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def best_params(wrapper_key):
    path = common.find_optuna_result(wrapper_key)
    if path is None:
        raise SystemExit(f"Sem best_trial.json para {wrapper_key}")
    return json.load(open(path, encoding="utf-8"))["best_params"]


def train_one(wrapper_key, hp_values, cond, train_df, val_df):
    name = common.wrapper_factory(wrapper_key)().name
    out_dir = OUT_BASE / f"{name}_{cond['slug']}"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        print(f"[{wrapper_key}/{cond['slug']}] já concluído, pulando")
        return json.load(open(metrics_path, encoding="utf-8"))

    K.clear_session()
    set_seed(SEED)
    wrapper = common.wrapper_factory(wrapper_key)()
    wrapper.prepare(
        train_df, val_df,
        input_steps=72, output_steps=36,
        target_col="ws100_wavelet",
        denoise=("ws100", "ws40", "v40"), denoise_level=1,
        decoder_mode="direct", persistence_gate=False,
    )
    if cond["override"]:
        wrapper.schedule_mode_override = cond["override"]
        steps_per_epoch = -(-wrapper.train["X_encoder"].shape[0] // BATCH)
        wrapper.schedule_total_steps = steps_per_epoch * EPOCHS

    started = time.perf_counter()
    wrapper.build(common.FixedHyperParameters(hp_values))
    callbacks = common.default_callbacks(wrapper, PATIENCE)
    history = wrapper.fit(epochs=EPOCHS, batch_size=BATCH, verbose=0,
                          callbacks=callbacks, use_validation=True)
    elapsed = time.perf_counter() - started
    losses = history.history.get("val_loss", [])
    best_val = float(min(losses)) if losses else None
    epochs_run = len(losses)
    print(f"[{wrapper_key}/{cond['slug']}] {epochs_run} épocas em {elapsed:.0f}s | "
          f"best val_loss={best_val:.5f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save(out_dir / "model.keras")

    def blk(df):
        pred = common.predict_all_horizons(wrapper, df)
        real = RAW_WS100.reindex(pred["timestamp"]).to_numpy()
        return {
            "mae": float(np.mean(np.abs(pred["predicted"] - real))),
            "mse": float(np.mean((pred["predicted"] - real) ** 2)),
            "r2": float(common.compute_metrics(real, pred["predicted"])["r2"]),
        }

    test_all = blk(TEST_DF)
    roll_pred, roll_act = wrapper.rolling_forecast(FULL_DF, test_start=TEST_START)
    roll = {
        "mae": float(np.mean(np.abs(roll_pred - roll_act))),
        "mse": float(np.mean((roll_pred - roll_act) ** 2)),
        "r2": float(common.compute_metrics(roll_act, roll_pred)["r2"]),
    }
    out = {
        "condicao": cond["label"], "epochs_run": epochs_run,
        "train_sec": round(elapsed, 1), "best_val_loss": best_val,
        "test_all_horizons": test_all, "rolling_h36": roll,
        "hyperparameters": hp_values,
    }
    json.dump(out, open(metrics_path, "w", encoding="utf-8"), indent=1)
    return out


def main():
    global RAW_WS100, TEST_DF, FULL_DF, TEST_START
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrappers", nargs="*", default=WRAPPERS)
    args = parser.parse_args()

    dataset = common.load_dataset(PROJECT_ROOT / "data/dataset.csv",
                                  keep_raw=("ws40", "ws100", "v40"))
    train_df, val_df, test_df = common.split_dataset(dataset, 0.75, 0.20)
    RAW_WS100 = dataset["ws100"]
    TEST_DF = test_df
    FULL_DF = dataset
    TEST_START = len(train_df) + len(val_df)

    results = {}
    for wrapper_key in args.wrappers:
        hp_values = best_params(wrapper_key)
        for cond in CONDITIONS:
            key = f"{wrapper_key}/{cond['slug']}"
            results[key] = train_one(wrapper_key, hp_values, cond,
                                     train_df, val_df)
            json.dump(results, open(OUT_BASE / "summary.json", "w",
                                    encoding="utf-8"), indent=1)

    print("\n=== Resumo: constante (ReduceLROnPlateau) vs warmup cosine ===")
    header = (f"{'modelo':<10} {'condição':<14} {'épocas':>7} {'val_loss':>10} "
              f"{'MAE denso':>10} {'MSE denso':>10} {'MAE roll':>9} {'R² roll':>8}")
    print(header)
    for wrapper_key in args.wrappers:
        for cond in CONDITIONS:
            r = results[f"{wrapper_key}/{cond['slug']}"]
            print(f"{wrapper_key:<10} {cond['slug']:<14} {r['epochs_run']:>7} "
                  f"{r['best_val_loss']:>10.5f} {r['test_all_horizons']['mae']:>10.4f} "
                  f"{r['test_all_horizons']['mse']:>10.4f} {r['rolling_h36']['mae']:>9.4f} "
                  f"{r['rolling_h36']['r2']:>8.4f}")
    print(f"\nResumo salvo em {OUT_BASE / 'summary.json'}")


if __name__ == "__main__":
    main()