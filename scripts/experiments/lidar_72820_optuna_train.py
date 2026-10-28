"""Treino LiDAR 72/8/20 com EarlyStopping + ReduceLROnPlateau (regime HPO).

Réplica do lidar_7525_train.py no protocolo do Optuna: split 72/8 (val
real), EarlyStopping(val_loss, patience, restore_best_weights) +
ReduceLROnPlateau via default_callbacks, HPs promovidos em
pipeline/tmp/optuna/ e loss por modelo (LSTM→ITILDE-Q, TCN→MSE). Avalia
no MESMO teste de 20% (grade idêntica ao 80/20 — split cronológico).

    python scripts/experiments/lidar_72820_optuna_train.py --static-dilations
    python reports/src/relatorio_resultados.py \
        --models-dir pipeline/tmp/trained_72820_optuna \
        --out-dir data/results/report_72820_optuna --no-baselines
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import common

WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
TRAIN_RATIO = 0.72
VAL_RATIO = 0.08
EPOCHS = 60
PATIENCE = 12
BATCH_SIZE = 32
SEED = 42
# Mesma política do lidar_7525_train.py (round 4: loss por modelo).
LOSS_BY_WRAPPER = {"lstm": "itilde_q", "lstm_bi": "itilde_q",
                   "tcn": "mse", "tcn_bi": "mse"}
STATIC_DILATIONS = {
    "encoder": [1, 2, 4, 8, 16, 32, 36],
    "decoder": [1, 2, 4, 8, 16],
    "encoder_depth": (4, 7),
    "decoder_depth": (3, 5),
}
OUTPUT_DIR = common.resolve("pipeline/tmp/trained_72820_optuna")
FEATURES_EVAL_DIR = common.resolve("data/results/lidar_72820_features")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                        choices=WRAPPERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--static-dilations", action="store_true")
    parser.add_argument("--legacy-hwmse", action="store_true")
    args = parser.parse_args()

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    dataset = common.load_dataset(common.resolve("data/dataset.csv"),
                                  keep_raw=("ws100",))
    train_df, val_df, test_df = common.split_dataset(
        dataset, args.train_ratio, args.val_ratio)
    split_label = (f"{round(args.train_ratio * 100)}/"
                   f"{round(args.val_ratio * 100)}/"
                   f"{round((1 - args.train_ratio - args.val_ratio) * 100)} "
                   f"com EarlyStopping(patience {args.patience}, "
                   f"restore_best) + ReduceLROnPlateau")
    print(f"Dataset: {len(dataset)} linhas | treino {len(train_df)} | "
          f"val {len(val_df)} | teste {len(test_df)}")

    output_dir = common.resolve(args.output_dir) if args.output_dir else OUTPUT_DIR
    eval_cache = FEATURES_EVAL_DIR / "all.json"
    evaluations = {}

    for wrapper_key in args.wrappers:
        wrapper_name = common.wrapper_factory(wrapper_key)().name
        hyperparameters, params_source = common.resolve_hyperparameters(
            wrapper_key,
            {"params_source": "optuna", "hyperparameters": {}},
            {"paths": {"tmp_dir": "pipeline/tmp", "results_dir": "data/results"}},
        )
        from keras import backend as K

        K.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        wrapper = common.wrapper_factory(wrapper_key)()
        wrapper.prepare(
            train_df, val_df,
            input_steps=72, output_steps=36, target_col="ws100_wavelet",
            denoise=("ws100",), denoise_level=2,
        )
        batch_size = int(hyperparameters.get("batch_size", BATCH_SIZE))
        if hasattr(wrapper, "schedule_total_steps"):
            steps = int((len(train_df) + batch_size - 1) // batch_size)
            wrapper.schedule_total_steps = steps * args.epochs
        wrapper.loss = hyperparameters.get("loss", "mse")
        if not args.legacy_hwmse:
            wrapper.loss_mode_override = LOSS_BY_WRAPPER.get(wrapper_key)
        if args.static_dilations and wrapper_key.startswith("tcn"):
            wrapper.static_dilations = STATIC_DILATIONS
        wrapper.build(common.FixedHyperParameters(hyperparameters))
        callbacks = common.default_callbacks(wrapper, args.patience)
        print(f"[{wrapper_key}] callbacks: "
              f"{[type(c).__name__ for c in callbacks]} | batch={batch_size}")

        print(f"[{wrapper_key}] treino {args.epochs} épocas (ES "
              f"patience {args.patience})...")
        started = time.perf_counter()
        history = wrapper.fit(
            epochs=args.epochs, batch_size=batch_size, verbose=0,
            callbacks=callbacks, use_validation=True,
        )
        elapsed = time.perf_counter() - started
        losses = history.history.get("loss", [])
        val_losses = history.history.get("val_loss", [])

        model_dir = output_dir / wrapper_key
        model_dir.mkdir(parents=True, exist_ok=True)
        wrapper.model.save(model_dir / "model.keras")
        common.write_json(model_dir / "model.json", {
            "model_class": common.wrapper_factory(wrapper_key).__name__,
            "wrapper_key": wrapper_key,
            "name": wrapper.name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hyperparameters": hyperparameters,
            "params_source": params_source,
            "input_steps": 72, "output_steps": 36,
            "target_col": "ws100_wavelet",
            "denoise": ["ws100"], "denoise_level": 2, "features": None,
            "decoder_mode": getattr(wrapper, "decoder_mode", "direct"),
            "target_mode": getattr(wrapper, "target_mode", "absolute"),
            "normalize": bool(getattr(wrapper, "normalize", True)),
            "persistence_gate": False, "gate_mode": "static",
            "context_mode": getattr(wrapper, "context_mode", "repeat"),
            "loss": wrapper.loss,
            "training": {
                "epochs_requested": args.epochs,
                "epochs_run": len(losses),
                "batch_size": batch_size,
                "patience": args.patience,
                "early_stopping": True,
                "reduce_lr_on_plateau": True,
                "training_time_sec": round(elapsed, 3),
                "final_loss": float(losses[-1]) if losses else None,
                "final_val_loss": float(val_losses[-1]) if val_losses else None,
                "best_val_loss": float(min(val_losses)) if val_losses else None,
                "seed": SEED,
            },
            "dataset": {
                "source": "data/dataset.csv",
                "train_rows": len(train_df), "val_rows": len(val_df),
                "test_rows": len(test_df),
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "split": split_label,
            },
            "features_mode": "all",
        })
        print(f"[{wrapper_key}] {len(losses)}/{args.epochs} épocas em "
              f"{elapsed:.0f}s | loss final={losses[-1]:.4f} | "
              f"val_loss={min(val_losses):.6f} (best)" if val_losses else "")

        all_df = common.predict_all_horizons(wrapper, test_df)
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        metrics = common.compute_metrics(all_df["actual_raw"],
                                         all_df["predicted"])
        persistence = common.compute_metrics(all_df["actual_raw"],
                                             all_df["persistence"])
        skill = 1 - metrics["mse"] / persistence["mse"]
        per_horizon = {}
        for h in (1, 6, 12, 24, 36):
            group = all_df[all_df["horizon"] == h]
            if len(group):
                per_horizon[h] = float(np.mean(
                    (group["predicted"] - group["actual_raw"]) ** 2))
        evaluations[wrapper_key] = {
            "name": wrapper_name, "features_mode": "all",
            "features": int(dataset.shape[1]), **metrics,
            "persistence_mse": persistence["mse"], "skill": skill,
            "per_horizon_mse": per_horizon, "epochs": len(losses),
        }
        print(f"[{wrapper_key}] teste: MSE={metrics['mse']:.4f} "
              f"RMSE={metrics['rmse']:.4f} skill={skill:+.3f} "
              f"(persistencia {persistence['mse']:.4f})")

    eval_cache.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(eval_cache, evaluations)
    print(f"\nAvaliacao: {eval_cache}")
    print(f"Modelos: {output_dir}")


if __name__ == "__main__":
    main()
