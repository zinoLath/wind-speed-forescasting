"""A/B experiment: raw (ws100) vs wavelet-denoised (ws100_wavelet) target.

Trains the same wrapper on both targets with the same seed and compares
honest metrics computed against the RAW wind speed on the held-out test
split, so the smoothing effect is measured rather than assumed.

Run from the project root:  python tests/ab_raw_vs_wavelet.py
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common


def run_variant(wrapper_key, dataset, target_col, hp, epochs, seed):
    from keras import backend as K
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    train_df, val_df, test_df = common.split_dataset(dataset)
    tf = common.setup_tensorflow()
    tf.keras.utils.set_random_seed(seed)

    K.clear_session()
    wrapper = common.wrapper_factory(wrapper_key)()
    wrapper.prepare(
        train_df,
        val_df,
        target_col=target_col,
        denoise=("ws100",),
    )
    if hasattr(wrapper, "schedule_total_steps"):
        wrapper.schedule_total_steps = int(len(train_df) / 32 + 1) * epochs
    wrapper.build(common.FixedHyperParameters(hp))
    wrapper.fit(
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ],
        use_validation=True,
    )

    started_at = time.perf_counter()
    predictions = common.predict_all_horizons(wrapper, test_df)
    elapsed = time.perf_counter() - started_at

    raw = dataset["ws100"].reindex(predictions["timestamp"]).to_numpy()
    metrics = common.compute_metrics(raw, predictions["predicted"])
    metrics["mae_denoised"] = common.compute_metrics(
        predictions["actual"], predictions["predicted"]
    )["mae"]
    metrics["forecast_sec"] = round(elapsed, 2)
    metrics["train_epochs"] = len(wrapper.model.history.history.get("loss", []))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="A/B: raw vs wavelet-denoised target.")
    parser.add_argument("--wrapper", default="lstm_bi")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = common.load_dataset(common.resolve("data/dataset.csv"))
    hp = common.DEFAULT_HYPERPARAMETERS[args.wrapper]

    variants = {
        "raw (ws100)": "ws100",
        "wavelet (ws100_wavelet)": "ws100_wavelet",
    }
    results = {}
    for label, target_col in variants.items():
        print(f"\n=== {label} ===")
        results[label] = run_variant(
            args.wrapper, dataset, target_col, hp, args.epochs, args.seed
        )
        print(
            f"MAE(raw)={results[label]['mae']:.4f} RMSE(raw)={results[label]['rmse']:.4f} "
            f"R2(raw)={results[label]['r2']:.4f} MAE(denoised)={results[label]['mae_denoised']:.4f}"
        )

    print(f"\n{'=' * 70}\nA/B summary ({args.wrapper}, {args.epochs} epochs, metrics vs RAW ws100)")
    for label, m in results.items():
        print(
            f"  {label:<24} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
            f"R2={m['r2']:.4f}  epochs={m['train_epochs']}"
        )


if __name__ == "__main__":
    main()