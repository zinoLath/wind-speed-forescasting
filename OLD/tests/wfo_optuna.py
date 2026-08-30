"""Walk-forward evaluation using the best hyperparameters found by Optuna."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.common import (
    WRAPPERS,
    FixedHyperParameters,
    compute_metrics,
    load_best_params,
    load_dataset,
    split_dataset,
    wrapper_factory,
    write_json,
)


def evaluate_wrapper(wrapper_key, dataset, params_file, epochs, input_steps, output_steps):
    params_path, best_params = load_best_params(wrapper_key, params_file)
    train_df, val_df, _ = split_dataset(dataset)
    val_end = len(train_df) + len(val_df)

    K.clear_session()
    wrapper = wrapper_factory(wrapper_key)()
    wrapper.prepare(
        train_df,
        val_df,
        input_steps=input_steps,
        output_steps=output_steps,
        target_col="ws100_wavelet",
        denoise=["ws100"],
        denoise_level=2,
    )
    wrapper.build(FixedHyperParameters(best_params))

    if hasattr(wrapper, "schedule_total_steps"):
        n_train = len(train_df)
        steps_per_epoch = int(np.ceil(n_train / 32))
        wrapper.schedule_total_steps = steps_per_epoch * epochs

    history = wrapper.fit(
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ],
        use_validation=True,
    )

    started_at = time.perf_counter()
    predictions, actuals = wrapper.rolling_forecast(dataset, test_start=val_end)
    elapsed = time.perf_counter() - started_at
    predictions = predictions.ravel()
    actuals = actuals.ravel()

    metrics = {
        "wrapper": wrapper.name,
        "params_file": str(params_path),
        "best_params": best_params,
        "train_epochs": len(history.history.get("loss", [])),
        "prediction_time_sec": elapsed,
        **compute_metrics(actuals, predictions),
    }
    results_dir = PROJECT_ROOT / "data" / "results" / "wfo_optuna" / wrapper.name
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions, "actual": actuals}).to_csv(
        results_dir / "predictions.csv", index=False
    )
    write_json(results_dir / "metrics.json", metrics)
    print(f"{wrapper.name}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Optuna-selected models with walk-forward forecasting."
    )
    parser.add_argument("--wrappers", nargs="*", choices=WRAPPERS, default=list(WRAPPERS))
    parser.add_argument("--params-file", type=Path, help="Use one best_trial.json for a single wrapper.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-steps", type=int, default=72)
    parser.add_argument("--output-steps", type=int, default=36)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.params_file and len(args.wrappers) != 1:
        raise ValueError("--params-file requires exactly one wrapper.")
    dataset = load_dataset(PROJECT_ROOT / "data" / "dataset.csv")
    for wrapper_key in args.wrappers:
        evaluate_wrapper(
            wrapper_key,
            dataset,
            args.params_file,
            args.epochs,
            args.input_steps,
            args.output_steps,
        )


if __name__ == "__main__":
    main()
