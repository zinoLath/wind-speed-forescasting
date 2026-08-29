"""Walk-forward evaluation using the best hyperparameters found by Optuna."""

import argparse
import json
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.s2s_lstm_bi_wrapper import S2SLSTMBidirectionalWrapper
from src.models.s2s_lstm_wrapper import S2SLSTMWrapper
from src.models.s2s_tcn_bi_wrapper import S2STCNBidirectionalWrapper
from src.models.s2s_tcn_lstm_wrapper import S2STCNLSTMWrapper
from src.models.s2s_tcn_wrapper import S2STCNWrapper
from src.models.s2s_transformer_preln_wrapper import S2STransformerPrelnWrapper
from tests.optuna_all_hyperparameters import load_dataset, split_dataset


class FixedHyperParameters:
    """Adapter that gives a wrapper the values persisted by an Optuna trial."""

    def __init__(self, values):
        self.values = values

    def _get(self, name):
        if name not in self.values:
            # Fallbacks for new parameters added after the first Optuna searches.
            if name == "decoder_filters":
                return self._get("encoder_filters")
            raise KeyError(f"Optuna parameter '{name}' is missing.")

        # Compatibilidade com resultados antigos.
        if name == "encoder_filters" and "filters_power" in self.values:
            return 2 ** int(self.values["filters_power"])

        return self.values[name]

    def Float(self, name, min_value, max_value, step=None, sampling=None, default=None):
        return float(self._get(name))

    def Int(self, name, min_value, max_value, step=1, default=None):
        return int(self._get(name))

    def Choice(self, name, values, default=None):
        return self._get(name)


WRAPPERS = {
    "lstm": S2SLSTMWrapper,
    "lstm_bi": S2SLSTMBidirectionalWrapper,
    "tcn": S2STCNWrapper,
    "tcn_bi": S2STCNBidirectionalWrapper,
    "tcn_lstm": S2STCNLSTMWrapper,
    "transformer": S2STransformerPrelnWrapper,
}


def load_best_params(wrapper_key, params_file=None):
    path = (
        Path(params_file)
        if params_file
        else PROJECT_ROOT
        / "data"
        / "results"
        / "optuna"
        / WRAPPERS[wrapper_key]().name
        / "best_trial.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Optuna result not found: {path}. Run optuna_all_hyperparameters.py first."
        )
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    return path, result["best_params"]


def evaluate_wrapper(wrapper_key, dataset, params_file, epochs, input_steps, output_steps):
    params_path, best_params = load_best_params(wrapper_key, params_file)
    train_df, val_df = split_dataset(dataset)
    train_end = len(train_df)
    val_end = train_end + len(val_df)

    K.clear_session()
    wrapper = WRAPPERS[wrapper_key]()
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
        "test_samples": len(actuals),
        "mae": mean_absolute_error(actuals, predictions),
        "mse": mean_squared_error(actuals, predictions),
        "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
        "r2": r2_score(actuals, predictions),
        "prediction_time_sec": elapsed,
    }
    results_dir = PROJECT_ROOT / "data" / "results" / "wfo_optuna" / wrapper.name
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions, "actual": actuals}).to_csv(
        results_dir / "predictions.csv", index=False
    )
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
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
    dataset = load_dataset()
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
