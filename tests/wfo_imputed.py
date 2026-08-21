"""Walk-forward validation with periodic retraining on imputed wind data.

This script trains each requested Seq2Seq model on the last N days of data and
evaluates every horizon of each 36-step forecast over the next M days.

The input CSV is expected to contain an ``imputed`` boolean flag column indicating
which rows contained originally missing values. The script records whether the input
window or the output sample used for comparison came from imputed data.

Results are written to ``data/wfo_results_imputed/<MODEL_NAME>/``:
    - ``predictions_window_<N>.csv`` — per-window timestamped predictions vs actuals
    - ``metrics.csv`` — per-window metrics
    - ``metrics.json`` — overall settings and aggregated metrics

Usage:
    python tests/wfo_imputed.py
    python tests/wfo_imputed.py --models lstm lstm_bi --train-window-days 60 --test-window-days 14 --step-days 7

Only the test script is written; it is not executed automatically.
"""

# Default parameters are coalesced at the top of the file for easy editing.
DEFAULT_INPUT_FILE = "data/wind_data_imputed_rf.csv"
DEFAULT_OUTPUT_DIR = "data/wfo_results_imputed"
DEFAULT_TRAIN_WINDOW_DAYS = 60
DEFAULT_TEST_WINDOW_DAYS = 7
DEFAULT_STEP_DAYS = 7
DEFAULT_INPUT_STEPS = 72
DEFAULT_OUTPUT_STEPS = 36
DEFAULT_TARGET_COL = "ws100"
DEFAULT_DENOISE_LEVEL = 1
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 12
DEFAULT_MODELS = ["lstm", "lstm_bi", "tcn", "tcn_bi"]
DEFAULT_RANDOM_STATE = 42
DEFAULT_RAW_INPUT_FILE = "data/wind_data.csv"

# Default hyperparameters used for each model when no Optuna results are supplied.
DEFAULT_HP = {
    "lstm": {
        "learning_rate": 0.001249176597990083,
        "lstm_units": 256,
        "encoder_dropout_rate": 0.1,
        "decoder_dropout_rate": 0.1,
    },
    "lstm_bi": {
        "learning_rate": 0.006403023029268099,
        "lstm_units": 64,
        "encoder_dropout_rate": 0.05,
        "decoder_dropout_rate": 0.15,
    },
    "tcn": {
        "learning_rate": 0.001249176597990083,
        "encoder_filters": 256,
        "encoder_kernel_size": 3,
        "encoder_nb_stacks": 2,
        "encoder_dropout_rate": 0.2,
        "encoder_dilation_rate": 4,
        "decoder_kernel_size": 4,
        "decoder_nb_stacks": 2,
        "decoder_dropout_rate": 0.1,
        "decoder_dilation_rate": 4,
    },
    "tcn_bi": {
        "learning_rate": 0.003969484893321028,
        "encoder_filters": 64,
        "encoder_kernel_size": 2,
        "encoder_nb_stacks": 1,
        "encoder_dropout_rate": 0.2,
        "encoder_dilation_rate": 2,
        "decoder_filters": 80,
        "decoder_kernel_size": 2,
        "decoder_nb_stacks": 2,
        "decoder_dropout_rate": 0.0,
        "decoder_dilation_rate": 1,
    },
}

import argparse
import ctypes
import glob
import json
import os
import site
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def configure_tensorflow_gpu_runtime():
    """Configure CUDA libraries before importing TensorFlow."""
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    candidate_dirs = []

    for site_packages in site.getsitepackages():
        candidate_dirs.extend(glob.glob(os.path.join(site_packages, "nvidia", "*", "lib")))

    if "VIRTUAL_ENV" in os.environ:
        candidate_dirs.extend(
            glob.glob(
                os.path.join(
                    os.environ["VIRTUAL_ENV"],
                    "lib",
                    "python*",
                    "site-packages",
                    "nvidia",
                    "*",
                    "lib",
                )
            )
        )

    valid_dirs = []
    for lib_dir in candidate_dirs:
        if os.path.isdir(lib_dir) and lib_dir not in valid_dirs:
            valid_dirs.append(lib_dir)

    if valid_dirs:
        merged = ":".join(valid_dirs)
        os.environ["LD_LIBRARY_PATH"] = f"{merged}:{current_ld}" if current_ld else merged

        preload_libs = [
            "libcudart.so.12",
            "libcublas.so.12",
            "libcudnn.so.9",
            "libcusolver.so.11",
        ]
        for lib_name in preload_libs:
            for lib_dir in valid_dirs:
                lib_path = os.path.join(lib_dir, lib_name)
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
                    break

    return valid_dirs


configure_tensorflow_gpu_runtime()

import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.models.s2s_lstm_bi_wrapper import S2SLSTMBidirectionalWrapper
from src.models.s2s_lstm_wrapper import S2SLSTMWrapper
from src.models.s2s_tcn_bi_wrapper import S2STCNBidirectionalWrapper
from src.models.s2s_tcn_wrapper import S2STCNWrapper
from src.utils import wavelet_denoising


WRAPPERS = {
    "lstm": S2SLSTMWrapper,
    "lstm_bi": S2SLSTMBidirectionalWrapper,
    "tcn": S2STCNWrapper,
    "tcn_bi": S2STCNBidirectionalWrapper,
}

RETAINED_MODES = {
    "lstm_bi": ("direct", "residual"),
    "tcn_bi": ("teacher_forcing", "absolute"),
}

METADATA_COLUMNS = ["imputed", "_target_imputed"]


class FixedHyperParameters:
    """Adapter that feeds a fixed dict of hyperparameters into a wrapper build()."""

    def __init__(self, values):
        self.values = values

    def _get(self, name):
        if name == "encoder_filters" and "filters_power" in self.values:
            return 2 ** int(self.values["filters_power"])
        if name == "decoder_filters" and name not in self.values:
            return self._get("encoder_filters")
        if name not in self.values:
            raise KeyError(f"Parameter '{name}' is missing.")

        return self.values[name]

    def Float(self, name, min_value, max_value, step=None, sampling=None, default=None):
        return float(self._get(name))

    def Int(self, name, min_value, max_value, step=1, default=None):
        return int(self._get(name))

    def Choice(self, name, values, default=None):
        return self._get(name)


def load_imputed_dataset(path: Path) -> pd.DataFrame:
    """Load the imputed dataset and ensure the imputed flag is boolean.

    Returns a DataFrame with ``timestamp`` as the DatetimeIndex so that
    non-numeric columns never leak into model training.
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    if "imputed" in df.columns:
        if df["imputed"].dtype != bool:
            df["imputed"] = df["imputed"].astype(str).str.lower().eq("true")
    else:
        df["imputed"] = False
    print(df.head())
    return df


def add_target_imputation_flag(
    df: pd.DataFrame, raw_path: Path | None, target_col: str
) -> pd.DataFrame:
    """Mark targets that were missing in the original, non-imputed dataset."""
    result = df.copy()
    if raw_path is None or not raw_path.is_file():
        raise FileNotFoundError(
            "The original dataset is required for observed-target metrics: "
            f"{raw_path}"
        )

    raw = pd.read_csv(raw_path, usecols=["timestamp", target_col])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    raw = raw.set_index("timestamp").reindex(result.index)
    result["_target_imputed"] = raw[target_col].isna()
    return result


def encode_direction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Represent circular direction measurements without a 0/360 discontinuity."""
    result = df.copy()
    direction_columns = [col for col in result.columns if col.startswith("dir")]
    for col in direction_columns:
        radians = np.deg2rad(result[col].to_numpy(dtype=float))
        result[f"{col}_sin"] = np.sin(radians)
        result[f"{col}_cos"] = np.cos(radians)
    return result.drop(columns=direction_columns)


def retain_observed_target_sequences(prepared: dict, frame: pd.DataFrame) -> int:
    """Remove sequences trained against any originally missing target value."""
    flags = frame["_target_imputed"].to_numpy(dtype=bool)
    input_steps = prepared["X_encoder"].shape[1]
    output_steps = prepared["y_decoder"].shape[1]
    valid = np.array(
        [
            not flags[i + input_steps:i + input_steps + output_steps].any()
            for i in range(len(prepared["X_encoder"]))
        ]
    )
    for key in ("X_encoder", "X_decoder", "y_decoder"):
        prepared[key] = prepared[key][valid]
    return int(valid.sum())


def load_hyperparameters(wrapper_key: str, source: str) -> tuple[dict, str]:
    if source == "default":
        return DEFAULT_HP[wrapper_key].copy(), "built-in defaults"

    wrapper_name = WRAPPERS[wrapper_key]().name
    path = PROJECT_ROOT / "data" / "results" / "optuna" / wrapper_name / "best_trial.json"
    if not path.is_file():
        raise FileNotFoundError(f"Optuna parameters not found: {path}")
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    return result["best_params"], str(path)


def add_wavelet_target(df: pd.DataFrame, target_col: str, denoise_level: int) -> pd.DataFrame:
    """Add a wavelet-denoised target column derived from the raw target column."""
    wavelet_col = f"{target_col}_wavelet"
    if wavelet_col not in df.columns:
        df = df.copy()
        df[wavelet_col] = wavelet_denoising(df[target_col].values, level=denoise_level)
    return df


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MSE, RMSE, MAE, NMSE, NRMSE, NMAE, and R2."""
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()

    if len(actual) == 0:
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "nmse": np.nan,
            "nrmse": np.nan,
            "nmae": np.nan,
            "r2": np.nan,
            "samples": 0,
        }

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    mean_abs_actual = np.mean(np.abs(actual))
    mean_sq_actual = np.mean(actual ** 2)

    nmse = mse / mean_sq_actual if mean_sq_actual > 0 else np.nan
    nrmse = rmse / mean_abs_actual if mean_abs_actual > 0 else np.nan
    nmae = mae / mean_abs_actual if mean_abs_actual > 0 else np.nan

    r2 = r2_score(actual, predicted) if np.var(actual) > 0 else np.nan

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "nmse": float(nmse),
        "nrmse": float(nrmse),
        "nmae": float(nmae),
        "r2": float(r2),
        "samples": len(actual),
    }


def compute_horizon_metrics(predictions: pd.DataFrame) -> list[dict]:
    results = []
    for horizon, group in predictions.groupby("horizon", sort=True):
        observed = ~group["output_imputed"].to_numpy(dtype=bool)
        metrics = compute_metrics(
            group.loc[observed, "actual"], group.loc[observed, "predicted"]
        )
        persistence = compute_metrics(
            group.loc[observed, "actual"], group.loc[observed, "persistence"]
        )
        metrics["horizon"] = int(horizon)
        metrics["persistence_mae"] = persistence["mae"]
        results.append(metrics)
    return results


def apply_mean_horizon_mae(metrics: dict, predictions: pd.DataFrame) -> list[dict]:
    by_horizon = compute_horizon_metrics(predictions)
    metrics["pooled_mae"] = metrics["mae"]
    metrics["mae"] = float(np.nanmean([item["mae"] for item in by_horizon]))
    return by_horizon


def train_and_predict_window(
    wrapper,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hp,
    epochs: int,
    batch_size: int,
    patience: int,
    input_steps: int,
    output_steps: int,
    target_col: str,
    denoise_level: int,
    window_idx: int,
    decoder_mode: str,
    denoise_target: bool,
    include_imputed_training_targets: bool,
    target_mode: str,
    loss_name: str,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Train a fresh model on train_df and produce rolling predictions over test_df.

    Returns one row per forecast origin and horizon, plus all-horizon metrics.
    """
    model_target = f"{target_col}_wavelet" if denoise_target else target_col
    denoise_columns = [target_col] if denoise_target else []

    # Split train into train/val for the wrapper.
    train_split = int(len(train_df) * 0.8)
    train_split_df = train_df.iloc[:train_split].copy()
    val_df = train_df.iloc[train_split:].copy()

    K.clear_session()
    tf.keras.utils.set_random_seed(seed)
    wrapper.prepare(
        train_split_df.drop(columns=METADATA_COLUMNS, errors="ignore"),
        val_df.drop(columns=METADATA_COLUMNS, errors="ignore"),
        input_steps=input_steps,
        output_steps=output_steps,
        target_col=model_target,
        denoise=denoise_columns,
        denoise_level=denoise_level,
        decoder_mode=decoder_mode,
        target_mode=target_mode,
    )
    if not include_imputed_training_targets:
        retain_observed_target_sequences(wrapper.train, train_split_df)
        retain_observed_target_sequences(wrapper.val, val_df)
    #print(wrapper.train)
    if decoder_mode == "teacher_forcing":
        # Early stopping must use the decoder inputs available at inference.
        wrapper.val["X_decoder"][:, :, 0] = wrapper.val["X_decoder"][:, :1, 0]
    wrapper.build(hp)
    if loss_name == "mae":
        wrapper.model.compile(
            optimizer=wrapper.model.optimizer, loss="mse", metrics=["mse"]
        )
    elif loss_name == "huber":
        wrapper.model.compile(
            optimizer=wrapper.model.optimizer,
            loss=tf.keras.losses.Huber(delta=0.05),
            metrics=["mae", "mse"],
        )

    history = wrapper.fit(
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience // 2), min_lr=1e-6),
        ],
        use_validation=True,
    )

    evaluation_df = pd.concat([train_df.iloc[-input_steps:], test_df])
    evaluation_numeric = evaluation_df.drop(columns=METADATA_COLUMNS, errors="ignore")
    prepared, _ = wrapper.prepare_data(
        evaluation_numeric,
        input_steps,
        output_steps,
        model_target,
        scaler_target=wrapper.scaler_target,
        scaler_other=wrapper.scaler_other,
        denoise=denoise_columns,
        decoder_mode=decoder_mode,
        target_mode=target_mode,
    )
    if decoder_mode == "teacher_forcing":
        prepared["X_decoder"][:, :, 0] = prepared["X_decoder"][:, :1, 0]
    predicted_scaled = wrapper.model.predict(
        [prepared["X_encoder"], prepared["X_decoder"]],
        batch_size=batch_size,
        verbose=0,
    )[:, :, 0]
    actual_scaled = prepared["y_decoder"][:, :, 0]
    persistence_scaled = prepared["X_encoder"][:, -1, wrapper.target_col_index]
    if target_mode == "residual":
        predicted_scaled = predicted_scaled + persistence_scaled[:, None]
        actual_scaled = actual_scaled + persistence_scaled[:, None]
    predictions = wrapper.scaler_target.inverse_transform(
        predicted_scaled.reshape(-1, 1)
    ).reshape(predicted_scaled.shape)
    actuals = wrapper.scaler_target.inverse_transform(
        actual_scaled.reshape(-1, 1)
    ).reshape(actual_scaled.shape)
    persistence = wrapper.scaler_target.inverse_transform(
        persistence_scaled.reshape(-1, 1)
    ).ravel()

    sequence_count = len(predictions)
    target_positions = (
        np.arange(sequence_count)[:, None]
        + input_steps
        + np.arange(output_steps)[None, :]
    )
    timestamps = evaluation_df.index.to_numpy()[target_positions]
    origins = evaluation_df.index.to_numpy()[
        np.arange(sequence_count) + input_steps - 1
    ]
    output_imputed_flags = evaluation_df["_target_imputed"].to_numpy()[target_positions]
    row_imputed = evaluation_df["imputed"].to_numpy(dtype=bool)
    input_imputed_flags = np.array(
        [row_imputed[i:i + input_steps].any() for i in range(sequence_count)]
    )

    pred_df = pd.DataFrame(
        {
            "origin": np.repeat(origins, output_steps),
            "timestamp": timestamps.ravel(),
            "horizon": np.tile(np.arange(1, output_steps + 1), sequence_count),
            "actual": actuals.ravel(),
            "predicted": predictions.ravel(),
            "persistence": np.repeat(persistence, output_steps),
            "input_imputed": np.repeat(input_imputed_flags, output_steps),
            "output_imputed": output_imputed_flags.ravel(),
        }
    )

    observed = ~pred_df["output_imputed"].to_numpy(dtype=bool)
    metrics = compute_metrics(pred_df.loc[observed, "actual"], pred_df.loc[observed, "predicted"])
    by_horizon = apply_mean_horizon_mae(metrics, pred_df)
    all_metrics = compute_metrics(actuals.ravel(), predictions.ravel())
    persistence_metrics = compute_metrics(
        pred_df.loc[observed, "actual"], pred_df.loc[observed, "persistence"]
    )
    persistence_metrics["pooled_mae"] = persistence_metrics["mae"]
    persistence_metrics["mae"] = float(
        np.nanmean([item["persistence_mae"] for item in by_horizon])
    )
    metrics.update({f"all_{key}": value for key, value in all_metrics.items()})
    metrics.update({f"persistence_{key}": value for key, value in persistence_metrics.items()})
    metrics["window"] = window_idx
    metrics["train_epochs"] = len(history.history.get("loss", []))

    return pred_df, metrics


def evaluate_wrapper(
    wrapper_key: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Run the full walk-forward evaluation for one model."""
    wrapper = WRAPPERS[wrapper_key]()
    decoder_mode = args.decoder_mode
    if decoder_mode == "auto":
        decoder_mode = RETAINED_MODES.get(
            wrapper_key, ("teacher_forcing", "absolute")
        )[0]
    target_mode = args.target_mode
    if target_mode == "auto":
        target_mode = RETAINED_MODES.get(
            wrapper_key, ("teacher_forcing", "absolute")
        )[1]
    loss_name = args.loss
    if loss_name == "auto":
        loss_name = "mse"
    hp_values, hp_source = load_hyperparameters(wrapper_key, args.params_source)
    if args.tcn_encoder_dilation_rate is not None and wrapper_key.startswith("tcn"):
        hp_values["encoder_dilation_rate"] = args.tcn_encoder_dilation_rate
    hp = FixedHyperParameters(hp_values)
    run_name = args.run_name or (
        f"{decoder_mode}_{target_mode}_{loss_name}_"
        f"{'wavelet' if args.denoise_target else 'raw'}_"
        f"train{args.train_window_days}_test{args.test_window_days}_seed{args.seed}"
    )
    model_dir = output_dir / wrapper.name / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    train_rows = int(args.train_window_days * 24 * 6)
    test_rows = int(args.test_window_days * 24 * 6)
    step_rows = int(args.step_days * 24 * 6)

    n = len(df)
    window_records = []
    all_predictions = []

    # First test window starts after the initial training window.
    test_start = train_rows
    window_idx = 0
    completed_windows = 0

    while test_start + test_rows <= n:
        if window_idx < args.start_window:
            window_idx += 1
            test_start += step_rows
            continue
        if args.max_windows is not None and completed_windows >= args.max_windows:
            break

        train_start = max(0, test_start - train_rows)
        train_end = test_start
        test_end = test_start + test_rows

        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        print(
            f"[{wrapper.name}] window {window_idx}: "
            f"train={df.index[train_start]}->{df.index[train_end - 1]}, "
            f"test={df.index[test_start]}->{df.index[test_end - 1]}"
        )

        window_start = time.perf_counter()
        pred_df, metrics = train_and_predict_window(
            wrapper,
            train_df,
            test_df,
            hp,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            target_col=args.target_col,
            denoise_level=args.denoise_level,
            window_idx=window_idx,
            decoder_mode=decoder_mode,
            denoise_target=args.denoise_target,
            include_imputed_training_targets=args.include_imputed_training_targets,
            target_mode=target_mode,
            loss_name=loss_name,
            seed=args.seed + window_idx,
        )
        elapsed = time.perf_counter() - window_start

        metrics.update(
            {
                "train_start": str(df.index[train_start]),
                "train_end": str(df.index[train_end - 1]),
                "test_start": str(df.index[test_start]),
                "test_end": str(df.index[test_end - 1]),
                "elapsed_sec": elapsed,
                "input_imputed_count": int(
                    pred_df.drop_duplicates("origin")["input_imputed"].sum()
                ),
                "output_imputed_count": int(pred_df["output_imputed"].sum()),
            }
        )
        window_records.append(metrics)
        all_predictions.append(pred_df)

        pred_df.to_csv(model_dir / f"predictions_window_{window_idx:04d}.csv", index=False)
        print(
            f"[{wrapper.name}] window {window_idx}: MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.6f}, "
            f"elapsed={elapsed:.1f}s"
        )

        window_idx += 1
        completed_windows += 1
        test_start += step_rows

    if not window_records:
        print(f"[{wrapper.name}] no windows generated; check window sizes.")
        return

    metrics_df = pd.DataFrame(window_records)
    metrics_df.to_csv(model_dir / "metrics.csv", index=False)

    if all_predictions:
        all_pred_df = pd.concat(all_predictions, ignore_index=True)
        all_pred_df.to_csv(model_dir / "predictions_all.csv", index=False)
        observed = ~all_pred_df["output_imputed"].astype(bool)
        overall = compute_metrics(
            all_pred_df.loc[observed, "actual"], all_pred_df.loc[observed, "predicted"]
        )
        overall["by_horizon"] = apply_mean_horizon_mae(overall, all_pred_df)
        overall["all"] = compute_metrics(all_pred_df["actual"], all_pred_df["predicted"])
        overall["persistence"] = compute_metrics(
            all_pred_df.loc[observed, "actual"], all_pred_df.loc[observed, "persistence"]
        )
        overall["persistence"]["pooled_mae"] = overall["persistence"]["mae"]
        overall["persistence"]["mae"] = float(
            np.nanmean(
                [item["persistence_mae"] for item in overall["by_horizon"]]
            )
        )
        overall["input_imputed_count"] = int(
            all_pred_df.drop_duplicates("origin")["input_imputed"].sum()
        )
        overall["output_imputed_count"] = int(all_pred_df["output_imputed"].sum())
    else:
        overall = {}

    summary = {
        "settings": {
            "wrapper": wrapper.name,
            "input_file": str(args.input),
            "train_window_days": args.train_window_days,
            "test_window_days": args.test_window_days,
            "step_days": args.step_days,
            "input_steps": args.input_steps,
            "output_steps": args.output_steps,
            "target_col": args.target_col,
            "denoise_level": args.denoise_level,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "decoder_mode": decoder_mode,
            "denoise_target": args.denoise_target,
            "circular_directions": args.circular_directions,
            "include_imputed_training_targets": args.include_imputed_training_targets,
            "target_mode": target_mode,
            "loss": loss_name,
            "params_source": hp_source,
            "resolved_params": hp_values,
            "seed": args.seed,
        },
        "windows": window_records,
        "overall": overall,
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"[{wrapper.name}] {len(window_records)} windows completed.")
    print(f"[{wrapper.name}] overall: {overall}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward validation with retraining on imputed wind data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_INPUT_FILE,
        help=f"Path to the imputed CSV (default: {DEFAULT_INPUT_FILE}).",
    )
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_RAW_INPUT_FILE,
        help="Original CSV used to identify whether the target itself was imputed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_OUTPUT_DIR,
        help=f"Directory for results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(WRAPPERS),
        default=DEFAULT_MODELS,
        help=f"Models to evaluate (default: {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--train-window-days",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_DAYS,
        help=f"Training window in days (default: {DEFAULT_TRAIN_WINDOW_DAYS}).",
    )
    parser.add_argument(
        "--test-window-days",
        type=int,
        default=DEFAULT_TEST_WINDOW_DAYS,
        help=f"Test window in days (default: {DEFAULT_TEST_WINDOW_DAYS}).",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=DEFAULT_STEP_DAYS,
        help=f"Step between consecutive test windows in days (default: {DEFAULT_STEP_DAYS}).",
    )
    parser.add_argument(
        "--input-steps",
        type=int,
        default=DEFAULT_INPUT_STEPS,
        help=f"Number of input timesteps (default: {DEFAULT_INPUT_STEPS}).",
    )
    parser.add_argument(
        "--output-steps",
        type=int,
        default=DEFAULT_OUTPUT_STEPS,
        help=f"Number of output timesteps per prediction (default: {DEFAULT_OUTPUT_STEPS}).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=DEFAULT_TARGET_COL,
        help=f"Raw target column (default: {DEFAULT_TARGET_COL}).",
    )
    parser.add_argument(
        "--denoise-level",
        type=int,
        default=DEFAULT_DENOISE_LEVEL,
        help=f"Wavelet denoising level (default: {DEFAULT_DENOISE_LEVEL}).",
    )
    parser.add_argument("--denoise-target", action="store_true")
    parser.add_argument("--circular-directions", action="store_true")
    parser.add_argument(
        "--exclude-imputed-training-targets",
        action="store_false",
        dest="include_imputed_training_targets",
        default=True,
    )
    parser.add_argument(
        "--decoder-mode",
        choices=["auto", "teacher_forcing", "direct"],
        default="auto",
    )
    parser.add_argument(
        "--target-mode", choices=["auto", "absolute", "residual"], default="auto"
    )
    parser.add_argument(
        "--loss", choices=["auto", "mse", "mae", "huber"], default="auto"
    )
    parser.add_argument(
        "--tcn-encoder-dilation-rate", type=int, choices=range(1, 6)
    )
    parser.add_argument(
        "--params-source", choices=["optuna", "default"], default="optuna"
    )
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs per window (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience (default: {DEFAULT_PATIENCE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed (default: {DEFAULT_RANDOM_STATE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    if args.step_days < args.test_window_days:
        raise ValueError("--step-days must be at least --test-window-days.")

    print(f"Loading {args.input}...")
    df = load_imputed_dataset(args.input)
    df = add_target_imputation_flag(df, args.raw_input, args.target_col)
    if args.circular_directions:
        df = encode_direction_columns(df)
    print(f"Dataset shape: {df.shape}")
    print(f"Rows with imputed values: {df['imputed'].sum()}")
    print(f"Timestamp range: {df.index.min()} to {df.index.max()}")

    for wrapper_key in args.models:
        evaluate_wrapper(wrapper_key, df, args, args.output_dir)


if __name__ == "__main__":
    main()
