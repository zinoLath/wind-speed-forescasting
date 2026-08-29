"""Shared helpers for the pipeline stages.

This module is side-effect free at import time: TensorFlow is configured
explicitly by ``setup_tensorflow`` before any model class is imported.
"""

import ctypes
import glob
import importlib
import json
import os
import site
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root_on_path():
    """Make the project root importable when a stage runs as a standalone script."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def resolve(path):
    """Resolve a config path; relative paths are relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_tensorflow_gpu_runtime():
    """Load the CUDA libraries bundled with the venv before importing TensorFlow."""
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

    valid_dirs = [path for path in dict.fromkeys(candidate_dirs) if os.path.isdir(path)]
    if not valid_dirs:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(
        valid_dirs + ([current_ld] if current_ld else [])
    )
    for lib_name in ("libcudart.so.12", "libcublas.so.12", "libcudnn.so.9", "libcusolver.so.11"):
        for lib_dir in valid_dirs:
            lib_path = os.path.join(lib_dir, lib_name)
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


def setup_tensorflow(gpu_enabled=True):
    """Configure CUDA libraries and GPU memory growth, then import TensorFlow."""
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if gpu_enabled:
        configure_tensorflow_gpu_runtime()

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            pass
    return tf


WRAPPERS = {
    "lstm": ("src.models.s2s_lstm_wrapper", "S2SLSTMWrapper"),
    "lstm_bi": ("src.models.s2s_lstm_bi_wrapper", "S2SLSTMBidirectionalWrapper"),
    "tcn": ("src.models.s2s_tcn_wrapper", "S2STCNWrapper"),
    "tcn_bi": ("src.models.s2s_tcn_bi_wrapper", "S2STCNBidirectionalWrapper"),
    "tcn_lstm": ("src.models.s2s_tcn_lstm_wrapper", "S2STCNLSTMWrapper"),
    "transformer": ("src.models.s2s_transformer_preln_wrapper", "S2STransformerPrelnWrapper"),
}


def wrapper_factory(key):
    """Return the wrapper class for a wrapper key (imported lazily)."""
    module_path, class_name = WRAPPERS[key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class FixedHyperParameters:
    """Adapter that feeds a fixed dict of hyperparameters into a wrapper.build()."""

    def __init__(self, values):
        self.values = dict(values)

    def _get(self, name):
        if name == "encoder_filters" and "filters_power" in self.values:
            return 2 ** int(self.values["filters_power"])
        if name == "decoder_filters" and name not in self.values:
            return self._get("encoder_filters")
        if name not in self.values:
            raise KeyError(f"Hyperparameter '{name}' is missing.")
        return self.values[name]

    def Float(self, name, min_value, max_value, step=None, sampling=None, default=None):
        return float(self._get(name))

    def Int(self, name, min_value, max_value, step=1, default=None):
        return int(self._get(name))

    def Choice(self, name, values, default=None):
        return self._get(name)


class OptunaHyperParameters:
    """Adapter that maps the wrapper hyperparameter API onto an Optuna trial."""

    def __init__(self, trial):
        self.trial = trial

    def Float(self, name, min_value, max_value, step=None, sampling=None, default=None):
        if sampling == "LOG":
            return self.trial.suggest_float(name, min_value, max_value, log=True)
        return self.trial.suggest_float(name, min_value, max_value, step=step)

    def Int(self, name, min_value, max_value, step=1, default=None):
        return self.trial.suggest_int(name, min_value, max_value, step=step)

    def Choice(self, name, values, default=None):
        return self.trial.suggest_categorical(name, values)


# Hyperparameters used when no Optuna result is available.
DEFAULT_HYPERPARAMETERS = {
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
    "tcn_lstm": {
        "learning_rate": 0.001,
        "encoder_filters": 128,
        "encoder_kernel_size": 3,
        "encoder_nb_stacks": 1,
        "encoder_dropout_rate": 0.1,
        "encoder_dilation_rate": 4,
        "lstm_units": 128,
        "decoder_dropout_rate": 0.1,
    },
    "transformer": {
        "learning_rate": 0.0006,
        "d_model": 128,
        "num_heads": 8,
        "num_layers": 3,
        "ff_dim": 256,
        "dropout_rate": 0.1,
    },
}


def load_dataset(path):
    """Load a forecast dataset (dataset.csv) into a clean DataFrame.

    Returns a DataFrame indexed by timestamp with the heavy redundant columns
    removed and directions/verticals renamed to short names.
    """
    dataset = pd.read_csv(path)

    for col in dataset.columns:
        if col in {"year", "month", "day", "hour", "minute", "id"}:
            continue
        if pd.api.types.is_numeric_dtype(dataset[col]):
            continue
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
        median_value = dataset[col].median()
        if pd.notna(median_value):
            dataset[col] = dataset[col].fillna(median_value)

    dataset["timestamp"] = pd.to_datetime(dataset["id"], format="mixed")
    dataset = dataset.sort_values("timestamp").set_index("timestamp")

    redundant_columns = [
        "year", "month", "day", "hour", "minute", "press", "humid", "temp", "id",
        "cis1", "cis2", "cis3", "cis4", "cis5", "cis6", "cis7", "cis8", "cis9",
        "cis10", "cis11", "cis12", "cis13", "cis14", "cis15", "cis16", "cis17",
        "cis18", "cis19",
        "wdisp40", "wdisp50", "wdisp60", "wdisp70", "wdisp80", "wdisp90",
        "wdisp100", "wdisp110", "wdisp120", "wdisp130", "wdisp140", "wdisp150",
        "wdisp160", "wdisp170", "wdisp180", "wdisp190", "wdisp200", "wdisp220",
        "wdisp240", "wdisp260",
        "vertdisp40", "vertdisp50", "vertdisp60", "vertdisp70", "vertdisp80",
        "vertdisp90", "vertdisp100", "vertdisp110", "vertdisp120", "vertdisp130",
        "vertdisp140", "vertdisp150", "vertdisp160", "vertdisp170", "vertdisp180",
        "vertdisp190", "vertdisp200", "vertdisp220", "vertdisp240", "vertdisp260",
        "wdir150", "wdir160", "wdir170", "wdir180", "wdir190", "wdir200",
        "wdir220", "wdir240", "wdir260",
        "verts150", "verts160", "verts170", "verts180", "verts190", "verts200",
        "verts220", "verts240", "verts260",
        "ws150", "ws160", "ws170", "ws180", "ws190", "ws200", "ws220", "ws240",
        "ws260",
    ]
    dataset = dataset.drop(
        columns=[c for c in redundant_columns if c in dataset.columns], errors="ignore"
    )

    heights = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    rename = {f"wdir{h}": f"dir{h}" for h in heights}
    rename.update({f"verts{h}": f"v{h}" for h in heights})
    dataset = dataset.rename(columns=rename)

    dataset = dataset.apply(pd.to_numeric, errors="coerce")
    dataset = dataset.interpolate(limit_direction="both").ffill().bfill()
    return dataset


def split_dataset(dataset, train_ratio=0.75, val_ratio=0.20):
    """Split a dataset into train, validation and test slices (in chronological order)."""
    train_end = int(len(dataset) * train_ratio)
    val_end = train_end + int(len(dataset) * val_ratio)
    return (
        dataset.iloc[:train_end].copy(),
        dataset.iloc[train_end:val_end].copy(),
        dataset.iloc[val_end:].copy(),
    )


def compute_metrics(actual, predicted):
    """Compute a full set of regression metrics for a pair of arrays."""
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()

    if len(actual) == 0:
        return {
            "samples": 0, "mae": np.nan, "mse": np.nan, "rmse": np.nan,
            "nmse": np.nan, "nrmse": np.nan, "nmae": np.nan, "r2": np.nan,
        }

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mean_abs = np.mean(np.abs(actual))
    mean_sq = np.mean(actual ** 2)

    return {
        "samples": len(actual),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "nmse": float(mse / mean_sq) if mean_sq > 0 else np.nan,
        "nrmse": float(rmse / mean_abs) if mean_abs > 0 else np.nan,
        "nmae": float(mae / mean_abs) if mean_abs > 0 else np.nan,
        "r2": float(r2_score(actual, predicted)) if np.var(actual) > 0 else np.nan,
    }


def predict_all_horizons(wrapper, eval_df):
    """Predict every forecast horizon for every origin inside *eval_df*.

    *eval_df* must be a DataFrame with the same columns the wrapper was
    prepared on. Returns a long-format DataFrame with origin, timestamp,
    horizon, actual, predicted and persistence columns. Decoder inputs are
    set to the inference-time value (last observed target) when the wrapper
    uses teacher forcing.
    """
    input_steps = wrapper.input_steps
    output_steps = wrapper.output_steps

    prepared, _ = wrapper.prepare_data(
        eval_df,
        input_steps,
        output_steps,
        wrapper.target_col,
        scaler_target=wrapper.scaler_target,
        scaler_other=wrapper.scaler_other,
        denoise=wrapper.denoise,
        decoder_mode=wrapper.decoder_mode,
        target_mode=wrapper.target_mode,
    )
    if wrapper.decoder_mode == "teacher_forcing":
        prepared["X_decoder"][:, :, 0] = prepared["X_decoder"][:, :1, 0]

    predicted_scaled = wrapper.model.predict(
        [prepared["X_encoder"], prepared["X_decoder"]], verbose=0
    )[:, :, 0]
    actual_scaled = prepared["y_decoder"][:, :, 0]
    persistence_scaled = prepared["X_encoder"][:, -1, wrapper.target_col_index]

    if wrapper.target_mode == "residual":
        predicted_scaled = predicted_scaled + persistence_scaled[:, None]
        actual_scaled = actual_scaled + persistence_scaled[:, None]

    def _inverse(values):
        return wrapper.scaler_target.inverse_transform(
            values.reshape(-1, 1)
        ).reshape(values.shape)

    predicted = _inverse(predicted_scaled)
    actual = _inverse(actual_scaled)
    persistence = wrapper.scaler_target.inverse_transform(
        persistence_scaled.reshape(-1, 1)
    ).ravel()

    sequence_count = len(predicted)
    target_positions = (
        np.arange(sequence_count)[:, None]
        + input_steps
        + np.arange(output_steps)[None, :]
    )
    origins = eval_df.index.to_numpy()[np.arange(sequence_count) + input_steps - 1]

    return pd.DataFrame(
        {
            "origin": np.repeat(origins, output_steps),
            "timestamp": eval_df.index.to_numpy()[target_positions].ravel(),
            "horizon": np.tile(np.arange(1, output_steps + 1), sequence_count),
            "actual": actual.ravel(),
            "predicted": predicted.ravel(),
            "persistence": np.repeat(persistence, output_steps),
        }
    )


def horizon_metrics(df):
    """Aggregate metrics per forecast horizon (with a persistence MAE column)."""
    rows = []
    for horizon, group in df.groupby("horizon"):
        row = compute_metrics(group["actual"], group["predicted"])
        row["horizon"] = int(horizon)
        row["persistence_mae"] = compute_metrics(
            group["actual"], group["persistence"]
        )["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_json(path, data):
    """Write a dict as pretty JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def find_optuna_result(wrapper_key, config):
    """Locate an Optuna best_trial.json for a wrapper, checking tmp then results."""
    wrapper_name = wrapper_factory(wrapper_key)().name
    candidates = [
        Path(config["paths"]["tmp_dir"]) / "optuna" / wrapper_name / "best_trial.json",
        Path(config["paths"]["results_dir"]) / "optuna" / wrapper_name / "best_trial.json",
    ]
    for candidate in candidates:
        path = resolve(candidate)
        if path.is_file():
            return path
    return None


def resolve_hyperparameters(wrapper_key, section, config):
    """Resolve hyperparameters for a wrapper from explicit config, Optuna, or defaults.

    *section* is a stage config dict with ``params_source`` and ``hyperparameters``
    keys. Returns (hyperparameters, source_description).
    """
    cfg = section["hyperparameters"]
    source = section["params_source"]

    if source == "explicit":
        if not cfg:
            raise ValueError("params_source='explicit' requires 'hyperparameters' in the config.")
        return dict(cfg), "explicit config"

    if source == "optuna":
        path = find_optuna_result(wrapper_key, config)
        if path is not None:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)["best_params"], str(path)
        raise FileNotFoundError(
            f"No Optuna result found for {wrapper_key}. Run the optuna stage first "
            "or set params_source to 'default'."
        )

    return dict(DEFAULT_HYPERPARAMETERS[wrapper_key]), "built-in defaults"