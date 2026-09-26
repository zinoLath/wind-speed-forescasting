"""Shared helpers for training, evaluation and the pipeline stages.

This module is side-effect free at import time: TensorFlow is configured
explicitly by ``setup_tensorflow`` before any model class is imported. It is
the single source of truth for GPU setup, dataset loading, the wrapper
registry, hyperparameter adapters and metrics.
"""

import ctypes
import glob
import importlib
import json
import os
import re
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


# --- Canonical feature naming standard --------------------------------------
# Every stored or derived feature uses one short lowercase pattern:
#   ws{h}      wind speed at height h (m/s)            canonical, kept
#   v{h}       vertical wind component at height h     canonical, kept
#   dir{h}     wind direction at height h (degrees)    canonical, kept
#   disp{h}    wind-speed dispersion at height h       dropped by load_dataset
#   vdisp{h}   vertical dispersion at height h         dropped by load_dataset
# Derived columns append suffixes: ws{h}_wavelet, dir{h}_sin, dir{h}_cos,
# hour_sin, hour_cos, doy_sin, doy_cos. Raw instrument files may carry legacy
# aliases (wdir{h}, verts{h}, wdisp{h}, vertdisp{h}); canonicalize_columns
# maps them to the standard so consumers never see two names for one feature.
CANONICAL_HEIGHTS = (40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140)
RAW_HEIGHTS = CANONICAL_HEIGHTS + (150, 160, 170, 180, 190, 200, 220, 240, 260)
COLUMN_ALIASES = {
    f"{legacy}{height}": f"{canonical}{height}"
    for height in RAW_HEIGHTS
    for legacy, canonical in (
        ("wdir", "dir"),
        ("verts", "v"),
        ("wdisp", "disp"),
        ("vertdisp", "vdisp"),
    )
}
# Non-feature columns of the forecast dataset (metadata/instrument columns).
METADATA_COLUMNS = ("year", "month", "day", "hour", "minute", "id", "press", "humid", "temp")


def canonicalize_columns(df):
    """Rename legacy feature columns to the canonical standard (in place).

    Unknown columns are left untouched, so the function is safe to apply to
    any DataFrame (forecast dataset, wind data, imputed outputs).
    """
    return df.rename(columns={col: COLUMN_ALIASES[col] for col in df.columns if col in COLUMN_ALIASES})


def forecast_redundant_columns(columns, keep=()):
    """Columns of the forecast dataset that never feed the models.

    Covers metadata columns, the shear indicators (``cis*``), the dispersion
    families and every ws/v/dir column outside the canonical heights.
    """
    keep = set(keep)
    drop = set(METADATA_COLUMNS)
    for col in columns:
        name = str(col)
        if name.startswith("cis"):
            drop.add(name)
            continue
        match = re.match(r"^(ws|v|dir|disp|vdisp)(\d+)$", name)
        if not match:
            continue
        prefix, height = match.group(1), int(match.group(2))
        if (prefix in ("disp", "vdisp") or height not in CANONICAL_HEIGHTS) \
                and name not in keep:
            drop.add(name)
    return drop


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
    "lstm_original": ("src.models.s2s_lstm_original_wrapper", "S2SLSTMOriginalWrapper"),
    "lstm": ("src.models.s2s_lstm_wrapper", "S2SLSTMWrapper"),
    "lstm_bi": ("src.models.s2s_lstm_bi_wrapper", "S2SLSTMBidirectionalWrapper"),
    "gru": ("src.models.s2s_gru_wrapper", "S2SGRUWrapper"),
    "gru_bi": ("src.models.s2s_gru_bi_wrapper", "S2SGRUBidirectionalWrapper"),
    "lstm_cnn": ("src.models.s2s_lstm_cnn_wrapper", "S2SLSTMCNNWrapper"),
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


def validate_wrapper_names():
    """Ensure every registered wrapper has a unique ``.name``.

    Wrapper names double as results-directory names, so duplicates would make
    runs silently overwrite each other's artifacts.
    """
    names = [wrapper_factory(key)().name for key in WRAPPERS]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate wrapper names in the registry: {duplicates}")


class FixedHyperParameters:
    """Adapter that feeds a fixed dict of hyperparameters into a wrapper.build().

    Names missing from the dict fall back to the ``default`` declared by the
    wrapper, so hyperparameter dicts saved by older Optuna searches stay
    compatible when the search space grows.
    """

    def __init__(self, values):
        self.values = dict(values)

    def _get(self, name, default=None):
        if name == "encoder_filters" and "filters_power" in self.values:
            return 2 ** int(self.values["filters_power"])
        if name == "decoder_filters" and name not in self.values:
            return self._get("encoder_filters")
        if name not in self.values:
            if default is not None:
                return default
            raise KeyError(f"Hyperparameter '{name}' is missing.")
        return self.values[name]

    def Float(self, name, min_value, max_value, step=None, sampling=None, default=None):
        return float(self._get(name, default))

    def Int(self, name, min_value, max_value, step=1, default=None):
        return int(self._get(name, default))

    def Choice(self, name, values, default=None):
        return self._get(name, default)


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


def default_callbacks(wrapper, patience, min_lr=1e-6):
    """Standard training callbacks for a prepared/built wrapper.

    ``EarlyStopping`` always monitors ``val_loss``. ``ReduceLROnPlateau`` is
    only attached when the wrapper chose a constant LR: with a
    ``warmup_cosine`` schedule the annealing is already handled, and Keras 3
    makes the LR of a schedule-based optimizer read-only, so the plateau
    callback would raise ``TypeError`` on the first plateau.
    """
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
    ]
    if getattr(wrapper, "lr_schedule_mode", "constant") != "warmup_cosine":
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(2, patience // 2),
                min_lr=min_lr,
            )
        )
    return callbacks


def optuna_pruning_callback(trial):
    """Keras callback that reports val_mae to Optuna for trial pruning.

    The reported value must be comparable across trials: ``val_loss`` is
    sampled per trial (mse/mae/huber) and each function lives on a different
    scale, which biases the pruner by loss choice. ``val_mae`` is computed by
    every wrapper (``metrics=['mae']``) in scaled units under the inference
    decoder convention, so it is a single consistent signal for pruning.

    Created through a factory so this module stays import-free of TensorFlow.
    """
    from keras.callbacks import Callback

    class _PruningCallback(Callback):
        def __init__(self, trial):
            super().__init__()
            self.trial = trial

        def on_epoch_end(self, epoch, logs=None):
            val_mae = (logs or {}).get("val_mae")
            if val_mae is None:
                return
            self.trial.report(float(val_mae), step=epoch)
            if self.trial.should_prune():
                import optuna

                raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")

    return _PruningCallback(trial)


# Hyperparameters used when no Optuna result is available. ``loss``,
# ``lr_schedule`` and ``weight_decay`` mirror the searchable keys added to
# every wrapper; older best_trial.json files without them fall back to these
# values through FixedHyperParameters.
DEFAULT_HYPERPARAMETERS = {
    "lstm_original": {
        "learning_rate": 0.001,
        "lstm_units": 256,
        "encoder_dropout_rate": 0.1,
        "decoder_dropout_rate": 0.1,
        "loss": "mse",
        "lr_schedule": "constant",
        "weight_decay": 0.0,
    },
    "lstm": {
        "learning_rate": 0.001249176597990083,
        "lstm_units": 256,
        "encoder_dropout_rate": 0.1,
        "decoder_dropout_rate": 0.1,
        "loss": "mse",
        "lr_schedule": "constant",
        "weight_decay": 0.0,
    },
    "lstm_bi": {
        "learning_rate": 0.006403023029268099,
        "lstm_units": 64,
        "encoder_dropout_rate": 0.05,
        "decoder_dropout_rate": 0.15,
        "loss": "mse",
        "lr_schedule": "constant",
        "weight_decay": 0.0,
    },
    "gru": {
        "learning_rate": 0.001249176597990083,
        "gru_units": 256,
        "encoder_dropout_rate": 0.1,
        "decoder_dropout_rate": 0.1,
        "loss": "mse",
        "lr_schedule": "constant",
        "weight_decay": 0.0,
    },
    "gru_bi": {
        "learning_rate": 0.006403023029268099,
        "gru_units": 64,
        "encoder_dropout_rate": 0.05,
        "decoder_dropout_rate": 0.15,
        "loss": "mse",
        "lr_schedule": "constant",
        "weight_decay": 0.0,
    },
    "lstm_cnn": {
        "learning_rate": 0.001,
        "encoder_layers": 1,
        "lstm_units": 128,
        "encoder_dropout_rate": 0.1,
        "decoder_filters": 48,
        "decoder_kernel_size": 2,
        "decoder_nb_stacks": 1,
        "decoder_dropout_rate": 0.1,
        "decoder_dilation_rate": 4,
        "loss": "mse",
        "lr_schedule": "constant",
        "weight_decay": 0.0,
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
        "loss": "mse",
        "lr_schedule": "constant",
        "weight_decay": 0.0,
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
        "loss": "mse",
        "lr_schedule": "constant",
        "weight_decay": 0.0,
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
        "loss": "mse",
        "lr_schedule": "constant",
        "weight_decay": 0.0,
    },
    "transformer": {
        "learning_rate": 0.0006,
        "d_model": 128,
        "num_heads": 8,
        "num_layers": 3,
        "ff_dim": 256,
        "dropout_rate": 0.1,
        "loss": "mse",
        "lr_schedule": "warmup_cosine",
        "weight_decay": 0.0,
    },
}


def add_cyclic_features(dataset):
    """Add calendar and wind-direction cyclic features, in place.

    - ``hour_sin``/``hour_cos``: position within the day (period 24 h);
    - ``doy_sin``/``doy_cos``: position within the year (period 365 days);
    - ``dir{h}_sin``/``dir{h}_cos``: wind direction at height *h* encoded on
      the unit circle (raw degree columns are dropped, removing the 359°→1°
      discontinuity).

    Requires a DatetimeIndex. Safe to call more than once.
    """
    if not isinstance(dataset.index, pd.DatetimeIndex):
        return dataset

    if "hour_sin" not in dataset.columns:
        minutes_of_day = dataset.index.hour * 60 + dataset.index.minute
        hour_angle = 2 * np.pi * minutes_of_day / (24 * 60)
        dataset["hour_sin"] = np.sin(hour_angle)
        dataset["hour_cos"] = np.cos(hour_angle)

    if "doy_sin" not in dataset.columns:
        year_angle = 2 * np.pi * (dataset.index.dayofyear - 1) / 365.0
        dataset["doy_sin"] = np.sin(year_angle)
        dataset["doy_cos"] = np.cos(year_angle)

    if "day_sin" not in dataset.columns:
        # Ciclo semanal (dataset de ~52 dias; dia do mês quase não cicla).
        day_angle = 2 * np.pi * dataset.index.dayofweek / 7.0
        dataset["day_sin"] = np.sin(day_angle)
        dataset["day_cos"] = np.cos(day_angle)

    direction_columns = [
        col for col in dataset.columns
        if col.startswith("dir") and col[3:].isdigit() and not col.endswith(("_sin", "_cos"))
    ]
    for col in direction_columns:
        radians = np.deg2rad(pd.to_numeric(dataset[col], errors="coerce"))
        dataset[f"{col}_sin"] = np.sin(radians)
        dataset[f"{col}_cos"] = np.cos(radians)
    if direction_columns:
        dataset = dataset.drop(columns=direction_columns)
    return dataset


def load_dataset(path, keep_raw=()):
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

    dataset = canonicalize_columns(dataset)
    redundant = forecast_redundant_columns(dataset.columns, keep=keep_raw)
    dataset = dataset.drop(
        columns=[c for c in redundant if c in dataset.columns], errors="ignore"
    )

    dataset = add_cyclic_features(dataset)
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
        features=getattr(wrapper, "features", None),
        decoder_extra=getattr(wrapper, "decoder_extra", None),
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


def find_optuna_result(
    wrapper_key=None,
    tmp_dir="pipeline/tmp",
    results_dir="data/results",
    wrapper_name=None,
):
    """Locate an Optuna best_trial.json for a wrapper, checking tmp then results.

    The result directory is named after the wrapper's ``.name`` attribute,
    taken from the registry via *wrapper_key* or given directly as
    *wrapper_name* (for wrappers outside the registry).
    """
    if wrapper_name is None:
        wrapper_name = wrapper_factory(wrapper_key)().name
    for base in (tmp_dir, results_dir):
        path = resolve(Path(base) / "optuna" / wrapper_name / "best_trial.json")
        if path.is_file():
            return path
    return None


def load_best_params(wrapper_key=None, params_file=None, wrapper_name=None):
    """Load the best hyperparameters saved by an Optuna search.

    Returns (path, best_params).
    """
    path = (
        Path(params_file)
        if params_file
        else find_optuna_result(wrapper_key, wrapper_name=wrapper_name)
    )
    if path is None or not path.is_file():
        raise FileNotFoundError(f"No Optuna result found for wrapper '{wrapper_key or wrapper_name}'.")
    with open(path, encoding="utf-8") as handle:
        return path, json.load(handle)["best_params"]


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
        path = find_optuna_result(
            wrapper_key, config["paths"]["tmp_dir"], config["paths"]["results_dir"]
        )
        if path is not None:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)["best_params"], str(path)
        raise FileNotFoundError(
            f"No Optuna result found for {wrapper_key}. Run the optuna stage first "
            "or set params_source to 'default'."
        )

    return dict(DEFAULT_HYPERPARAMETERS[wrapper_key]), "built-in defaults"