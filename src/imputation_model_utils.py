"""Shared model-based imputation for the focused wind variables."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.impute_wind_data import load_wind_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = ["ws100", "ws40", "v40", "dir40", "ws50"]
DIRECTION_COLUMNS = ["dir40"]
MODEL_NAMES = ["random_forest", "lightgbm", "neural_network"]
DEFAULT_CACHE = PROJECT_ROOT / "data" / "imputation_optuna" / "best_model.json"


def circular_mean_degrees(values: pd.Series) -> float:
    """Return the circular mean without adding sine/cosine model features."""
    radians = np.deg2rad(values.dropna().to_numpy(dtype=float))
    if len(radians) == 0:
        raise ValueError("Cannot calculate a direction mean from only missing values.")
    mean = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float(np.mod(np.rad2deg(mean), 360.0))


def wrap_offset(values: Any, center: float) -> Any:
    """Map angles to signed offsets in [-180, 180) around center."""
    return (values - center + 180.0) % 360.0 - 180.0


def center_directions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    centered = df.copy()
    centers = {}
    for col in DIRECTION_COLUMNS:
        centers[col] = circular_mean_degrees(centered[col])
        centered[col] = wrap_offset(centered[col], centers[col])
    return centered, centers


def restore_directions(df: pd.DataFrame, centers: dict[str, float]) -> pd.DataFrame:
    restored = df.copy()
    for col, center in centers.items():
        restored[col] = np.mod(restored[col] + center, 360.0)
    return restored


def build_estimator(model: str, params: dict[str, Any], seed: int) -> Any:
    excluded = {"max_iter", "hidden_layer_1", "hidden_layer_2", "estimator_max_iter"}
    estimator_params = {key: value for key, value in params.items() if key not in excluded}
    if model == "random_forest":
        return RandomForestRegressor(
            **estimator_params, random_state=seed, n_jobs=-1
        )
    if model == "lightgbm":
        return LGBMRegressor(
            **estimator_params,
            random_state=seed,
            n_jobs=-1,
            subsample_freq=1,
            verbosity=-1,
        )
    if model == "neural_network":
        layers = [int(params["hidden_layer_1"])]
        if int(params["hidden_layer_2"]) > 0:
            layers.append(int(params["hidden_layer_2"]))
        mlp = MLPRegressor(
            **estimator_params,
            hidden_layer_sizes=tuple(layers),
            max_iter=int(params["estimator_max_iter"]),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )
        regressor = make_pipeline(StandardScaler(), mlp)
        return TransformedTargetRegressor(
            regressor=regressor, transformer=StandardScaler()
        )
    raise ValueError(f"Unknown model: {model}")


def impute_focused(
    corrupted: pd.DataFrame,
    model: str,
    params: dict[str, Any],
    seed: int = 42,
    train_sample: int | None = 3000,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Impute the five focused columns with a cached model configuration."""
    missing = [col for col in TARGET_COLUMNS if col not in corrupted.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    centered, direction_centers = center_directions(corrupted[TARGET_COLUMNS])
    if train_sample is not None and train_sample < len(centered):
        required_indices = []
        for col in centered.columns:
            if centered[col].isna().any():
                index = centered.index[centered[col].isna()][0]
                if index not in required_indices:
                    required_indices.append(index)
        if train_sample < len(required_indices):
            raise ValueError(
                f"train_sample={train_sample} is too small to cover all "
                f"{len(required_indices)} columns with missing values."
            )
        remaining = centered.drop(index=required_indices)
        sampled = remaining.sample(
            n=train_sample - len(required_indices), random_state=seed
        )
        train = pd.concat([centered.loc[required_indices], sampled])
    else:
        train = centered

    iterative = IterativeImputer(
        estimator=build_estimator(model, params, seed),
        max_iter=int(params["max_iter"]),
        random_state=seed,
        skip_complete=True,
    )
    iterative.fit(train)
    values = iterative.transform(centered)
    result = pd.DataFrame(values, index=centered.index, columns=centered.columns)
    return restore_directions(result, direction_centers), direction_centers


def load_cached_params(path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Optuna cache not found: {path}. Run the comparison script with --optuna."
        )
    with path.open(encoding="utf-8") as handle:
        cached = json.load(handle)
    if cached.get("settings", {}).get("columns") != TARGET_COLUMNS:
        raise ValueError(f"Optuna cache has unexpected target columns: {path}")
    for model in MODEL_NAMES:
        params = cached.get(model, {}).get("params", {})
        if "max_iter" not in params:
            raise ValueError(f"Optuna cache is missing {model}.params: {path}")
    return cached


def generate_imputed_csv(model: str, description: str, default_output: Path) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "wind_data.csv"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-sample",
        type=int,
        default=3000,
        help="Rows used to fit the imputer; 0 uses all rows.",
    )
    args = parser.parse_args()

    data = load_wind_data(args.input)
    cached = load_cached_params(args.cache)
    original_missing = data[TARGET_COLUMNS].isna()
    train_sample = args.train_sample if args.train_sample > 0 else None
    print(f"Imputing {TARGET_COLUMNS} with {model}...")
    focused, _ = impute_focused(
        data[TARGET_COLUMNS],
        model,
        cached[model]["params"],
        seed=args.seed,
        train_sample=train_sample,
    )
    result = data.copy()
    result[TARGET_COLUMNS] = focused
    result["imputed"] = original_missing.any(axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)
    print(f"Imputed {int(original_missing.sum().sum())} values; saved {args.output}")
