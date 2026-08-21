"""Impute missing values in wind_data.csv using a per-row Random Forest model.

This script uses sklearn's IterativeImputer with a RandomForestRegressor to fill
missing values using the other columns at the same timestamp. Direction columns
(`dir*`) are converted to sin/cos components before imputation and reconstructed
afterwards to handle the 0/360° wrap-around.

Rows that are completely missing cannot be filled row-wise. For those rows the
script falls back to column-wise linear interpolation.

Usage:
    python src/impute_wind_data_rf.py
    python src/impute_wind_data_rf.py --input data/wind_data.csv --output data/wind_data_imputed_rf.csv --estimators 200 --max-iter 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impute_wind_data import _is_direction_column, load_wind_data


def _direction_to_components(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert direction columns in degrees to sin/cos components."""
    result = df.copy()
    for col in cols:
        if col in result.columns:
            rad = np.deg2rad(result[col].to_numpy(dtype=float))
            result[f"{col}_sin"] = np.sin(rad)
            result[f"{col}_cos"] = np.cos(rad)
            result.drop(columns=[col], inplace=True)
    return result


def _components_to_direction(df: pd.DataFrame, original_cols: list[str]) -> pd.DataFrame:
    """Convert sin/cos components back to direction columns in degrees."""
    result = df.copy()
    for col in original_cols:
        sin_col = f"{col}_sin"
        cos_col = f"{col}_cos"
        if sin_col in result.columns and cos_col in result.columns:
            angle = np.rad2deg(np.arctan2(result[sin_col], result[cos_col]))
            result[col] = np.mod(angle, 360.0)
            result.drop(columns=[sin_col, cos_col], inplace=True)
    return result


def _impute_rf(
    df: pd.DataFrame,
    n_estimators: int = 100,
    max_iter: int = 10,
    random_state: int = 42,
    verbose: int = 0,
    train_sample: int | None = None,
) -> pd.DataFrame:
    """Impute missing values per-row using IterativeImputer + RandomForestRegressor.

    If ``train_sample`` is set, the imputer is fitted on a random sample of that
    many rows and then applied to the full dataset. This keeps the runtime
    reasonable for large wind datasets while still learning the cross-column
    relationships.

    Rows that are entirely NaN are left as NaN so the caller can apply a fallback.
    """
    direction_cols = [c for c in df.columns if _is_direction_column(c)]
    transformed = _direction_to_components(df, direction_cols)

    numeric_cols = transformed.select_dtypes(include=[np.number]).columns.tolist()
    transformed_numeric = transformed[numeric_cols]

    rng = np.random.default_rng(random_state)
    if train_sample is not None and train_sample < len(transformed_numeric):
        sample_idx = rng.choice(
            transformed_numeric.index,
            size=train_sample,
            replace=False,
        )
        train_df = transformed_numeric.loc[sample_idx]
    else:
        train_df = transformed_numeric

    estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )
    imputer.fit(train_df)
    imputed_values = imputer.transform(transformed_numeric)
    imputed = pd.DataFrame(
        imputed_values, index=transformed.index, columns=transformed_numeric.columns
    )

    for col in transformed.columns:
        if col not in imputed.columns:
            imputed[col] = transformed[col]

    imputed = _components_to_direction(imputed, direction_cols)
    return imputed[df.columns]


def impute_dataframe_rf(
    df: pd.DataFrame,
    n_estimators: int = 100,
    max_iter: int = 10,
    random_state: int = 42,
    verbose: int = 0,
    train_sample: int | None = None,
) -> pd.DataFrame:
    """Impute missing values using RF, then fill any remaining NaNs with linear interpolation."""
    imputed = _impute_rf(
        df,
        n_estimators=n_estimators,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
        train_sample=train_sample,
    )

    # Rows that were completely NaN remain NaN because IterativeImputer cannot fill
    # them without any features. Fall back to column-wise linear interpolation.
    fully_missing_rows = imputed.isna().all(axis=1)
    if fully_missing_rows.any():
        print(
            f"{fully_missing_rows.sum()} rows were fully missing; "
            "falling back to column-wise linear interpolation for those rows."
        )
        imputed = imputed.interpolate(method="linear", axis=0, limit_direction="both")

    return imputed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Impute missing values in wind_data.csv using Random Forest (row-wise)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "wind_data.csv",
        help="Path to the input CSV (default: data/wind_data.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "wind_data_imputed_rf.csv",
        help="Path to the output CSV (default: data/wind_data_imputed_rf.csv).",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=100,
        help="Number of trees in RandomForestRegressor (default: 100).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="Number of IterativeImputer rounds (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level for IterativeImputer (default: 0).",
    )
    parser.add_argument(
        "--train-sample",
        type=int,
        default=5000,
        help="Number of rows to use for fitting the imputer (default: 5000). Set to 0 to use the full dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.input}...")
    df = load_wind_data(args.input)
    print(f"Shape after regularisation: {df.shape}")
    print(f"Missing values before imputation: {df.isna().sum().sum()}")

    train_sample = args.train_sample if args.train_sample > 0 else None
    print(
        f"Running Random Forest imputation "
        f"(trees={args.estimators}, max_iter={args.max_iter}, train_sample={args.train_sample})..."
    )
    imputed = impute_dataframe_rf(
        df,
        n_estimators=args.estimators,
        max_iter=args.max_iter,
        random_state=args.seed,
        verbose=args.verbose,
        train_sample=train_sample,
    )
    print(f"Missing values after imputation: {imputed.isna().sum().sum()}")

    # Add a flag column indicating whether any value in the row was originally missing.
    imputed = imputed.assign(imputed=df.isna().any(axis=1))
    print(f"Rows with at least one imputed value: {imputed['imputed'].sum()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imputed.to_csv(args.output)
    print(f"Imputed data saved to {args.output}")


if __name__ == "__main__":
    main()
