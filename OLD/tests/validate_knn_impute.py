"""Validate KNN imputation for wind_data.csv by injecting artificial gaps.

This script uses a per-row (multivariate) imputation: KNNImputer fills each missing
value using the other non-missing columns at the same timestamp. Direction columns
(`dir*`) are converted to sin/cos before imputation and reconstructed afterwards to
handle the 0/360 wrap-around.

Usage:
    python tests/validate_knn_impute.py
    python tests/validate_knn_impute.py --n-samples 20 --gap-lengths 1 2 3 5 10 --neighbors 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.validate_impute_wind_data import (
    _direction_error,
    find_complete_segments,
    inject_gaps,
    load_wind_data,
    _is_direction_column,
)


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


def _impute_knn(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    weights: str = "distance",
) -> pd.DataFrame:
    """Impute missing values per-row using KNNImputer.

    Rows that are entirely NaN cannot be imputed row-wise and are left unchanged.
    """
    direction_cols = [c for c in df.columns if _is_direction_column(c)]
    transformed = _direction_to_components(df, direction_cols)

    numeric_cols = transformed.select_dtypes(include=[np.number]).columns.tolist()
    transformed_numeric = transformed[numeric_cols]

    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed_values = imputer.fit_transform(transformed_numeric)
    imputed = pd.DataFrame(
        imputed_values, index=transformed.index, columns=transformed_numeric.columns
    )

    # Reattach any non-numeric columns (e.g., timestamp) if present.
    for col in transformed.columns:
        if col not in imputed.columns:
            imputed[col] = transformed[col]

    imputed = _components_to_direction(imputed, direction_cols)

    # Ensure only originally numeric columns remain in the output.
    return imputed[df.columns]


def evaluate_knn_imputation(
    corrupted: pd.DataFrame,
    ground_truth: dict[tuple[int, int, str], float],
    n_neighbors: int = 5,
    weights: str = "distance",
) -> dict:
    """Run KNN imputation and compute errors."""
    imputed = _impute_knn(corrupted, n_neighbors=n_neighbors, weights=weights)

    gap_info = {}
    for (gap_id, idx, col) in ground_truth.keys():
        gap_info.setdefault((gap_id, col), []).append(idx)

    records = []
    for (gap_id, col), rows in gap_info.items():
        gap_len = len(rows)
        for idx in rows:
            predicted = imputed.at[imputed.index[idx], col]
            if np.isnan(predicted):
                continue
            true_value = ground_truth[(gap_id, idx, col)]
            if _is_direction_column(col):
                error = _direction_error(true_value, predicted)
            else:
                error = abs(true_value - predicted)
            records.append(
                {
                    "gap_id": gap_id,
                    "gap_len": gap_len,
                    "row": idx,
                    "col": col,
                    "true": true_value,
                    "predicted": predicted,
                    "error": error,
                }
            )

    metrics = {}
    if not records:
        metrics["overall"] = {"samples": 0, "mae": None, "rmse": None}
        metrics["by_column"] = {}
        metrics["by_gap_length"] = {}
        return metrics

    df_errors = pd.DataFrame(records)
    overall_mae = mean_absolute_error(df_errors["true"], df_errors["predicted"])
    overall_rmse = np.sqrt(
        mean_squared_error(df_errors["true"], df_errors["predicted"])
    )
    metrics["overall"] = {
        "samples": len(df_errors),
        "mae": float(overall_mae),
        "rmse": float(overall_rmse),
    }
    metrics["by_column"] = (
        df_errors.groupby("col")
        .agg(
            samples=("col", "size"),
            mae=("error", "mean"),
            rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
        )
        .sort_values("mae")
        .to_dict(orient="index")
    )
    metrics["by_gap_length"] = (
        df_errors.groupby("gap_len")
        .agg(
            samples=("gap_len", "size"),
            mae=("error", "mean"),
            rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
        )
        .to_dict(orient="index")
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate KNN row-wise imputation by injecting artificial gaps."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "wind_data.csv",
        help="Path to the input CSV (default: data/wind_data.csv).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of artificial gaps to inject per gap length (default: 10).",
    )
    parser.add_argument(
        "--gap-lengths",
        type=int,
        nargs="*",
        default=[1, 2, 3, 5, 10],
        help="Gap lengths to test, in timesteps (default: 1 2 3 5 10).",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNNImputer (default: 5).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="distance",
        choices=["uniform", "distance"],
        help="Weight function for KNNImputer (default: distance).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible gap injection (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "knn_imputation_validation.json",
        help="Where to save the JSON report (default: data/knn_imputation_validation.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.input}...")
    df = load_wind_data(args.input)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Shape: {df.shape}, numeric columns: {len(numeric_cols)}")

    min_length = max(args.gap_lengths) + 100
    segments = find_complete_segments(df, min_length)
    if not segments:
        raise RuntimeError(
            f"No complete segment of length >= {min_length} found; reduce --gap-lengths."
        )

    start, end = max(segments, key=lambda s: s[1] - s[0])
    clean_df = df.iloc[start:end].copy()
    print(f"Using clean segment rows {start}:{end} ({len(clean_df)} rows)")

    print(
        f"Injecting {args.n_samples} gaps for each length {args.gap_lengths} "
        f"across {len(numeric_cols)} columns..."
    )
    corrupted, ground_truth = inject_gaps(
        clean_df, args.gap_lengths, args.n_samples, rng
    )

    print(f"Running KNN imputation (k={args.neighbors}, weights={args.weights})...")
    metrics = evaluate_knn_imputation(
        corrupted,
        ground_truth,
        n_neighbors=args.neighbors,
        weights=args.weights,
    )

    print("\nOverall metrics:")
    print(json.dumps(metrics["overall"], indent=2))
    print("\nMetrics by gap length:")
    for gap_len, vals in sorted(metrics["by_gap_length"].items()):
        print(f"  len={gap_len}: MAE={vals['mae']:.4f}, RMSE={vals['rmse']:.4f}, n={vals['samples']}")
    print("\nWorst columns by MAE:")
    by_col = sorted(
        metrics["by_column"].items(), key=lambda kv: kv[1]["mae"], reverse=True
    )
    for col, vals in by_col[:10]:
        print(f"  {col}: MAE={vals['mae']:.4f}, RMSE={vals['rmse']:.4f}, n={vals['samples']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "settings": {
                    "input": str(args.input),
                    "n_samples": args.n_samples,
                    "gap_lengths": args.gap_lengths,
                    "neighbors": args.neighbors,
                    "weights": args.weights,
                    "seed": args.seed,
                    "clean_segment": (int(start), int(end)),
                },
                "metrics": metrics,
            },
            handle,
            indent=2,
        )
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
