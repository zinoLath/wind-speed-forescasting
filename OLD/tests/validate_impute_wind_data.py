"""Validate the wavelet imputation by injecting artificial gaps and measuring error.

The script selects contiguous segments of the original data where no missing values
exist, randomly masks short gaps of configurable lengths, runs the imputation
routine, and compares the imputed values to the held-out ground truth. Metrics are
reported per gap length and per column.

Usage:
    python tests/validate_impute_wind_data.py
    python tests/validate_impute_wind_data.py --input data/wind_data.csv --n-samples 20 --gap-lengths 1 2 3 5 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impute_wind_data import impute_dataframe, load_wind_data, _is_direction_column


def _direction_error(true: float, predicted: float) -> float:
    """Circular absolute error for 0-360 degree directions."""
    return abs((true - predicted + 180.0) % 360.0 - 180.0)


def find_complete_segments(df: pd.DataFrame, min_length: int) -> list[tuple[int, int]]:
    """Return index ranges [start, end) where no numeric column is missing."""
    complete = ~df.isna().any(axis=1)
    segments = []
    start = None
    for i, flag in enumerate(complete):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            if i - start >= min_length:
                segments.append((start, i))
            start = None
    if start is not None and len(complete) - start >= min_length:
        segments.append((start, len(complete)))
    return segments


def inject_gaps(
    df: pd.DataFrame,
    gap_lengths: list[int],
    samples_per_length: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[tuple[int, int, str], float]]:
    """Inject short NaN gaps into an otherwise complete dataframe.

    Returns a tuple (corrupted_df, ground_truth) where ground_truth maps
    (gap_id, row_index, column) to the original value.
    """
    corrupted = df.copy()
    ground_truth: dict[tuple[int, int, str], float] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_rows = len(df)

    gap_id = 0
    for gap_len in gap_lengths:
        for _ in range(samples_per_length):
            start = rng.integers(0, total_rows - gap_len)
            end = start + gap_len
            col = rng.choice(numeric_cols)
            for idx in range(start, end):
                ground_truth[(gap_id, idx, col)] = float(corrupted.at[corrupted.index[idx], col])
                corrupted.iat[idx, corrupted.columns.get_loc(col)] = np.nan
            gap_id += 1

    return corrupted, ground_truth


def evaluate_imputation(
    corrupted: pd.DataFrame,
    ground_truth: dict[tuple[int, int, str], float],
    max_gap: int,
    window_factor: float,
    wavelet: str,
    level: int,
    max_iter: int,
) -> dict:
    """Run imputation and compute errors per gap length and per column."""
    imputed = impute_dataframe(
        corrupted,
        max_gap=max_gap,
        window_factor=window_factor,
        wavelet=wavelet,
        level=level,
        max_iter=max_iter,
    )

    # Infer the injected gap length for each (gap_id, col) pair.
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
        description="Validate wavelet imputation by injecting artificial gaps and measuring error."
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
        "--max-gap",
        type=int,
        default=12,
        help="Maximum gap length the imputer will fill (default: 12).",
    )
    parser.add_argument(
        "--window-factor",
        type=float,
        default=3.0,
        help="Window factor passed to the imputer (default: 3.0).",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="sym18",
        help="Wavelet passed to the imputer (default: sym18).",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=2,
        help="Wavelet level passed to the imputer (default: 2).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Imputation iterations passed to the imputer (default: 3).",
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
        default=PROJECT_ROOT / "data" / "imputation_validation.json",
        help="Where to save the JSON report (default: data/imputation_validation.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.input}...")
    df = load_wind_data(args.input)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Shape: {df.shape}, numeric columns: {len(numeric_cols)}")

    # Find a clean subset that is long enough for all requested gap lengths.
    min_length = max(args.gap_lengths) + 100
    segments = find_complete_segments(df, min_length)
    if not segments:
        raise RuntimeError(
            f"No complete segment of length >= {min_length} found; reduce --gap-lengths."
        )

    # Use the largest complete segment as a clean test bed.
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

    print("Running imputation...")
    metrics = evaluate_imputation(
        corrupted,
        ground_truth,
        max_gap=args.max_gap,
        window_factor=args.window_factor,
        wavelet=args.wavelet,
        level=args.level,
        max_iter=args.max_iter,
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
                    "max_gap": args.max_gap,
                    "window_factor": args.window_factor,
                    "wavelet": args.wavelet,
                    "level": args.level,
                    "max_iter": args.max_iter,
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
