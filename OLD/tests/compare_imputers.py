"""Compare KNN, Random Forest, LightGBM, and Kalman row-wise imputation on the same injected gaps.

This script runs the validation logic from `validate_knn_impute.py`,
`validate_rf_impute.py`, `validate_lgbm_impute.py`, and `validate_kalman_impute.py`
on the *same* corrupted dataset, so the metrics are directly comparable. A side-by-side
JSON report is saved.

Usage:
    python tests/compare_imputers.py
    python tests/compare_imputers.py --n-samples 30 --gap-lengths 1 2 3 5 10 20 36
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.validate_impute_wind_data import (
    find_complete_segments,
    inject_gaps,
    load_wind_data,
)
from tests.validate_kalman_impute import evaluate_kalman_imputation
from tests.validate_knn_impute import evaluate_knn_imputation
from tests.validate_lgbm_impute import evaluate_lgbm_imputation
from tests.validate_rf_impute import evaluate_rf_imputation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare KNN, Random Forest, LightGBM, and Kalman imputation on the same injected gaps."
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
        default=[1, 2, 3, 5, 10, 20, 36],
        help="Gap lengths to test, in timesteps (default: 1 2 3 5 10 20 36).",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNNImputer (default: 5).",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=100,
        help="Number of trees for RandomForestRegressor (default: 100).",
    )
    parser.add_argument(
        "--rf-max-iter",
        type=int,
        default=10,
        help="Number of IterativeImputer rounds for RF (default: 10).",
    )
    parser.add_argument(
        "--lgbm-estimators",
        type=int,
        default=100,
        help="Number of boosting rounds for LGBMRegressor (default: 100).",
    )
    parser.add_argument(
        "--lgbm-max-iter",
        type=int,
        default=10,
        help="Number of IterativeImputer rounds for LightGBM (default: 10).",
    )
    parser.add_argument(
        "--kalman-q-ratio",
        type=float,
        default=0.05,
        help="Kalman process-noise ratio (default: 0.05).",
    )
    parser.add_argument(
        "--kalman-min-r",
        type=float,
        default=1e-6,
        help="Kalman minimum observation-noise variance (default: 1e-6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "imputation_comparison.json",
        help="Where to save the comparison report (default: data/imputation_comparison.json).",
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

    print("\n[1/4] Running KNN imputation...")
    knn_metrics = evaluate_knn_imputation(
        corrupted,
        ground_truth,
        n_neighbors=args.knn_neighbors,
        weights="distance",
    )

    print("\n[2/3] Running Random Forest imputation...")
    rf_metrics = evaluate_rf_imputation(
        corrupted,
        ground_truth,
        n_estimators=args.rf_estimators,
        max_iter=args.rf_max_iter,
        random_state=args.seed,
    )

    print("\n[3/4] Running LightGBM imputation...")
    lgbm_metrics = evaluate_lgbm_imputation(
        corrupted,
        ground_truth,
        n_estimators=args.lgbm_estimators,
        max_iter=args.lgbm_max_iter,
        random_state=args.seed,
    )

    print("\n[4/4] Running Kalman imputation...")
    kalman_metrics = evaluate_kalman_imputation(
        corrupted,
        ground_truth,
        q_ratio=args.kalman_q_ratio,
        min_r=args.kalman_min_r,
    )

    print("\n" + "=" * 95)
    print("COMPARISON: KNN vs Random Forest vs LightGBM vs Kalman")
    print("=" * 95)
    print(f"{'Metric':<20} {'KNN':>15} {'Random Forest':>15} {'LightGBM':>15} {'Kalman':>15}")
    print("-" * 95)
    print(
        f"{'Samples':<20} {knn_metrics['overall']['samples']:>15} "
        f"{rf_metrics['overall']['samples']:>15} "
        f"{lgbm_metrics['overall']['samples']:>15} "
        f"{kalman_metrics['overall']['samples']:>15}"
    )
    print(
        f"{'MAE':<20} {knn_metrics['overall']['mae']:>15.4f} "
        f"{rf_metrics['overall']['mae']:>15.4f} "
        f"{lgbm_metrics['overall']['mae']:>15.4f} "
        f"{kalman_metrics['overall']['mae']:>15.4f}"
    )
    print(
        f"{'RMSE':<20} {knn_metrics['overall']['rmse']:>15.4f} "
        f"{rf_metrics['overall']['rmse']:>15.4f} "
        f"{lgbm_metrics['overall']['rmse']:>15.4f} "
        f"{kalman_metrics['overall']['rmse']:>15.4f}"
    )
    print("=" * 95)

    print("\nPer-gap-length MAE:")
    print(f"{'Gap length':<12} {'KNN MAE':>12} {'RF MAE':>12} {'LGBM MAE':>12} {'Kalman MAE':>12}")
    all_lengths = sorted(
        set(knn_metrics["by_gap_length"].keys())
        | set(rf_metrics["by_gap_length"].keys())
        | set(lgbm_metrics["by_gap_length"].keys())
        | set(kalman_metrics["by_gap_length"].keys())
    )
    for gap_len in all_lengths:
        knn_val = knn_metrics["by_gap_length"].get(gap_len, {})
        rf_val = rf_metrics["by_gap_length"].get(gap_len, {})
        lgbm_val = lgbm_metrics["by_gap_length"].get(gap_len, {})
        kalman_val = kalman_metrics["by_gap_length"].get(gap_len, {})
        print(
            f"{gap_len:<12} {knn_val.get('mae', float('nan')):>12.4f} "
            f"{rf_val.get('mae', float('nan')):>12.4f} "
            f"{lgbm_val.get('mae', float('nan')):>12.4f} "
            f"{kalman_val.get('mae', float('nan')):>12.4f}"
        )

    print("\nTop 10 worst columns by KNN MAE:")
    for col, vals in sorted(
        knn_metrics["by_column"].items(), key=lambda kv: kv[1]["mae"], reverse=True
    )[:10]:
        print(f"  {col}: MAE={vals['mae']:.4f}, RMSE={vals['rmse']:.4f}, n={vals['samples']}")

    print("\nTop 10 worst columns by RF MAE:")
    for col, vals in sorted(
        rf_metrics["by_column"].items(), key=lambda kv: kv[1]["mae"], reverse=True
    )[:10]:
        print(f"  {col}: MAE={vals['mae']:.4f}, RMSE={vals['rmse']:.4f}, n={vals['samples']}")

    print("\nTop 10 worst columns by LightGBM MAE:")
    for col, vals in sorted(
        lgbm_metrics["by_column"].items(), key=lambda kv: kv[1]["mae"], reverse=True
    )[:10]:
        print(f"  {col}: MAE={vals['mae']:.4f}, RMSE={vals['rmse']:.4f}, n={vals['samples']}")

    print("\nTop 10 worst columns by Kalman MAE:")
    for col, vals in sorted(
        kalman_metrics["by_column"].items(), key=lambda kv: kv[1]["mae"], reverse=True
    )[:10]:
        print(f"  {col}: MAE={vals['mae']:.4f}, RMSE={vals['rmse']:.4f}, n={vals['samples']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "settings": {
                    "input": str(args.input),
                    "n_samples": args.n_samples,
                    "gap_lengths": args.gap_lengths,
                    "knn_neighbors": args.knn_neighbors,
                    "rf_estimators": args.rf_estimators,
                    "rf_max_iter": args.rf_max_iter,
                    "lgbm_estimators": args.lgbm_estimators,
                    "lgbm_max_iter": args.lgbm_max_iter,
                    "kalman_q_ratio": args.kalman_q_ratio,
                    "kalman_min_r": args.kalman_min_r,
                    "seed": args.seed,
                    "clean_segment": (int(start), int(end)),
                },
                "knn": knn_metrics,
                "random_forest": rf_metrics,
                "lightgbm": lgbm_metrics,
                "kalman": kalman_metrics,
            },
            handle,
            indent=2,
        )
    print(f"\nComparison report saved to {args.output}")


if __name__ == "__main__":
    main()
