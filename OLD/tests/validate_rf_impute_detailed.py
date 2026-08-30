"""Detailed statistical validation of Random Forest imputation for wind_data.csv.

This script injects a large number of artificial gaps into clean data segments,
runs the Random Forest imputer, and computes detailed metrics per column and per
gap length: MAE, MSE, RMSE, R², and bias. Results are written to CSV and several
plots are saved to ``data/imputation_results/``.

Usage:
    python tests/validate_rf_impute_detailed.py
    python tests/validate_rf_impute_detailed.py --n-samples 50 --gap-lengths 1 2 3 5 10 20 36
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.validate_impute_wind_data import (
    _direction_error,
    _is_direction_column,
    find_complete_segments,
    inject_gaps,
    load_wind_data,
)
from src.impute_wind_data_rf import _impute_rf


def evaluate_rf_imputation_detailed(
    corrupted: pd.DataFrame,
    ground_truth: dict[tuple[int, int, str], float],
    n_estimators: int = 100,
    max_iter: int = 10,
    random_state: int = 42,
    train_sample: int | None = None,
) -> pd.DataFrame:
    """Run RF imputation and return a detailed per-sample DataFrame."""
    imputed = _impute_rf(
        corrupted,
        n_estimators=n_estimators,
        max_iter=max_iter,
        random_state=random_state,
        train_sample=train_sample,
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
                error = predicted - true_value
            records.append(
                {
                    "gap_id": gap_id,
                    "gap_len": gap_len,
                    "row": idx,
                    "col": col,
                    "true": true_value,
                    "predicted": predicted,
                    "error": error,
                    "abs_error": abs(error),
                    "squared_error": error * error,
                }
            )

    return pd.DataFrame(records)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per column and gap length."""
    summary = []
    for (col, gap_len), group in df.groupby(["col", "gap_len"]):
        valid = group.dropna(subset=["true", "predicted"])
        mae = group["abs_error"].mean()
        mse = group["squared_error"].mean()
        rmse = np.sqrt(mse)
        bias = group["error"].mean()
        if len(valid) >= 2 and valid["true"].nunique() > 1:
            r2 = r2_score(valid["true"], valid["predicted"])
        else:
            r2 = np.nan
        summary.append(
            {
                "col": col,
                "gap_len": gap_len,
                "samples": len(group),
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "bias": bias,
                "r2": r2,
            }
        )
    return pd.DataFrame(summary)


def plot_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save metric plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics by gap length (averaged over columns).
    by_gap = (
        summary.groupby("gap_len")
        .agg({"mae": "mean", "mse": "mean", "rmse": "mean", "bias": "mean", "r2": "mean"})
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Random Forest imputation metrics by gap length")
    by_gap.plot(x="gap_len", y="mae", ax=axes[0, 0], marker="o", legend=False)
    axes[0, 0].set_title("MAE")
    axes[0, 0].set_xlabel("Gap length (timesteps)")
    axes[0, 0].set_ylabel("MAE")
    axes[0, 0].grid(True)

    by_gap.plot(x="gap_len", y="rmse", ax=axes[0, 1], marker="o", legend=False)
    axes[0, 1].set_title("RMSE")
    axes[0, 1].set_xlabel("Gap length (timesteps)")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].grid(True)

    by_gap.plot(x="gap_len", y="bias", ax=axes[1, 0], marker="o", legend=False)
    axes[1, 0].set_title("Bias (predicted - true)")
    axes[1, 0].set_xlabel("Gap length (timesteps)")
    axes[1, 0].set_ylabel("Bias")
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].grid(True)

    by_gap.plot(x="gap_len", y="r2", ax=axes[1, 1], marker="o", legend=False)
    axes[1, 1].set_title("R²")
    axes[1, 1].set_xlabel("Gap length (timesteps)")
    axes[1, 1].set_ylabel("R²")
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "metrics_by_gap_length.png", dpi=150)
    plt.close(fig)

    # 2. MAE by column (overall average across gap lengths).
    by_col = summary.groupby("col").agg({"mae": "mean", "samples": "sum"}).sort_values("mae", ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(by_col["col"], by_col["mae"])
    ax.set_title("Overall MAE by column (Random Forest imputation)")
    ax.set_xlabel("MAE")
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(output_dir / "mae_by_column.png", dpi=150)
    plt.close(fig)

    # 3. Heatmap of MAE by column and gap length.
    pivot = summary.pivot(index="col", columns="gap_len", values="mae")
    fig, ax = plt.subplots(figsize=(12, 14))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Gap length (timesteps)")
    ax.set_ylabel("Column")
    ax.set_title("MAE heatmap by column and gap length")
    plt.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(output_dir / "mae_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. Bias by column and gap length (top biased columns).
    bias_summary = summary.copy()
    bias_summary["abs_bias"] = bias_summary["bias"].abs()
    top_bias = (
        bias_summary.groupby("col")
        .agg({"abs_bias": "mean", "samples": "sum"})
        .sort_values("abs_bias", ascending=False)
        .head(20)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top_bias["col"], top_bias["abs_bias"])
    ax.set_title("Top 20 columns by absolute mean bias")
    ax.set_xlabel("|bias|")
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(output_dir / "bias_by_column.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detailed statistical validation of Random Forest imputation."
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
        default=50,
        help="Number of artificial gaps to inject per gap length (default: 50).",
    )
    parser.add_argument(
        "--gap-lengths",
        type=int,
        nargs="*",
        default=[1, 2, 3, 5, 10, 20, 36],
        help="Gap lengths to test, in timesteps (default: 1 2 3 5 10 20 36).",
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
        "--train-sample",
        type=int,
        default=3000,
        help="Number of rows to use for fitting the imputer (default: 3000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "imputation_results",
        help="Directory to save results and plots (default: data/imputation_results).",
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

    n_missing_total = args.n_samples * sum(args.gap_lengths) * len(numeric_cols)
    print(
        f"Injecting {args.n_samples} gaps per length across {len(numeric_cols)} columns "
        f"({len(args.gap_lengths)} lengths) -> ~{n_missing_total} missing values to impute..."
    )
    corrupted, ground_truth = inject_gaps(
        clean_df, args.gap_lengths, args.n_samples, rng
    )

    print(f"Running Random Forest imputation (trees={args.estimators}, max_iter={args.max_iter}, train_sample={args.train_sample})...")
    detailed = evaluate_rf_imputation_detailed(
        corrupted,
        ground_truth,
        n_estimators=args.estimators,
        max_iter=args.max_iter,
        random_state=args.seed,
        train_sample=args.train_sample,
    )
    print(f"Evaluated {len(detailed)} imputed values.")

    summary = compute_summary(detailed)
    valid_overall = detailed.dropna(subset=["true", "predicted"])
    overall = {
        "samples": len(detailed),
        "mae": float(detailed["abs_error"].mean()),
        "mse": float(detailed["squared_error"].mean()),
        "rmse": float(np.sqrt(detailed["squared_error"].mean())),
        "bias": float(detailed["error"].mean()),
        "r2": float(
            r2_score(valid_overall["true"], valid_overall["predicted"])
            if len(valid_overall) >= 2 and valid_overall["true"].nunique() > 1
            else np.nan
        ),
    }

    print("\nOverall metrics:")
    print(json.dumps(overall, indent=2))
    print("\nTop 10 worst columns by MAE:")
    for _, row in summary.groupby("col").agg({"mae": "mean", "samples": "sum"}).sort_values("mae", ascending=False).head(10).iterrows():
        print(f"  {row.name}: MAE={row['mae']:.4f}, n={int(row['samples'])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "rf_imputation_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "rf_imputation_summary.csv", index=False)
    with open(args.output_dir / "rf_imputation_overall.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "settings": {
                    "input": str(args.input),
                    "n_samples": args.n_samples,
                    "gap_lengths": args.gap_lengths,
                    "estimators": args.estimators,
                    "max_iter": args.max_iter,
                    "seed": args.seed,
                    "train_sample": args.train_sample,
                    "clean_segment": (int(start), int(end)),
                },
                "overall": overall,
            },
            handle,
            indent=2,
        )

    print(f"\nSaved detailed results to {args.output_dir / 'rf_imputation_detailed.csv'}")
    print(f"Saved summary to {args.output_dir / 'rf_imputation_summary.csv'}")
    print(f"Saved overall metrics to {args.output_dir / 'rf_imputation_overall.json'}")

    print("Generating plots...")
    plot_metrics(summary, args.output_dir)
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
