"""Detailed comparison of RF, LightGBM, and neural-network imputation.

The comparison is restricted to ws100, ws40, v40, dir40, and ws50. Direction
values remain one-dimensional: dir40 is represented as its wrapped offset from
the circular mean instead of being expanded into sine/cosine features.

Run Optuna and cache the best parameters before the first comparison:
    python tests/compare_rf_lgbm_impute_detailed.py --optuna

Reuse the cached parameters:
    python tests/compare_rf_lgbm_impute_detailed.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.impute.base import find_complete_segments, load_wind_data
from src.imputation_model_utils import (
    DIRECTION_COLUMNS,
    MODEL_NAMES,
    TARGET_COLUMNS,
    circular_mean_degrees,
    impute_focused,
    load_cached_params,
    wrap_offset,
)


def inject_focused_gaps(
    df: pd.DataFrame,
    gap_lengths: list[int],
    samples_per_length: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[tuple[int, int, str], float]]:
    """Inject the same number of gaps into every requested target column."""
    corrupted = df.copy()
    ground_truth = {}
    gap_id = 0
    requested_lengths = np.repeat(gap_lengths, samples_per_length)
    required_rows = int(requested_lengths.sum() + len(requested_lengths) - 1)
    if required_rows > len(df):
        raise ValueError(
            f"The requested gaps need at least {required_rows} rows per column "
            f"when separated, but the clean segment has {len(df)}. Reduce "
            "--n-samples or --gap-lengths."
        )

    for col in TARGET_COLUMNS:
        ordered_lengths = rng.permutation(requested_lengths)
        spare_rows = len(df) - required_rows
        padding = rng.multinomial(
            spare_rows, np.full(len(ordered_lengths) + 1, 1.0 / (len(ordered_lengths) + 1))
        )
        start = int(padding[0])
        for position, gap_len in enumerate(ordered_lengths):
            gap_len = int(gap_len)
            for row in range(start, start + gap_len):
                ground_truth[(gap_id, row, col)] = float(df.iloc[row][col])
                corrupted.iat[row, corrupted.columns.get_loc(col)] = np.nan
            gap_id += 1
            start += gap_len + int(padding[position + 1])
            if position < len(ordered_lengths) - 1:
                start += 1
    return corrupted, ground_truth


def detailed_records(
    model: str,
    imputed: pd.DataFrame,
    ground_truth: dict[tuple[int, int, str], float],
    direction_centers: dict[str, float],
) -> pd.DataFrame:
    gap_rows: dict[tuple[int, str], list[int]] = {}
    for gap_id, row, col in ground_truth:
        gap_rows.setdefault((gap_id, col), []).append(row)

    records = []
    for (gap_id, col), rows in gap_rows.items():
        for row in rows:
            true = ground_truth[(gap_id, row, col)]
            predicted = float(imputed.iloc[row][col])
            if col in direction_centers:
                center = direction_centers[col]
                true_normalized = float(wrap_offset(true, center))
                predicted_normalized = float(wrap_offset(predicted, center))
                error = float(wrap_offset(predicted - true, 0.0))
            else:
                true_normalized = true
                predicted_normalized = predicted
                error = predicted - true
            records.append(
                {
                    "model": model,
                    "gap_id": gap_id,
                    "gap_len": len(rows),
                    "row": row,
                    "col": col,
                    "true": true,
                    "predicted": predicted,
                    "true_normalized": true_normalized,
                    "predicted_normalized": predicted_normalized,
                    "error": error,
                    "abs_error": abs(error),
                    "squared_error": error**2,
                }
            )
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, col, gap_len), group in records.groupby(["model", "col", "gap_len"]):
        r2 = np.nan
        if len(group) >= 2 and group["true_normalized"].nunique() > 1:
            r2 = r2_score(group["true_normalized"], group["predicted_normalized"])
        mse = float(group["squared_error"].mean())
        rows.append(
            {
                "model": model,
                "col": col,
                "gap_len": gap_len,
                "samples": len(group),
                "mae": float(group["abs_error"].mean()),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "bias": float(group["error"].mean()),
                "r2": float(r2),
            }
        )
    return pd.DataFrame(rows)


def overall_metrics(records: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for model, group in records.groupby("model"):
        mse = float(group["squared_error"].mean())
        result[model] = {
            "samples": int(len(group)),
            "mae": float(group["abs_error"].mean()),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "bias": float(group["error"].mean()),
            "r2": float(r2_score(group["true_normalized"], group["predicted_normalized"])),
        }
    return result


def objective_score(records: pd.DataFrame, clean: pd.DataFrame) -> float:
    """Average per-column normalized RMSE so degrees do not dominate tuning."""
    scales = clean.copy()
    for col in DIRECTION_COLUMNS:
        center = circular_mean_degrees(scales[col])
        scales[col] = wrap_offset(scales[col], center)
    standard_deviations = scales.std().replace(0, 1.0)
    per_column_rmse = records.groupby("col")["squared_error"].mean().pow(0.5)
    return float((per_column_rmse / standard_deviations[per_column_rmse.index]).mean())


def suggest_rf(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        "max_iter": trial.suggest_int("max_iter", 2, 10),
    }


def suggest_lgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 2, 10),
    }


def suggest_neural_network(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "hidden_layer_1": trial.suggest_int("hidden_layer_1", 32, 256, step=32),
        "hidden_layer_2": trial.suggest_categorical(
            "hidden_layer_2", [0, 16, 32, 64, 128]
        ),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
        "alpha": trial.suggest_float("alpha", 1e-7, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 1e-4, 1e-2, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "estimator_max_iter": trial.suggest_int(
            "estimator_max_iter", 100, 400, step=50
        ),
        "max_iter": trial.suggest_int("max_iter", 2, 10),
    }


def run_optuna(
    clean: pd.DataFrame,
    gap_lengths: list[int],
    samples: int,
    trials: int,
    seed: int,
    train_sample: int | None,
    cache_path: Path,
) -> dict[str, Any]:
    tuning_corrupted, tuning_truth = inject_focused_gaps(
        clean, gap_lengths, samples, np.random.default_rng(seed + 1)
    )
    suggesters: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
        "random_forest": suggest_rf,
        "lightgbm": suggest_lgbm,
        "neural_network": suggest_neural_network,
    }
    cached: dict[str, Any] = {
        "settings": {
            "columns": TARGET_COLUMNS,
            "gap_lengths": gap_lengths,
            "samples_per_length": samples,
            "trials_per_model": trials,
            "seed": seed,
            "train_sample": train_sample,
            "objective": "mean per-column RMSE divided by column standard deviation",
        }
    }

    for model in MODEL_NAMES:
        print(f"Running {trials} Optuna trials for {model}...")

        def objective(trial: optuna.Trial) -> float:
            params = suggesters[model](trial)
            imputed, centers = impute_focused(
                tuning_corrupted, model, params, seed, train_sample
            )
            records = detailed_records(model, imputed, tuning_truth, centers)
            return objective_score(records, clean)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=trials, gc_after_trial=True)
        cached[model] = {
            "params": study.best_trial.params,
            "objective_score": float(study.best_value),
            "best_trial": int(study.best_trial.number),
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(cached, handle, indent=2)
    print(f"Cached best hyperparameters in {cache_path}")
    return cached


def plot_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "random_forest": "#315c8c",
        "lightgbm": "#d9792b",
        "neural_network": "#4b8f61",
    }
    by_gap = summary.groupby(["model", "gap_len"], as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"), r2=("r2", "mean")
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for model in MODEL_NAMES:
        values = by_gap[by_gap["model"] == model]
        for ax, metric in zip(axes.flat, ["mae", "rmse", "bias", "r2"]):
            ax.plot(values["gap_len"], values[metric], marker="o", label=model, color=colors[model])
            ax.set_title(metric.upper())
            ax.set_xlabel("Gap length (timesteps)")
            ax.grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].legend()
    fig.suptitle("Model-based imputation metrics by gap length")
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_by_gap_length.png", dpi=150)
    plt.close(fig)

    by_column = summary.groupby(["model", "col"], as_index=False)["mae"].mean()
    pivot = (
        by_column.pivot(index="col", columns="model", values="mae")
        .reindex(TARGET_COLUMNS)
        .reindex(columns=MODEL_NAMES)
    )
    ax = pivot.plot.bar(figsize=(12, 7), color=[colors.get(col, "gray") for col in pivot.columns])
    ax.set_title("Mean absolute error by column")
    ax.set_ylabel("MAE (degrees for dir40)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "mae_by_column.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(20, 6), constrained_layout=True)
    vmax = summary["mae"].max()
    for ax, model in zip(axes, MODEL_NAMES):
        pivot = (
            summary[summary["model"] == model]
            .pivot(index="col", columns="gap_len", values="mae")
            .reindex(TARGET_COLUMNS)
        )
        image = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(model)
        ax.set_xlabel("Gap length")
    fig.colorbar(image, ax=axes, label="MAE")
    fig.suptitle("MAE heatmaps")
    fig.savefig(output_dir / "mae_heatmaps.png", dpi=150)
    plt.close(fig)

    mae = summary.pivot(index=["col", "gap_len"], columns="model", values="mae")
    comparisons = [
        ("random_forest", "lightgbm"),
        ("random_forest", "neural_network"),
        ("lightgbm", "neural_network"),
    ]
    matrices = [
        (mae[left] - mae[right]).unstack("gap_len").reindex(TARGET_COLUMNS)
        for left, right in comparisons
    ]
    limit = max(max(abs(matrix.min().min()), abs(matrix.max().max())) for matrix in matrices)
    limit = limit or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for ax, matrix, (left, right) in zip(axes, matrices, comparisons):
        image = ax.imshow(
            matrix.values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xlabel("Gap length")
        ax.set_title(f"{left} - {right}")
    fig.colorbar(image, ax=axes, label="MAE difference (positive favors second model)")
    fig.suptitle("Pairwise MAE differences")
    fig.savefig(output_dir / "mae_difference.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detailed RF, LightGBM, and neural-network imputation comparison."
    )
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "wind_data.csv")
    parser.add_argument("--n-samples", type=int, default=30, help="Gaps per column and gap length.")
    parser.add_argument("--gap-lengths", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 36])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sample", type=int, default=3000, help="Rows used to fit each imputer; 0 uses all rows.")
    parser.add_argument(
        "--optuna", action="store_true", help="Tune all models before comparison."
    )
    parser.add_argument("--optuna-trials", type=int, default=30, help="Trials per model (default: 30).")
    parser.add_argument("--optuna-samples", type=int, default=5, help="Tuning gaps per column and gap length.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "imputation_optuna" / "best_model.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "imputation_results" / "model_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples < 1 or args.optuna_samples < 1 or args.optuna_trials < 1:
        raise ValueError("Sample and trial counts must be positive.")
    if any(length < 1 for length in args.gap_lengths):
        raise ValueError("Gap lengths must be positive.")

    loaded = load_wind_data(args.input)
    missing = [col for col in TARGET_COLUMNS if col not in loaded.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    focused = loaded[TARGET_COLUMNS]
    min_length = max(args.gap_lengths) + 100
    segments = find_complete_segments(focused, min_length)
    if not segments:
        raise RuntimeError(f"No complete target-column segment of at least {min_length} rows.")
    start, end = max(segments, key=lambda pair: pair[1] - pair[0])
    clean = focused.iloc[start:end].copy()
    train_sample = args.train_sample if args.train_sample > 0 else None
    print(f"Using complete rows {start}:{end} ({len(clean)} rows), columns: {TARGET_COLUMNS}")

    if args.optuna:
        cached = run_optuna(
            clean,
            args.gap_lengths,
            args.optuna_samples,
            args.optuna_trials,
            args.seed,
            train_sample,
            args.cache,
        )
    else:
        cached = load_cached_params(args.cache)
        print(f"Loaded hyperparameters from {args.cache}")

    corrupted, truth = inject_focused_gaps(
        clean, args.gap_lengths, args.n_samples, np.random.default_rng(args.seed + 10_000)
    )
    all_records = []
    direction_centers = {}
    for model in MODEL_NAMES:
        params = cached[model]["params"]
        print(f"Running {model} with {params}")
        imputed, centers = impute_focused(
            corrupted, model, params, args.seed, train_sample
        )
        direction_centers[model] = centers
        all_records.append(detailed_records(model, imputed, truth, centers))

    records = pd.concat(all_records, ignore_index=True)
    summary = summarize(records)
    overall = overall_metrics(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_dir / "imputation_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "imputation_summary.csv", index=False)
    report = {
        "settings": {
            "input": str(args.input),
            "columns": TARGET_COLUMNS,
            "direction_representation": "wrapped offset from circular mean in degrees",
            "n_samples": args.n_samples,
            "gap_lengths": args.gap_lengths,
            "seed": args.seed,
            "train_sample": train_sample,
            "clean_segment": [int(start), int(end)],
            "optuna_cache": str(args.cache),
            "direction_centers": direction_centers,
        },
        "hyperparameters": {model: cached[model] for model in MODEL_NAMES},
        "overall": overall,
    }
    with (args.output_dir / "imputation_overall.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    plot_comparison(summary, args.output_dir)

    print("\nOverall metrics:")
    print(json.dumps(overall, indent=2))
    print(f"Saved detailed comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
