"""Shared utilities for the model-based imputation implementations.

This module holds the pieces common to every imputer: data loading, circular
direction handling, the iterative imputer wrapper, artificial-gap validation
and the metric plots. Each concrete imputer (RandomForest, LightGBM, ...)
lives in its own module and only provides the scikit-learn estimator.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer as _enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score

NUMERIC_SEED = 42


def is_direction_column(col):
    """Direction columns wrap around 0/360 degrees and need circular handling."""
    return str(col).lower().startswith("dir")


def load_wind_data(path):
    """Load a wind CSV and regularise it to a 10-minute DatetimeIndex.

    Legacy column aliases (wdir{h}, verts{h}, ...) are canonicalised so every
    consumer sees one name per feature (dir{h}, v{h}, ...).
    """
    from ..common import canonicalize_columns

    df = pd.read_csv(path)
    df = canonicalize_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    full_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="10min",
    )
    df = df.set_index("timestamp").reindex(full_index)
    df.index.name = "timestamp"
    return df


def direction_to_components(df, cols):
    """Convert degree direction columns to sin/cos components."""
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        radians = np.deg2rad(result[col].to_numpy(dtype=float))
        result[f"{col}_sin"] = np.sin(radians)
        result[f"{col}_cos"] = np.cos(radians)
        result = result.drop(columns=[col])
    return result


def components_to_direction(df, cols):
    """Convert sin/cos components back to degree direction columns."""
    result = df.copy()
    for col in cols:
        sin_col, cos_col = f"{col}_sin", f"{col}_cos"
        if sin_col not in result.columns or cos_col not in result.columns:
            continue
        angle = np.rad2deg(np.arctan2(result[sin_col], result[cos_col]))
        result[col] = np.mod(angle, 360.0)
        result = result.drop(columns=[sin_col, cos_col])
    return result


def direction_error(true, predicted):
    """Circular absolute error for 0-360 degree directions."""
    return abs((true - predicted + 180.0) % 360.0 - 180.0)


def impute_with_estimator(df, estimator, max_iter=10, random_state=42, train_sample=3000):
    """Fill missing values per row using IterativeImputer.

    Direction columns are imputed through their sin/cos components. When
    ``train_sample`` is set, the imputer is fitted on a random sample of rows
    and then applied to the whole dataset, keeping the runtime reasonable.
    """
    direction_cols = [c for c in df.columns if is_direction_column(c)]
    transformed = direction_to_components(df, direction_cols)
    numeric_cols = transformed.select_dtypes(include=[np.number]).columns.tolist()
    numeric = transformed[numeric_cols]

    if train_sample is not None and train_sample < len(numeric):
        train = numeric.sample(n=train_sample, random_state=random_state)
    else:
        train = numeric

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
    )
    imputer.fit(train)
    values = imputer.transform(numeric)
    imputed = pd.DataFrame(values, index=numeric.index, columns=numeric.columns)

    for col in transformed.columns:
        if col not in imputed.columns:
            imputed[col] = transformed[col]

    imputed = components_to_direction(imputed, direction_cols)
    return imputed[df.columns]


def fill_remaining_nan(df):
    """Fill any residual NaNs (e.g. fully missing rows) with column-wise interpolation."""
    if df.isna().all(axis=1).any():
        return df.interpolate(method="linear", axis=0, limit_direction="both")
    return df


def validate(
    imputer,
    method_name,
    df,
    n_samples=30,
    gap_lengths=(1, 2, 3, 5, 10, 20, 36),
    seed=42,
    imputer_kwargs=None,
):
    """Inject artificial gaps into a clean segment and measure the imputation error.

    *imputer* is a callable ``DataFrame -> DataFrame``. Returns
    (detailed_df, summary_df, overall_metrics, settings).
    """
    imputer_kwargs = imputer_kwargs or {}
    rng = np.random.default_rng(seed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    min_length = max(gap_lengths) + 100
    segments = find_complete_segments(df, min_length)
    if not segments:
        raise RuntimeError(
            f"No complete segment of length >= {min_length} found; reduce gap-lengths."
        )
    start, end = max(segments, key=lambda s: s[1] - s[0])
    clean = df.iloc[start:end].copy()

    corrupted, ground_truth = inject_gaps(clean, gap_lengths, n_samples, rng)
    imputed = imputer(corrupted, **imputer_kwargs)
    detailed = evaluate_imputed(imputed, ground_truth)
    summary = summarize(detailed)
    overall = overall_metrics(detailed)
    settings = {
        "method": method_name,
        "n_samples": n_samples,
        "gap_lengths": list(gap_lengths),
        "seed": seed,
        "clean_segment": (int(start), int(end)),
        "columns_imputed": len(numeric_cols),
        "imputer_kwargs": imputer_kwargs,
    }
    return detailed, summary, overall, settings


def find_complete_segments(df, min_length):
    """Return index ranges [start, end) with no missing value in any column."""
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


def inject_gaps(df, gap_lengths, samples_per_length, rng):
    """Mask short artificial gaps in a complete DataFrame.

    Returns (corrupted_df, ground_truth) where ground_truth maps
    (gap_id, row_index, column) to the original value.
    """
    corrupted = df.copy()
    ground_truth = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_rows = len(df)

    gap_id = 0
    for gap_len in gap_lengths:
        for _ in range(samples_per_length):
            start = int(rng.integers(0, total_rows - gap_len))
            end = start + gap_len
            col = str(rng.choice(numeric_cols))
            for idx in range(start, end):
                ground_truth[(gap_id, idx, col)] = float(
                    corrupted.at[corrupted.index[idx], col]
                )
                corrupted.iat[idx, corrupted.columns.get_loc(col)] = np.nan
            gap_id += 1

    return corrupted, ground_truth


def evaluate_imputed(imputed, ground_truth):
    """Compare imputed values to the injected ground truth.

    Returns a long-format DataFrame with one row per (gap, position, column):
    true value, predicted value, absolute and squared errors.
    """
    gap_info = {}
    for (gap_id, idx, col) in ground_truth:
        gap_info.setdefault((gap_id, col), []).append(idx)

    records = []
    for (gap_id, col), rows in gap_info.items():
        gap_len = len(rows)
        for idx in rows:
            predicted = imputed.at[imputed.index[idx], col]
            if np.isnan(predicted):
                continue
            true_value = ground_truth[(gap_id, idx, col)]
            if is_direction_column(col):
                error = direction_error(true_value, predicted)
            else:
                error = predicted - true_value
            records.append({
                "gap_id": gap_id,
                "gap_len": gap_len,
                "row": idx,
                "col": col,
                "true": true_value,
                "predicted": float(predicted),
                "error": error,
                "abs_error": abs(error),
                "squared_error": error * error,
            })

    return pd.DataFrame(records)


def summarize(detailed):
    """Aggregate per-sample errors into per-column/gap-length metrics."""
    if detailed.empty:
        return pd.DataFrame()
    rows = []
    for (col, gap_len), group in detailed.groupby(["col", "gap_len"]):
        mse = float(group["squared_error"].mean())
        valid = group.dropna(subset=["true", "predicted"])
        r2 = np.nan
        if len(valid) >= 2 and valid["true"].nunique() > 1:
            r2 = float(r2_score(valid["true"], valid["predicted"]))
        rows.append({
            "col": col,
            "gap_len": gap_len,
            "samples": len(group),
            "mae": float(group["abs_error"].mean()),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "bias": float(group["error"].mean()),
            "r2": r2,
        })
    return pd.DataFrame(rows)


def overall_metrics(detailed):
    """Compute a single overall metrics dict for the whole evaluation."""
    if detailed.empty:
        return {"samples": 0, "mae": np.nan, "mse": np.nan, "rmse": np.nan,
                "bias": np.nan, "r2": np.nan}
    mse = float(detailed["squared_error"].mean())
    valid = detailed.dropna(subset=["true", "predicted"])
    r2 = np.nan
    if len(valid) >= 2 and valid["true"].nunique() > 1:
        r2 = float(r2_score(valid["true"], valid["predicted"]))
    return {
        "samples": len(detailed),
        "mae": float(detailed["abs_error"].mean()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": float(detailed["error"].mean()),
        "r2": r2,
    }


def plot_metrics(summary, output_dir):
    """Generate the standard metric plots from the summary DataFrame."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary.empty:
        return []

    by_gap = (
        summary.groupby("gap_len")
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"),
             bias=("bias", "mean"), r2=("r2", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Imputation metrics by gap length")
    for ax, column, title in [
        (axes[0, 0], "mae", "MAE"),
        (axes[0, 1], "rmse", "RMSE"),
        (axes[1, 0], "bias", "Bias (predicted - true)"),
        (axes[1, 1], "r2", "R²"),
    ]:
        by_gap.plot(x="gap_len", y=column, ax=ax, marker="o", legend=False)
        ax.set_title(title)
        ax.set_xlabel("Gap length (timesteps)")
        ax.set_ylabel(column)
        ax.grid(True)
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 1].set_ylim(-0.1, 1.1)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_by_gap_length.png", dpi=150)
    plt.close(fig)

    by_col = (
        summary.groupby("col")
        .agg(mae=("mae", "mean"), samples=("samples", "sum"))
        .sort_values("mae", ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(by_col))))
    ax.barh(by_col["col"], by_col["mae"])
    ax.set_title("Overall MAE by column")
    ax.set_xlabel("MAE")
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(output_dir / "mae_by_column.png", dpi=150)
    plt.close(fig)

    pivot = summary.pivot(index="col", columns="gap_len", values="mae")
    fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(pivot))))
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

    return ["metrics_by_gap_length.png", "mae_by_column.png", "mae_heatmap.png"]


def save_report(output_dir, settings, detailed, summary, overall):
    """Persist every validation artifact (CSV/JSON) so plots can be regenerated later."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed.to_csv(output_dir / "detailed.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"settings": settings, "overall": overall},
            handle,
            indent=2,
            default=str,
        )

    return {
        "detailed_csv": str(output_dir / "detailed.csv"),
        "summary_csv": str(output_dir / "summary.csv"),
        "metrics_json": str(output_dir / "metrics.json"),
    }