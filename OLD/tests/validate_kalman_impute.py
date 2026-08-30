"""Validate linear interpolation + Kalman filter imputation for wind_data.csv.

This script uses a column-wise (univariate) imputation: each missing value is first
filled with linear interpolation, then a simple scalar Kalman random-walk smoother is
applied to the whole column, and only the originally missing positions are replaced
by the smoothed estimates. Direction columns (`dir*`) are converted to sin/cos and
reconstructed to handle the 0/360 wrap-around.

Usage:
    python tests/validate_kalman_impute.py
    python tests/validate_kalman_impute.py --n-samples 20 --gap-lengths 1 2 3 5 10 20 36 --q-ratio 0.1
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

from tests.validate_impute_wind_data import (
    _direction_error,
    find_complete_segments,
    inject_gaps,
    load_wind_data,
    _is_direction_column,
)


def _kalman_smoother(
    series: np.ndarray,
    q_ratio: float = 0.05,
    min_r: float = 1e-6,
    use_trend: bool = True,
) -> np.ndarray:
    """Apply a Kalman smoother to a 1D signal.

    By default a local linear trend model (state = [position, velocity]) is used;
    this captures momentum better than a simple random walk. Set ``use_trend=False``
    to fall back to a random walk.

    Missing values (NaN) are treated as missing observations in the filter. The
    returned array is the smoothed estimate at every time step.

    Parameters
    ----------
    series : np.ndarray
        1-D signal, possibly containing NaNs.
    q_ratio : float
        The process noise variance is scaled by ``q_ratio * var(diff(interpolated))``.
        Smaller values produce smoother estimates; larger values track the data more
        closely.
    min_r : float
        Minimum observation noise variance R.
    use_trend : bool
        If True, use a local linear trend model; otherwise use a random walk.

    Returns
    -------
    np.ndarray
        Smoothed signal of the same length as ``series``.
    """
    n = len(series)
    if n < 5:
        return pd.Series(series).interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)

    interpolated = pd.Series(series).interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)

    # Estimate observation noise R from residual w.r.t. a rolling median.
    window = min(25, n // 4 + 1)
    if window >= 3:
        rolling_median = (
            pd.Series(interpolated)
            .rolling(window=window, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
    else:
        rolling_median = interpolated
    r = max(np.nanvar(interpolated - rolling_median), min_r)

    if use_trend:
        # Local linear trend model: state = [x, v]
        # Transition: x_t = x_{t-1} + v_{t-1}, v_t = v_{t-1}
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])

        diff_var = np.nanvar(np.diff(interpolated))
        q_pos = max(q_ratio * diff_var, min_r * 1e-3)
        q_vel = max(q_ratio * diff_var / (n ** 2), min_r * 1e-6)
        Q = np.diag([q_pos, q_vel])
        R = np.array([[r]])

        # Initialize state and covariance.
        x = np.array([interpolated[0], 0.0])
        P = np.diag([np.nanvar(interpolated), diff_var])

        x_pred = np.zeros((n, 2))
        P_pred = np.zeros((n, 2, 2))
        x_post = np.zeros((n, 2))
        P_post = np.zeros((n, 2, 2))

        for t in range(n):
            if t == 0:
                x_pred[t] = x
                P_pred[t] = P
            else:
                x_pred[t] = F @ x_post[t - 1]
                P_pred[t] = F @ P_post[t - 1] @ F.T + Q

            if not np.isnan(series[t]):
                y = np.array([series[t]])
                S = H @ P_pred[t] @ H.T + R
                K = P_pred[t] @ H.T @ np.linalg.inv(S)
                x_post[t] = x_pred[t] + K @ (y - H @ x_pred[t])
                P_post[t] = (np.eye(2) - K @ H) @ P_pred[t]
            else:
                x_post[t] = x_pred[t]
                P_post[t] = P_pred[t]

        # Backward smoother.
        smoothed = np.zeros((n, 2))
        smoothed[-1] = x_post[-1]
        P_smooth = np.zeros((n, 2, 2))
        P_smooth[-1] = P_post[-1]

        for t in range(n - 2, -1, -1):
            J = P_post[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
            smoothed[t] = x_post[t] + J @ (smoothed[t + 1] - x_pred[t + 1])
            P_smooth[t] = P_post[t] + J @ (P_smooth[t + 1] - P_pred[t + 1]) @ J.T

        return smoothed[:, 0]

    else:
        # Simple random walk.
        q = max(q_ratio * np.nanvar(np.diff(interpolated)), min_r * 1e-3)

        x_pred = np.zeros(n)
        p_pred = np.zeros(n)
        x_post = np.zeros(n)
        p_post = np.zeros(n)

        x_post[0] = interpolated[0]
        p_post[0] = r

        for t in range(1, n):
            x_pred[t] = x_post[t - 1]
            p_pred[t] = p_post[t - 1] + q

            if not np.isnan(series[t]):
                k = p_pred[t] / (p_pred[t] + r)
                x_post[t] = x_pred[t] + k * (series[t] - x_pred[t])
                p_post[t] = (1.0 - k) * p_pred[t]
            else:
                x_post[t] = x_pred[t]
                p_post[t] = p_pred[t]

        smoothed = np.zeros(n)
        p_smooth = np.zeros(n)
        smoothed[-1] = x_post[-1]
        p_smooth[-1] = p_post[-1]

        for t in range(n - 2, -1, -1):
            j = p_post[t] / p_pred[t + 1]
            smoothed[t] = x_post[t] + j * (smoothed[t + 1] - x_pred[t + 1])
            p_smooth[t] = p_post[t] + j * (p_smooth[t + 1] - p_pred[t + 1]) * j

        return smoothed


def _impute_column_kalman(
    series: pd.Series,
    q_ratio: float = 0.05,
    min_r: float = 1e-6,
) -> pd.Series:
    """Fill NaN gaps in a single column using interpolation + Kalman smoothing.

    Only the originally missing positions are overwritten; observed positions are
    left unchanged.
    """
    values = series.to_numpy(dtype=float)
    missing = np.isnan(values)
    if not missing.any():
        return series.copy()

    smoothed = _kalman_smoother(values, q_ratio=q_ratio, min_r=min_r)
    result = values.copy()
    result[missing] = smoothed[missing]
    return pd.Series(result, index=series.index, name=series.name)


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


def _impute_direction_column_kalman(
    series: pd.Series,
    q_ratio: float = 0.05,
    min_r: float = 1e-6,
) -> pd.Series:
    """Impute a circular direction column using circular linear interpolation.

    Kalman smoothing on circular angles is unstable in this setting (noisy wind
    direction with frequent wrap-around), so direction gaps are filled with linear
    interpolation on unwrapped angles and then wrapped back to [0, 360).
    """
    values = series.to_numpy(dtype=float)
    missing = np.isnan(values)
    if not missing.any():
        return series.copy()

    # Linear interpolation on unwrapped angles to preserve circular continuity.
    interp = (
        pd.Series(values)
        .interpolate(method="linear", limit_direction="both")
        .to_numpy(dtype=float)
    )
    unwrapped = np.unwrap(np.deg2rad(interp))
    wrapped = np.mod(np.rad2deg(unwrapped), 360.0)

    result = values.copy()
    result[missing] = wrapped[missing]
    return pd.Series(result, index=series.index, name=series.name)


def impute_dataframe_kalman(
    df: pd.DataFrame,
    q_ratio: float = 0.05,
    min_r: float = 1e-6,
) -> pd.DataFrame:
    """Impute all numeric columns using interpolation + Kalman smoothing."""
    imputed = df.copy()
    numeric_cols = imputed.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if _is_direction_column(col):
            imputed[col] = _impute_direction_column_kalman(
                imputed[col], q_ratio=q_ratio, min_r=min_r
            )
        else:
            imputed[col] = _impute_column_kalman(
                imputed[col], q_ratio=q_ratio, min_r=min_r
            )

    return imputed


def evaluate_kalman_imputation(
    corrupted: pd.DataFrame,
    ground_truth: dict[tuple[int, int, str], float],
    q_ratio: float = 0.05,
    min_r: float = 1e-6,
) -> dict:
    """Run Kalman imputation and compute errors."""
    imputed = impute_dataframe_kalman(corrupted, q_ratio=q_ratio, min_r=min_r)

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
        description="Validate interpolation + Kalman filter imputation by injecting artificial gaps."
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
        "--q-ratio",
        type=float,
        default=0.05,
        help="Ratio used to set Kalman process noise Q (default: 0.05).",
    )
    parser.add_argument(
        "--min-r",
        type=float,
        default=1e-6,
        help="Minimum observation noise variance R (default: 1e-6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "kalman_imputation_validation.json",
        help="Where to save the JSON report (default: data/kalman_imputation_validation.json).",
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

    print(f"Running Kalman imputation (q_ratio={args.q_ratio}, min_r={args.min_r})...")
    metrics = evaluate_kalman_imputation(
        corrupted,
        ground_truth,
        q_ratio=args.q_ratio,
        min_r=args.min_r,
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
                    "q_ratio": args.q_ratio,
                    "min_r": args.min_r,
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
