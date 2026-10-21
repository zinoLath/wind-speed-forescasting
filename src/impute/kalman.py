"""Column-wise Kalman imputation (interpolation + Kalman smoothing)."""

import numpy as np
import pandas as pd

from src.impute import base


def _kalman_smoother(series, q_ratio=0.05, min_r=1e-6, use_trend=True):
    """Apply a Kalman smoother to a 1D signal, treating NaNs as missing observations.

    By default a local linear trend model (state = [position, velocity]) is
    used; ``use_trend=False`` falls back to a random walk. ``q_ratio`` scales
    the process noise by the variance of the first differences of the linearly
    interpolated signal; smaller values produce smoother estimates.
    """
    n = len(series)
    if n < 5:
        return pd.Series(series).interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)

    interpolated = pd.Series(series).interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)

    # Observation noise R estimated from the residual w.r.t. a rolling median.
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

    diff_var = np.nanvar(np.diff(interpolated))

    if use_trend:
        # Local linear trend model: state = [position, velocity].
        transition = np.array([[1.0, 1.0], [0.0, 1.0]])
        observation = np.array([[1.0, 0.0]])
        q = np.diag([
            max(q_ratio * diff_var, min_r * 1e-3),
            max(q_ratio * diff_var / (n ** 2), min_r * 1e-6),
        ])
        big_q = q
        big_r = np.array([[r]])

        x = np.array([interpolated[0], 0.0])
        p = np.diag([np.nanvar(interpolated), diff_var])

        identity = np.eye(2)
        x_pred, p_pred = np.zeros((n, 2)), np.zeros((n, 2, 2))
        x_post, p_post = np.zeros((n, 2)), np.zeros((n, 2, 2))

        for t in range(n):
            if t == 0:
                x_pred[t], p_pred[t] = x, p
            else:
                x_pred[t] = transition @ x_post[t - 1]
                p_pred[t] = transition @ p_post[t - 1] @ transition.T + big_q

            if not np.isnan(series[t]):
                innovation = series[t] - observation @ x_pred[t]
                gain = p_pred[t] @ observation.T @ np.linalg.inv(
                    observation @ p_pred[t] @ observation.T + big_r
                )
                x_post[t] = x_pred[t] + (gain @ innovation).ravel()
                p_post[t] = (identity - gain @ observation) @ p_pred[t]
            else:
                x_post[t], p_post[t] = x_pred[t], p_pred[t]

        # Rauch-Tung-Striebel backward smoother.
        smoothed = np.zeros((n, 2))
        smoothed[-1] = x_post[-1]
        for t in range(n - 2, -1, -1):
            j = p_post[t] @ transition.T @ np.linalg.inv(p_pred[t + 1])
            smoothed[t] = x_post[t] + j @ (smoothed[t + 1] - x_pred[t + 1])

        return smoothed[:, 0]

    # Simple random walk.
    q = max(q_ratio * diff_var, min_r * 1e-3)
    x_post, p_post = np.zeros(n), np.zeros(n)
    x_pred, p_pred = np.zeros(n), np.zeros(n)
    x_post[0], p_post[0] = interpolated[0], r

    for t in range(1, n):
        x_pred[t] = x_post[t - 1]
        p_pred[t] = p_post[t - 1] + q
        if not np.isnan(series[t]):
            gain = p_pred[t] / (p_pred[t] + r)
            x_post[t] = x_pred[t] + gain * (series[t] - x_pred[t])
            p_post[t] = (1.0 - gain) * p_pred[t]
        else:
            x_post[t], p_post[t] = x_pred[t], p_pred[t]

    smoothed = np.zeros(n)
    smoothed[-1] = x_post[-1]
    for t in range(n - 2, -1, -1):
        j = p_post[t] / p_pred[t + 1]
        smoothed[t] = x_post[t] + j * (smoothed[t + 1] - x_pred[t + 1])
    return smoothed


def _impute_column(series, q_ratio, min_r):
    """Fill NaN gaps in one column; observed positions stay untouched."""
    values = series.to_numpy(dtype=float)
    missing = np.isnan(values)
    if not missing.any():
        return series.copy()

    smoothed = _kalman_smoother(values, q_ratio=q_ratio, min_r=min_r)
    values[missing] = smoothed[missing]
    return pd.Series(values, index=series.index, name=series.name)


def _impute_direction_column(series, q_ratio, min_r):
    """Fill direction gaps with circular interpolation.

    Kalman smoothing on circular angles is unstable here (frequent 0/360
    wrap-around), so gaps are linearly interpolated on unwrapped angles.
    """
    values = series.to_numpy(dtype=float)
    missing = np.isnan(values)
    if not missing.any():
        return series.copy()

    interp = pd.Series(values).interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)
    wrapped = np.mod(np.rad2deg(np.unwrap(np.deg2rad(interp))), 360.0)
    values[missing] = wrapped[missing]
    return pd.Series(values, index=series.index, name=series.name)


def impute_dataframe(df, q_ratio=0.05, min_r=1e-6):
    """Impute every numeric column: Kalman smoothing (circular interp for directions)."""
    imputed = df.copy()
    for col in imputed.select_dtypes(include=[np.number]).columns:
        if base.is_direction_column(col):
            imputed[col] = _impute_direction_column(imputed[col], q_ratio, min_r)
        else:
            imputed[col] = _impute_column(imputed[col], q_ratio, min_r)
    return imputed