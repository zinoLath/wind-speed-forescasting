"""Impute short missing-value gaps in wind_data.csv using wavelet denoising.

The script reads the raw CSV, ensures a regular 10-minute timestamp index, identifies
consecutive NaN gaps per numeric column, and fills only short gaps (default <= 12
timesteps, i.e. 2 hours) with values derived from a wavelet-denoised reconstruction
of the surrounding signal. Longer gaps are preserved as NaN so callers can decide
whether to drop them or use another imputation method.

Usage:
    python src/impute_wind_data.py
    python src/impute_wind_data.py --input data/wind_data.csv --output data/wind_data_imputed.csv --max-gap 12 --window-factor 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import wavelet_denoising


def find_gaps(series: pd.Series) -> list[tuple[int, int]]:
    """Return a list of (start, end) integer indices for consecutive NaN runs."""
    missing = series.isna().to_numpy()
    if not missing.any():
        return []

    gaps = []
    start = None
    for i, flag in enumerate(missing):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            gaps.append((start, i))
            start = None
    if start is not None:
        gaps.append((start, len(series)))
    return gaps


def _choose_level(window_len: int, wavelet: str, requested_level: int) -> int:
    """Choose the largest usable wavelet decomposition level for the window."""
    try:
        w = pywt.Wavelet(wavelet)
    except ValueError:
        return 1
    max_level = pywt.dwt_max_level(window_len, w.dec_len)
    return max(1, min(requested_level, max_level))


def impute_short_gaps(
    series: pd.Series,
    max_gap: int = 12,
    window_factor: float = 3.0,
    wavelet: str = "sym18",
    level: int = 2,
    max_iter: int = 3,
    min_window: int = 144,
) -> pd.Series:
    """Fill short NaN gaps in *series* using iterative wavelet denoising.

    For each gap of length <= *max_gap*:
    1. Build a window of at least ``min_window`` samples (or
       ``window_factor * gap_len`` if larger) around the gap.
    2. Initialise the gap with linear interpolation.
    3. Apply wavelet denoising (``src.utils.wavelet_denoising``) to the window,
       automatically reducing the decomposition level if the window is too short.
    4. Replace only the originally missing positions with the denoised values.
    5. Repeat up to *max_iter* times, using the previous result as the new
       interpolation seed.

    Gaps longer than *max_gap* are left as NaN. Gaps without enough surrounding
    valid data (less than two valid observations inside the window) are also left
    as NaN.
    """
    result = series.copy()
    values = result.to_numpy(dtype=float)
    gaps = find_gaps(series)

    for start, end in gaps:
        gap_len = end - start
        if gap_len > max_gap:
            continue

        window_len = max(min_window // 2 - gap_len, int(window_factor * gap_len))
        win_start = max(0, start - window_len)
        win_end = min(len(values), end + window_len)
        window = values[win_start:win_end].copy()
        local_gap_start = start - win_start
        local_gap_end = end - win_start

        # We need at least two valid observations around the gap to anchor a
        # reconstruction.
        valid_mask = ~np.isnan(window)
        if valid_mask.sum() < 2:
            continue

        # Initial interpolation seed.
        interp = pd.Series(window).interpolate(method="linear", limit_direction="both")
        interp = interp.to_numpy(dtype=float)

        effective_level = _choose_level(len(window), wavelet, level)
        if effective_level < 1:
            # Not enough data for any wavelet decomposition; keep linear interpolation.
            values[start:end] = interp[local_gap_start:local_gap_end]
            continue

        for _ in range(max_iter):
            try:
                denoised = wavelet_denoising(interp, wavelet=wavelet, level=effective_level)
            except ValueError:
                # Wavelet decomposition can fail for very small windows; fall back
                # to the interpolated seed.
                break

            # Clamp the denoised estimate to the range of valid observations to
            # avoid runaway wavelet artefacts.
            valid_values = window[valid_mask]
            lo, hi = valid_values.min(), valid_values.max()
            denoised = np.clip(denoised, lo, hi)

            # Keep non-missing positions untouched; refine the gap estimate.
            refined = window.copy()
            refined[np.isnan(refined)] = denoised[np.isnan(refined)]
            interp = refined

        imputed_values = interp[local_gap_start:local_gap_end]
        values[start:end] = imputed_values

    return pd.Series(values, index=series.index, name=series.name)


def load_wind_data(path: Path) -> pd.DataFrame:
    """Load the CSV, canonicalise feature names and regularise the timestamp."""
    from src.common import canonicalize_columns

    df = pd.read_csv(path)
    df = canonicalize_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Reindex to a regular 10-minute grid, inserting NaNs for missing timestamps.
    full_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="10min",
    )
    df = df.set_index("timestamp").reindex(full_index)
    df.index.name = "timestamp"
    return df


def _is_direction_column(col: str) -> bool:
    return col.lower().startswith("dir")


def _impute_direction_column(
    series: pd.Series,
    max_gap: int = 12,
    window_factor: float = 3.0,
    wavelet: str = "sym18",
    level: int = 2,
    max_iter: int = 3,
    min_window: int = 144,
) -> pd.Series:
    """Impute a circular direction column via sin/cos decomposition.

    Direction columns (names starting with ``dir``) are wrapped around 0/360.
    Imputing them in degrees directly causes artefacts near the wrap-around. We
    instead impute the sine and cosine components independently and reconstruct
    the angle.
    """
    radians = np.deg2rad(series.to_numpy(dtype=float))
    sin_component = pd.Series(np.sin(radians), index=series.index)
    cos_component = pd.Series(np.cos(radians), index=series.index)

    sin_imputed = impute_short_gaps(
        sin_component,
        max_gap=max_gap,
        window_factor=window_factor,
        wavelet=wavelet,
        level=level,
        max_iter=max_iter,
        min_window=min_window,
    )
    cos_imputed = impute_short_gaps(
        cos_component,
        max_gap=max_gap,
        window_factor=window_factor,
        wavelet=wavelet,
        level=level,
        max_iter=max_iter,
        min_window=min_window,
    )

    reconstructed = np.rad2deg(np.arctan2(sin_imputed.to_numpy(), cos_imputed.to_numpy()))
    reconstructed = np.mod(reconstructed, 360.0)
    return pd.Series(reconstructed, index=series.index, name=series.name)


def impute_dataframe(
    df: pd.DataFrame,
    max_gap: int = 12,
    window_factor: float = 3.0,
    wavelet: str = "sym18",
    level: int = 2,
    max_iter: int = 3,
    min_window: int = 144,
) -> pd.DataFrame:
    """Impute short gaps in every numeric column of *df*."""
    imputed = df.copy()
    numeric_cols = imputed.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if _is_direction_column(col):
            imputed[col] = _impute_direction_column(
                imputed[col],
                max_gap=max_gap,
                window_factor=window_factor,
                wavelet=wavelet,
                level=level,
                max_iter=max_iter,
                min_window=min_window,
            )
        else:
            imputed[col] = impute_short_gaps(
                imputed[col],
                max_gap=max_gap,
                window_factor=window_factor,
                wavelet=wavelet,
                level=level,
                max_iter=max_iter,
                min_window=min_window,
            )

    return imputed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Impute short missing-value gaps in wind_data.csv using wavelet denoising."
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
        default=PROJECT_ROOT / "data" / "wind_data_imputed.csv",
        help="Path to the output CSV (default: data/wind_data_imputed.csv).",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=12,
        help="Maximum gap length (in timesteps) to impute with wavelets (default: 12).",
    )
    parser.add_argument(
        "--window-factor",
        type=float,
        default=3.0,
        help="Multiplier for the gap length used to define the surrounding window (default: 3.0).",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="sym18",
        help="Wavelet used for denoising (default: sym18).",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=2,
        help="Wavelet decomposition level (default: 2).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Number of wavelet imputation iterations (default: 3).",
    )
    parser.add_argument(
        "--min-window",
        type=int,
        default=144,
        help="Minimum window size (in samples) around a gap for wavelet denoising (default: 144).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.input}...")
    df = load_wind_data(args.input)
    print(f"Shape after regularisation: {df.shape}")
    print(f"Missing values before imputation: {df.isna().sum().sum()}")

    print("Imputing short gaps...")
    imputed = impute_dataframe(
        df,
        max_gap=args.max_gap,
        window_factor=args.window_factor,
        wavelet=args.wavelet,
        level=args.level,
        max_iter=args.max_iter,
        min_window=args.min_window,
    )
    print(f"Missing values after imputation: {imputed.isna().sum().sum()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imputed.to_csv(args.output)
    print(f"Imputed data saved to {args.output}")


if __name__ == "__main__":
    main()
