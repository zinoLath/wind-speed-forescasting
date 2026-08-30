"""Detection and export of complete (gap-free) periods in the wind data.

Recreates the logic that previously lived only in ``data/zino-view.ipynb``:
contiguous runs where the focus columns are all present (zero is treated as
missing) are exported as ``complete_period_<i>.csv`` slices plus a
``complete_periods_summary.json`` index.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

FOCUS_COLUMNS = ["ws40", "ws100", "v40", "dir40", "ws50"]


def find_complete_periods(df, columns=None, min_length=1):
    """Return the contiguous periods where all *columns* are present.

    Zero values count as missing (sensor dropouts report 0). Returns a list
    of dicts with keys period, start, end and length, in chronological order.
    """
    columns = list(columns or FOCUS_COLUMNS)
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in the dataset: {missing_cols}")

    focus = df[columns].replace(0, pd.NA)
    complete = focus.notna().all(axis=1).to_numpy()

    periods = []
    start = None
    for i, flag in enumerate(complete):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            if i - start >= min_length:
                periods.append((start, i))
            start = None
    if start is not None and len(complete) - start >= min_length:
        periods.append((start, len(complete)))

    return [
        {
            "period": i,
            "start": str(df.index[s]),
            "end": str(df.index[e - 1]),
            "length": e - s,
        }
        for i, (s, e) in enumerate(periods)
    ]


def save_complete_periods(df, output_dir, columns=None, min_length=1):
    """Export the complete periods as CSV slices plus a JSON summary index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    periods = find_complete_periods(df, columns=columns, min_length=min_length)
    for info in periods:
        s = df.index.get_loc(pd.Timestamp(info["start"]))
        e = df.index.get_loc(pd.Timestamp(info["end"])) + 1
        df.iloc[s:e].to_csv(output_dir / f"complete_period_{info['period']}.csv")

    with open(output_dir / "complete_periods_summary.json", "w", encoding="utf-8") as handle:
        json.dump(periods, handle, indent=4)
    return periods


def main():
    parser = argparse.ArgumentParser(description="Export complete (gap-free) periods.")
    parser.add_argument("--input", type=Path, default=Path("data/wind_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/series_list"))
    parser.add_argument("--min-length", type=int, default=108,
                        help="Minimum period length in rows (default: 108 = 18h).")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    periods = save_complete_periods(df, args.output_dir, min_length=args.min_length)
    print(f"{len(periods)} complete periods saved to {args.output_dir}")


if __name__ == "__main__":
    main()
