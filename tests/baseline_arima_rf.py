"""Statistical baselines (ARIMA and Random Forest) on the seq2seq protocol.

Runs the classical baselines missing from the model comparison, using the
same data, split and rolling-forecast protocol as the pipeline's evaluate
stage, so the numbers land on the same table as the deep models:

- dataset: ``data/dataset.csv`` loaded through ``common.load_dataset``
  (the same cyclic sin/cos features the seq2seq encoders consume);
- split: chronological 75/20/5 via ``common.split_dataset``;
- target: RAW ``ws100`` (headline convention; no wavelet smoothing —
  denoising is part of the deep models' method, not of the baselines);
- protocol: for every origin ``i`` from ``val_end`` onward (the same grid
  as ``wrapper.rolling_forecast(dataset, test_start=val_end)``) the model
  observes only the history before ``i`` and forecasts the next 36 steps;
  persistence (last observed value repeated) is measured on the same grid.

ARIMA is a univariate SARIMAX: the order is selected by AIC on the training
slice (small p/q grid), fitted once, and then rolled forward with
``append(refit=False)`` + ``forecast(36)`` per origin — no refitting, so no
information from validation/test ever leaks into estimation.

Random Forest is a direct multi-output regressor: input is the flattened
multivariate window of the last ``--rf-window`` rows (same columns the
encoders see), output is the next 36 target values; it is fitted only on
windows fully contained in the training slice.

Outputs (default ``data/results/baselines/arima_rf/``): metrics.json,
per-horizon metrics CSV, long-format predictions CSV and comparison plots.

Standalone script (no pytest): ``python tests/baseline_arima_rf.py``.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import common


def parse_args():
    parser = argparse.ArgumentParser(description="ARIMA/RF baselines on the seq2seq rolling protocol.")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to dataset.csv (default: paths.dataset_csv layout, i.e. data/dataset.csv).")
    parser.add_argument("--target", default="ws100",
                        help="Raw target column (default: ws100).")
    parser.add_argument("--input-steps", type=int, default=72,
                        help="Observed history length in samples (default: 72 = 12 h).")
    parser.add_argument("--output-steps", type=int, default=36,
                        help="Forecast horizon in samples (default: 36 = 6 h).")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--arima-order", default="auto",
                        help="ARIMA order 'p,d,q' or 'auto' for the AIC grid search (default: auto).")
    parser.add_argument("--rf-window", type=int, default=72,
                        help="Multivariate window length flattened for the RF input (default: 72).")
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--limit-origins", type=int, default=None,
                        help="Smoke-test switch: evaluate only the first N origins.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data/results/baselines/arima_rf"))
    return parser.parse_args()


def build_origin_grid(dataset, target_col, val_end, output_steps, limit=None):
    """Long-format skeleton shared by every model: one row per origin x horizon."""
    n = len(dataset)
    origins = np.arange(val_end, n - output_steps + 1)
    if limit is not None:
        origins = origins[:limit]
    sequence_count = len(origins)
    target_positions = origins[:, None] + np.arange(output_steps)[None, :]

    return pd.DataFrame({
        "origin": np.repeat(dataset.index.to_numpy()[origins], output_steps),
        "timestamp": dataset.index.to_numpy()[target_positions].ravel(),
        "horizon": np.tile(np.arange(1, output_steps + 1), sequence_count),
        "actual": dataset[target_col].to_numpy()[target_positions].ravel(),
        "persistence": np.repeat(dataset[target_col].to_numpy()[origins - 1], output_steps),
    })


def select_arima_order(train_target, max_p=3, max_q=3, d_values=(0, 1)):
    """Small AIC grid search over non-seasonal (p, d, q) on the training slice."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    table = []
    for d in d_values:
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    fit = SARIMAX(
                        train_target, order=(p, d, q),
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).fit(disp=False)
                    table.append({"order": f"{p},{d},{q}", "aic": float(fit.aic)})
                except Exception as error:  # noqa: BLE001 - a failed cell must not stop the search
                    table.append({"order": f"{p},{d},{q}", "aic": None, "error": str(error)})
                print(f"  ARIMA({p},{d},{q}) aic={table[-1]['aic']}")

    valid = [row for row in table if row["aic"] is not None and np.isfinite(row["aic"])]
    if not valid:
        raise RuntimeError("Every ARIMA candidate failed during the AIC grid search.")
    best = min(valid, key=lambda row: row["aic"])
    return tuple(int(v) for v in best["order"].split(",")), table


def run_arima(full_target, train_target, val_end, origins, output_steps, order):
    """Roll ARIMA forward: fit once on train, append/reforecast per origin."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit = SARIMAX(
        train_target, order=order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    print(f"  fitted ARIMA{order} on {len(train_target)} train samples")

    predictions = np.empty((len(origins), output_steps))
    appended_to = val_end
    started = time.time()
    for k, origin in enumerate(origins):
        if origin > appended_to:
            fit = fit.append(full_target[appended_to:origin], refit=False)
            appended_to = origin
        predictions[k] = fit.forecast(output_steps)
        if (k + 1) % 100 == 0:
            rate = (k + 1) / (time.time() - started)
            print(f"  origin {k + 1}/{len(origins)} ({rate:.1f} origins/s)")
    return predictions


def run_rf(dataset, target_col, rf_window, output_steps, val_end, origins, n_estimators, seed):
    """Direct multi-output Random Forest on flattened multivariate windows."""
    from sklearn.ensemble import RandomForestRegressor

    values = dataset.to_numpy(dtype=np.float32)
    n_rows, n_features = values.shape
    target = values[:, dataset.columns.get_loc(target_col)]

    # enc_windows[k] covers rows [k, k + rf_window), so the input window for
    # origin i sits at index i - rf_window.
    enc_windows = sliding_window_view(values, (rf_window, n_features))[:, 0]
    target_windows = sliding_window_view(target, rf_window + output_steps)
    n_train_sequences = val_end - rf_window - output_steps + 1
    if n_train_sequences <= 0:
        raise ValueError("Training slice shorter than rf_window + output_steps.")

    X_train = np.ascontiguousarray(enc_windows[:n_train_sequences]).reshape(n_train_sequences, -1)
    y_train = np.ascontiguousarray(target_windows[:n_train_sequences, rf_window:])
    print(f"  RF fit: {X_train.shape[0]} windows x {X_train.shape[1]} features -> {output_steps} outputs")

    positions = origins - rf_window
    if positions.min() < 0:
        raise ValueError("An origin precedes the first complete RF window.")
    X_pred = np.ascontiguousarray(enc_windows[positions]).reshape(len(origins), -1)

    regressor = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, random_state=seed,
    )
    regressor.fit(X_train, y_train)
    return regressor.predict(X_pred)


def summarize(frame, output_steps):
    """Pooled, rolling (last horizon) and per-horizon metrics for one model."""
    pooled = common.compute_metrics(frame["actual"], frame["predicted"])
    rolling = frame[frame["horizon"] == output_steps]
    rolling_metrics = common.compute_metrics(rolling["actual"], rolling["predicted"])
    per_horizon = common.horizon_metrics(frame)
    return pooled, rolling_metrics, per_horizon


def plot_comparison(per_horizon, out_dir):
    """Per-horizon MAE/RMSE lines for every {model}_mae column of *per_horizon*."""
    models = [c[:-4] for c in per_horizon.columns if c.endswith("_mae")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in models:
        axes[0].plot(per_horizon["horizon"], per_horizon[f"{name}_mae"], marker="o", label=name)
        axes[1].plot(per_horizon["horizon"], per_horizon[f"{name}_rmse"], marker="o", label=name)
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        ax.set_title(f"{metric} by forecast horizon")
        ax.set_xlabel("Horizon (10-min steps)")
        ax.set_ylabel(f"{metric} (m/s)")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_horizon_comparison.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    dataset_path = args.dataset or common.resolve("data/dataset.csv")
    dataset = common.load_dataset(dataset_path)
    if args.target not in dataset.columns:
        raise ValueError(f"Target '{args.target}' not found in the loaded dataset.")

    _, _, test_df = common.split_dataset(dataset, args.train_ratio, args.val_ratio)
    val_end = len(dataset) - len(test_df)
    target = dataset[args.target].to_numpy(dtype=np.float64)
    print(
        f"Dataset: {len(dataset)} rows | train={val_end} test={len(test_df)} "
        f"| target={args.target} | window {args.input_steps}->{args.output_steps}"
    )

    frame = build_origin_grid(dataset, args.target, val_end, args.output_steps, args.limit_origins)
    origins = np.arange(val_end, val_end + frame["origin"].nunique())

    results = {}

    if args.arima_order != "skip":
        print("[ARIMA]")
        if args.arima_order == "auto":
            order, selection_table = select_arima_order(target[:val_end])
            print(f"  selected order {order} by AIC")
        else:
            order = tuple(int(v) for v in args.arima_order.split(","))
            selection_table = None
        started = time.time()
        predictions = run_arima(target, target[:val_end], val_end, origins, args.output_steps, order)
        frame["arima"] = predictions.ravel()
        print(f"  rolling forecasts done in {time.time() - started:.1f}s")
        results["arima"] = {"order": list(order), "aic_selection": selection_table}

    if args.rf_estimators > 0:
        print("[Random Forest]")
        started = time.time()
        predictions = run_rf(
            dataset, args.target, args.rf_window, args.output_steps,
            val_end, origins, args.rf_estimators, args.seed,
        )
        frame["rf"] = predictions.ravel()
        print(f"  fit+predict done in {time.time() - started:.1f}s")
        results["rf"] = {"n_estimators": args.rf_estimators, "window": args.rf_window}

    out_dir = common.resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_horizon_frames = []
    for model in ("arima", "rf", "persistence"):
        if model not in frame.columns:
            continue
        model_frame = frame.rename(columns={model: "predicted"})
        if "persistence" not in model_frame.columns:
            # horizon_metrics expects the persistence reference column.
            model_frame["persistence"] = model_frame["predicted"]
        pooled, rolling_metrics, per_horizon = summarize(model_frame, args.output_steps)
        entry = results.setdefault(model, {})
        entry["all_horizons"] = pooled
        entry["rolling"] = rolling_metrics
        flat = per_horizon[["horizon", "mae", "rmse", "r2"]].copy()
        flat.columns = ["horizon", f"{model}_mae", f"{model}_rmse", f"{model}_r2"]
        per_horizon_frames.append(flat)
        print(
            f"[{model}] all-horizon MAE={pooled['mae']:.4f} RMSE={pooled['rmse']:.4f} R2={pooled['r2']:.4f} | "
            f"rolling(h={args.output_steps}) MAE={rolling_metrics['mae']:.4f} RMSE={rolling_metrics['rmse']:.4f}"
        )

    combined = per_horizon_frames[0]
    for extra in per_horizon_frames[1:]:
        combined = combined.merge(extra, on="horizon")
    combined.to_csv(out_dir / "per_horizon_metrics.csv", index=False)
    plot_comparison(combined, out_dir)

    frame.to_csv(out_dir / "predictions_long.csv", index=False)
    results["config"] = {
        "dataset": str(dataset_path),
        "target": args.target,
        "input_steps": args.input_steps,
        "output_steps": args.output_steps,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "val_end": val_end,
        "n_origins": int(frame["origin"].nunique()),
        "rf_window": args.rf_window,
        "seed": args.seed,
        "protocol": "origins from val_end; history = everything before the origin; persistence on the same grid",
    }
    common.write_json(out_dir / "metrics.json", results)
    print(f"Results saved to {out_dir}")
    return results


if __name__ == "__main__":
    main()
