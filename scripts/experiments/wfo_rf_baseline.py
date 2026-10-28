"""Baseline Random Forest para o walk-forward LightGBM (mesmo protocolo).

Para cada janela: treina um RandomForestRegressor MULTI-OUTPUT (100 arvores,
seed 42) mapeando o encoder de 12 h (72 x features, mesmas colunas dos
modelos seq2seq) nos 36 passos futuros de ws100 bruto; rolling forecast com
as MESMAS origens dos modelos seq2seq (a cada 6 amostras) e a mesma censura
de alvos que cruzam o fim do bloco.

Saida em data/results/wfo_lightgbm/rf_baseline/ no mesmo formato dos modelos
(predictions_window_<block>.csv + window_metrics_<block>.json + metrics.json)
com as mesmas flags de imputacao — o relatorio consome como pseudo-modelo.

Usage:
    python scripts/experiments/wfo_rf_baseline.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor

from src import common
import wfo_lightgbm_common as proto
from wfo_notify import notify

OUTPUT_DIR = proto.RESULTS_DIR / "rf_baseline"
BASELINE_NAME = "RF (baseline)"
WS100_INDEX = list(proto.SERIES_COLUMNS).index("ws100")


def month_files(window):
    return (
        OUTPUT_DIR / f"predictions_window_{window}.csv",
        OUTPUT_DIR / f"window_metrics_{window}.json",
    )


def window_matrix(frame, positions):
    """(len(positions), INPUT_STEPS x F) encoder flatten + (len, OUTPUT_STEPS) alvos."""
    values = frame.to_numpy(dtype=np.float32)
    blocks = sliding_window_view(values, proto.INPUT_STEPS + proto.OUTPUT_STEPS, axis=0)
    selected = blocks[positions]                     # (m, F, 108)
    encoder = selected[:, :, : proto.INPUT_STEPS]    # (m, F, 72) — janela no ultimo eixo
    targets = selected[:, WS100_INDEX, proto.INPUT_STEPS:]  # (m, 36)
    return (
        np.ascontiguousarray(encoder.reshape(len(positions), -1)),
        np.ascontiguousarray(targets),
    )


def run_window(frame, flags, window):
    train = frame.loc[window["train_start"]: window["train_end"] - pd.Timedelta(minutes=10)]
    test_grid = frame.index[
        (frame.index >= window["test_start"]) & (frame.index <= window["test_end"])
    ]
    origins = test_grid[::proto.STRIDE]

    started = time.perf_counter()
    position = {ts: i for i, ts in enumerate(frame.index)}
    first_train = position[train.index[0]] + proto.INPUT_STEPS - 1
    last_train = position[train.index[-1]] - proto.OUTPUT_STEPS
    train_positions = np.arange(first_train, last_train + 1)
    X_train, y_train = window_matrix(frame, train_positions)

    forest = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    forest.fit(X_train, y_train)

    test_positions = np.array([position[o] for o in origins])
    X_test, _ = window_matrix(frame, test_positions)
    forecast = forest.predict(X_test)  # (n_origins, 36)

    rows = []
    for i, origin in enumerate(origins):
        keep_rows = []
        for horizon in range(1, proto.OUTPUT_STEPS + 1):
            ts = origin + pd.Timedelta(minutes=10 * horizon)
            if ts > window["test_end"]:  # mesma censura dos modelos
                break
            keep_rows.append((horizon, ts, float(forecast[i, horizon - 1])))
        for horizon, ts, pred in keep_rows:
            rows.append({
                "origin": origin,
                "timestamp": ts,
                "horizon": horizon,
                "predicted": pred,
                "actual_raw": float(frame.at[ts, "ws100"]),
            })

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("nenhuma previsao gerada")
    predictions = predictions.sort_values(["origin", "horizon"]).reset_index(drop=True)

    predictions["actual"] = predictions["actual_raw"]
    predictions["persistence"] = frame["ws100"].reindex(predictions["origin"]).to_numpy()
    predictions["output_imputed"] = flags["ws100"].reindex(
        predictions["timestamp"]).fillna(False).to_numpy(dtype=bool)

    flags_window = flags.reindex(frame.loc[
        window["test_start"] - pd.Timedelta(minutes=720): window["test_end"]
    ].index).fillna(False)
    flags_any = flags_window.any(axis=1).to_numpy(dtype=float)
    frac_cell = flags_window.to_numpy(dtype=float).mean(axis=1)
    grid_position = {ts: i for i, ts in enumerate(flags_window.index)}
    input_any, input_frac = [], []
    for origin in predictions["origin"].unique():
        p = grid_position[origin]
        start = p - proto.INPUT_STEPS + 1
        input_any.append(flags_any[start:p + 1].max())
        input_frac.append(frac_cell[start:p + 1].mean())
    n_origins = predictions["origin"].nunique()
    predictions["input_imputed_any"] = np.repeat(
        np.asarray(input_any, dtype=bool), proto.OUTPUT_STEPS)[: len(predictions)]
    predictions["input_imputed_frac"] = np.repeat(
        np.asarray(input_frac), proto.OUTPUT_STEPS)[: len(predictions)]

    metrics = proto.metrics_with_persistence(
        predictions["actual_raw"], predictions["predicted"], predictions["persistence"]
    )
    observed = ~predictions["output_imputed"]
    if observed.any():
        obs = common.compute_metrics(
            predictions.loc[observed, "actual_raw"], predictions.loc[observed, "predicted"]
        )
        metrics["mse_observed"], metrics["mae_observed"], metrics["rmse_observed"] = (
            obs["mse"], obs["mae"], obs["rmse"])
    if (~observed).any():
        imp = common.compute_metrics(
            predictions.loc[~observed, "actual_raw"], predictions.loc[~observed, "predicted"]
        )
        metrics["mse_imputed"], metrics["mae_imputed"], metrics["rmse_imputed"] = (
            imp["mse"], imp["mae"], imp["rmse"])
    metrics.update({
        "window": window["window"],
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "test_start": str(window["test_start"]),
        "test_end": str(window["test_end"]),
        "train_rows": int(len(train)),
        "test_rows": int(len(test_grid)),
        "n_origins": int(n_origins),
        "epochs_trained": None,
        "train_seconds": round(time.perf_counter() - started, 1),
        "params_source": "RandomForestRegressor(100, multioutput)",
        "hyperparameters": {"n_estimators": 100, "random_state": 42},
    })
    return predictions, metrics


def finalize(args):
    windows = sorted(OUTPUT_DIR.glob("window_metrics_*.json"))
    if not windows:
        print("[RF] finalize: nenhuma janela.")
        return None
    records, predictions = [], []
    for json_path in windows:
        with open(json_path, encoding="utf-8") as handle:
            record = json.load(handle)
        pred_path = OUTPUT_DIR / f"predictions_window_{record['window']}.csv"
        if not pred_path.is_file():
            continue
        records.append(record)
        predictions.append(pd.read_csv(pred_path, parse_dates=["origin", "timestamp"]))
    all_df = pd.concat(predictions, ignore_index=True)
    all_df.to_csv(OUTPUT_DIR / "predictions_all.csv", index=False)

    overall = proto.metrics_with_persistence(
        all_df["actual_raw"], all_df["predicted"], all_df["persistence"]
    )
    horizon_rows = []
    for horizon, group in all_df.groupby("horizon"):
        row = proto.metrics_with_persistence(
            group["actual_raw"], group["predicted"], group["persistence"]
        )
        row["horizon"] = int(horizon)
        horizon_rows.append(row)
    summary = {
        "settings": {
            "wrapper": BASELINE_NAME,
            "wrapper_key": "rf",
            "primary_metric": "mse",
            "protocol": {"train_months": args.train_months, "stride": proto.STRIDE},
        },
        "windows": records,
        "overall": overall,
        "by_horizon": pd.DataFrame(horizon_rows).sort_values("horizon").to_dict(orient="records"),
    }
    common.write_json(OUTPUT_DIR / "metrics.json", summary)
    print(f"[RF] finalize: {len(records)} janelas | MSE={overall['mse']:.4f} "
          f"skill={overall['skill']:.3f}")
    return OUTPUT_DIR / "metrics.json"


def main():
    global OUTPUT_DIR
    from wfo_lightgbm_run import load_series

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--train-months", type=int, default=proto.TRAIN_MONTHS)
    parser.add_argument("--series", default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: data/results/wfo_lightgbm/rf_baseline")
    args = parser.parse_args()

    if args.series:
        proto.SERIES_CSV = common.resolve(args.series)
    if args.output_dir:
        OUTPUT_DIR = common.resolve(args.output_dir)
    if not proto.SERIES_CSV.is_file():
        raise SystemExit(f"{proto.SERIES_CSV} ausente — rode o prepare primeiro.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    series, flags = load_series(proto.DATA_DIR)
    windows = proto.plan_windows(series.index, args.train_months)
    if args.max_windows:
        windows = windows[: args.max_windows]

    print(f"RF baseline: {len(windows)} janelas | saida: {OUTPUT_DIR}")
    for window in windows:
        pred_path, json_path = month_files(window["window"])
        if pred_path.is_file() and json_path.is_file():
            print(f"[{window['window']}] ja concluida — pulando")
            continue
        print(f"[{window['window']}] treino {window['train_start']:%F} -> "
              f"{window['train_end'] - pd.Timedelta(minutes=10):%F}")
        predictions, metrics = run_window(series, flags, window)
        predictions.to_csv(pred_path, index=False)
        common.write_json(json_path, metrics)
        print(f"[{window['window']}] MSE={metrics['mse']:.4f} "
              f"skill={metrics['skill']:.3f} ({metrics['train_seconds']:.0f}s)")

    result = finalize(args)
    if result is not None:
        with open(result, encoding="utf-8") as handle:
            overall = json.load(handle)["overall"]
        notify("WFO RF OK",
               f"🌲 Baseline Random Forest concluido: MSE={overall['mse']:.4f} "
               f"skill={overall['skill']:.3f} — regere o relatorio para incluir.")


if __name__ == "__main__":
    main()
