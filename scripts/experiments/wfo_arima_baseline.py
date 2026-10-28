"""Baseline ARIMA para o walk-forward LightGBM (mesmo protocolo).

Para cada janela mensal: treina ARIMA nos 3 meses civis anteriores (ordem
selecionada por AIC entre ORDERS), e faz rolling forecast com as MESMAS
origens dos modelos seq2seq (a cada 6 amostras), prevendo 36 passos e
descartando alvos que cruzam o fim do mes (mesma censura dos modelos).
Observacoes intermediarias sao incorporadas via append(refit=False).

Saida em data/results/wfo_lightgbm/arima_baseline/ no mesmo formato dos
modelos (predictions_window_<YYYYMM>.csv + window_metrics_<YYYYMM>.json +
metrics.json), com as mesmas colunas de flag de imputacao — o relatorio
consumo como um pseudo-modelo.

Usage:
    python scripts/experiments/wfo_arima_baseline.py [--wrappers arima]  # idempotente
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
from statsmodels.tsa.arima.model import ARIMA

from src import common
import wfo_lightgbm_common as proto
from wfo_notify import notify

ORDERS = ((2, 0, 2), (3, 0, 1), (2, 1, 2), (1, 1, 1))
OUTPUT_DIR = proto.RESULTS_DIR / "arima_baseline"
BASELINE_NAME = "ARIMA (baseline)"


def month_files(window):
    return (
        OUTPUT_DIR / f"predictions_window_{window}.csv",
        OUTPUT_DIR / f"window_metrics_{window}.json",
    )


def fit_order(train):
    """Escolhe a ordem por AIC (menor) entre os candidatos."""
    best = None
    for order in ORDERS:
        try:
            res = ARIMA(train, order=order, enforce_stationarity=False,
                        enforce_invertibility=False).fit()
        except Exception as exc:
            print(f"  ordem {order} falhou: {exc!r}")
            continue
        if best is None or res.aic < best[1].aic:
            best = (order, res)
    return best


def run_window(series, flags, window):
    train = series.loc[window["train_start"]:window["train_end"] - pd.Timedelta(minutes=10)]
    test_grid = series.index[
        (series.index >= window["test_start"]) & (series.index <= window["test_end"])
    ]
    origins = test_grid[::proto.STRIDE]

    started = time.perf_counter()
    picked, res = fit_order(train)
    print(f"  ordem={picked} (aic={res.aic:.0f})")

    rows = []
    flags_window = flags.reindex(series.loc[window["test_start"] - pd.Timedelta(minutes=720):
                                            window["test_end"]].index).fillna(False)
    # incorpora as observacoes ate a PRIMEIRA origem (o ultimo ponto observado
    # dela), senao o append criaria um buraco apos o fim do treino
    current = res.append(
        series.loc[train.index[-1] + pd.Timedelta(minutes=10):origins[0]], refit=False
    )
    previous = origins[0]
    for origin in origins:
        if origin != previous:
            new_obs = series.loc[previous + pd.Timedelta(minutes=10):origin]
            current = current.append(new_obs, refit=False)
        forecast = current.forecast(proto.OUTPUT_STEPS)
        keep = forecast.index <= window["test_end"]  # mesma censura dos modelos
        for horizon, (ts, pred) in enumerate(forecast[keep].items(), start=1):
            rows.append({
                "origin": origin,
                "timestamp": ts,
                "horizon": horizon,
                "predicted": float(pred),
                "actual_raw": float(series.loc[ts]),
            })
        previous = origin

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("nenhuma previsao gerada")
    predictions = predictions.sort_values(["origin", "horizon"]).reset_index(drop=True)

    # flags identicas aos modelos: alvo imputado + imputacao na janela de entrada
    predictions["actual"] = predictions["actual_raw"]
    predictions["persistence"] = series.reindex(predictions["origin"]).to_numpy()
    predictions["output_imputed"] = flags["ws100"].reindex(
        predictions["timestamp"]).fillna(False).to_numpy(dtype=bool)
    flags_any = flags_window.any(axis=1).to_numpy(dtype=float)
    frac_cell = flags_window.to_numpy(dtype=float).mean(axis=1)
    origin_positions = np.searchsorted(flags_window.index, predictions["origin"].unique())
    input_any, input_frac = [], []
    for position in origin_positions:
        start = position - proto.INPUT_STEPS + 1
        input_any.append(flags_any[start:position + 1].max())
        input_frac.append(frac_cell[start:position + 1].mean())
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
        "params_source": f"ARIMA{picked} (AIC)",
        "hyperparameters": {"order": list(picked)},
    })
    return predictions, metrics


def finalize(args):
    windows = sorted(OUTPUT_DIR.glob("window_metrics_*.json"))
    if not windows:
        print("[ARIMA] finalize: nenhuma janela.")
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
            "wrapper_key": "arima",
            "primary_metric": "mse",
            "protocol": {"train_months": args.train_months, "stride": proto.STRIDE},
        },
        "windows": records,
        "overall": overall,
        "by_horizon": pd.DataFrame(horizon_rows).sort_values("horizon").to_dict(orient="records"),
    }
    common.write_json(OUTPUT_DIR / "metrics.json", summary)
    print(f"[ARIMA] finalize: {len(records)} janelas | MSE={overall['mse']:.4f} "
          f"skill={overall['skill']:.3f}")
    return OUTPUT_DIR / "metrics.json"


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--train-months", type=int, default=proto.TRAIN_MONTHS)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: data/results/wfo_lightgbm/arima_baseline")
    args = parser.parse_args()

    if args.output_dir:
        OUTPUT_DIR = common.resolve(args.output_dir)
    if not proto.SERIES_CSV.is_file():
        raise SystemExit(f"{proto.SERIES_CSV} ausente — rode o prepare primeiro.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(proto.SERIES_CSV, index_col="timestamp", parse_dates=True)["ws100"]
    # freq explicita: o append do statsmodels exige que os novos indices
    # estendam o indice do modelo com a mesma frequencia (10 min)
    series.index = pd.DatetimeIndex(series.index, freq="10min")
    flags = pd.read_csv(proto.FLAGS_CSV, index_col="timestamp",
                        parse_dates=True, dtype=bool)["ws100"].to_frame()
    windows = proto.plan_windows(series.index, args.train_months)
    if args.max_windows:
        windows = windows[: args.max_windows]

    print(f"ARIMA baseline: {len(windows)} janelas (treino {args.train_months} meses) "
          f"| ordens candidatas {ORDERS} | saida: {OUTPUT_DIR}")
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
        notify("WFO ARIMA OK",
               f"📈 Baseline ARIMA concluido: MSE={overall['mse']:.4f} "
               f"skill={overall['skill']:.3f} — regere o relatorio para incluir.")


if __name__ == "__main__":
    main()
