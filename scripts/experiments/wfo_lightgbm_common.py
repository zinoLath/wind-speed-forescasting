"""Protocolo compartilhado do experimento walk-forward com imputacao LightGBM.

Serie de entrada: LiDAR (data/dataset.csv, set-nov/2021, alturas 40-140 m)
concatenado com SODAR (data/wind_data.csv, nov/2021-out/2022), tudo imputado
com LightGBM e flag por celula. Protocolo: janela de treino = 3 meses civis
anteriores, teste = 1 mes civil, avanco mes a mes; rolling forecast com
origens a cada STRIDE amostras (6 x 10 min = 1 h) e horizonte de 36 passos.

Este modulo NAO importa TensorFlow (seguro para o relatorio e para o collect).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common  # noqa: E402

# --- Protocolo ---------------------------------------------------------------
WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
LOCAL_WRAPPERS = ("lstm", "lstm_bi")
KAGGLE_WRAPPERS = ("tcn", "tcn_bi")
TRAIN_MONTHS = 3          # janela de treino: 3 meses civis anteriores
STRIDE = 6                # origens do rolling forecast: a cada 6 amostras (1 h)
INPUT_STEPS = 72          # 12 h de encoder
OUTPUT_STEPS = 36         # horizonte de 6 h
TARGET_COL = "ws100_wavelet"
DENOISE = ("ws100",)
DENOISE_LEVEL = 2
BATCH_SIZE = 32
EPOCHS = 60
PATIENCE = 12
TRAIN_VAL_RATIO = 0.8
SEED = 42
ROWS_PER_DAY = 24 * 6     # dado de 10 minutos

# --- Caminhos ----------------------------------------------------------------
DATA_DIR = common.resolve("data/wfo_lightgbm")
SERIES_CSV = DATA_DIR / "series_imputed.csv"
FLAGS_CSV = DATA_DIR / "imputed_flags.csv"
META_JSON = DATA_DIR / "series_meta.json"
VALIDATION_DIR = DATA_DIR / "imputation_validation"
RESULTS_DIR = common.resolve("data/results/wfo_lightgbm")
PROGRESS_JSON = RESULTS_DIR / "progress.json"
# O relatório curado mora no hub (reports/imputacao/<slug>/report.html);
# data/results guarda apenas os agregados do walk-forward.
REPORT_HTML = (common.resolve("reports/imputacao")
               / RESULTS_DIR.name.replace("_", "-") / "report.html")

# Colunas numericas da serie combinada (iguais nos dois instrumentos).
SERIES_COLUMNS = tuple(
    f"{prefix}{h}"
    for prefix in ("ws", "v", "dir")
    for h in common.CANONICAL_HEIGHTS
)


def plan_windows(index, train_months=TRAIN_MONTHS, test_days=30):
    """Janelas do rolling forecast em blocos de *test_days* dias.

    Treino = os *train_months* meses (~30 d cada) imediatamente anteriores a
    cada bloco de teste; o primeiro bloco comeca apos o treino inicial e os
    blocos avancam *test_days* por vez, sem sobreposicao.
    """
    train_days = train_months * 30
    start = index.min().normalize()
    first_test = start + pd.Timedelta(days=train_days)
    windows = []
    while True:
        test_start = first_test
        test_end = test_start + pd.Timedelta(days=test_days) - pd.Timedelta(minutes=10)
        if test_end > index.max():
            break
        train_start = test_start - pd.Timedelta(days=train_days)
        expected_train_rows = int((test_start - train_start) / pd.Timedelta(minutes=10))
        in_train = index[(index >= train_start) & (index < test_start)]
        in_test = index[(index >= test_start) & (index <= test_end)]
        windows.append({
            "window": test_start.strftime("%Y%m%d"),
            "train_start": train_start,
            "train_end": test_start,          # exclusivo
            "test_start": test_start,
            "test_end": test_end,             # inclusivo
            "train_rows": len(in_train),
            "test_rows": len(in_test),
            "train_coverage": len(in_train) / expected_train_rows if expected_train_rows else np.nan,
        })
        first_test += pd.Timedelta(days=test_days)
    return windows


def month_window_files(result_dir, window):
    """Caminhos dos artefatos de uma janela (CSV de predicoes + JSON de metricas).

    O JSON e gravado por ultimo e funciona como marcador de conclusao.
    """
    result_dir = Path(result_dir)
    return (
        result_dir / f"predictions_window_{window}.csv",
        result_dir / f"window_metrics_{window}.json",
    )


def window_is_complete(result_dir, window):
    """Janela pronta = predictions CSV + metrics JSON legivel no disco."""
    pred_path, json_path = month_window_files(result_dir, window)
    if not (pred_path.is_file() and json_path.is_file()):
        return False
    try:
        with open(json_path, encoding="utf-8") as handle:
            json.load(handle)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def used_ram_gb():
    """RAM usada (MemTotal - MemAvailable) em GB, lida de /proc/meminfo."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            info = {
                line.split(":")[0]: int(line.split()[1])
                for line in handle if ":" in line
            }
        return (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024
    except (OSError, KeyError):
        return float("nan")


def update_progress(wrapper, done, total, status="running", extra=None):
    """Atualiza o heartbeat global data/results/wfo_lightgbm/progress.json."""
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    progress = {}
    if PROGRESS_JSON.is_file():
        try:
            with open(PROGRESS_JSON, encoding="utf-8") as handle:
                progress = json.load(handle)
        except json.JSONDecodeError:
            progress = {}
    progress.setdefault("wrappers", {})[wrapper] = {
        "done": done,
        "total": total,
        "status": status,
        "ram_gb": round(used_ram_gb(), 2),
        "updated": pd.Timestamp.now().isoformat(),
        **(extra or {}),
    }
    progress["updated"] = pd.Timestamp.now().isoformat()
    with open(PROGRESS_JSON, "w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2, default=str)


def predict_all_horizons_strided(wrapper, eval_df, stride=STRIDE, batch_size=256):
    """Versao com passo de *stride* origens de ``common.predict_all_horizons``.

    Mesmo contrato do original (long-format origin/timestamp/horizon/actual/
    predicted/persistence), porem so origens multiplas de *stride* sao
    previstas — uma unica chamada batched de ``model.predict``.
    """
    input_steps = wrapper.input_steps
    output_steps = wrapper.output_steps

    prepared, _ = wrapper.prepare_data(
        eval_df,
        input_steps,
        output_steps,
        wrapper.target_col,
        scaler_target=wrapper.scaler_target,
        scaler_other=wrapper.scaler_other,
        denoise=wrapper.denoise,
        decoder_mode=wrapper.decoder_mode,
        target_mode=wrapper.target_mode,
        features=getattr(wrapper, "features", None),
    )
    if wrapper.decoder_mode == "teacher_forcing":
        prepared["X_decoder"][:, :, 0] = prepared["X_decoder"][:, :1, 0]

    n_total = len(prepared["X_encoder"])
    if n_total <= 0:
        return pd.DataFrame(
            columns=["origin", "timestamp", "horizon", "actual", "predicted", "persistence"]
        )
    selected = np.arange(0, n_total, stride)
    X_encoder = prepared["X_encoder"][selected]
    X_decoder = prepared["X_decoder"][selected]

    predicted_scaled = wrapper.model.predict(
        [X_encoder, X_decoder], batch_size=batch_size, verbose=0
    )[:, :, 0]
    actual_scaled = prepared["y_decoder"][selected][:, :, 0]
    persistence_scaled = X_encoder[:, -1, wrapper.target_col_index]

    if wrapper.target_mode == "residual":
        predicted_scaled = predicted_scaled + persistence_scaled[:, None]
        actual_scaled = actual_scaled + persistence_scaled[:, None]

    def _inverse(values):
        return wrapper.scaler_target.inverse_transform(
            values.reshape(-1, 1)
        ).reshape(values.shape)

    predicted = _inverse(predicted_scaled)
    actual = _inverse(actual_scaled)
    persistence = wrapper.scaler_target.inverse_transform(
        persistence_scaled.reshape(-1, 1)
    ).ravel()

    origins = eval_df.index.to_numpy()[selected + input_steps - 1]
    target_positions = (
        selected[:, None] + input_steps + np.arange(output_steps)[None, :]
    )

    return pd.DataFrame({
        "origin": np.repeat(origins, output_steps),
        "timestamp": eval_df.index.to_numpy()[target_positions].ravel(),
        "horizon": np.tile(np.arange(1, output_steps + 1), len(selected)),
        "actual": actual.ravel(),
        "predicted": predicted.ravel(),
        "persistence": np.repeat(persistence, output_steps),
    })

def metrics_with_persistence(actual, predicted, persistence):
    """Metricas do forecast com MSE como metrica principal.

    skill = 1 - MSE_modelo / MSE_persistencia (rmse_skill e mae_skill ficam
    como secundarias, na mesma formula com RMSE e MAE).
    """
    model = common.compute_metrics(actual, predicted)
    persistence_metrics = common.compute_metrics(actual, persistence)
    model["persistence_mse"] = persistence_metrics["mse"]
    model["persistence_rmse"] = persistence_metrics["rmse"]
    model["persistence_mae"] = persistence_metrics["mae"]
    model["skill"] = (
        1 - model["mse"] / persistence_metrics["mse"]
        if persistence_metrics["mse"] else np.nan
    )
    model["rmse_skill"] = (
        1 - model["rmse"] / persistence_metrics["rmse"]
        if persistence_metrics["rmse"] else np.nan
    )
    model["mae_skill"] = (
        1 - model["mae"] / persistence_metrics["mae"]
        if persistence_metrics["mae"] else np.nan
    )
    return model


def finalize_wrapper(wrapper_key, results_dir=None, **protocol):
    """Agrega as janelas concluidas de um wrapper (idempotente).

    *results_dir* None = o valor ATUAL de RESULTS_DIR (avaliado na chamada —
    nao capturado no import, para respeitar --output-dir).
    """
    wrapper_name = common.wrapper_factory(wrapper_key)().name
    wrapper_dir = Path(results_dir or RESULTS_DIR) / wrapper_name
    windows = sorted(wrapper_dir.glob("window_metrics_*.json"))
    if not windows:
        print(f"[{wrapper_name}] finalize: nenhuma janela concluida.")
        return None

    window_records = []
    predictions = []
    for json_path in windows:
        with open(json_path, encoding="utf-8") as handle:
            record = json.load(handle)
        pred_path = wrapper_dir / f"predictions_window_{record['window']}.csv"
        if not pred_path.is_file():
            continue
        window_records.append(record)
        predictions.append(pd.read_csv(pred_path, parse_dates=["origin", "timestamp"]))

    all_df = pd.concat(predictions, ignore_index=True)

    # Re-calcula por janela (RMSE principal) a partir das predicoes salvas.
    for record in window_records:
        group = all_df[
            (all_df["timestamp"] >= pd.Timestamp(record["test_start"]))
            & (all_df["timestamp"] <= pd.Timestamp(record["test_end"]))
        ]
        if group.empty:
            continue
        record.update(metrics_with_persistence(
            group["actual_raw"], group["predicted"], group["persistence"]
        ))
        observed = ~(group["output_imputed"].astype(str) == "True")
        if observed.any():
            obs = common.compute_metrics(
                group.loc[observed, "actual_raw"], group.loc[observed, "predicted"]
            )
            record["mse_observed"] = obs["mse"]
            record["mae_observed"] = obs["mae"]
            record["rmse_observed"] = obs["rmse"]
        if (~observed).any():
            imp = common.compute_metrics(
                group.loc[~observed, "actual_raw"], group.loc[~observed, "predicted"]
            )
            record["mse_imputed"] = imp["mse"]
            record["mae_imputed"] = imp["mae"]
            record["rmse_imputed"] = imp["rmse"]

    window_records.sort(key=lambda r: r["window"])
    metrics_df = pd.DataFrame(window_records)
    metrics_df.to_csv(wrapper_dir / "metrics.csv", index=False)
    all_df.to_csv(wrapper_dir / "predictions_all.csv", index=False)

    overall = metrics_with_persistence(
        all_df["actual_raw"], all_df["predicted"], all_df["persistence"]
    )
    observed = ~(all_df["output_imputed"].astype(str) == "True")
    if observed.any():
        obs = common.compute_metrics(
            all_df.loc[observed, "actual_raw"], all_df.loc[observed, "predicted"]
        )
        overall["mse_observed"] = obs["mse"]
        overall["mae_observed"] = obs["mae"]
        overall["rmse_observed"] = obs["rmse"]
    if (~observed).any():
        imp = common.compute_metrics(
            all_df.loc[~observed, "actual_raw"], all_df.loc[~observed, "predicted"]
        )
        overall["mse_imputed"] = imp["mse"]
        overall["mae_imputed"] = imp["mae"]
        overall["rmse_imputed"] = imp["rmse"]

    # Por horizonte, com skill RMSE-vs-persistencia (RMSE como principal).
    horizon_rows = []
    for horizon, group in all_df.groupby("horizon"):
        row = metrics_with_persistence(
            group["actual_raw"], group["predicted"], group["persistence"]
        )
        row["horizon"] = int(horizon)
        horizon_rows.append(row)
    by_horizon = pd.DataFrame(horizon_rows).sort_values("horizon").to_dict(orient="records")

    summary = {
        "settings": {
            "wrapper": wrapper_name,
            "wrapper_key": wrapper_key,
            "primary_metric": "mse",
            "protocol": {
                "train_months": protocol.get("train_months", TRAIN_MONTHS),
                "stride": protocol.get("stride", STRIDE),
                "input_steps": protocol.get("input_steps", INPUT_STEPS),
                "output_steps": protocol.get("output_steps", OUTPUT_STEPS),
                "epochs": protocol.get("epochs", EPOCHS),
                "patience": protocol.get("patience", PATIENCE),
                "batch_size": protocol.get("batch_size", BATCH_SIZE),
                "seed": protocol.get("seed", SEED),
            },
            "series_csv": str(SERIES_CSV),
            "params_source": window_records[0].get("params_source"),
        },
        "windows": window_records,
        "overall": overall,
        "by_horizon": by_horizon,
    }
    common.write_json(wrapper_dir / "metrics.json", summary)
    print(
        f"[{wrapper_name}] finalize: {len(window_records)} janelas | "
        f"MSE={overall['mse']:.4f} RMSE={overall['rmse']:.4f} "
        f"skill={overall['skill']:.3f}"
    )
    return wrapper_dir / "metrics.json"
