"""Walk-forward validation (LightGBM-imputado) com checkpoint/resume.

Protocolo: para cada mes civil totalmente coberto pela serie, treina nos 3
meses civis anteriores (split interno 80/20 treino/validacao) e prevê o mês
de teste com origens a cada ``STRIDE`` amostras (6 = 1 h) e horizonte de 36
passos (6 h). Cada janela grava ``predictions_window_<YYYYMM>.csv`` +
``window_metrics_<YYYYMM>.json`` (gravado por ultimo = marcador de conclusao),
entao interrupcoes/OOM retomam de onde pararam ao reexecutar o script.

Hiperparametros: best_trial.json do Optuna (pipeline/tmp/optuna) com queda
para defaults embutidos. Metricas de topo contra ws100 bruto; splits
observado vs imputado ficam nas colunas de flag das predicoes.

Usage:
    python scripts/experiments/wfo_lightgbm_run.py                        # 4 wrappers
    python scripts/experiments/wfo_lightgbm_run.py --wrappers lstm        # um wrapper
    python scripts/experiments/wfo_lightgbm_run.py --finalize-only        # so agregar
    python scripts/experiments/wfo_lightgbm_run.py --max-windows 1 --epochs 2  # smoke
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import common
import wfo_lightgbm_common as proto
from wfo_notify import notify


def _supervised():
    """True quando o processo esta sob o watchdog local (que reinicia crashes)."""
    return os.environ.get("WFO_SUPERVISED") == "1"


def load_series(data_dir):
    """Carrega serie imputada + flags por celula + flags derivadas por linha."""
    series = pd.read_csv(proto.SERIES_CSV, index_col="timestamp", parse_dates=True)
    flags = pd.read_csv(proto.FLAGS_CSV, index_col="timestamp",
                        parse_dates=True, dtype=bool)
    if not series.index.equals(flags.index):
        flags = flags.reindex(series.index).fillna(False).astype(bool)
    # Mesmo preprocessamento dos experimentos LiDAR (load_dataset) e do
    # step_walkforward do pipeline: hora/dia/ano e direcoes em componentes
    # senoidais (os graus crus sao descartados).
    series = common.add_cyclic_features(series)
    return series, flags


def window_flags_for_origins(flags_window, origin_positions, input_steps):
    """Por origem: (alguma celula imputada no encoder, fracao de celulas imputadas)."""
    any_row = flags_window.any(axis=1).to_numpy(dtype=float)
    frac_cell = flags_window.to_numpy(dtype=float).mean(axis=1)
    any_list, frac_list = [], []
    for position in origin_positions:
        start = position - input_steps + 1
        any_list.append(any_row[start:position + 1].max())
        frac_list.append(frac_cell[start:position + 1].mean())
    return np.asarray(any_list), np.asarray(frac_list)


def train_and_predict_window(wrapper_key, hyperparameters, params_source, train_df, eval_df,
                             series, flags, cfg):
    """Treina um wrapper na janela de treino e prevê a janela de teste.

    Espelha o protocolo de pipeline/step_walkforward.py (split 80/20, callbacks
    padrao, decoder direct) com origens a cada *stride* amostras. Retorna
    (predictions_df, window_metrics_dict).
    """
    from keras import backend as K

    tf = cfg["_tf"]
    wrapper_class = common.wrapper_factory(wrapper_key)
    wrapper = wrapper_class()

    train_split = int(len(train_df) * cfg["train_val_ratio"])
    train_part = train_df.iloc[:train_split].copy()
    val_part = train_df.iloc[train_split:].copy()

    K.clear_session()
    tf.keras.utils.set_random_seed(cfg["seed"] + int(cfg["window"].replace("-", "")) % 10_000)
    wrapper.prepare(
        train_part,
        val_part,
        input_steps=cfg["input_steps"],
        output_steps=cfg["output_steps"],
        target_col=proto.TARGET_COL,
        denoise=proto.DENOISE,
        denoise_level=cfg["denoise_level"],
        persistence_gate=False,
    )
    batch_size = int(hyperparameters.get("batch_size", cfg["batch_size"]))
    if hasattr(wrapper, "schedule_total_steps"):
        steps_per_epoch = int((len(train_part) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps_per_epoch * cfg["epochs"]
    wrapper.loss = hyperparameters.get("loss", "mse")

    wrapper.build(common.FixedHyperParameters(hyperparameters))
    started_at = time.perf_counter()
    history = wrapper.fit(
        epochs=cfg["epochs"],
        batch_size=batch_size,
        verbose=cfg.get("verbose", 0),
        callbacks=common.default_callbacks(wrapper, cfg["patience"]),
        use_validation=True,
    )
    train_seconds = time.perf_counter() - started_at

    eval_window = pd.concat([train_df.iloc[-cfg["input_steps"]:], eval_df])
    predictions = proto.predict_all_horizons_strided(wrapper, eval_window, cfg["stride"])

    raw_col = proto.TARGET_COL.removesuffix("_wavelet")
    predictions["actual_raw"] = series[raw_col].reindex(predictions["timestamp"]).to_numpy()

    # Flags de imputacao: alvo (celula ws100 imputada no timestamp previsto) e
    # janela de entrada (qualquer celula / fracao de celulas imputadas).
    output_imputed = flags[raw_col].reindex(predictions["timestamp"]).fillna(False)
    predictions["output_imputed"] = output_imputed.to_numpy(dtype=bool)
    origin_positions = np.searchsorted(eval_window.index, predictions["origin"].unique())
    input_any, input_frac = window_flags_for_origins(
        flags.reindex(eval_window.index).fillna(False).astype(bool),
        origin_positions,
        cfg["input_steps"],
    )
    n_origins = len(predictions) // cfg["output_steps"]
    predictions["input_imputed_any"] = np.repeat(input_any.astype(bool), cfg["output_steps"])
    predictions["input_imputed_frac"] = np.repeat(input_frac, cfg["output_steps"])
    assert n_origins == len(origin_positions), "desalinho origem/flags"

    metrics = proto.metrics_with_persistence(
        predictions["actual_raw"], predictions["predicted"], predictions["persistence"]
    )
    observed = ~predictions["output_imputed"]
    if observed.any():
        obs = common.compute_metrics(
            predictions.loc[observed, "actual_raw"], predictions.loc[observed, "predicted"]
        )
        metrics["mae_observed"] = obs["mae"]
        metrics["rmse_observed"] = obs["rmse"]
    if (~observed).any():
        imp = common.compute_metrics(
            predictions.loc[~observed, "actual_raw"], predictions.loc[~observed, "predicted"]
        )
        metrics["mae_imputed"] = imp["mae"]
        metrics["rmse_imputed"] = imp["rmse"]
    for flag_name in ("input_imputed_any",):
        for value in (True, False):
            selected = predictions[predictions[flag_name] == value]
            key = "com" if value else "sem"
            if len(selected):
                split = common.compute_metrics(selected["actual_raw"], selected["predicted"])
                metrics[f"mae_input_{key}_imputacao"] = split["mae"]

    metrics.update({
        "window": cfg["window"],
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "test_start": str(eval_df.index.min()),
        "test_end": str(eval_df.index.max()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(eval_df)),
        "n_origins": int(n_origins),
        "epochs_trained": len(history.history.get("loss", [])),
        "train_seconds": round(train_seconds, 1),
        "params_source": params_source,
        "hyperparameters": {k: v for k, v in hyperparameters.items() if k != "batch_size"},
    })

    K.clear_session()
    gc.collect()
    return predictions, metrics


def run_wrapper(wrapper_key, series, flags, cfg):
    """Executa (ou retoma) o walk-forward completo de um wrapper."""
    wrapper_name = common.wrapper_factory(wrapper_key)().name
    wrapper_dir = proto.RESULTS_DIR / wrapper_name
    wrapper_dir.mkdir(parents=True, exist_ok=True)

    try:
        hyperparameters, params_source = common.resolve_hyperparameters(
            wrapper_key,
            {"params_source": cfg["params_source"], "hyperparameters": {}},
            {"paths": {"tmp_dir": "pipeline/tmp", "results_dir": "data/results"}},
        )
    except FileNotFoundError as exc:
        print(f"[{wrapper_name}] AVISO: {exc} Usando defaults embutidos.")
        hyperparameters = dict(common.DEFAULT_HYPERPARAMETERS[wrapper_key])
        params_source = "built-in defaults (optuna ausente)"

    windows = proto.plan_windows(series.index, cfg["train_months"])
    if cfg["max_windows"] is not None:
        windows = windows[: cfg["max_windows"]]
    pending = [
        w for w in windows
        if not proto.window_is_complete(wrapper_dir, w["window"])
    ]
    print(f"[{wrapper_name}] {len(windows)} janelas | pendentes: {len(pending)}")
    proto.update_progress(wrapper_key, len(windows) - len(pending), len(windows),
                          status="running")
    notify(
        f"WFO {wrapper_key} inicio",
        f"▶ {wrapper_name}: walk-forward iniciado — {len(windows)} janelas "
        f"({len(windows) - len(pending)} ja concluidas)",
    )

    for window in pending:
        train_df = series.loc[window["train_start"]:window["train_end"] - pd.Timedelta(minutes=10)]
        eval_df = series.loc[window["test_start"]:window["test_end"]]
        cfg = {**cfg, "window": window["window"]}
        print(
            f"[{wrapper_name}] janela {window['window']}: treino "
            f"{train_df.index.min()} -> {train_df.index.max()} ({len(train_df)} linhas) | "
            f"teste {len(eval_df)} linhas"
        )

        pred_path, json_path = proto.month_window_files(wrapper_dir, window["window"])
        try:
            predictions, metrics = train_and_predict_window(
                wrapper_key, hyperparameters, params_source, train_df, eval_df, series, flags, cfg
            )
        except KeyboardInterrupt:
            print(f"[{wrapper_name}] interrompido na janela {window['window']}; "
                  "estado preservado — reexecute para retomar.")
            raise
        except Exception as exc:  # janela com problema nao derruba a corrida
            print(f"[{wrapper_name}] ERRO na janela {window['window']}: {exc!r}")
            notify(
                f"WFO {wrapper_key} janela com erro",
                f"⚠ {wrapper_name}: janela {window['window']} falhou ({exc!r}) — "
                "seguindo para as demais (refaca depois se necessario)",
                priority="high", tags="warning",
            )
            proto.update_progress(
                wrapper_key, len(windows) - len(pending), len(windows),
                status="window_error", extra={"last_error": repr(exc), "window": window["window"]},
            )
            continue

        predictions.to_csv(pred_path, index=False)
        common.write_json(json_path, metrics)
        print(
            f"[{wrapper_name}] janela {window['window']}: MSE={metrics['mse']:.4f} "
            f"RMSE={metrics['rmse']:.4f} skill={metrics['skill']:.3f} "
            f"epochs={metrics['epochs_trained']} ({metrics['train_seconds']:.0f}s)"
        )
        done = sum(1 for w in windows if proto.window_is_complete(wrapper_dir, w["window"]))
        proto.update_progress(wrapper_key, done, len(windows), status="running")

    result_path = proto.finalize_wrapper(
        wrapper_key, results_dir=proto.RESULTS_DIR, protocol={
            "train_months": cfg["train_months"], "stride": cfg["stride"],
            "input_steps": cfg["input_steps"], "output_steps": cfg["output_steps"],
            "epochs": cfg["epochs"], "patience": cfg["patience"],
            "batch_size": cfg["batch_size"], "seed": cfg["seed"],
        },
    )
    if result_path is not None:
        with open(result_path, encoding="utf-8") as handle:
            overall = json.load(handle)["overall"]
        done = sum(1 for w in windows if proto.window_is_complete(wrapper_dir, w["window"]))
        skill = overall.get("skill")
        detail = (
            f"MSE={overall['mse']:.4f} RMSE={overall['rmse']:.4f} skill={skill:.3f}"
            if skill is not None and skill == skill
            else f"MSE={overall['mse']:.4f} RMSE={overall['rmse']:.4f}"
        )
        notify(
            f"WFO {wrapper_key} OK",
            f"✅ {wrapper_name}: {done}/{len(windows)} janelas | {detail}",
        )
    return result_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrappers", nargs="+", default=list(proto.WRAPPERS),
                        choices=proto.WRAPPERS)
    parser.add_argument("--train-months", type=int, default=proto.TRAIN_MONTHS)
    parser.add_argument("--stride", type=int, default=proto.STRIDE)
    parser.add_argument("--input-steps", type=int, default=proto.INPUT_STEPS)
    parser.add_argument("--output-steps", type=int, default=proto.OUTPUT_STEPS)
    parser.add_argument("--epochs", type=int, default=proto.EPOCHS)
    parser.add_argument("--patience", type=int, default=proto.PATIENCE)
    parser.add_argument("--batch-size", type=int, default=proto.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=proto.SEED)
    parser.add_argument("--denoise-level", type=int, default=proto.DENOISE_LEVEL)
    parser.add_argument("--params-source", default="optuna",
                        choices=("optuna", "default"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="diretorio de resultados (default: data/results/wfo_lightgbm)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="limita o numero de janelas (smoke test)")
    parser.add_argument("--finalize-only", action="store_true",
                        help="so agrega as janelas existentes (pos-collect Kaggle)")
    parser.add_argument("--series", default=None,
                        help="CSV da serie (default: data/wfo_lightgbm/series_imputed.csv)")
    parser.add_argument("--results-dir", default=None,
                        help="diretorio de resultados (default: data/results/wfo_lightgbm)")
    parser.add_argument("--list-windows", action="store_true",
                        help="imprime o plano de janelas e sai")
    args = parser.parse_args()

    if args.series:
        proto.SERIES_CSV = common.resolve(args.series)
    if args.results_dir:
        proto.RESULTS_DIR = common.resolve(args.results_dir)
        proto.PROGRESS_JSON = proto.RESULTS_DIR / "progress.json"

    if args.list_windows:
        if not proto.SERIES_CSV.is_file():
            raise SystemExit(f"{proto.SERIES_CSV} ausente — rode wfo_lightgbm_prepare.py")
        index = pd.read_csv(proto.SERIES_CSV, index_col="timestamp",
                            parse_dates=True).index
        for window in proto.plan_windows(index, args.train_months):
            print(f"{window['window']}: treino {window['train_start']:%F} -> "
                  f"{window['train_end'] - pd.Timedelta(minutes=10):%F} "
                  f"({window['train_rows']} linhas) | teste {window['test_rows']} linhas")
        return

    cfg = {
        "train_months": args.train_months,
        "stride": args.stride,
        "input_steps": args.input_steps,
        "output_steps": args.output_steps,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "denoise_level": args.denoise_level,
        "train_val_ratio": proto.TRAIN_VAL_RATIO,
        "params_source": args.params_source,
        "max_windows": args.max_windows,
        "verbose": 0,
    }

    if args.finalize_only:
        results_dir = (common.resolve(args.output_dir)
                       if args.output_dir else proto.RESULTS_DIR)
        for wrapper_key in args.wrappers:
            proto.finalize_wrapper(wrapper_key, results_dir=results_dir, protocol=cfg)
        return

    if not proto.SERIES_CSV.is_file() or not proto.FLAGS_CSV.is_file():
        raise SystemExit(
            f"{proto.SERIES_CSV} / {proto.FLAGS_CSV} ausentes — rode "
            "scripts/experiments/wfo_lightgbm_prepare.py primeiro."
        )

    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    cfg["_tf"] = tf

    series, flags = load_series(proto.DATA_DIR)
    print(f"Serie: {series.shape} | {series.index.min()} -> {series.index.max()}")
    if args.output_dir:
        proto.RESULTS_DIR = common.resolve(args.output_dir)

    supervised = _supervised()
    try:
        for wrapper_key in args.wrappers:
            print(f"\n=== Walk-forward LightGBM-imputado: {wrapper_key} ===")
            run_wrapper(wrapper_key, series, flags, {**cfg})
    except KeyboardInterrupt:
        notify(
            "WFO interrompido",
            f"⏹ interrompido manualmente ({args.wrappers}) — "
            + ("watchdog reinicia do checkpoint." if supervised
               else "SEM auto-restart: reexecute para retomar."),
            priority="high", tags="warning",
        )
        raise
    except Exception as exc:
        notify(
            "WFO crash",
            ("💥 crash em " + str(args.wrappers) + f": {exc!r} — watchdog reinicia do checkpoint."
             if supervised else
             f"❌ CRASH em {args.wrappers}: {exc!r} — SEM auto-restart, acao manual necessaria."),
            priority="high", tags="rotating_light",
        )
        raise
    progress = json.loads(proto.PROGRESS_JSON.read_text()) if proto.PROGRESS_JSON.is_file() else {}
    for wrapper_key in args.wrappers:
        if wrapper_key in progress.get("wrappers", {}):
            progress["wrappers"][wrapper_key]["status"] = "finished"
    progress["updated"] = pd.Timestamp.now().isoformat()
    proto.PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    proto.PROGRESS_JSON.write_text(json.dumps(progress, indent=2, default=str))
    print("\nConcluido.")


if __name__ == "__main__":
    main()
