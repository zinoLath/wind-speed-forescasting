"""Avaliação + figuras dos modelos seq2seq treinados em data/dataset.csv.

Carrega os wrappers Keras já treinados, mede métricas e tempos de inferência
no conjunto de teste e gera figuras + evaluation.json consumidos pelo
relatório curado do hub (reports/lidar/<protocolo>/report.py). O HTML
standalone legado (relatorio_resultados.html) foi removido — a página do hub
o substitui.

Toda run é gravada no banco SQLite (data/predictions.db, ver
src/predictions_db.py): métricas por horizonte, hp, features e as predições
linha a linha — os relatórios do hub leem do banco antes do evaluation.json,
e --from-db regenera figuras/DM/MCS a partir das predições gravadas, sem GPU.

Foco: lstm, lstm_bi, tcn, tcn_bi (série LiDAR, dataset.csv).

Executar a partir da raiz do projeto:
    python tests/relatorio_resultados.py
    python tests/relatorio_resultados.py --from-db lidar/8020  # sem GPU

Saída:
    <out>/images/*.png
    <out>/evaluation.json
    data/predictions.db (run gravada; --no-db para pular)
"""

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import common, predictions_db
from src.models.tcn_hp import apply_static_dilations
from src.utils import diebold_mariano, diebold_mariano_loss_diff, \
    model_confidence_set

DATA_FILE = PROJECT_ROOT / "data" / "dataset.csv"
MODELS_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "trained_compare"
NOGATE_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "trained_compare_nogate"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "report"
# Ratios de split lidos do model.json do primeiro modelo (evaluate_models os
# atualiza) — permite relatorios com protocolos diferentes (ex.: 75/25 sem
# validacao) sem duplicar o script.
SPLIT_RATIOS = {"train_ratio": 0.75, "val_ratio": 0.20}
IMG_DIR = OUT_DIR / "images"
EVAL_CACHE = OUT_DIR / "evaluation.json"
BASELINES_DIR = PROJECT_ROOT / "data" / "results" / "baselines" / "arima_rf"
BASELINE_STYLE = {
    "arima": ("ARIMA", "#7f7f7f"),
    "rf": ("Random Forest", "#8c564b"),
    "persistence": ("Persistência", "#b0b0b0"),
}


def load_baselines():
    """Métricas dos baselines estatísticos (geradas por tests/baseline_arima_rf.py)."""
    if not (BASELINES_DIR / "metrics.json").is_file():
        return None
    out = {"metrics": json.load(open(BASELINES_DIR / "metrics.json"))}
    ph = BASELINES_DIR / "per_horizon_metrics.csv"
    if ph.is_file():
        out["per_horizon"] = pd.read_csv(ph)
    return out


def trim_boundary_origins(df, n, label):
    """Descarta as primeiras *n* origens do grid de teste.

    Origens cuja janela de entrada cruza a fronteira treino/teste (72
    passos) consomem dados de treino no encoder; para eliminar qualquer
    argumento de vazamento, o protocolo permite removê-las de TODOS os
    métodos (DL e baselines) — mesma grade, comparação justa.
    """
    if n <= 0 or df is None or len(df) == 0:
        return df
    origins = np.sort(df["origin"].unique())
    if len(origins) <= n:
        print(f"[trim-boundary] {label}: {len(origins)} origens <= n={n}; "
              "nada restaria — corte ignorado")
        return df
    cut = origins[n - 1]
    kept = df[df["origin"] > cut]
    print(f"[trim-boundary] {label}: {len(origins)} -> "
          f"{kept['origin'].nunique()} origens (descartadas as primeiras "
          f"{n}, ate {pd.Timestamp(cut)})")
    return kept


def recompute_baseline_metrics(base_preds, base_json):
    """Recalcula as métricas dos baselines na grade já cortada."""
    out = {"metrics": json.loads(json.dumps(base_json))}
    for key in ("arima", "rf", "persistence"):
        if key not in base_preds.columns:
            continue
        m_all = common.compute_metrics(base_preds["actual"],
                                       base_preds[key])
        h_max = int(base_preds["horizon"].max())
        roll = base_preds[base_preds["horizon"] == h_max]
        m_roll = common.compute_metrics(roll["actual"], roll[key])
        if key in out["metrics"]:
            out["metrics"][key]["all_horizons"] = m_all
            out["metrics"][key]["rolling"] = m_roll
        else:
            out["metrics"][key] = {"all_horizons": m_all, "rolling": m_roll}
    rows = []
    for h, g in base_preds.groupby("horizon"):
        row = {"horizon": int(h)}
        for key in ("arima", "rf", "persistence"):
            m = common.compute_metrics(g["actual"], g[key])
            row[f"{key}_mae"] = m["mae"]
            row[f"{key}_rmse"] = m["rmse"]
            row[f"{key}_r2"] = m["r2"]
        rows.append(row)
    out["per_horizon"] = pd.DataFrame(rows).sort_values("horizon")
    return out


def load_baseline_predictions():
    """Predições long (origin, timestamp, horizon, actual, persistence, arima, rf)."""
    path = BASELINES_DIR / "predictions_long.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path, parse_dates=["origin", "timestamp"])
    # As origens dos baselines vêm com drift de microssegundos (ex.:
    # 00:39:59.999998) e não casam com a grade regularizada do dataset. Sem o
    # arredondamento ao minuto, qualquer alinhamento DL × baselines por
    # `origin` (DM, MCS) colapsa para poucas origens.
    for col in ("origin", "timestamp"):
        df[col] = df[col].dt.round("min")
    return df

FOCUS_KEYS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
SEED = 42
# Nível da wavelet da run corrente; None = derivar do model.json dos modelos
# (avaliação na GPU). --from-db fixa o valor gravado na run.
DENOISE_LEVEL = None
# Origens de fronteira a descartar (--trim-boundary-origins); 0 = mantém
TRIM_BOUNDARY = 0


def current_denoise_level() -> int:
    if DENOISE_LEVEL is not None:
        return int(DENOISE_LEVEL)
    return int(load_metadata()[0]["meta"].get("denoise_level", 2))

COLORS = {
    "Seq2Seq_LSTM": "#1f77b4",
    "Seq2Seq_LSTM_Bidirectional": "#9467bd",
    "Seq2Seq_TCN": "#ff7f0e",
    "Seq2Seq_TCN_Bidirectional": "#d62728",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.autolayout": True,
})


def short_name(name):
    return name.replace("Seq2Seq_", "").replace("_", " ")


def savefig(fig, name):
    fig.savefig(IMG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


def card(num, lab):
    return f'<div class="card"><div class="num">{num}</div><div class="lab">{lab}</div></div>'


def fig_html(src, caption):
    return (f'<figure><img src="images/{src}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def latex_button(latex_source):
    esc = (latex_source.replace("&", "&amp;")
           .replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;"))
    return (
        '<button type="button" class="btn-copy" data-latex="'
        f'{esc}" onclick="copiarLatex(this)">copiar como LaTeX</button>'
    )


def latex_table(columns, rows, highlight_first_col_bold=True):
    head_html = "".join(f"<th>{c}</th>" for c in columns)
    body_html = ""
    for row in rows:
        cells = []
        for j, cell in enumerate(row):
            if j == 0 and highlight_first_col_bold:
                cells.append(f"<td>{cell}</td>")
            else:
                cells.append(f"<td>{cell}</td>")
        body_html += "<tr>" + "".join(cells) + "</tr>"
    html_table = f"<table><tr>{head_html}</tr>{body_html}</table>"

    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\begin{tabular}{l" + "c" * (len(columns) - 1) + "}\n"
    latex += " & ".join(columns) + " \\\\ \\hline\n"
    for row in rows:
        latex += " & ".join(str(cell).replace(",", ".") for cell in row) + " \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\caption{...}\n\\end{table}"
    return html_table, latex


def section(sec_id, title, sub, body, images=(), fig_grid=False):
    imgs = "".join(fig_html(p, t) for p, t in images if (IMG_DIR / p).is_file())
    if fig_grid and imgs:
        imgs = f'<div class="fig-grid">{imgs}</div>'
    anchor = f' id="{sec_id}"' if sec_id else ""
    return (f"<section{anchor}><h2>{title}</h2><div class='sub'>{sub}</div>"
            f"{body}{imgs}</section>")


# ----------------------------------------------------------------------------
# Coleta: avaliação dos modelos salvos
# ----------------------------------------------------------------------------
def load_metadata(base_dir=None):
    base = Path(base_dir) if base_dir else MODELS_DIR
    models = []
    for model_dir in sorted(base.iterdir()):
        meta_path = model_dir / "model.json"
        if not meta_path.is_file():
            continue
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta["wrapper_key"] not in FOCUS_KEYS:
            continue
        models.append({"dir": model_dir, "meta": meta})
    if not models:
        raise FileNotFoundError(
            f"Nenhum modelo encontrado em {base} para {FOCUS_KEYS}. "
            "Rode pipeline/pipeline.py --stage train (config trained_compare) primeiro."
        )
    return models


def build_wrapper(meta, train_df, val_df, model_path, require_weights=True):
    from keras import backend as K

    K.clear_session()
    wrapper = common.wrapper_factory(meta["wrapper_key"])()
    wrapper.gate_mode = meta.get("gate_mode", "static")
    wrapper.context_mode = meta.get("context_mode", "repeat")
    # prepare(create_sequences=False) nao popula future_indices — derivamos das
    # colunas (prepare_data anexa time_delta ao fim; add_cyclic_features ja
    # proveu hour_sin/cos no dataset, respeitando o features-mode).
    prepared_cols = list(train_df.columns)
    if "time_delta" not in prepared_cols:
        prepared_cols.append("time_delta")
    names = tuple(n for n in ("time_delta", "hour_sin", "hour_cos")
                  if n in prepared_cols)
    wrapper.future_indices = (tuple(prepared_cols.index(n) for n in names)
                              or None)
    wrapper.future_covariate_names = names
    wrapper.prepare(
        train_df,
        val_df,
        input_steps=meta["input_steps"],
        output_steps=meta["output_steps"],
        target_col=meta["target_col"],
        denoise=meta["denoise"],
        denoise_level=meta["denoise_level"],
        features=meta.get("features"),
        create_sequences=False,
        persistence_gate=bool(meta.get("persistence_gate", False)),
        decoder_mode=meta.get("decoder_mode", "direct"),
        target_mode=meta.get("target_mode", "absolute"),
        normalize=bool(meta.get("normalize", True)),
    )
    # Compat com geracoes anteriores de pesos: tenta o decoder atual (width 5),
    # depois time_delta sozinho (width 3) e por fim sem covariadas (width 2).
    base_future = wrapper.future_indices
    base_names = names
    attempts = []
    if base_future:
        attempts.append((base_future, base_names))
        if base_names and base_names[0] == "time_delta":
            attempts.append(((base_future[0],), ("time_delta",)))
    attempts.append((None, ()))
    last_error = None
    for future_indices, names in attempts:
        wrapper.future_indices = future_indices
        wrapper.future_covariate_names = names
        wrapper.num_decoder_features = (
            (2 if wrapper.decoder_mode == "direct" else 1)
            + len(future_indices or ())
            + (1 if getattr(wrapper, "persistence_gate", False) else 0)
        )
        apply_static_dilations(wrapper, meta.get("hyperparameters"))
        wrapper.build(common.FixedHyperParameters(meta["hyperparameters"]))
        try:
            wrapper.model.load_weights(model_path)
            break
        except ValueError as error:
            last_error = error
    else:
        if require_weights:
            raise last_error
        # Diagramas de arquitetura dependem apenas da estrutura; modelos
        # antigos (ex.: trained_compare_nogate) podem ter shapes incompatíveis.
        print(f"  aviso: pesos de {model_path} nao carregados ({last_error})")
    return wrapper


def timed_inference(wrapper, test_df):
    prepared, _ = wrapper.prepare_data(
        test_df,
        wrapper.input_steps,
        wrapper.output_steps,
        wrapper.target_col,
        scaler_target=wrapper.scaler_target,
        scaler_other=wrapper.scaler_other,
        denoise=wrapper.denoise,
        decoder_mode=wrapper.decoder_mode,
        features=getattr(wrapper, "features", None),
        target_mode=wrapper.target_mode,
    )
    if wrapper.decoder_mode == "teacher_forcing":
        prepared["X_decoder"][:, :, 0] = prepared["X_decoder"][:, :1, 0]
    common.align_decoder_channels(wrapper, prepared)

    X_enc = prepared["X_encoder"]
    X_dec = prepared["X_decoder"]
    model = wrapper.model

    model.predict([X_enc, X_dec], batch_size=256, verbose=0)
    t0 = time.perf_counter()
    model.predict([X_enc, X_dec], batch_size=256, verbose=0)
    batch_sec = time.perf_counter() - t0
    n = len(X_enc)

    enc1, dec1 = X_enc[:1], X_dec[:1]
    for _ in range(5):
        model.predict([enc1, dec1], verbose=0)
    t0 = time.perf_counter()
    reps = 30
    for _ in range(reps):
        model.predict([enc1, dec1], verbose=0)
    single_ms = (time.perf_counter() - t0) / reps * 1000.0

    return {
        "windows": int(n),
        "batch_total_ms": float(batch_sec * 1000.0),
        "per_window_ms": float(batch_sec / n * 1000.0),
        "throughput_wps": float(n / batch_sec),
        "single_shot_ms": float(single_ms),
    }


def rolling_frame(wrapper, dataset, val_end):
    roll_pred, roll_actual = wrapper.rolling_forecast(dataset, test_start=val_end)
    offset = val_end + wrapper.output_steps - 1
    roll_df = pd.DataFrame(
        {
            "origin": dataset.index[val_end - 1: val_end - 1 + len(roll_actual)],
            "timestamp": dataset.index[offset: offset + len(roll_actual)],
            "actual_denoised": roll_actual.ravel(),
            "predicted": roll_pred.ravel(),
        }
    )
    roll_df["actual_raw"] = dataset["ws100"].reindex(roll_df["timestamp"]).to_numpy()
    return roll_df


def evaluate_models(dataset):
    tf = common.setup_tensorflow(True)
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    models = load_metadata()
    global SPLIT_RATIOS
    _ratios0 = models[0]["meta"].get("dataset", {})
    SPLIT_RATIOS = {"train_ratio": _ratios0.get("train_ratio", 0.75),
                    "val_ratio": _ratios0.get("val_ratio", 0.20)}
    # amostras interpoladas pela regularizacao da serie nunca entram nas
    # metricas/DM — apenas timestamps com registro real sao avaliados
    observed = common.observed_timestamps(DATA_FILE)
    results = {}
    extras = {}

    # lineup misto: cada modelo declara "feature_keep" no model.json; o
    # dataset carregado é a união, e aqui cada modelo recebe só as colunas
    # com as quais foi treinado
    extras_all = set()
    for entry in models:
        extras_all |= set(entry["meta"].get("feature_keep", ()))

    for entry in models:
        meta = entry["meta"]
        name = meta["name"]
        key = meta["wrapper_key"]
        ratios = meta.get("dataset", {})
        train_ratio = ratios.get("train_ratio", 0.75)
        val_ratio = ratios.get("val_ratio", 0.20)
        drop_cols = extras_all - set(meta.get("feature_keep", ()))
        df_model = dataset.drop(columns=drop_cols) if drop_cols else dataset
        train_df, val_df, test_df = common.split_dataset(df_model, train_ratio, val_ratio)
        val_end = len(train_df) + len(val_df)
        # prepare() com create_sequences=False nao usa o val, mas o denoise
        # (wavelet) sobre val vazio quebra o pywt — passe o tail do treino.
        val_for_prepare = val_df if len(val_df) else train_df.tail(400)

        print(f"[{key}] carregando pesos de {entry['dir'].name} ...")
        wrapper = build_wrapper(meta, train_df, val_for_prepare, entry["dir"] / "model.keras")

        t0 = time.perf_counter()
        # mesma grade de origens dos baselines: primeira origem = primeira
        # linha do teste, encoder consome o fim da fronteira val/teste
        eval_df = df_model.iloc[max(0, val_end - wrapper.input_steps):]
        all_df = common.predict_all_horizons(wrapper, eval_df)
        eval_sec = time.perf_counter() - t0
        all_df["actual_raw"] = dataset["ws100"].reindex(all_df["timestamp"]).to_numpy()
        if observed is not None:
            all_df = all_df[all_df["timestamp"].isin(observed)]
        all_df = trim_boundary_origins(all_df, TRIM_BOUNDARY, f"{key}/dense")

        roll_df = rolling_frame(wrapper, df_model, val_end)
        if observed is not None:
            roll_df = roll_df[roll_df["timestamp"].isin(observed)]
        roll_df = trim_boundary_origins(roll_df, TRIM_BOUNDARY, f"{key}/rolling")
        timing = timed_inference(wrapper, test_df)

        per_horizon = common.horizon_metrics(all_df.assign(actual=all_df["actual_raw"]))
        per_horizon_den = common.horizon_metrics(all_df.assign(actual=all_df["actual"]))

        m_all = common.compute_metrics(all_df["actual_raw"], all_df["predicted"])
        m_all_den = common.compute_metrics(all_df["actual"], all_df["predicted"])
        m_roll = common.compute_metrics(roll_df["actual_raw"], roll_df["predicted"])
        m_roll_den = common.compute_metrics(roll_df["actual_denoised"], roll_df["predicted"])

        results[name] = {
            "key": key,
            "name": name,
            "short": short_name(name),
            "color": COLORS[name],
            "loss": meta.get("loss"),
            "hp": meta["hyperparameters"],
            "training": meta["training"],
            "params": int(wrapper.model.count_params()),
            "size_mb": float((entry["dir"] / "model.keras").stat().st_size / 1e6),
            "eval_sec": eval_sec,
            "features": list(df_model.columns),
            "output_steps": int(meta["output_steps"]),
            "all": m_all,
            "all_denoised": m_all_den,
            "rolling": m_roll,
            "rolling_denoised": m_roll_den,
            "timing": timing,
        }
        extras[name] = {
            "all_df": all_df,
            "roll_df": roll_df,
            "per_horizon": per_horizon,
            "per_horizon_den": per_horizon_den,
        }
        results[name]["per_horizon"] = per_horizon.to_dict("records")
        results[name]["per_horizon_den"] = per_horizon_den.to_dict("records")
        print(
            f"[{key}] MSE={m_all['mse']:.4f} roll={m_roll['mse']:.4f} "
            f"params={results[name]['params']:,} {timing['per_window_ms']:.2f} ms/janela"
        )

    return results, extras


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def export_architectures(dataset, base_dir=None):
    tf = common.setup_tensorflow(True)
    from keras import backend as K

    records = []
    for entry in load_metadata(base_dir):
        meta = entry["meta"]
        ratios = meta.get("dataset", {})
        train_df, val_df, _ = common.split_dataset(
            dataset, ratios.get("train_ratio", 0.75), ratios.get("val_ratio", 0.20)
        )
        print(f"[{meta['wrapper_key']}] exportando diagrama de arquitetura ...")
        wrapper = build_wrapper(meta, train_df,
                                val_df if len(val_df) else train_df.tail(400),
                                entry["dir"] / "model.keras",
                                require_weights=False)
        out = IMG_DIR / f"15_arquitetura_{meta['wrapper_key']}.png"
        tf.keras.utils.plot_model(
            wrapper.model,
            to_file=str(out),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=False,
            rankdir="TB",
            dpi=88,
        )
        comp = Counter(type(layer).__name__ for layer in wrapper.model.layers)
        records.append({
            "name": meta["name"],
            "key": meta["wrapper_key"],
            "short": short_name(meta["name"]),
            "layers": len(wrapper.model.layers),
            "composition": dict(comp),
            "params": int(wrapper.model.count_params()),
        })
        K.clear_session()
    return records


def build_split_table(dataset_info):
    rows = []
    for s in dataset_info["splits"]:
        rows.append((
            s["name"],
            s["start"],
            s["end"],
            fmt_int(s["rows"]),
            fmt(s["pct"], 1),
        ))
    columns = ["Divisão", "Início", "Fim", "Amostras", "Participação (%)"]
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def build_arch_table(arch_records):
    counter = Counter()
    for rec in arch_records:
        counter.update(rec["composition"])
    type_cols = [name for name, _ in counter.most_common()]
    rows = []
    for rec in arch_records:
        rows.append(tuple(
            [f"{rec['short']} ({rec['layers']})"]
            + [rec["composition"].get(t, 0) for t in type_cols]
        ))
    columns = ["Modelo (camadas)"] + type_cols
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def plot_split_timeline(dataset, recs):
    n = len(dataset)
    train_ratio = SPLIT_RATIOS["train_ratio"]
    val_ratio = SPLIT_RATIOS["val_ratio"]
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(dataset.index, dataset["ws100"], lw=0.4, color="#1b6ca8")
    ax.axvspan(dataset.index[0], dataset.index[train_end - 1],
               color="#2166ac", alpha=0.12,
               label=f"treino {train_ratio:.0%} ({train_end})")
    if val_end > train_end:
        ax.axvspan(dataset.index[train_end], dataset.index[val_end - 1],
                   color="#e0a800", alpha=0.14,
                   label=f"validação {val_ratio:.0%} ({val_end - train_end})")
    ax.axvspan(dataset.index[val_end], dataset.index[-1],
               color="#b03030", alpha=0.14,
               label=f"teste {1 - train_ratio - val_ratio:.0%} ({n - val_end})")
    ax.set_ylabel("ws100 (m/s)")
    ax.set_title("Divisão temporal do dataset LiDAR (sem embaralhamento)")
    ax.legend(loc="upper right", ncol=3)
    savefig(fig, "01_split_temporal.png")


def plot_comparison_bars(recs, baselines=None):
    names = [r["short"] for r in recs]
    mse = [r["all"]["mse"] for r in recs]
    rmse = [r["all"]["rmse"] for r in recs]
    colors = [r["color"] for r in recs]
    hatches = [""] * len(recs)
    if baselines:
        for key in ("arima", "rf", "persistence"):
            m = baselines["metrics"].get(key)
            if not m:
                continue
            label, color = BASELINE_STYLE[key]
            names.append(f"{label} (baseline)")
            mse.append(m["all_horizons"]["mse"])
            rmse.append(m["all_horizons"]["rmse"])
            colors.append(color)
            hatches.append("//")
    y = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4 + 0.4 * len(hatches)))
    axes[0].barh(y, mse, color=colors)
    for bar, h in zip(axes[0].containers[0], hatches):
        bar.set_hatch(h)
    for yi, v in zip(y, mse):
        axes[0].text(v, yi, f" {v:.3f}", va="center", fontsize=8.5)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MSE ((m/s)²)")
    axes[0].set_title("MSE todos os horizontes (1–36 passos)")
    axes[0].set_xlim(0, max(mse) * 1.22)

    axes[1].barh(y, rmse, color=colors)
    for bar, h in zip(axes[1].containers[0], hatches):
        bar.set_hatch(h)
    for bar, h in zip(axes[0].containers[0], hatches):
        bar.set_hatch(h)
    for yi, v in zip(y, rmse):
        axes[1].text(v, yi, f" {v:.3f}", va="center", fontsize=8.5)
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RMSE (m/s)")
    axes[1].set_title("RMSE todos os horizontes (1–36 passos)")
    axes[1].set_xlim(0, max(rmse) * 1.22)
    savefig(fig, "02_comparacao_mse_rmse.png")


def plot_error_summary(recs, extras, baselines=None):
    """MAE, MSE, R2 (todos os horizontes) e loss de treino por modelo + baselines."""
    first = next(iter(extras.values()))
    rows = []
    if baselines:
        for key in ("arima", "rf", "persistence"):
            m = baselines["metrics"].get(key)
            if m:
                label, color = BASELINE_STYLE[key]
                rows.append((f"{label} (baseline)", m["all_horizons"], color, "//"))
    if not any("Persistência" in r[0] for r in rows):
        rows.append(("Persistência (baseline)",
                     common.compute_metrics(first["all_df"]["actual_raw"],
                                            first["all_df"]["persistence"]),
                     BASELINE_STYLE["persistence"][1], "//"))
    for r in recs:
        rows.append((r["short"], r["all"], r["color"], ""))
    rows.sort(key=lambda x: x[1]["mse"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = ((axes[0][0], "mae", "MAE (m/s)",
               "MAE todos os horizontes (1–36 passos)", False),
              (axes[0][1], "mse", "MSE ((m/s)²)",
               "MSE todos os horizontes (1–36 passos)", False),
              (axes[1][0], "r2", "R²",
               "R² todos os horizontes (1–36 passos)", True))
    for ax, key, xlabel, title, _best_high in panels:
        y = np.arange(len(rows))
        vals = [row[1].get(key, np.nan) for row in rows]
        bars = ax.barh(y, vals, color=[row[2] for row in rows])
        for bar, row in zip(bars, rows):
            bar.set_hatch(row[3])
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(max(v, 0), yi, f" {v:.3f}", va="center", fontsize=8.5)
        ax.set_yticks(y, [row[0] for row in rows])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        ax.set_xlim(min(0, lo) - 0.05 * abs(hi - lo or 1),
                    hi + 0.18 * abs(hi - lo or 1))

    ax = axes[1][1]
    losses = {r["loss"] for r in recs}
    loss_label = "/".join(sorted(str(v) for v in losses if v)) or "mse"
    loss_recs = sorted(recs, key=lambda r: r["training"].get("final_loss") or 0)
    y = np.arange(len(loss_recs))
    finals = [r["training"].get("final_loss") for r in loss_recs]
    ax.barh(y, finals, color=[r["color"] for r in loss_recs])
    for yi, v in zip(y, finals):
        ax.text(v, yi, f" {v:.4f}", va="center", fontsize=8.5)
    ax.set_yticks(y, [r["short"] for r in loss_recs])
    ax.invert_yaxis()
    ax.set_xlabel("loss de treino final (época 60)")
    ax.set_title(f"Função de perda: {loss_label} — valor final por modelo")
    savefig(fig, "02b_erro_geral.png")
    return rows


def plot_skill(recs, baselines=None):
    names = [r["short"] for r in recs]
    colors = [r["color"] for r in recs]
    hatches = [""] * len(recs)
    if baselines:
        for key in ("arima", "rf", "persistence"):
            m = baselines["metrics"].get(key)
            if not m:
                continue
            label, color = BASELINE_STYLE[key]
            names.append(f"{label} (baseline)")
            colors.append(color)
            hatches.append("//")
    y = np.arange(len(names))
    w = 0.38

    def _vals(regime):
        vals = [r[regime]["mse"] for r in recs]
        r2s = [r[regime]["r2"] for r in recs]
        if baselines:
            src_key = "all_horizons" if regime == "all" else regime
            for key in ("arima", "rf", "persistence"):
                m = baselines["metrics"].get(key)
                if m:
                    vals.append(m[src_key]["mse"])
                    r2s.append(m[src_key]["r2"])
        return vals, r2s

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4 + 0.4 * len(names)))
    mse, _ = _vals("all")
    axes[0].barh(y - w / 2, mse, height=w, color=colors, label="todos horizontes")
    mse_roll, _ = _vals("rolling")
    axes[0].barh(y + w / 2, mse_roll, height=w, color=colors, alpha=0.45,
                 hatch="//", label="rolling (h=36)")
    for yi, v in zip(y, mse):
        axes[0].text(v, yi - w / 2, f" {v:.3f}", va="center", fontsize=8)
    for yi, v in zip(y, mse_roll):
        axes[0].text(v, yi + w / 2, f" {v:.3f}", va="center", fontsize=8)
    for bar, h in zip(axes[0].containers[0], hatches):
        bar.set_hatch(h)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MSE ((m/s)²)")
    axes[0].set_title("MSE (vs ws100 bruto)")
    axes[0].legend(loc="lower right")

    _, r2 = _vals("all")
    axes[1].barh(y - w / 2, r2, height=w, color=colors, label="todos horizontes")
    _, r2_roll = _vals("rolling")
    axes[1].barh(y + w / 2, r2_roll, height=w, color=colors, alpha=0.45,
                 hatch="//", label="rolling (h=36)")
    for yi, v in zip(y, r2):
        axes[1].text(v, yi - w / 2, f" {v:.3f}", va="center", fontsize=8)
    for yi, v in zip(y, r2_roll):
        axes[1].text(v, yi + w / 2, f" {v:.3f}", va="center", fontsize=8)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("R²")
    axes[1].set_title("R² (vs ws100 bruto)")
    axes[1].legend(loc="lower right")
    savefig(fig, "03_mse_r2.png")


def plot_radar(recs, baselines=None):
    metric_rows = [
        ("MSE", True),
        ("RMSE", True),
        ("R²", False),
        ("MSE rolling", True),
        ("Parâmetros", True),
        ("Tempo treino (s)", True),
    ]

    def raw_values(r):
        return np.array([
            r["all"]["mse"],
            r["all"]["rmse"],
            r["all"]["r2"],
            r["rolling"]["mse"],
            float(r["params"]),
            r["training"]["training_time_sec"],
        ])

    base_rows = []
    if baselines:
        for key in ("arima", "rf", "persistence"):
            m = baselines["metrics"].get(key)
            if m:
                base_rows.append((f"{BASELINE_STYLE[key][0]} (baseline)",
                                  BASELINE_STYLE[key][1],
                                  np.array([m["all_horizons"]["mse"],
                                            m["all_horizons"]["rmse"],
                                            m["all_horizons"]["r2"],
                                            m["rolling"]["mse"],
                                            np.nan, np.nan])))

    arr = np.vstack([raw_values(r) for r in recs])
    if base_rows:
        arr = np.vstack([arr] + [b[2] for b in base_rows])
    norm = np.full_like(arr, np.nan)
    for j, (_, lower_better) in enumerate(metric_rows):
        col = arr[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        valid = np.isfinite(col)
        scaled = np.zeros_like(col)
        if hi - lo < 1e-12:
            scaled[valid] = 1.0
        elif lower_better:
            scaled[valid] = (hi - col[valid]) / (hi - lo)
        else:
            scaled[valid] = (col[valid] - lo) / (hi - lo)
        norm[valid, j] = 0.15 + 0.85 * scaled[valid]

    n_models = len(recs)
    n = len(metric_rows)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for i, r in enumerate(recs):
        values = norm[i].tolist() + [float(norm[i][0])]
        ax.plot(angles, values, color=r["color"], lw=1.8, label=r["short"])
        ax.fill(angles, values, color=r["color"], alpha=0.08)
    for k, (label, color, _raw) in enumerate(base_rows):
        row = norm[n_models + k]
        values = row[:4].tolist() + [row[0]]
        base_angles = angles[:4] + [angles[0]]
        ax.plot(base_angles, values, color=color, lw=1.2, ls="--",
                label=label)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metric_rows], fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Perfil dos métodos (eixos normalizados, mais longe do centro = melhor;\n"
                 "baselines plotados só nos eixos de erro)",
                 pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08), fontsize=8)
    savefig(fig, "04_radar.png")


def plot_per_horizon(recs, extras, baselines=None):
    variants = [
        ("per_horizon", "05_mae_por_horizonte_real.png",
         "MAE vs dado real (ws100 bruto)"),
        ("per_horizon_den", "05_mae_por_horizonte_wavelet.png",
         "MAE vs alvo wavelet (nível treinado)"),
    ]
    for key, fname, title in variants:
        fig, ax = plt.subplots(figsize=(13, 5.0))
        for r in recs:
            ph = extras[r["name"]][key]
            ax.plot(ph["horizon"], ph["mae"], color=r["color"], lw=1.7,
                    marker="o", ms=2.6, label=r["short"])
        if baselines and "per_horizon" in baselines:
            bh = baselines["per_horizon"]
            ax.plot(bh["horizon"], bh["persistence_mae"], color=BASELINE_STYLE["persistence"][1],
                    lw=1.2, ls=":", label="Persistência (baseline)")
            ax.plot(bh["horizon"], bh["arima_mae"], color=BASELINE_STYLE["arima"][1],
                    lw=1.4, ls="--", label="ARIMA (baseline)")
            ax.plot(bh["horizon"], bh["rf_mae"], color=BASELINE_STYLE["rf"][1],
                    lw=1.4, ls="--", label="Random Forest (baseline)")
        ax.set_xlabel("Horizonte (passos de 10 min)")
        ax.set_ylabel("MAE (m/s)")
        ax.set_title(f"MAE por horizonte de previsão — {title}")
        ax.set_xticks(range(1, 37, 2))
        ax.grid(alpha=0.3)
        sec = ax.secondary_xaxis("top", functions=(lambda x: x / 6, lambda x: x * 6))
        sec.set_xlabel("Horizonte (horas)")
        ax.legend(ncol=2, fontsize=8)
        savefig(fig, fname)


def plot_horizon_heatmap(recs, extras, baselines=None):
    mses = []
    labels = [r["short"] for r in recs]
    for r in recs:
        ph = extras[r["name"]]["per_horizon"]
        mses.append(ph["mse"].to_numpy())
    if baselines and "per_horizon" in baselines:
        bh = baselines["per_horizon"]
        for mae_col, rmse_col, label in (("arima_mae", "arima_rmse", "ARIMA"),
                                         ("rf_mae", "rf_rmse", "Random Forest"),
                                         ("persistence_mae", "persistence_rmse",
                                          "Persistência")):
            mses.append((bh[rmse_col] ** 2).to_numpy())
            labels.append(f"{label} (baseline)")
    mat = np.vstack(mses)

    # Fontes grandes: no artigo a figura entra a \linewidth de coluna
    # (≈0,41× a largura original), então o tamanho efetivo ≈ fontsize × 0,41.
    fig, ax = plt.subplots(figsize=(9, 1.7 + 0.34 * len(labels)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=float(mat.min()), vmax=float(mat.max()),
                   interpolation="nearest")
    ax.set_xlim(-0.5, 36.6)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(labelsize=14)
    _ticks = [5, 11, 17, 23, 29, 35]
    ax.set_xticks(_ticks, [str(6 * (k + 1)) for k in range(6)])
    ax.set_xlabel("Horizonte (passos de 10 min)", fontsize=15)
    ax.set_title("MSE ((m/s)²) por modelo e horizonte (verde = menor erro)",
                 fontsize=16)
    for j in _ticks:
        for i in range(len(labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=13, color="black",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.75, lw=0))
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("MSE ((m/s)²)", fontsize=15)
    cbar.ax.tick_params(labelsize=13)
    savefig(fig, "06_heatmap_mse_horizonte.png")


def plot_horizon_r2_heatmap(recs, extras, baselines=None):
    r2s = []
    labels = [r["short"] for r in recs]
    for r in recs:
        ph = extras[r["name"]]["per_horizon"]
        r2s.append(ph["r2"].to_numpy())
    if baselines and "per_horizon" in baselines:
        bh = baselines["per_horizon"]
        for col, label in (("arima_r2", "ARIMA (baseline)"),
                           ("rf_r2", "Random Forest (baseline)"),
                           ("persistence_r2", "Persistência (baseline)")):
            r2s.append(bh[col].to_numpy())
            labels.append(label)
    mat = np.vstack(r2s)

    # Fontes grandes: ver justificativa em plot_horizon_heatmap.
    fig, ax = plt.subplots(figsize=(9, 1.7 + 0.34 * len(labels)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                   vmin=float(mat.min()), vmax=float(mat.max()),
                   interpolation="nearest")
    ax.set_xlim(-0.5, 36.6)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(labelsize=14)
    _ticks = [5, 11, 17, 23, 29, 35]
    ax.set_xticks(_ticks, [str(6 * (k + 1)) for k in range(6)])
    ax.set_xlabel("Horizonte (passos de 10 min)", fontsize=15)
    ax.set_title("R² por modelo e horizonte (verde = melhor; 0 = equivalente à média)",
                 fontsize=16)
    for j in _ticks:
        for i in range(len(labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=13, color="black",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.75, lw=0))
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("R²", fontsize=15)
    cbar.ax.tick_params(labelsize=13)
    savefig(fig, "06_heatmap_r2_horizonte.png")


def compute_bootstrap_ci_matrix(recs, extras, base_preds=None,
                                horizons=(6, 12, 24, 36), n_boot=1000,
                                block=36, alpha=0.05, seed=20261023):
    """ΔMSE (linha − coluna) com IC por bootstrap em blocos, por horizonte.

    Complementa o DM: em vez de p-valor assintótico, cada célula recebe o
    diferencial de perda quadrática e o IC95% por bootstrap circular em
    blocos de ``block`` origens (preserva a autocorrelação das janelas
    sobrepostas e dos regimes de vento). Célula marcada = IC exclui zero.

    Alinhamento: os baselines (h=1 mira a própria origem) são deslocados
    em +1 horizonte para casar o alvo com o DL (h=1 mira origem+1 passo).
    """
    def error_wide_shifted(df, pred_col, shift=0):
        if "actual_raw" not in df:
            df = df.assign(actual_raw=df["actual"])
        e = df["actual_raw"] - df[pred_col]
        w = (df.assign(e=e)
               .pivot(index="origin", columns="horizon", values="e")
               .sort_index().sort_index(axis=1))
        if shift:
            w.columns = [h - shift for h in w.columns]
        return w

    series, names, labels = {}, [], []
    for r in recs:
        series[r["name"]] = error_wide_shifted(
            extras[r["name"]]["all_df"], "predicted")
        names.append(r["name"])
        labels.append(r["short"])
    series["__persistence__"] = error_wide_shifted(
        extras[recs[0]["name"]]["all_df"], "persistence")
    names.append("__persistence__")
    labels.append("Pers.")
    if base_preds is not None:
        series["__arima__"] = error_wide_shifted(base_preds, "arima", shift=1)
        series["__rf__"] = error_wide_shifted(base_preds, "rf", shift=1)
        names += ["__arima__", "__rf__"]
        labels += ["ARIMA", "RF"]

    rng = np.random.default_rng(seed)
    matrices = {}
    for h in horizons:
        present = [k for k in names if h in series[k].columns]
        common = None
        for k in present:
            w = series[k][[h]]
            ii = w.index[w[h].notna()]
            common = ii if common is None else common.intersection(ii)
        E = np.column_stack([series[k].loc[common, h].to_numpy() ** 2
                             for k in present])
        n = len(common)
        k_all = len(names)
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(n_boot, n_blocks))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n)
        idx = idx.reshape(n_boot, -1)[:, :n]
        delta = np.full((k_all, k_all), np.nan)
        sig = np.zeros((k_all, k_all), bool)
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                dd = E[:, a] - E[:, b]
                boot = dd[idx].mean(axis=1)
                lo, hi = np.percentile(boot, [100 * alpha / 2,
                                              100 * (1 - alpha / 2)])
                i, j = names.index(present[a]), names.index(present[b])
                delta[i, j] = dd.mean()
                delta[j, i] = -dd.mean()
                sig[i, j] = sig[j, i] = not (lo < 0 < hi)
        matrices[h] = {"delta": delta, "sig": sig, "n": n}
    return {"horizons": list(horizons), "labels": labels,
            "matrices": matrices, "n_boot": n_boot, "block": block,
            "alpha": alpha}


def plot_bootstrap_ci_matrix(bs, fname="16e_bootstrap_ci.png"):
    """Grade 2×2 dos ΔMSE com IC por bootstrap em blocos (* = IC exclui 0)."""
    labels = bs["labels"]
    k = len(labels)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.07)
    vmax = max(np.nanmax(np.abs(m["delta"])) for m in bs["matrices"].values())
    for ax in axes.ravel():
        h = bs["horizons"][list(axes.ravel()).index(ax)]
        m = bs["matrices"][h]
        im = ax.imshow(m["delta"], aspect="equal", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"h={h}", fontsize=14)
        ax.set_yticks(range(k), labels, fontsize=11)
        ax.set_xticks(range(k), labels, rotation=45, ha="right", fontsize=11)
        for i in range(k):
            for j in range(k):
                if i == j or not np.isfinite(m["delta"][i, j]):
                    continue
                star = "*" if m["sig"][i, j] else ""
                ax.text(j, i, f"{m['delta'][i, j]:+.2f}{star}",
                        ha="center", va="center", fontsize=9,
                        color="black", fontweight="bold" if star else "normal",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  alpha=0.7, lw=0))
    cbar = fig.colorbar(im, ax=axes, pad=0.04, shrink=0.9)
    cbar.set_label("ΔMSE (linha − coluna) (m/s)²")
    fig.suptitle(f"ΔMSE com IC95% por bootstrap em blocos "
                 f"({bs['n_boot']} reamostragens, bloco {bs['block']} "
                 f"origens) — * = IC exclui 0; positivo = linha pior")
    savefig(fig, fname)


def compute_dm_analysis(recs, extras, alpha=0.05, power=2):
    """Testes de Diebold–Mariano pareados (perda quadrática vs ws100 bruto).

    Erros de cada modelo alinhados por (origin, horizon) no mesmo teste.
    Por horizonte, a série de diferenciais ordenada por origin carrega
    correlação MA(h-1) (janelas sobrepostas) — o estimador de variância do
    DM usa defasagens até h-1. O agregado usa a média das perdas dos 36
    horizontes por origin, com h=36. Pares: modelo × modelo e cada modelo ×
    persistência (coluna comum a todos os all_df).
    """
    def error_matrix(df, pred_col):
        e = df["actual_raw"] - df[pred_col]
        wide = (df.assign(e=e)
                  .pivot(index="origin", columns="horizon", values="e")
                  .sort_index())
        return wide.sort_index(axis=1)

    matrices = {r["name"]: error_matrix(extras[r["name"]]["all_df"],
                                        "predicted") for r in recs}
    df0 = extras[recs[0]["name"]]["all_df"]
    persistence_name = "__persistence__"
    matrices[persistence_name] = error_matrix(df0, "persistence")

    common_idx = None
    for wide in matrices.values():
        common_idx = (wide.index if common_idx is None
                      else common_idx.intersection(wide.index))
    common_idx = common_idx.sort_values()
    horizons = list(matrices[recs[0]["name"]].columns)
    n_h = len(horizons)
    aligned = {name: wide.loc[common_idx].to_numpy(dtype=float)
               for name, wide in matrices.items()}

    mse_persistence = float(np.mean((df0["actual_raw"] - df0["persistence"]) ** 2))
    shorts = {r["name"]: r["short"] for r in recs}
    mses = {r["name"]: r["all"]["mse"] for r in recs}
    mses[persistence_name] = mse_persistence
    shorts[persistence_name] = "Persistência"

    def pair_entry(name_a, name_b):
        e_a = aligned[name_a]
        e_b = aligned[name_b]
        per_h = [diebold_mariano(e_a[:, j], e_b[:, j], h=h, power=power)
                 for j, h in enumerate(horizons)]
        d_origin = (np.abs(e_a) ** power).mean(axis=1) \
                 - (np.abs(e_b) ** power).mean(axis=1)
        aggregate = diebold_mariano_loss_diff(d_origin, h=n_h)
        return {
            "name_a": name_a, "name_b": name_b,
            "short_a": shorts[name_a], "short_b": shorts[name_b],
            "mse_a": mses[name_a], "mse_b": mses[name_b],
            "dm_h": np.array([r["dm"] for r in per_h], dtype=float),
            "p_h": np.array([r["p_value"] for r in per_h], dtype=float),
            "sig_count": int(sum(r["p_value"] < alpha for r in per_h)),
            "aggregate": aggregate,
        }

    entries = [pair_entry(r1["name"], r2["name"])
               for r1, r2 in combinations(recs, 2)]
    entries += [pair_entry(r["name"], persistence_name) for r in recs]
    return {"pairs": entries, "n_origins": int(len(common_idx)),
            "horizons": n_h, "alpha": alpha, "power": power}


def evaluate_loss_ab_variants(dataset, base_dir, keys=("lstm", "lstm_bi")):
    """Avalia variantes de loss (mesmo protocolo do models-dir principal)
    para o painel A/B do teste DM. Retorna {wrapper_key: all_df}."""
    variants = {}
    if base_dir is None or not Path(base_dir).is_dir():
        return variants
    try:
        by_key = {m["meta"]["wrapper_key"]: m for m in load_metadata(base_dir)}
    except FileNotFoundError:
        return variants
    for key in keys:
        entry = by_key.get(key)
        if entry is None:
            print(f"  aviso: variante de loss '{key}' ausente em {base_dir}")
            continue
        meta = entry["meta"]
        ratios = meta.get("dataset", {})
        train_df, val_df, test_df = common.split_dataset(
            dataset, ratios.get("train_ratio", 0.75),
            ratios.get("val_ratio", 0.20))
        val_end = len(train_df) + len(val_df)
        print(f"[{key}] A/B loss: carregando {entry['dir'].name} ...")
        wrapper = build_wrapper(meta, train_df,
                                val_df if len(val_df) else train_df.tail(400),
                                entry["dir"] / "model.keras")
        eval_df = dataset.iloc[max(0, val_end - wrapper.input_steps):]
        all_df = common.predict_all_horizons(wrapper, eval_df)
        observed = common.observed_timestamps(DATA_FILE)
        if observed is not None:
            all_df = all_df[all_df["timestamp"].isin(observed)]
        all_df["actual_raw"] = dataset["ws100"].reindex(
            all_df["timestamp"]).to_numpy()
        variants[key] = all_df
    return variants


def compute_dm_horizon_matrix(recs, extras, base_preds=None,
                              horizons=(6, 12, 24, 36), alpha=0.05, swap=None):
    """Matrizes quadradas de DM (linha × coluna) para horizontes selecionados.

    Célula (i, j) = DM entre o modelo da linha (A) e o da coluna (B) naquele
    horizonte, perda quadrática vs ws100 bruto, origens comuns ordenadas
    (correlação MA(h-1)). Positivo = linha pior; (j, i) = -DM de (i, j);
    diagonal vazia. Inclui ARIMA, RF e Persistência quando as predições dos
    baselines estão disponíveis (base_preds). `swap` troca a série de erros
    de um wrapper_key por uma variante: {key: (sufixo_de_label, all_df)}.
    """
    def error_wide(df, pred_col):
        if "actual_raw" not in df:
            df = df.assign(actual_raw=df["actual"])
        e = df["actual_raw"] - df[pred_col]
        return (df.assign(e=e)
                  .pivot(index="origin", columns="horizon", values="e")
                  .sort_index().sort_index(axis=1))

    series = {}
    names = []
    labels = []
    for r in recs:
        if swap and r["key"] in swap:
            suffix, alt_df = swap[r["key"]]
            series[r["key"]] = error_wide(alt_df, "predicted")
            names.append(r["key"])
            labels.append(r["short"] + suffix)
        else:
            series[r["name"]] = error_wide(extras[r["name"]]["all_df"],
                                           "predicted")
            names.append(r["name"])
            labels.append(r["short"])
    series["__persistence__"] = error_wide(
        extras[recs[0]["name"]]["all_df"], "persistence")
    if base_preds is not None:
        series["__arima__"] = error_wide(base_preds, "arima")
        series["__rf__"] = error_wide(base_preds, "rf")

    for key, label in (("__arima__", "ARIMA"), ("__rf__", "RF"),
                       ("__persistence__", "Pers.")):
        if key in series:
            names.append(key)
            labels.append(label)

    matrices = {}
    for h in horizons:
        common_idx = None
        for wide in series.values():
            idx = wide.index[wide[h].notna()]
            common_idx = idx if common_idx is None else common_idx.intersection(idx)
        common_idx = common_idx.sort_values()
        e_cols = {n: series[n].loc[common_idx, h].to_numpy(dtype=float)
                  for n in names}
        k = len(names)
        dm_mat = np.full((k, k), np.nan)
        p_mat = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(i + 1, k):
                res = diebold_mariano(e_cols[names[i]], e_cols[names[j]], h=h)
                dm_mat[i, j] = res["dm"]
                if np.isfinite(res["dm"]):
                    dm_mat[j, i] = -res["dm"]
                p_mat[i, j] = p_mat[j, i] = res["p_value"]
        matrices[h] = {"dm": dm_mat, "p": p_mat, "n": int(len(common_idx))}
    return {"labels": labels, "names": names, "horizons": list(horizons),
            "alpha": alpha, "matrices": matrices}


DM_GRID_TITLES = {6: "h6 (1 h)", 12: "h12 (2 h)", 24: "h24 (4 h)", 36: "h36 (6 h)"}


def _dm_grid(dmx, key, cmap, vmin, vmax, cbar_label, fig_title, fname,
             cell_text):
    """Grade 2×2 compartilhada das matrizes DM: legenda à direita, ticks
    com os nomes dos métodos em todos os painéis."""
    labels = dmx["labels"]
    k = len(labels)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.07)
    for ax, h in zip(axes.ravel(), dmx["horizons"]):
        m = dmx["matrices"][h]
        im = ax.imshow(np.ma.masked_invalid(m[key]), aspect="equal", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(DM_GRID_TITLES.get(h, f"h{h}"), fontsize=14)
        ax.set_yticks(range(k), labels, fontsize=11)
        ax.set_xticks(range(k), labels, rotation=45, ha="right", fontsize=11)
        for i in range(k):
            for j in range(k):
                if i == j or not np.isfinite(m[key][i, j]):
                    continue
                ax.text(j, i, cell_text(m, i, j), ha="center", va="center",
                        fontsize=9.5, color="black",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  alpha=0.7, lw=0))
    cbar = fig.colorbar(im, ax=axes, pad=0.04, shrink=0.9)
    cbar.set_label(cbar_label)
    fig.suptitle(fig_title)
    savefig(fig, fname)


def plot_dm_matrix(dmx):
    """Painel 2×2 com matrizes DM linha × coluna para h6, h12, h24 e h36."""
    vmax = np.nanmax([np.nanmax(np.abs(m["dm"])) for m in dmx["matrices"].values()])
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0

    def cell_text(m, i, j):
        p = m["p"][i, j]
        stars = ""
        if np.isfinite(p):
            if p < 0.01:
                stars = "**"
            elif p < dmx["alpha"]:
                stars = "*"
        return f"{m['dm'][i, j]:+.1f}{stars}"

    _dm_grid(
        dmx, "dm", "RdBu_r", -vmax, vmax,
        "DM corrigido (t) — positivo: modelo da linha tem perda maior (pior)",
        "Teste de Diebold–Mariano por horizonte "
        f"(* p < {dmx['alpha']:.2f}; ** p < 0,01)",
        "16_significancia_dm.png", cell_text,
    )


def plot_dm_pvalue_matrix(dmx, fname="16b_significancia_dm_pvalor.png",
                          suptitle=None):
    """Mesma grade da figura de DM, com o p-valor de cada célula."""
    def cell_text(m, i, j):
        p = m["p"][i, j]
        return "<0,001" if p < 0.001 else fmt(p, 2)

    _dm_grid(
        dmx, "p", "RdYlGn_r", 0.0, 0.10,
        "p-valor (verde < 0,05 = diferença significativa; escala até 0,10)",
        suptitle or "p-valores do teste de Diebold–Mariano por horizonte",
        fname, cell_text,
    )


R2_TABLE_HORIZONS = (1, 2, 3, 6, 9, 12, 18, 24, 30, 36)


def build_r2_per_horizon_table(recs, extras, baselines=None):
    """Tabela compacta de R² por horizonte (vs ws100 bruto), com baselines."""
    index = {r["name"]: extras[r["name"]]["per_horizon"].set_index("horizon")
             for r in recs}
    columns = ["Horizonte (passos / h)"] + [r["short"] for r in recs]
    if baselines and "per_horizon" in baselines:
        bh = baselines["per_horizon"].set_index("horizon")
        columns += ["ARIMA", "Random Forest", "Persistência"]
    rows = []
    for h in R2_TABLE_HORIZONS:
        hour = h / 6
        label = f"h{h} ({hour:.0f} h)" if hour == int(hour) else f"h{h}"
        row = [label]
        row += [fmt(index[r["name"]].loc[h, "r2"], 3) for r in recs]
        if baselines and "per_horizon" in baselines:
            row += [fmt(bh.loc[h, c], 3) for c in ("arima_r2", "rf_r2",
                                                   "persistence_r2")]
        rows.append(row)
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def plot_fan(recs, extras, dataset, base_preds=None):
    all_df = extras[recs[0]["name"]]["all_df"]
    origins = all_df["origin"].unique()
    idx = np.linspace(0, len(origins) - 1, 6).astype(int)
    # cores por modelo + viridis apenas para as linhas de origem
    model_colors = {r["short"]: r["color"] for r in recs}
    base_lines = (("persistence", "Persistência", "#7a7a7a", ":"),
                  ("arima", "ARIMA", BASELINE_STYLE["arima"][1], "--"),
                  ("rf", "Random Forest", BASELINE_STYLE["rf"][1], "--"))

    t_start = pd.Timestamp(origins[idx[0]])
    t_end = all_df["timestamp"].max()

    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5), sharex=False,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.12})
    handles = []
    for k, (ax, o) in enumerate(zip(axes.ravel(), origins[idx])):
        o_ts = pd.Timestamp(o)
        span = dataset.loc[o_ts - pd.Timedelta(hours=2):o_ts + pd.Timedelta(hours=6),
                           "ws100"]
        ax.plot(span.index, span.values, color="black", lw=1.0, alpha=0.6,
                label="ws100 observado")
        for r in recs:
            sub = extras[r["name"]]["all_df"]
            sub = sub[sub["origin"] == o].sort_values("horizon")
            line, = ax.plot(sub["timestamp"], sub["predicted"],
                            color=model_colors[r["short"]], lw=1.5,
                            label=r["short"])
            if k == 0:
                handles.append(line)
        if base_preds is not None:
            bsub = base_preds[base_preds["origin"] == o].sort_values("horizon")
            for col, label, color, ls in base_lines:
                line, = ax.plot(bsub["timestamp"], bsub[col], color=color,
                                lw=1.1, ls=ls, label=label)
                if k == 0:
                    handles.append(line)
        ax.axvline(o_ts, color="gray", lw=0.7, ls=":", alpha=0.7)
        ax.set_title(f"origem {o_ts:%d/%m %H:%M}", fontsize=9.5)
        ax.set_ylabel("m/s", fontsize=8.5)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    fig.suptitle("Previsões de 6 h à frente de todos os modelos, 6 origens do teste",
                 fontsize=12)
    handles = [plt.Line2D([], [], color="black", lw=1.0, alpha=0.6,
                          label="ws100 observado")] + handles
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    savefig(fig, "07_fan_chart.png")


def plot_worst_predictions(recs, extras, dataset, top_n=12):
    """Piores trajetorias completas (36 passos) da LSTM, sem sobreposicao.

    Selecao gulosa: origens ordenadas por MSE da trajetoria inteira; uma
    origem so entra se distar >= 6 h (36 passos) de todas as ja escolhidas,
    garantindo que nenhum alvo apareca em duas predicoes.
    """
    try:
        lstm = next(r for r in recs if r["key"] == "lstm")
    except StopIteration:
        return None
    df = extras[lstm["name"]]["all_df"]
    df = df.dropna(subset=("actual_raw", "predicted"))
    per_origin = df.groupby("origin").apply(
        lambda g: pd.Series({
            "mse": float(np.mean((g["predicted"] - g["actual_raw"]) ** 2)),
            "mae": float(np.mean(np.abs(g["predicted"] - g["actual_raw"]))),
            "worst_h": int(g.loc[(g["predicted"] - g["actual_raw"]).abs().idxmax(),
                                 "horizon"]),
            "n": len(g),
        }), include_groups=False)
    per_origin = per_origin[per_origin["n"] == df["horizon"].max()].copy()
    per_origin = per_origin.sort_values("mse", ascending=False)

    min_gap = pd.Timedelta(minutes=10 * df["horizon"].max())
    selected = []
    for origin, row in per_origin.iterrows():
        ts = pd.Timestamp(origin)
        if all(abs((ts - pd.Timestamp(s)).total_seconds()) >= min_gap.total_seconds()
               for s in selected):
            selected.append(origin)
            if len(selected) >= top_n:
                break

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12.5))
    rows = []
    for k, (ax, origin) in enumerate(zip(axes.ravel(), selected)):
        o_ts = pd.Timestamp(origin)
        sub = df[df["origin"] == origin].sort_values("horizon")
        span = dataset.loc[o_ts - pd.Timedelta(hours=2):o_ts + pd.Timedelta(hours=6),
                           "ws100"]
        ax.plot(span.index, span.values, color="black", lw=1.1, alpha=0.65,
                label="ws100 observado")
        ax.plot(sub["timestamp"], sub["predicted"], color=lstm["color"], lw=1.6,
                label="LSTM (36 passos)")
        ax.axvline(o_ts, color="gray", lw=0.7, ls=":", alpha=0.7)
        ax.set_title(f"#{k + 1} · origem {o_ts:%d/%m %H:%M} · "
                     f"MSE {row['mse']:.2f} (m/s)²", fontsize=9)
        ax.set_ylabel("m/s", fontsize=8.5)
        ax.tick_params(labelsize=7.5, labelrotation=20)
        ax.grid(alpha=0.25)
        rows.append({"rank": k + 1, "origem": f"{o_ts:%d/%m/%Y %H:%M}",
                     "mse": row["mse"], "mae": row["mae"],
                     "pior_h": row["worst_h"]})
    handles = [plt.Line2D([], [], color="black", lw=1.1, alpha=0.65,
                          label="ws100 observado"),
               plt.Line2D([], [], color=lstm["color"], lw=1.6,
                          label="LSTM (36 passos)")]
    fig.suptitle("As 12 piores previsões completas da LSTM — trajetórias de 6 h "
                 "sem nenhum alvo compartilhado", fontsize=12)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.002))
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    savefig(fig, "10_piores_predicoes_lstm.png")
    return rows


def plot_rolling(recs, extras, base_preds=None):
    level = current_denoise_level()
    base_roll = None
    if base_preds is not None:
        sub = base_preds[base_preds["horizon"] == base_preds["horizon"].max()]
        base_roll = sub.sort_values("origin")
    base_lines = (("persistence", "Persistência", "#7a7a7a", ":"),
                  ("arima", "ARIMA", BASELINE_STYLE["arima"][1], "--"),
                  ("rf", "Random Forest", BASELINE_STYLE["rf"][1], "--"))
    variants = [
        ("actual_raw", "08_rolling_forecast_real.png",
         "observado (bruto)", "Erro vs dado bruto"),
        ("actual_denoised", "08_rolling_forecast_wavelet.png",
         f"alvo wavelet nível {level}", "Erro vs alvo wavelet"),
    ]
    for actual_col, fname, ref_label, err_label in variants:
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
        roll0 = extras[recs[0]["name"]]["roll_df"]
        axes[0].plot(roll0["timestamp"], roll0[actual_col], color="black", lw=1.2,
                     label=ref_label, zorder=5)
        for r in recs:
            roll = extras[r["name"]]["roll_df"]
            axes[0].plot(roll["timestamp"], roll["predicted"], color=r["color"],
                         lw=1.1, alpha=0.9,
                         label=f"{r['short']} (MAE {r['rolling']['mae']:.2f})")
        if base_roll is not None and len(base_roll):
            for col, label, color, ls in base_lines:
                err = base_roll[col] - base_roll["actual"]
                axes[0].plot(base_roll["timestamp"], base_roll[col], color=color,
                             lw=1.0, ls=ls, alpha=0.9,
                             label=f"{label} (MAE {np.mean(np.abs(err)):.2f})")
                resid = (base_roll[col] - base_roll["actual"]).rolling(
                    12, min_periods=1).mean()
                axes[1].plot(base_roll["timestamp"], resid, color=color, lw=1.0,
                             ls=ls, alpha=0.85)
        axes[0].set_ylabel("Velocidade (m/s)")
        axes[0].set_title(f"Previsão autônoma (rolling, h=36) vs {ref_label} — teste")
        axes[0].legend(ncol=3, fontsize=8)

        for r in recs:
            roll = extras[r["name"]]["roll_df"]
            resid = (roll["predicted"] - roll[actual_col]).rolling(12, min_periods=1).mean()
            axes[1].plot(roll["timestamp"], resid, color=r["color"], lw=1.3,
                         label=r["short"])
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_ylabel(f"{err_label}\n(média móvel 2 h)")
        axes[1].set_xlabel("Timestamp")
        savefig(fig, fname)


def _pacf_values(x, nlags):
    """PACF por Durbin-Levinson sobre a ACF amostral (equivalente a ywm)."""
    x = x - x.mean()
    acf = np.array([1.0] + [np.corrcoef(x[:-k], x[k:])[0, 1]
                            for k in range(1, nlags + 1)])
    phi = np.zeros((nlags + 1, nlags + 1))
    pacf = np.ones(nlags + 1)
    phi[1, 1] = acf[1]
    pacf[1] = phi[1, 1]
    for k in range(2, nlags + 1):
        den = 1.0 - np.dot(phi[k - 1, 1:k], acf[1:k])
        num = acf[k] - np.dot(phi[k - 1, 1:k], acf[k - 1:0:-1])
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf[k] = phi[k, k]
    return pacf


def plot_diagnostics(recs, extras, base_preds=None):
    lo, hi = np.inf, -np.inf
    for r in recs:
        df = extras[r["name"]]["all_df"]
        lo = min(lo, float(df["actual_raw"].min()), float(df["predicted"].min()))
        hi = max(hi, float(df["actual_raw"].max()), float(df["predicted"].max()))

    base_frames = []
    if base_preds is not None:
        for col, label, color in (("persistence", "Persistência", "#7a7a7a"),
                                  ("arima", "ARIMA", BASELINE_STYLE["arima"][1]),
                                  ("rf", "Random Forest", BASELINE_STYLE["rf"][1])):
            base_frames.append((label, color,
                                base_preds[["actual", col]].rename(
                                    columns={"actual": "actual_raw", col: "predicted"})))
    n_panels = len(recs) + len(base_frames)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    flat = np.asarray(axes).ravel()
    for ax, r in zip(flat, recs):
        df = extras[r["name"]]["all_df"]
        hb = ax.hexbin(df["actual_raw"], df["predicted"], gridsize=38,
                       cmap="YlGnBu", mincnt=1)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, zorder=5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Observado (m/s)")
        ax.set_ylabel("Previsto (m/s)")
        ax.set_title(f"Previsto vs observado — {r['short']} (todos os horizontes)",
                     fontsize=10.5)
        fig.colorbar(hb, ax=ax, pad=0.01).set_label("contagem")
    for ax, (label, color, df) in zip(flat[len(recs):], base_frames):
        ax.scatter(df["actual_raw"], df["predicted"], s=1.2, alpha=0.15, color=color)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, zorder=5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Observado (m/s)")
        ax.set_ylabel("Previsto (m/s)")
        ax.set_title(f"Previsto vs observado — {label} (baseline)",
                     fontsize=10.5)
    for ax in flat[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, "09_diagnostico_dispersao.png")

    err_all = [
        (r, extras[r["name"]]["all_df"]["predicted"]
            - extras[r["name"]]["all_df"]["actual_raw"])
        for r in recs
    ] + [(("baselines", label), df["predicted"] - df["actual_raw"])
         for label, _color, df in base_frames]
    bins = np.linspace(min(e.min() for _, e in err_all),
                       max(e.max() for _, e in err_all), 61)
    families = [("recorrentes (LSTM)", lambda r: isinstance(r, tuple) or not r["key"].startswith("tcn")),
                ("convolucionais (TCN)", lambda r: isinstance(r, tuple) or r["key"].startswith("tcn"))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (label, cond) in zip(axes, families):
        for item, err in err_all:
            if not cond(item):
                continue
            if isinstance(item, tuple):
                ax.hist(err, bins=bins, density=True, histtype="step", lw=1.3,
                        ls="--", color=BASELINE_STYLE.get(
                            "persistence" if item[1] == "Persistência"
                            else ("arima" if item[1] == "ARIMA" else "rf"))[1],
                        label=f"{item[1]} (baseline)")
                continue
            r = item
            mu, sigma = float(err.mean()), float(err.std())
            ax.hist(err, bins=bins, density=True, histtype="step", lw=1.7,
                    color=r["color"],
                    label=f"{r['short']} (μ={mu:+.2f}, σ={sigma:.2f})")
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_xlabel("Erro (m/s)")
        ax.set_title(f"Distribuição dos erros — {label}")
        ax.legend()
    axes[0].set_ylabel("Densidade")
    savefig(fig, "09_diagnostico_histograma.png")

    all_frames = [(r["short"], r["color"], extras[r["name"]]["all_df"])
                  for r in recs] + [
        (f"{label} (baseline)", color, df)
        for label, color, df in base_frames]
    alo = min(float(df["actual_raw"].min()) for _, _, df in all_frames)
    ahi = max(float(df["actual_raw"].max()) for _, _, df in all_frames)
    vbins = np.arange(np.floor(alo), np.ceil(ahi) + 1e-9, 1.0)
    vcenters = 0.5 * (vbins[:-1] + vbins[1:])
    n = len(all_frames)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for i, (label, color, df) in enumerate(all_frames):
        err = df["predicted"] - df["actual_raw"]
        bias = err.groupby(pd.cut(df["actual_raw"], vbins),
                           observed=False).mean().to_numpy()
        ax.bar(vcenters + (i - (n - 1) / 2) * (0.8 / n),
               np.nan_to_num(bias, nan=0.0), width=0.8 / n,
               color=color, alpha=0.9, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(np.floor(alo), np.ceil(ahi) + 1))
    ax.set_xlabel("Velocidade observada (m/s)")
    ax.set_ylabel("Erro médio (m/s)")
    ax.set_title("Viés por regime de vento (erro > 0 = superestima)")
    ax.legend(ncol=2, fontsize=8)
    savefig(fig, "09_diagnostico_vies.png")

    lags = np.arange(1, 37)
    resid = {r["name"]: (extras[r["name"]]["roll_df"]["predicted"]
                         - extras[r["name"]]["roll_df"]["actual_raw"]).to_numpy()
             for r in recs}
    ci = 1.96 / np.sqrt(len(next(iter(resid.values()))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    for r in recs:
        rd = resid[r["name"]]
        acf = np.array([np.corrcoef(rd[:-lag], rd[lag:])[0, 1] for lag in lags])
        pacf = _pacf_values(rd, 36)[1:]
        for ax, vals in ((axes[0], acf), (axes[1], pacf)):
            ax.plot(lags, vals, color=r["color"], lw=1.4, marker="o", ms=3,
                    label=r["short"])
    for ax, ttl, ylab in ((axes[0], "ACF", "Autocorrelação"),
                          (axes[1], "PACF", "Autocorrelação parcial")):
        ax.axhspan(-ci, ci, color="gray", alpha=0.18, lw=0)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel("Defasagem (passos de 10 min)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ttl} dos resíduos rolling (h=36) — todos os modelos")
        ax.set_xticks(range(0, 37, 3))
    axes[0].legend()
    savefig(fig, "09_diagnostico_acf_pacf.png")


def plot_early_stopping(recs):
    names = [r["name"] for r in recs]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    eps_orig = [r["training"]["epochs_run"] for r in recs]
    budget = max((r["training"].get("epochs_requested") or 0) for r in recs) or 150
    has_val = any(r["training"].get("best_val_loss") is not None for r in recs)
    axes[0].barh(y, eps_orig, height=0.6, color=[r["color"] for r in recs])
    axes[0].axvline(budget, color="black", ls="--", lw=1.2,
                    label=f"orçamento ({budget} épocas)")
    for yi, v in zip(y, eps_orig):
        axes[0].text(v, yi, f" {v:.0f}", va="center", fontsize=8)
    axes[0].set_yticks(y, [short_name(n) for n in names])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Épocas executadas")
    axes[0].set_title("Épocas executadas vs orçamento"
                      + ("" if has_val else " (sem validação: orçamento fixo)"))
    axes[0].legend(loc="lower right")

    best_orig = [r["training"].get("best_val_loss") or np.nan for r in recs]
    final_orig = [r["training"].get("final_val_loss") or np.nan for r in recs]
    axes[1].barh(y - 0.19, best_orig, height=0.38,
                 color=[r["color"] for r in recs], label="melhor val_loss")
    axes[1].barh(y + 0.19, final_orig, height=0.38,
                 color=[r["color"] for r in recs], alpha=0.45,
                 label="val_loss final (restaurada p/ melhor)")
    for yi, v in zip(y, best_orig):
        if np.isfinite(v):
            axes[1].text(v, yi - 0.19, f" {v:.4f}", va="center", fontsize=8)
    if not has_val:
        axes[1].text(0.5, 0.5, "sem partição de validação\n(split 75/25)",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=10, color="#666")
        axes[1].set_xticks([])
    axes[1].set_yticks(y, [short_name(n) for n in names])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("val_loss")
    axes[1].set_title("val_loss no treino original (best vs final)")
    axes[1].legend(loc="lower right")
    savefig(fig, "11_early_stopping.png")


def plot_training_time(recs):
    names = [r["name"] for r in recs]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    total_orig = [r["training"]["training_time_sec"] for r in recs]
    ax.barh(y, total_orig, color=[r["color"] for r in recs])
    for yi, v in zip(y, total_orig):
        ax.text(v, yi, f" {v:.0f}s", va="center", fontsize=8.5)
    ax.set_yticks(y, [short_name(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Tempo total de treino (s)")
    ax.set_title("Tempo de treinamento")
    savefig(fig, "12_tempo_treinamento.png")


def plot_inference(recs):
    names = [r["short"] for r in recs]
    y = np.arange(len(recs))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    per_win = [r["timing"]["per_window_ms"] for r in recs]
    axes[0].barh(y, per_win, color=[r["color"] for r in recs])
    for yi, v in zip(y, per_win):
        axes[0].text(v, yi, f" {v:.2f} ms", va="center", fontsize=8.5)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("ms por janela (36 previsões)")
    axes[0].set_title(f"Inferência em lote (GPU, {recs[0]['timing']['windows']} janelas)")

    single = [r["timing"]["single_shot_ms"] for r in recs]
    axes[1].barh(y, single, color=[r["color"] for r in recs])
    for yi, v in zip(y, single):
        axes[1].text(v, yi, f" {v:.1f} ms", va="center", fontsize=8.5)
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ms por chamada (1 janela)")
    axes[1].set_title("Latência single-shot (modelo, 1 janela)")

    for r in recs:
        axes[2].scatter(r["params"], r["size_mb"], s=140, color=r["color"],
                        edgecolor="black", lw=0.6, zorder=3)
        axes[2].annotate(r["short"], (r["params"], r["size_mb"]),
                         textcoords="offset points", xytext=(7, 4), fontsize=8)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Parâmetros treináveis")
    axes[2].set_ylabel("Tamanho do model.keras (MB)")
    axes[2].set_title("Tamanho do modelo")
    savefig(fig, "13_inferencia.png")


def plot_tradeoff(recs):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    anns = []
    x_all = [r["training"]["training_time_sec"] for r in recs]
    x_lo, x_hi = min(x_all), max(x_all)
    for r in recs:
        x, y = r["training"]["training_time_sec"], r["all"]["mse"]
        ax.scatter(x, y, s=max(60.0, r["params"] / 250.0), color=r["color"],
                   edgecolor="black", lw=0.7, zorder=3, alpha=0.9)
        near_right = x > x_lo + 0.75 * (x_hi - x_lo)
        anns.append(ax.annotate(
            f"{r['short']}\n({fmt_int(r['params'])} parâm.)", (x, y),
            textcoords="offset points", xytext=(-12, 8) if near_right else (10, 8),
            ha="right" if near_right else "left", fontsize=8.5,
        ))
    ax.set_xscale("log")
    ax.margins(y=0.24)
    ax.set_xlabel("Tempo de treinamento (s, escala log)")
    ax.set_ylabel("MSE todos os horizontes ((m/s)²)")
    ax.set_title("Custo de treinamento × erro (área = nº de parâmetros)")

    # Resolve sobreposição: bolhas contam como obstáculos e cada rótulo
    # desce até um espaço livre (a folga superior protege o título).
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    obstacles = []
    for r in recs:
        cx, cy = ax.transData.transform(
            (r["training"]["training_time_sec"], r["all"]["mse"]))
        rad = (max(60.0, r["params"] / 250.0) ** 0.5) / 2 + 3
        rad *= fig.dpi / 72
        obstacles.append(Bbox([[cx - rad, cy - rad], [cx + rad, cy + rad]]))
    placed = []
    for ann in anns:
        box = ann.get_window_extent(renderer)
        moves = 0
        while moves < 80 and (any(box.overlaps(b) for b in placed)
                              or any(box.overlaps(b) for b in obstacles)):
            dx, dy = ann.xyann
            ann.set_position((dx, dy - 12))
            fig.canvas.draw()
            box = ann.get_window_extent(renderer)
            moves += 1
        placed.append(box)
    savefig(fig, "14_custo_beneficio.png")


# ----------------------------------------------------------------------------
# Relatório HTML
# ----------------------------------------------------------------------------
CSS = """
:root{--accent:#1b6ca8;--bg:#f4f6f9;--card:#ffffff;}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI',system-ui,Arial,sans-serif;background:var(--bg);color:#222}
header{background:linear-gradient(135deg,#0f3d5e,#1b6ca8);color:#fff;padding:28px 32px}
header h1{margin:0 0 6px;font-size:24px}
header p{margin:0;opacity:.9;font-size:14px}
nav{background:#0f3d5e;padding:10px 32px}
nav a{color:#cfe6f7;text-decoration:none;font-size:13px;margin-right:18px}
nav a:hover{color:#fff;text-decoration:underline}
.container{max-width:1100px;margin:0 auto;padding:24px 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid #e3e8ee;border-radius:10px;padding:14px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card .num{font-size:20px;font-weight:700;color:var(--accent)}
.card .lab{font-size:12px;color:#555;margin-top:2px}
section{background:var(--card);border:1px solid #e3e8ee;border-radius:10px;margin:22px 0;padding:20px 24px;box-shadow:0 1px 3px rgba(0,0,0,.06);scroll-margin-top:12px}
section h2{margin:0 0 4px;font-size:18px;color:#0f3d5e;border-bottom:2px solid #e8eef4;padding-bottom:8px}
section .sub{color:#666;font-size:13px;margin:8px 0 14px;line-height:1.5}
figure{margin:14px auto;text-align:center}
figure img{max-width:100%;height:auto;border:1px solid #e3e8ee;border-radius:8px}
figcaption{font-size:12px;color:#666;margin-top:5px}
.fig-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 16px;align-items:start}
.fig-grid figure{margin:0}
.fig-grid figure img{max-width:100%;max-height:560px;width:auto;margin:0 auto}
@media (max-width:900px){.fig-grid{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{border:1px solid #dfe5ec;padding:5px 9px;text-align:right}
th{background:#eef3f8;color:#0f3d5e}
td:first-child,th:first-child{text-align:left;font-weight:600}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
.btn-copy{background:#1b6ca8;color:#fff;border:none;border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer;margin:8px 0 0;font-family:inherit}
.btn-copy:hover{background:#0f3d5e}
.btn-copy:active{opacity:.8}
.hl{background:#fff7e0;padding:10px 14px;border-left:4px solid #e0a800;border-radius:0 6px 6px 0;font-size:13px;line-height:1.5;margin:10px 0}
.note{background:#eef6ee;padding:10px 14px;border-left:4px solid #3d8b57;border-radius:0 6px 6px 0;font-size:13px;line-height:1.5;margin:10px 0}
.good{color:#1b7a3d;font-weight:700}
.bad{color:#b03030;font-weight:700}
section ul{margin:8px 0;padding-left:20px}
section li{margin:7px 0;font-size:13px;line-height:1.55;color:#333}
section li b{color:#0f3d5e}
section p{font-size:13px;line-height:1.6;color:#333}
code{background:#eef3f8;border:1px solid #dfe5ec;border-radius:4px;padding:1px 5px;font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#0f3d5e}
footer{color:#888;text-align:center;font-size:12px;padding:18px}
@media(max-width:760px){.two{grid-template-columns:1fr}}
"""

JS = """
function copiarLatex(btn){
  var latex = btn.getAttribute("data-latex");
  function done(){ btn.textContent = "copiado!"; setTimeout(function(){ btn.textContent = "copiar como LaTeX"; }, 1500); }
  if(navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(latex).then(done).catch(function(){ fallback(latex); done(); });
  } else { fallback(latex); done(); }
}
function fallback(text){
  var ta = document.createElement("textarea");
  ta.value = text; document.body.appendChild(ta); ta.select();
  try{ document.execCommand("copy"); }catch(e){}
  document.body.removeChild(ta);
}
"""


def build_results_table(recs):
    rows = []
    for r in recs:
        rows.append((
            r["short"],
            fmt(r["all"]["mse"], 3),
            fmt(r["all"]["rmse"], 3),
            fmt(r["all"]["r2"], 3),
            fmt(r["rolling"]["mse"], 3),
            fmt(r["rolling"]["rmse"], 3),
            fmt(r["training"]["training_time_sec"], 0),
            fmt_int(r["params"]),
            fmt(r["size_mb"], 1),
            fmt(r["timing"]["per_window_ms"], 2),
        ))
    columns = ["Modelo", "MSE", "RMSE", "R²", "MSE rolling (6 h)", "RMSE rolling (6 h)",
               "Treino (s)", "Parâmetros", "MB", "ms/janela"]
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def compute_mcs(recs, extras, base_preds=None, alpha=0.10, n_boot=2000,
                mean_block=36, seed=SEED):
    """Model Confidence Set agregando a perda sobre a predição inteira.

    Matriz de perdas: média do erro quadrático nos 36 horizontes por
    origem (mesmo agregado do DM), para as redes + Persistência (+ ARIMA/RF
    quando as predições dos baselines estão disponíveis).
    """
    def loss_per_origin(df):
        e = (df["actual_raw"] - df["predicted"]) ** 2
        return e.groupby(df["origin"]).mean().sort_index()

    series, labels, mses = [], [], []
    for r in recs:
        series.append(loss_per_origin(extras[r["name"]]["all_df"]))
        labels.append(r["short"])
        mses.append(r["all"]["mse"])
    df0 = extras[recs[0]["name"]]["all_df"]
    series.append(loss_per_origin(
        df0.assign(predicted=df0["persistence"])))
    labels.append("Persistência")
    mses.append(float(np.mean((df0["actual_raw"] - df0["persistence"]) ** 2)))
    if base_preds is not None:
        for col, label in (("arima", "ARIMA"), ("rf", "Random Forest")):
            series.append(loss_per_origin(
                base_preds.assign(actual_raw=base_preds["actual"],
                                  predicted=base_preds[col])))
            labels.append(f"{label} (baseline)")
            mses.append(float(np.mean((base_preds["actual"]
                                       - base_preds[col]) ** 2)))

    common_idx = series[0].index
    for s in series[1:]:
        common_idx = common_idx.intersection(s.index)
    common_idx = common_idx.sort_values()
    mat = np.column_stack([s.loc[common_idx].to_numpy(dtype=float)
                           for s in series])
    res = model_confidence_set(mat, names=labels, alpha=alpha,
                               n_boot=n_boot, mean_block=mean_block,
                               seed=seed)
    res["mse"] = dict(zip(labels, mses))
    return res


def plot_mcs(mcs, fname="16d_mcs.png"):
    """Barra horizontal dos p-valores de exclusão do MCS (α em linha)."""
    items = sorted(mcs["exclusion_p"].items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    pvals = [v for _, v in items]
    colors = ["#1c7c3c" if v >= mcs["alpha"] else "#c0504d" for v in pvals]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 2.2 + 0.55 * len(labels)))
    ax.barh(y, pvals, color=colors, alpha=0.9)
    ax.axvline(mcs["alpha"], color="black", ls="--", lw=1.2,
               label=f"α = {mcs['alpha']:.2f}")
    for yi, v in zip(y, pvals):
        txt = f"{v:.3f}" if v >= 0.001 else "<0,001"
        ax.text(min(v + 0.01, 0.98), yi, txt, va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("p-valor de exclusão do MCS "
                  f"({mcs['n_boot']} bootstrap, bloco médio "
                  f"{mcs['mean_block']})")
    ax.set_title("Model Confidence Set — verde: pertence ao conjunto "
                 f"({', '.join(mcs['mcs'])})")
    ax.legend(loc="lower right")
    savefig(fig, fname)


def build_mcs_table(mcs):
    """Tabela do MCS: modelo, MSE, p de exclusão e status."""
    rows = []
    for label, p in sorted(mcs["exclusion_p"].items(), key=lambda kv: kv[1]):
        inside = p >= mcs["alpha"]
        p_txt = fmt(p, 3) if p >= 0.001 else "&lt; 0,001"
        rows.append((
            label,
            fmt(mcs["mse"].get(label, float("nan")), 3),
            p_txt,
            "pertence ao MCS" if inside else "excluído",
        ))
    columns = ["Modelo", "MSE (m/s)²",
               "p-valor de exclusão",
               f"Status (α = {fmt(mcs['alpha'], 2)})"]
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def build_dm_table(dm):
    """Tabela de significância Diebold–Mariano (agregado + por horizonte)."""
    rows = []
    for p in dm["pairs"]:
        agg = p["aggregate"]
        p_val = agg["p_value"]
        if p_val < 0.001:
            p_txt = "&lt; 0,001"
        else:
            p_txt = fmt(p_val, 3)
        if not np.isfinite(p_val) or p_val >= dm["alpha"]:
            verdict = "Sem diferença significativa"
        elif agg["dm"] > 0:
            verdict = f"<b>{p['short_b']}</b> melhor"
        else:
            verdict = f"<b>{p['short_a']}</b> melhor"
        rows.append((
            f"{p['short_a']} × {p['short_b']}",
            fmt(p["mse_a"], 3),
            fmt(p["mse_b"], 3),
            fmt(agg["dm"], 2) if np.isfinite(agg["dm"]) else "—",
            p_txt,
            f"{p['sig_count']}/{dm['horizons']}",
            verdict,
        ))
    columns = ["Comparação (A × B)", "MSE A", "MSE B", "DM agregado",
               "p-valor", f"Horizontes signif. (p&lt;{dm['alpha']:.2f})",
               f"Conclusão (α = {fmt(dm['alpha'], 2)})"]
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


class _SpaceRecorder:
    """Registra o espaço de busca imitando a API de OptunaHyperParameters."""

    def __init__(self):
        self.entries = []

    def Int(self, name, min_value, max_value, step=1, default=None):
        rng = f"int {min_value}–{max_value}" + (f" (passo {step})" if step > 1 else "")
        self.entries.append((name, rng, "int"))
        return default if default is not None else min_value

    def Float(self, name, min_value, max_value, step=None, sampling=None, default=None):
        rng = f"{min_value}–{max_value}"
        if sampling == "LOG":
            rng += " (escala log)"
        if step:
            rng += f" (passo {step})"
        self.entries.append((name, rng, "float"))
        return default if default is not None else min_value

    def Choice(self, name, values, default=None):
        self.entries.append((name, " | ".join(str(v) for v in values), "categórico"))
        return default if default is not None else values[0]


def collect_search_space(wrapper_key):
    """Executa build() com um recorder para derivar o espaço de busca do código."""
    from keras import backend as K

    wrapper = common.wrapper_factory(wrapper_key)()
    wrapper.train, wrapper.val = {}, {}
    wrapper.input_steps, wrapper.output_steps = 72, 36
    wrapper.num_encoder_features = 3
    wrapper.num_decoder_features = 4
    wrapper.target_col_index = 0
    wrapper.persistence_gate = False
    recorder = _SpaceRecorder()
    wrapper.build(recorder)
    K.clear_session()
    # batch_size é fixo no round3 (config), fora do build().
    return list(recorder.entries) + [("batch_size", "32 (fixo)", "int")]


def build_space_table(wrapper_key, title):
    rows = [(n, t, r) for n, r, t in collect_search_space(wrapper_key)]
    html_table, latex = latex_table(
        ["Hiperparâmetro", "Tipo", "Espaço buscado"], rows
    )
    return f"<h3>{title}</h3>" + latex_button(latex) + html_table


def build_wavelet_section(recs, extras, dataset):
    """Seção dedicada: comparação contra o dado real e contra a wavelet treinada."""
    from src.utils import wavelet_denoising

    n = len(dataset)
    train_ratio = SPLIT_RATIOS["train_ratio"]
    val_ratio = SPLIT_RATIOS["val_ratio"]
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    level = current_denoise_level()

    ws100 = dataset["ws100"].to_numpy()
    ws_w = wavelet_denoising(ws100, wavelet="sym18", level=level)[:n]
    raw, den = ws100[val_end:], ws_w[val_end:]
    floor_mae = float(np.mean(np.abs(raw - den)))
    floor_rmse = float(np.sqrt(np.mean((raw - den) ** 2)))

    from src.models.losses import horizon_weights

    # Perda de treinamento (MSE ponderado por horizonte) avaliada no teste:
    # L = (1/H) * soma_h w_h * mse_h, com w_h = 1 + h/H normalizado.
    w_h = horizon_weights(36)

    def weighted_loss(per_horizon_key):
        return [
            float(np.sum(w_h * extras[r["name"]][per_horizon_key]["mse"].to_numpy())
                  / len(w_h))
            for r in recs
        ]

    def transposed_table(regime, roll_regime, loss_vals):
        """Modelos em colunas; métricas (all horizons + h36) e perda em linhas."""
        header = ["Métrica"] + [r["short"] for r in recs]
        rows = []

        def add(label, key, metric):
            rows.append((label, *[fmt(r[key][metric]) for r in recs]))

        for label, metric in (("MAE", "mae"), ("MSE", "mse"),
                              ("RMSE", "rmse"), ("R²", "r2")):
            add(f"{label} — todos os horizontes (1–36)", regime, metric)
            add(f"{label} — rolling autônomo (h36)", roll_regime, metric)
        rows.append(("Perda ponderada por horizonte (teste)",
                     *[fmt(v, 4) for v in loss_vals]))
        html_table, latex = latex_table(header, rows)
        return latex_button(latex) + html_table

    t_deno = transposed_table("all_denoised", "rolling_denoised",
                              weighted_loss("per_horizon_den"))
    t_raw = transposed_table("all", "rolling",
                             weighted_loss("per_horizon"))

    # figura: último trecho do teste — bruto vs wavelet vs previsões de todos
    t1 = dataset.index[-1] - pd.Timedelta(days=7)
    m = dataset.index >= t1
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(dataset.index[m], ws100[m], color="#2ca02c", lw=0.8, alpha=0.22,
            label="ws100 observado (bruto)")
    ax.plot(dataset.index[m], ws_w[m], color="black", lw=1.6,
            label=f"ws100 wavelet sym18 nível {level} (alvo de treino)")
    for r in recs:
        rr = extras[r["name"]]["roll_df"]
        rr = rr[rr["timestamp"] >= t1]
        ax.plot(rr["timestamp"], rr["predicted"], color=r["color"], lw=1.1,
                alpha=0.85, label=f"rolling h36 — {r['short']}")
    ax.set_ylabel("Velocidade (m/s)")
    ax.set_title("Dado bruto × alvo wavelet × previsões rolling (últimos 7 dias do teste)")
    ax.legend(ncol=3, fontsize=7.5)
    ax.grid(alpha=0.3)
    savefig(fig, "16_wavelet_vs_real.png")

    body = (
        "<p>Os modelos são treinados para prever a série <em>suavizada</em> pela "
        f"wavelet sym18 no nível {level}; a avaliação oficial, porém, é contra o "
        "ws100 bruto. A distância entre as duas séries é um <b>piso de suavização</b> "
        "que nenhuma previsão consegue vencer contra o dado bruto: no split de teste, "
        f"MAE = {fmt(floor_mae, 3)} m/s e RMSE = {fmt(floor_rmse, 3)} m/s. As tabelas "
        "abaixo separam os dois alvos: quanto o modelo erra contra a realidade "
        "observada e quanto erra contra o sinal que de fato tentou reconstruir. A "
        "última linha de cada tabela avalia no teste a função de perda de "
        "treinamento — MSE ponderado por horizonte, com peso linear crescente "
        "(~0,7 no passo 1 até ~1,3 no passo 36, média 1) — contra o mesmo alvo da "
        "tabela, em (m/s)².</p>"
        + "<h3>Alvo wavelet (treinado)</h3>" + t_deno
        + "<h3>Dado bruto (ws100)</h3>" + t_raw
    )
    images = [("16_wavelet_vs_real.png",
               "Série bruta, alvo wavelet (nível treinado) e previsões rolling de todos os modelos")]
    return body, images


def build_hp_table(recs):
    canonical = {"lstm": 0, "lstm_bi": 1, "tcn": 0, "tcn_bi": 1}
    recs = sorted(recs, key=lambda r: canonical.get(r["key"], 9))
    families = [
        ("Recorrentes (LSTM / LSTM Bidirecional)",
         [r for r in recs if not r["key"].startswith("tcn")],
         [("lstm_units", "Units do encoder (decoder = 2× no bi)"),
          ("encoder_dropout_rate", "Dropout do encoder"),
          ("decoder_dropout_rate", "Dropout do decoder")]),
        ("Convolucionais (TCN / TCN Bidirecional)",
         [r for r in recs if r["key"].startswith("tcn")],
         [("filters", "Filtros (encoder = decoder)"),
          ("encoder_kernel_size", "Kernel do encoder"),
          ("encoder_nb_stacks", "Stacks do encoder"),
          ("encoder_dilation_rate", "Dilatação do encoder"),
          ("encoder_dropout_rate", "Dropout do encoder"),
          ("decoder_kernel_size", "Kernel do decoder"),
          ("decoder_nb_stacks", "Stacks do decoder"),
          ("decoder_dilation_rate", "Dilatação do decoder"),
          ("decoder_dropout_rate", "Dropout do decoder"),
          ("context_pooling", "Pooling do contexto")]),
    ]

    def cell(value):
        if isinstance(value, float) and 0 < abs(value) < 1e-2:
            return f"{value:.2e}"
        return str(value)

    parts = []
    for title, group, arch_cols in families:
        if not group:
            continue
        rows = [("Loss", *[str(r["hp"].get("loss", r["loss"])) for r in group]),
                ("LR", *[f"{float(r['hp'].get('learning_rate', 0)):.2e}" for r in group]),
                ("Batch", *[str(r["hp"].get("batch_size", 32)) for r in group]),
                ("Schedule", *[str(r["hp"].get("lr_schedule", "constant")) for r in group]),
                ("Weight decay", *[cell(r["hp"].get("weight_decay", 0.0)) for r in group])]
        rows += [(label, *[cell(r["hp"].get(col, "—")) for r in group])
                 for col, label in arch_cols]
        columns = ["Hiperparâmetro"] + [r["short"] for r in group]
        html_table, latex = latex_table(columns, rows)
        parts.append(f"<h3>{title}</h3>" + latex_button(latex) + html_table)
    return "".join(parts)


def build_training_summary_table(recs):
    rows = []
    for r in recs:
        t = r["training"]
        rows.append((
            r["short"],
            f"{t['epochs_run']} / {t['epochs_requested']}",
            fmt(t["best_val_loss"], 5),
            fmt(t["final_val_loss"], 5),
            fmt(t["training_time_sec"], 0),
        ))
    columns = ["Modelo", "Épocas (executadas / orçamento)",
               "melhor val_loss", "val_loss final", "Tempo (s)"]
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def build_best_table(best):
    r = best
    rows = [
        ("MSE ((m/s)²)", fmt(r["all"]["mse"], 4), fmt(r["all_denoised"]["mse"], 4),
         fmt(r["rolling"]["mse"], 4)),
        ("RMSE (m/s)", fmt(r["all"]["rmse"], 4), fmt(r["all_denoised"]["rmse"], 4),
         fmt(r["rolling"]["rmse"], 4)),
        ("R²", fmt(r["all"]["r2"], 4), fmt(r["all_denoised"]["r2"], 4),
         fmt(r["rolling"]["r2"], 4)),
    ]
    columns = ["Métrica", "Todos horizontes (ws100 bruto)",
               "Todos horizontes (alvo denoised)", "Rolling h=36 (ws100 bruto)"]
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def build_worst_table(worst_rows):
    """Tabela das piores trajetorias da LSTM (ordem do grafico 10)."""
    rows = [(w["rank"], w["origem"], fmt(w["mse"], 3), fmt(w["mae"], 3),
             str(w["pior_h"])) for w in worst_rows]
    columns = ["Rank", "Origem", "MSE trajetória ((m/s)²)", "MAE trajetória (m/s)",
               "Horizonte do pior passo"]
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def build_metric_tables(recs, extras, baselines=None):
    """Tabelas compactas (R2/MSE/MAE/RMSE): agregado e horizonte h=36."""
    display = {"lstm": "LSTM", "lstm_bi": "LSTM - Bi",
               "tcn": "TCN", "tcn_bi": "TCN - Bi"}

    def row_cells(m):
        return (fmt(m["r2"], 3), fmt(m["mse"], 3), fmt(m["mae"], 3),
                fmt(m["rmse"], 3))

    # ---- agregado (todos os horizontes) ----
    rows_all = []
    if baselines:
        for key in ("arima", "rf", "persistence"):
            m = baselines["metrics"].get(key)
            if m:
                rows_all.append((BASELINE_STYLE[key][0], m["all_horizons"]))
    for r in recs:
        rows_all.append((display.get(r["key"], r["short"]), r["all"]))
    rows_all.sort(key=lambda x: x[1]["mse"])
    html_all, latex_all = latex_table(
        ["Método", "R²", "MSE ((m/s)²)", "MAE (m/s)", "RMSE (m/s)"],
        [(n, *row_cells(m)) for n, m in rows_all])

    # ---- h = 36 ----
    rows_h36 = []
    if baselines and "per_horizon" in baselines:
        bh = baselines["per_horizon"]
        last = bh[bh["horizon"] == bh["horizon"].max()].iloc[0]
        for key in ("arima", "rf", "persistence"):
            rows_h36.append((BASELINE_STYLE[key][0], {
                "r2": last[f"{key}_r2"], "mse": last[f"{key}_rmse"] ** 2,
                "mae": last[f"{key}_mae"], "rmse": last[f"{key}_rmse"]}))
    for r in recs:
        ph = extras[r["name"]]["per_horizon"]
        hmax = ph[ph["horizon"] == ph["horizon"].max()].iloc[0]
        rows_h36.append((display.get(r["key"], r["short"]), hmax))
    rows_h36.sort(key=lambda x: x[1]["mse"])
    html_h36, latex_h36 = latex_table(
        ["Método", "R²", "MSE ((m/s)²)", "MAE (m/s)", "RMSE (m/s)"],
        [(n, *row_cells(m)) for n, m in rows_h36])

    html = (f'<div class="note">Tabela agregada — todos os horizontes '
            f"(1–36), mesmo recorte dos heatmaps.</div>{html_all}"
            f'<div class="note">No horizonte máximo h = 36 (6 h à frente), '
            f"mesmo recorte da última coluna dos heatmaps.</div>{html_h36}")
    latex = latex_all + "\n\n" + latex_h36
    return latex_button(latex) + html


def build_error_table(error_rows, recs):
    """Tabela MAE/MSE/R2/loss por modelo + baselines (ordem do grafico 02b)."""
    loss_by_short = {r["short"]: r["loss"] for r in recs}
    final_by_short = {r["short"]: r["training"].get("final_loss") for r in recs}
    rows = []
    for name, m, _color, _hatch in error_rows:
        key = name.replace(" (baseline)", "")
        rows.append((
            name,
            fmt(m["mae"], 4), fmt(m["mse"], 4), fmt(m["r2"], 4),
            loss_by_short.get(key, "—"),
            fmt(final_by_short[key], 4) if key in final_by_short
            and final_by_short[key] is not None else "—",
        ))
    columns = ["Método", "MAE (m/s)", "MSE ((m/s)²)", "R²",
               "Função de perda", "Loss final (treino)"]
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def build_html(recs, dataset_info, hz_stats, arch_records, extras, dataset,
               baselines=None, error_rows=None, worst_rows=None, dm=None,
               dm_ab=None, mcs=None):
    best = recs[0]
    best_roll = min(recs, key=lambda r: r["rolling"]["mse"])
    total_train = sum(r["training"]["training_time_sec"] for r in recs)
    fastest = min(recs, key=lambda r: r["timing"]["per_window_ms"])
    lightest = min(recs, key=lambda r: r["params"])

    cards = "".join([
        card(f"{len(recs)}", "arquiteturas comparadas"),
        card(f"{fmt(best['all']['mse'], 3)} (m/s)²", f"melhor MSE — {best['short']}"),
        card(f"{fmt(best['all']['r2'], 3)}", f"melhor R² — {best['short']}"),
        card(f"{fmt(best_roll['rolling']['mse'], 3)} (m/s)²",
             f"melhor rolling 6 h — {best_roll['short']}"),
        card("6 h", "horizonte máximo (36 × 10 min)"),
        card(f"{total_train / 60:.0f} min", "tempo total de treinamento"),
        card(f"{fmt(fastest['timing']['per_window_ms'], 2)} ms",
             f"janela mais rápida — {fastest['short']}"),
    ])

    parts = []

    windows = recs[0]["timing"]["windows"]
    pairs = windows * 36
    test_rows = dataset_info["test_rows"]

    parts.append(section(
        "dados",
        "Dados e protocolo experimental",
        ("Série medida por LiDAR (<code>data/dataset.csv</code>), campanha "
         f"{dataset_info['start']} a {dataset_info['end']}, amostras de 10 min. "
         "Divisão cronológica "
         f"{SPLIT_RATIOS['train_ratio']:.0%}/{SPLIT_RATIOS['val_ratio']:.0%}/"
         f"{1 - SPLIT_RATIOS['train_ratio'] - SPLIT_RATIOS['val_ratio']:.0%} "
         "(treino/validação/teste) — janelas de 72 passos "
         "(12 h) de entrada e 36 passos (6 h) de saída. Todos os modelos são "
         "seq2seq com alvo <code>ws100_wavelet</code> (ws100 denoised por wavelet "
         "sym18, nível 1), features de entrada ws/ws_wavelet/direção sin-cos/ciclicidades "
         "e decoder <b>direto</b> (convenção de inferência: último valor observado + "
         "fração do horizonte — sem teacher forcing e sem vazamento de dados). "
         "O treinamento usa <b>MSE ponderado por horizonte</b>, dando mais peso aos "
         "horizontes longos. As métricas de referência comparam contra o "
         "<b>ws100 bruto</b>."),
        build_split_table(dataset_info) + (
        '<div class="hl">As '
         f"{test_rows} linhas de teste geram <b>{windows} origens</b> de previsão; "
         f"cada origem produz 36 previsões ({fmt_int(pairs)} pares previsto-observado). "
         'A avaliação &quot;todos os horizontes&quot; usa o decoder direto (o mesmo '
         'da inferência: o modelo nunca vê valores futuros reais); a avaliação '
         '&quot;rolling&quot; alimenta o modelo de forma totalmente '
         'autônoma por 6 h (36 passos consecutivos sem observar valores reais).</div>'),
        images=[("01_split_temporal.png",
                 "Divisão temporal treino/validação/teste sobre a série ws100")],
    ))

    results_table = build_results_table(recs)
    parts.append(section(
        "comparacao",
        "Comparação entre arquiteturas",
        ("Métricas no conjunto de teste, todos os horizontes e protocolo rolling. "
         "O <b>MSE é a métrica de foco</b> — o treinamento usa sua versão ponderada "
         "por horizonte, penalizando proporcionalmente erros grandes e dando "
         "prioridade aos horizontes longos, que são os mais difíceis."),
         results_table
         + (build_error_table(error_rows, recs) if error_rows else ""),
         images=[("02_comparacao_mse_rmse.png", "MSE e RMSE por modelo e baselines (todos os horizontes)"),
                 ("02b_erro_geral.png", "Erro geral: MAE, MSE, R² e loss de treino por método"),
                 ("03_mse_r2.png", "MSE e R² por modelo (todos os horizontes e rolling)"),
                 ("04_radar.png", "Perfil normalizado: erro, rolling, tamanho e custo")],
    ))

    if dm:
        n_orig = dm["n_origins"]
        parts.append(section(
            "significancia",
            "Significância estatística — teste de Diebold–Mariano",
            (f"O teste de Diebold–Mariano (1995) verifica se a diferença de "
             f"precisão entre dois modelos pareados é estatisticamente "
             f"distinta de zero: H₀: E[d_t] = 0, com d_t = perda(A) − perda(B) "
             "(perda quadrática, erro contra o ws100 bruto). As "
             f"{fmt_int(n_orig)} origens de teste geram janelas sobrepostas de "
             f"{dm['horizons']} passos, então o estimador de variância incorpora "
             "as autocovariâncias de d até a defasagem h−1 (correção para "
             "previsões h-passos correlacionadas) e a estatística recebe a "
             "correção de pequena amostra de Harvey et al. (1997), avaliada "
             "contra uma t(n−1). <b>DM &gt; 0 significa que o primeiro modelo "
             "da dupla tem perda maior (pior)</b>. A tabela usa o agregado "
             "(média das perdas dos 36 horizontes por origem, h=36); o painel "
              "2×2 traz a matriz DM <b>linha × coluna</b> nos horizontes h6, "
              "h12, h24 e h36 — cada eixo é um método (redes, ARIMA, RF e "
              "Persistência), a célula (i, j) compara o método da linha com o "
              "da coluna naquele horizonte e os asteriscos marcam p &lt; 0,05 "
              "(*) e p &lt; 0,01 (**). São comparações pontuais por horizonte, "
              "sem correção de multiplicidade."),
            build_dm_table(dm),
            images=[("16_significancia_dm.png",
                     "Matriz DM linha × coluna em h6, h12, h24 e h36 — redes, "
                     "ARIMA, RF e Persistência; * p < 0,05, ** p < 0,01"),
                    ("16b_significancia_dm_pvalor.png",
                     "p-valor de cada célula na mesma grade — verde < 0,05 "
                     "(diferença significativa), escala limitada em 0,10")]
                   + ([("16c_significancia_dm_ab_loss.png",
                        "A/B de função de perda: mesma grade com as LSTMs "
                        "trocadas pelas variantes treinadas com MSE simples — "
                        "verde < 0,05")] if dm_ab else []),
        ))

    if mcs:
        parts.append(section(
            "mcs",
            "Model Confidence Set — comparação agregada sobre a predição",
            (f"O <b>Model Confidence Set</b> (Hansen, Lunde &amp; Nason, 2011) "
             f"responde, de uma vez e sem multiplicidade de testes, "
             f"<b>quais modelos são estatisticamente indistinguíveis</b> "
             f"quando a comparação agrega a perda sobre a predição inteira: "
             f"a perda de cada origem é a média do erro quadrático nos "
             f"{dm['horizons']} horizontes (mesmo agregado do DM), e o "
             f"procedimento elimina sucessivamente o pior modelo enquanto o "
             f"p-valor de exclusão &lt; {fmt(mcs['alpha'], 2)}. Os p-valores "
             f"vêm do <i>stationary bootstrap</i> ({fmt_int(mcs['n_boot'])} "
             f"reamostragens, blocos de média {mcs['mean_block']} origens — "
             f"capturam a dependência das janelas sobrepostas de 6 h e as "
             f"caudas pesadas da perda quadrática). <b>Verde = pertence ao "
             f"MCS</b> (empate técnico que sobrevive ao α); vermelho = "
             f"excluído com evidência de perda maior. Modelos no MCS devem "
             f"ser escolhidos por custo (treino/inferência), não por MSE."),
            build_mcs_table(mcs),
            images=[("16d_mcs.png",
                     "p-valor de exclusão do Model Confidence Set por modelo "
                     "(agregado sobre os 36 horizontes; verde = no conjunto)")]
        ))

    parts.append(section(
        "arquitetura",
        "Arquitetura das redes",
        ("Todos os modelos seguem o esqueleto <b>encoder–decoder</b>: o encoder "
         "comprime a janela de 12 h de histórico (72 passos × variáveis) em um "
         "contexto, e o decoder emite os 36 passos futuros de uma vez "
         "(multi-horizon, decoder direto — sem teacher forcing). Nos LSTMs o contexto são "
         "os estados ocultos (com atenção sobre as saídas do encoder); nas TCNs o "
         "encoder e o decoder são redes convolucionais dilatadas com blocos "
         "residuais — na TCN Bidirecional o encoder é bidirecional e o contexto "
         "chega ao decoder pelo último passo do encoder (mapeamento direto, sem "
         "pooling). A saída é uma camada <code>TimeDistributed(Dense)</code> sobre "
         "as features combinadas do decoder. Os diagramas representam a "
         "<b>arquitetura definitiva, sem persistence gate</b>. Formas mostram "
         "<code>(passos, variáveis)</code>; ativações aparecem nas caixas."),
        build_arch_table(arch_records) + (
        '<div class="hl"><b>Última camada — <code>TimeDistributed(Dense(1))</code>:</b> '
        "o decoder entrega, para cada um dos 36 passos futuros, um vetor de features; "
        "uma <b>única camada densa de 1 neurônio</b> (pesos compartilhados) é aplicada "
        "a cada passo de forma independente. Internamente o tensor "
        "<code>(lote, 36, F)</code> é achatado para <code>(lote·36, F)</code>, "
        "projetado para 1 valor por linha e reformatado em <code>(lote, 36, 1)</code> "
        "— o equivalente a 36 camadas densas idênticas, porém com uma só cópia dos "
        "pesos: menos parâmetros e a mesma projeção linear em todos os horizontes.</div>"),
        images=[
            (f"15_arquitetura_{r['key']}.png",
             f"Diagrama de camadas — {r['short']} ({r['layers']} camadas, "
             f"{fmt_int(r['params'])} parâmetros)")
            for r in arch_records
        ],
        fig_grid=True,
    ))

    parts.append(section(
        "horizontes",
        "Desempenho por horizonte (10 min a 6 h)",
        ("O erro cresce monotonicamente com o horizonte: do passo 1 (10 min) ao "
         f"passo 36 (6 h) o MSE dos modelos aumenta "
         f"{fmt(hz_stats['growth_min'], 1)}× a {fmt(hz_stats['growth_max'], 1)}× — "
         "reflexo direto da perda de informação à medida que a previsão se afasta "
         "da última observação. As curvas se mantêm próximas até ~h12 e se "
         "separam nos horizontes longos, exatamente onde a extrapolação autônoma "
         "distingue as arquiteturas. O R² por horizonte (tabela e heatmap abaixo) "
         "completa o quadro: valores positivos indicam ganho de informação frente "
         "à média; R² ≤ 0 significa que a previsão não supera nem a climatologia."),
        build_r2_per_horizon_table(recs, extras, baselines)
        + build_metric_tables(recs, extras, baselines),
        images=[("05_mae_por_horizonte_real.png", "MAE por horizonte — vs dado real, com baselines"),
                ("05_mae_por_horizonte_wavelet.png", "MAE por horizonte — vs alvo wavelet, com baselines"),
                ("06_heatmap_mse_horizonte.png",
                 "Heatmap de MSE ((m/s)²) por modelo e horizonte — verde = menor erro"),
                ("06_heatmap_r2_horizonte.png",
                 "Heatmap de R² por modelo e horizonte — verde = melhor ajuste"),
                ("07_fan_chart.png",
                 "Leques de previsão de 6 h — todos os modelos, 6 origens do teste")],
    ))

    parts.append(section(
        "rolling",
        "Previsão autônoma (rolling, horizonte de 6 h)",
        (f"No protocolo rolling o modelo prevê 36 passos sem nenhuma observação "
         f"intermediária — o cenário real de operação. <b>{best_roll['short']}</b> lidera "
         f"com MSE {fmt(best_roll['rolling']['mse'], 3)} (m/s)². Ao serem alimentadas "
         "pelos próprios erros, as previsões tendem a regredir à média do treino: "
         "quanto maior a tendência de achatamento, maior a degradação entre a "
         "avaliação densa e a autônoma. O painel inferior mostra o erro médio móvel: "
         "períodos acima de zero indicam superestimação persistente."),
        ('<div class="note">A comparação &quot;todos os horizontes&quot; (denso) e '
         '&quot;rolling&quot; (autônomo) mede coisas diferentes: no denso cada passo recebe '
         'o histórico verdadeiro; no rolling, os próprios erros do modelo se acumulam. '
         'Por isso o ranking inverte: LSTM vence no denso, TCN no autônomo.</div>'),
        images=[("08_rolling_forecast_real.png",
                 "Rolling h=36 vs dado real (bruto), com erro médio móvel"),
                ("08_rolling_forecast_wavelet.png",
                 "Rolling h=36 vs alvo wavelet (nível treinado), com erro médio móvel")],
    ))

    if worst_rows:
        parts.append(section(
            "piores_lstm",
            "Piores previsões completas — LSTM (sem sobreposição)",
            ("As 12 trajetórias de 6 h (36 passos) com maior MSE da LSTM no teste. "
             "A seleção é <b>gulosa sem sobreposição</b>: uma origem só entra se "
             "distar no mínimo 6 h (36 passos) de todas as já escolhidas, de modo "
             "que nenhum alvo observado apareça em duas previsões — cada painel é "
             "um evento de falha independente. A linha preta mostra 2 h de história "
             "antes da origem (contexto do encoder) e as 6 h observadas depois; a "
             "curva colorida é a previsão completa da LSTM."),
            build_worst_table(worst_rows),
            images=[("10_piores_predicoes_lstm.png",
                     "12 piores trajetórias completas da LSTM, independentes entre si")],
        ))

    parts.append(section(
        "diagnostico",
        "Diagnóstico dos erros (todos os modelos)",
        ("Quatro visões do comportamento dos modelos: dispersão previsto vs "
         "observado em <b>heatmap</b> (um painel por modelo, mesma escala), "
         "<b>histograma de erros</b> por família (recorrentes à esquerda, "
         "convolucionais à direita), <b>viés por regime de vento</b> com os quatro "
         "modelos e <b>ACF/PACF</b> dos resíduos rolling. O padrão é comum às "
         "quatro arquiteturas: erros quase gaussianos centrados próximos de zero, "
         "superestimação em ventos fracos e subestimação em rajadas — o achatamento "
         "clássico da regressão à média, visível na compressão da faixa prevista "
         "acima da diagonal — e ACF/PACF com estrutura remanescente de longo prazo "
         f"que métodos adicionais (covariáveis, comitês) poderiam explorar. O "
         f"<b>{best['short']}</b>, melhor modelo denso (MSE {fmt(best['all']['mse'], 3)} "
         f"(m/s)², RMSE {fmt(best['all']['rmse'], 3)} m/s, R² {fmt(best['all']['r2'], 3)}), "
         "tem suas métricas detalhadas abaixo."),
        build_best_table(best),
        images=[
            ("09_diagnostico_dispersao.png",
             "Previsto vs observado (heatmap) — um painel por modelo, todos os horizontes"),
            ("09_diagnostico_histograma.png",
             "Distribuição dos erros — recorrentes (LSTM) à esquerda, convolucionais (TCN) à direita"),
            ("09_diagnostico_vies.png",
             "Viés por regime de vento — todos os modelos (erro > 0 = superestima)"),
            ("09_diagnostico_acf_pacf.png",
             "ACF e PACF dos resíduos rolling (h=36) — todos os modelos"),
        ],
        fig_grid=True,
    ))

    epochs_runs = [r["training"]["epochs_run"] for r in recs]
    budget = max((r["training"].get("epochs_requested") or 0) for r in recs) or 150
    has_val = any(r["training"].get("best_val_loss") is not None for r in recs)
    save_pct = [(1 - e / budget) * 100 for e in epochs_runs]
    if has_val:
        training_intro = (
            "Todos os modelos usam o mesmo orçamento de "
            f"{budget} épocas com early stopping "
            "(paciência 12) monitorando <code>val_loss</code>, cuja convenção de decoder "
            "espelha a inferência. O early stopping interrompe entre "
            f"{min(epochs_runs)} e {max(epochs_runs)} épocas — "
            f"{fmt(min(save_pct), 0)}% a {fmt(max(save_pct), 0)}% do orçamento "
            "economizado — e restaura os pesos da melhor época. Os registros abaixo "
            "referem-se ao treino original de cada modelo."
        )
    else:
        training_intro = (
            f"Neste experimento os modelos usam orçamento fixo de {budget} épocas "
            "<b>sem partição de validação</b> (split 75/25 treino/teste): todas as "
            "épocas são executadas, não há early stopping nem restauração de pesos, "
            "e o monitoramento é apenas a perda de treino. Os registros abaixo "
            "referem-se ao treino original de cada modelo."
        )
    parts.append(section(
        "treinamento",
        "Dinâmica de treinamento",
        training_intro,
        build_training_summary_table(recs),
        images=[("11_early_stopping.png",
                 "Épocas executadas vs orçamento; melhor vs final val_loss (treino original)"),
                ("12_tempo_treinamento.png",
                 "Tempo total de treinamento")],
    ))

    parts.append(section(
        "inferencia",
        "Custo computacional e inferência",
        (f"Medições na GPU do ambiente local: entrada em lote com "
         f"{recs[0]['timing']['windows']} janelas de teste por chamada; a latência "
         "single-shot isola uma janela por chamada (a sobrecarga de kernel do "
         "TensorFlow domina nesse regime). O modelo mais leve é o "
         f"<b>{lightest['short']}</b> ({fmt_int(lightest['params'])} parâmetros, "
         f"{fmt(lightest['size_mb'], 1)} MB) e o mais rápido por janela é o "
         f"<b>{fastest['short']}</b> ({fmt(fastest['timing']['per_window_ms'], 2)} ms "
         "por janela de 36 previsões)."),
        "",
        images=[("13_inferencia.png",
                 "Throughput em lote, latência single-shot e tamanho dos modelos"),
                ("14_custo_beneficio.png",
                 "Tempo de treinamento × MSE (área proporcional ao nº de parâmetros)")],
    ))

    parts.append(section(
        "hiperparametros",
        "Hiperparâmetros otimizados por Optuna",
        ("Cada modelo foi configurado pela melhor tentativa de seu estudo Optuna "
         "do round 3 (120 trials por wrapper; espaço compartilhado para TCN em "
         "<code>src/models/tcn_hp.py</code>, com filtros iguais entre encoder e "
         "decoder e sem pós-dropout). LR = taxa de aprendizado; o cronograma de LR "
         "é fixo (<code>constant</code> com ReduceLROnPlateau) e o batch, fixo em 32. "
         "As tabelas de espaço de busca listam os intervalos amostrados pelo "
         "estudo, derivados diretamente do código dos wrappers."),
        build_hp_table(recs)
        + build_space_table("lstm", "Espaço de busca — LSTM (recorrentes)")
        + build_space_table("tcn", "Espaço de busca — TCN (convolucionais)"),
    ))

    best_roll_family = "convolucional" if best_roll["key"].startswith("tcn") else "recorrente"
    concl = [
        f"<li><b>Melhor modelo denso: {best['short']}</b> — MSE "
        f"{fmt(best['all']['mse'], 3)} (m/s)², RMSE {fmt(best['all']['rmse'], 3)} m/s e "
        f"R² {fmt(best['all']['r2'], 3)}, com apenas {fmt_int(best['params'])} "
        f"parâmetros e {fmt(best['training']['training_time_sec'], 0)} s de treino.</li>",
        f"<li><b>Melhor modelo rolling: {best_roll['short']}</b> — MSE "
        f"{fmt(best_roll['rolling']['mse'], 3)} (m/s)² no horizonte autônomo de 6 h; a "
        f"família {best_roll_family} generaliza melhor quando os próprios erros do "
        "modelo se acumulam.</li>",
        "<li><b>O erro cresce com o horizonte</b>: do passo 1 ao passo 36 o MSE dos "
        f"modelos aumenta {fmt(hz_stats['growth_min'], 1)}× a "
        f"{fmt(hz_stats['growth_max'], 1)}×, e as arquiteturas se separam "
        "justamente nos horizontes longos, onde a previsão é mais difícil.</li>",
        f"<li><b>Custo de inferência é desprezível</b>: {fmt(fastest['timing']['per_window_ms'], 2)}–"
        f"{fmt(max(r['timing']['per_window_ms'] for r in recs), 2)} ms por janela em GPU "
        "(36 previsões de 6 h), viável para operação em tempo real.</li>",
    ]
    if has_val:
        concl.insert(3,
            f"<li><b>Early stopping economizou {fmt(min(save_pct), 0)}–"
            f"{fmt(max(save_pct), 0)}% do orçamento</b> de {budget} épocas; a validação "
            "na convenção de inferência permitiu parar cedo sem perder "
            "generalização.</li>")
    else:
        concl.insert(3,
            f"<li><b>Treino sem validação</b>: orçamento fixo de {budget} épocas "
            "(split 75/25) — sem early stopping, todas as épocas contribuem para "
            "os pesos finais.</li>")
    if "Seq2Seq_TCN_Bidirectional" in [r["name"] for r in recs]:
        tcnbi = next(r for r in recs if r["name"] == "Seq2Seq_TCN_Bidirectional")
        drop = tcnbi["hp"].get("encoder_dropout_rate", "—")
        concl.append(
            f"<li><b>{tcnbi['short']} degrada</b> (R² {fmt(tcnbi['all']['r2'], 2)} "
            f"denso): apesar do menor MSE ponderado no estudo Optuna "
            f"(dropout do encoder {drop}), a bidirecionalidade não se transferiu "
            "para a extrapolação autônoma no teste.</li>"
        )
    if dm:
        best_name = best["name"]
        rel = [p for p in dm["pairs"]
               if best_name in (p["name_a"], p["name_b"])]
        wins = [p for p in rel
                if p["aggregate"]["p_value"] < dm["alpha"]
                and ((p["name_a"] == best_name) != (p["aggregate"]["dm"] > 0))]
        if len(wins) == len(rel):
            concl.append(
                f"<li><b>Superioridade estatística (Diebold–Mariano)</b>: "
                f"{best['short']} supera todas as {len(rel)} comparações "
                "pareadas com p &lt; 0,05 no agregado (perda quadrática, "
                "correção de Harvey para janelas sobrepostas).</li>"
            )
        else:
            concl.append(
                f"<li><b>Superioridade estatística (Diebold–Mariano)</b>: "
                f"{best['short']} supera {len(wins)} de {len(rel)} comparações "
                "pareadas com p &lt; 0,05 no agregado — diferenças restantes "
                "não são estatisticamente distinguíveis.</li>"
            )
    wavelet_body, wavelet_imgs = build_wavelet_section(recs, extras, dataset)
    parts.append(section(
        "wavelet",
        "Wavelet × dado real",
        "Erro contra a série observada (bruta) e contra o alvo wavelet no nível treinado.",
        wavelet_body,
        images=wavelet_imgs,
    ))
    parts.append(section(
        "conclusoes",
        "Conclusões",
        "Síntese dos resultados para a seção de resultados do artigo.",
        f"<ul>{''.join(concl)}</ul>",
    ))

    html = (f'<!DOCTYPE html>\n<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>Resultados — Modelos Seq2Seq (LiDAR)</title>\n"
            f"<style>{CSS}</style>\n</head>\n<body>\n<header>\n"
            "<h1>Resultados — Previsão de Vento com Modelos Seq2Seq</h1>\n"
            f"<p>Série LiDAR (dataset.csv) · janela 72→36 passos (12 h → 6 h) · "
            f"teste {dataset_info['start']} a {dataset_info['end']} · Keras/TensorFlow</p>\n"
            "</header>\n<nav>"
            + " ".join(
                f'<a href="#{sid}">{label}</a>'                 for sid, label in [
                    ("dados", "Dados"),
                    ("comparacao", "Comparação"),
                    ("significancia", "Significância (DM)"),
                    ("arquitetura", "Arquiteturas"),
                    ("horizontes", "Horizontes"),
                    ("rolling", "Rolling"),
                    ("diagnostico", "Diagnóstico"),
                    ("treinamento", "Treinamento"),
                    ("inferencia", "Inferência"),
                    ("hiperparametros", "Hiperparâmetros"),
                    ("wavelet", "Wavelet × real"),
                    ("conclusoes", "Conclusões"),
                ])
            + "</nav>\n"
            f'<div class="container">\n<div class="cards">{cards}</div>\n'
            f"{''.join(parts)}\n</div>\n"
            "<footer>Relatório gerado automaticamente por tests/relatorio_resultados.py"
            " · métricas contra ws100 bruto · tempos medidos na GPU local</footer>\n"
            f"<script>{JS}</script>\n</body>\n</html>")
    return html


# ----------------------------------------------------------------------------
# Banco de runs (data/predictions.db)
# ----------------------------------------------------------------------------
def derive_report_key(out_dir) -> str | None:
    """Deriva o report_id do hub a partir da pasta de saída (heurística).

    report_8020 -> lidar/8020; report_homologado -> lidar/homologado.
    Pastas fora do padrão report_* não têm relatório associado.
    """
    name = Path(out_dir).name
    if name.startswith("report_"):
        return f"lidar/{name[len('report_'):].replace('_', '-')}"
    return None


def record_run(results, extras, args, models_dir, dataset, denoise_level):
    """Grava a run avaliada no banco SQLite; retorna o id (None se falhar)."""
    report_key = args.report or derive_report_key(models_dir) \
        or derive_report_key(OUT_DIR)
    category = args.run_category or (report_key.replace("/", " ")
                                     if report_key else "avulsa")
    title = args.run_title or OUT_DIR.name
    try:
        conn = predictions_db.connect()
    except sqlite3.Error as error:
        print(f"aviso: banco indisponível ({error}); run não gravada")
        return None
    try:
        run_id = predictions_db.record_run_from_evaluation(
            conn, results=results, extras=extras, category=category,
            title=title, description=args.run_description,
            report_id=report_key, models_dir=str(models_dir),
            dataset_rows=len(dataset), train_ratio=SPLIT_RATIOS["train_ratio"],
            val_ratio=SPLIT_RATIOS["val_ratio"],
            denoise_level=denoise_level,
            source_file="tests/relatorio_resultados.py",
            source_cmd=[sys.executable, *sys.argv],
            out_file=str(EVAL_CACHE),
            replace=not args.keep_prior)
    except sqlite3.Error as error:
        print(f"aviso: falha ao gravar a run no banco ({error})")
        return None
    finally:
        conn.close()
    print(f"Banco de runs: run #{run_id} em {predictions_db.DB_PATH} "
          f"(report={report_key or '—'})")
    return run_id


def run_from_database(key, dataset):
    """Reconstitui results/extras de uma run gravada no banco (sem GPU).

    O banco guarda origem/timestamp/horizonte/real/predito; as colunas
    derivadas (alvo wavelet, persistência) são recomputadas do dataset —
    determinísticas — e as métricas por horizonte voltam de run_model_stath.
    """
    from src.utils import wavelet_denoising

    conn = predictions_db.connect()
    try:
        run = predictions_db.resolve_run(conn, key)
        if run is None:
            raise SystemExit(
                f"Run não encontrada no banco ({predictions_db.DB_PATH}): {key}")
        results = predictions_db.results_dict(conn, run["id"])
        if not results:
            raise SystemExit(f"Run #{run['id']} não tem modelos gravados.")
        print(f"Banco de runs: run #{run['id']} — {run['title']} "
              f"({run['category']}, {_fmt_ts_short(run['created_at'])})")
        level = run["denoise_level"] or 2
        wave = pd.Series(
            wavelet_denoising(dataset["ws100"].to_numpy(), wavelet="sym18",
                              level=level),
            index=dataset.index)
        extras = {}
        for name, entry in results.items():
            dense = predictions_db.predictions_df(conn, entry["model_id"],
                                                  mode="dense")
            if dense.empty:
                raise SystemExit(
                    f"Run #{run['id']} / {name} não tem predições gravadas; "
                    "reavalie na GPU e regenere o banco.")
            dense["actual_raw"] = dense["actual"]
            dense["actual"] = wave.reindex(dense["timestamp"]).to_numpy()
            dense["persistence"] = wave.reindex(dense["origin"]).to_numpy()
            rolling = predictions_db.predictions_df(conn, entry["model_id"],
                                                    mode="rolling")
            rolling["actual_raw"] = rolling["actual"]
            rolling["actual_denoised"] = wave.reindex(
                rolling["timestamp"]).to_numpy()
            roll_df = rolling[["origin", "timestamp", "actual_denoised",
                               "predicted", "actual_raw"]]
            roll_df = roll_df.sort_values("timestamp").reset_index(drop=True)
            extras[name] = {
                "all_df": dense[["origin", "timestamp", "horizon", "actual",
                                 "predicted", "persistence",
                                 "actual_raw"]].reset_index(drop=True),
                "roll_df": roll_df,
                "per_horizon": pd.DataFrame(entry["per_horizon"]),
                "per_horizon_den": pd.DataFrame(entry["per_horizon_den"]),
            }
        ratios = {"train_ratio": run["train_ratio"] or 0.75,
                  "val_ratio": run["val_ratio"] or 0.0,
                  "denoise_level": run["denoise_level"]}
    finally:
        conn.close()
    return results, extras, ratios


def _fmt_ts_short(value) -> str:
    return str(value).replace("T", " ")[:19]


def main():
    global MODELS_DIR, OUT_DIR, IMG_DIR, EVAL_CACHE, SPLIT_RATIOS
    global DENOISE_LEVEL, TRIM_BOUNDARY
    parser = argparse.ArgumentParser(description="Relatório de resultados (HTML).")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="base dos modelos treinados (default: pipeline/tmp/trained_compare)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="diretório de saída (default: data/results/report)")
    parser.add_argument("--no-baselines", action="store_true",
                        help="nao incluir ARIMA/RF/persistencia (foram calculados no "
                             "teste antigo de 5%%; inconsistentes com outros splits)")
    parser.add_argument("--loss-ab-dir", type=Path, default=None,
                        help="diretorio com variantes de loss (mesmo protocolo) para "
                             "o painel A/B do DM — ex.: pipeline/tmp/trained_8020_mseloss")
    parser.add_argument("--from-db", metavar="RUN", default=None,
                        help="recarrega metricas/predicoes de uma run do banco "
                             "(id ou report_id, ex.: lidar/8020) em vez de "
                             "reavaliar os pesos na GPU")
    parser.add_argument("--no-db", action="store_true",
                        help="nao gravar a run avaliada no banco SQLite")
    parser.add_argument("--trim-boundary-origins", type=int, default=0,
                        help="descarta as primeiras N origens do teste "
                             "(cuja janela de entrada cruza a fronteira "
                             "treino/teste) de TODOS os metodos; default 0")
    parser.add_argument("--keep-prior", action="store_true",
                        help="nao substituir a run anterior do mesmo report_id")
    parser.add_argument("--report", default=None,
                        help="report_id no hub (ex.: lidar/8020); "
                             "default: derivado de --out-dir/--models-dir")
    parser.add_argument("--run-category", default=None,
                        help="categoria da run no banco (ex.: 'lidar 8020'); "
                             "default: derivada do report_id, senao 'avulsa'")
    parser.add_argument("--run-title", default=None,
                        help="titulo da run no banco (default: pasta de saida)")
    parser.add_argument("--run-description", default=None,
                        help="descricao livre da run no banco")
    args = parser.parse_args()

    TRIM_BOUNDARY = int(args.trim_boundary_origins)
    if TRIM_BOUNDARY > 0:
        print(f"Protocolo: descartando as primeiras {TRIM_BOUNDARY} origens "
              "do teste (janela de entrada cruza a fronteira treino/teste) "
              "para todos os metodos.")

    if args.models_dir:
        MODELS_DIR = PROJECT_ROOT / args.models_dir
    if args.out_dir:
        OUT_DIR = PROJECT_ROOT / args.out_dir
        IMG_DIR = OUT_DIR / "images"
        EVAL_CACHE = OUT_DIR / "evaluation.json"

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # runs com conjunto de atributos estendido declaram "feature_keep" no
    # model.json; o dataset precisa das mesmas colunas na avaliação
    feature_keep = {"ws40", "ws100", "v40"}
    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
    for mj in sorted(Path(models_dir).glob("*/model.json")):
        try:
            feature_keep |= set(json.load(
                open(mj, encoding="utf-8")).get("feature_keep", ()))
        except (OSError, json.JSONDecodeError):
            continue
    dataset = common.load_dataset(DATA_FILE, keep_raw=tuple(sorted(feature_keep)))
    print(f"Dataset: {len(dataset)} linhas | {dataset.index.min()} a {dataset.index.max()}")

    denoise_level = 2
    if args.from_db:
        results, extras, ratios = run_from_database(args.from_db, dataset)
        SPLIT_RATIOS = ratios
        denoise_level = int(ratios.get("denoise_level") or 2)
        DENOISE_LEVEL = denoise_level
    else:
        denoise_meta = load_metadata()[0]["meta"]
        denoise_level = int(denoise_meta.get("denoise_level", 2))
        results, extras = evaluate_models(dataset)

    if not args.no_db and not args.from_db:
        record_run(results, extras, args, models_dir, dataset, denoise_level)

    if EVAL_CACHE:
        common.write_json(EVAL_CACHE, {k: {kk: vv for kk, vv in v.items()
                                           if kk not in ("hp", "training")}
                                       for k, v in results.items()})

    recs = sorted(results.values(), key=lambda r: r["all"]["mse"])
    baselines = None if args.no_baselines else load_baselines()
    if baselines:
        print("Baselines estatísticos carregados: ARIMA, Random Forest, Persistência")

    base_preds = load_baseline_predictions()
    if TRIM_BOUNDARY > 0 and base_preds is not None:
        base_preds = trim_boundary_origins(base_preds, TRIM_BOUNDARY,
                                           "baselines")
        if baselines:
            baselines = recompute_baseline_metrics(base_preds,
                                                   baselines["metrics"])
    if baselines:
        common.write_json(OUT_DIR / "baselines.json", baselines["metrics"])

    plot_split_timeline(dataset, recs)
    plot_comparison_bars(recs, baselines)
    plot_error_summary(recs, extras, baselines)
    plot_skill(recs, baselines)
    plot_radar(recs, baselines)
    plot_per_horizon(recs, extras, baselines)
    plot_horizon_heatmap(recs, extras, baselines)
    plot_horizon_r2_heatmap(recs, extras, baselines)
    plot_fan(recs, extras, dataset, base_preds)
    plot_rolling(recs, extras, base_preds)
    plot_worst_predictions(recs, extras, dataset)
    plot_diagnostics(recs, extras, base_preds)
    plot_early_stopping(recs)
    plot_training_time(recs)
    plot_inference(recs)
    plot_tradeoff(recs)

    dm = compute_dm_analysis(recs, extras)
    dmx = compute_dm_horizon_matrix(recs, extras, base_preds)
    plot_dm_matrix(dmx)
    plot_dm_pvalue_matrix(dmx)
    bs = compute_bootstrap_ci_matrix(recs, extras, base_preds)
    plot_bootstrap_ci_matrix(bs)

    mcs = compute_mcs(recs, extras, base_preds)
    plot_mcs(mcs)
    common.write_json(OUT_DIR / "mcs.json", mcs)
    print(f"MCS (α={mcs['alpha']:.2f}): {', '.join(mcs['mcs'])} "
          f"({len(mcs['exclusion_p'])} candidatos, {mcs['n_boot']} bootstrap)")

    dm_ab = None
    if args.loss_ab_dir:
        loss_ab = evaluate_loss_ab_variants(
            dataset, PROJECT_ROOT / args.loss_ab_dir,
            keys=tuple(r["key"] for r in recs if r["key"].startswith("lstm")))
        if loss_ab:
            swap = {k: (" (MSE)", df) for k, df in loss_ab.items()}
            dm_ab = compute_dm_horizon_matrix(recs, extras, base_preds,
                                              swap=swap)
            plot_dm_pvalue_matrix(
                dm_ab, fname="16c_significancia_dm_ab_loss.png",
                suptitle=("A/B de loss: p-valores do DM com as LSTMs "
                          "trocadas pelas variantes com MSE simples "
                          "(verde < 0,05)"))
            print(f"A/B loss: {len(loss_ab)} variantes avaliadas "
                  f"({', '.join(loss_ab)})")
    n_sig = sum(p["aggregate"]["p_value"] < dm["alpha"] for p in dm["pairs"])
    print(f"Diebold–Mariano: {len(dm['pairs'])} pares, "
          f"{n_sig} significativos (p<0,05) no agregado")

    n_rows = len(dataset)
    train_ratio = SPLIT_RATIOS["train_ratio"]
    val_ratio = SPLIT_RATIOS["val_ratio"]
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    splits = [("Treino", 0, train_end)]
    if val_end > train_end:
        splits.append(("Validação", train_end, val_end))
    splits.append(("Teste", val_end, n_rows))
    print("Splits: " + ", ".join(
        f"{name} {a}-{b} ({(b - a) / n_rows * 100:.1f}%)"
        for name, a, b in splits))

    print(f"Figuras salvas em: {IMG_DIR}")
    print(f"Métricas salvas em: {OUT_DIR / 'evaluation.json'}")
    print("Relatório curado: reports/lidar/ (rebuild do hub para atualizar; "
          "a página lê do banco de runs antes do evaluation.json)")


if __name__ == "__main__":
    main()
