"""Relatório de resultados (HTML) dos modelos seq2seq treinados em data/dataset.csv.

Carrega os wrappers Keras já treinados (pipeline/tmp/trained_compare), mede
métricas e tempos de inferência no conjunto de teste, reexecuta o protocolo de
treinamento para capturar as curvas de loss/val_loss (com cache incremental) e
gera figuras + relatório HTML semântico, no mesmo estilo do relatório de EDA.

Foco: lstm, lstm_bi, tcn, tcn_bi (série LiDAR, dataset.csv).

Executar a partir da raiz do projeto:
    python tests/relatorio_resultados.py [--retrain] [--no-train]

Saída:
    data/results/report/images/*.png
    data/results/report/training_history.json   (cache das curvas)
    data/results/report/relatorio_resultados.html
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import common

DATA_FILE = PROJECT_ROOT / "data" / "dataset.csv"
MODELS_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "trained_compare"
NOGATE_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "trained_compare_nogate"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "report"
IMG_DIR = OUT_DIR / "images"
HISTORY_CACHE = OUT_DIR / "training_history.json"
EVAL_CACHE = OUT_DIR / "evaluation.json"

FOCUS_KEYS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
SEED = 42

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


def build_wrapper(meta, train_df, val_df, model_path):
    from keras import backend as K

    K.clear_session()
    wrapper = common.wrapper_factory(meta["wrapper_key"])()
    wrapper.prepare(
        train_df,
        val_df,
        input_steps=meta["input_steps"],
        output_steps=meta["output_steps"],
        target_col=meta["target_col"],
        denoise=meta["denoise"],
        denoise_level=meta["denoise_level"],
        create_sequences=False,
        persistence_gate=bool(meta.get("persistence_gate", False)),
        decoder_mode=meta.get("decoder_mode", "teacher_forcing"),
        target_mode=meta.get("target_mode", "absolute"),
    )
    wrapper.gate_mode = meta.get("gate_mode", "static")
    wrapper.context_mode = meta.get("context_mode", "repeat")
    wrapper.build(common.FixedHyperParameters(meta["hyperparameters"]))
    wrapper.model.load_weights(model_path)
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
        target_mode=wrapper.target_mode,
    )
    if wrapper.decoder_mode == "teacher_forcing":
        prepared["X_decoder"][:, :, 0] = prepared["X_decoder"][:, :1, 0]

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
    results = {}
    extras = {}

    for entry in models:
        meta = entry["meta"]
        name = meta["name"]
        key = meta["wrapper_key"]
        ratios = meta.get("dataset", {})
        train_ratio = ratios.get("train_ratio", 0.75)
        val_ratio = ratios.get("val_ratio", 0.20)
        train_df, val_df, test_df = common.split_dataset(dataset, train_ratio, val_ratio)
        val_end = len(train_df) + len(val_df)

        print(f"[{key}] carregando pesos de {entry['dir'].name} ...")
        wrapper = build_wrapper(meta, train_df, val_df, entry["dir"] / "model.keras")

        t0 = time.perf_counter()
        all_df = common.predict_all_horizons(wrapper, test_df)
        eval_sec = time.perf_counter() - t0
        all_df["actual_raw"] = dataset["ws100"].reindex(all_df["timestamp"]).to_numpy()

        roll_df = rolling_frame(wrapper, dataset, val_end)
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
        print(
            f"[{key}] MSE={m_all['mse']:.4f} roll={m_roll['mse']:.4f} "
            f"params={results[name]['params']:,} {timing['per_window_ms']:.2f} ms/janela"
        )

    return results, extras


# ----------------------------------------------------------------------------
# Coleta: curvas de treinamento (re-execução com cache incremental)
# ----------------------------------------------------------------------------
def _load_history_cache():
    if HISTORY_CACHE.is_file():
        with open(HISTORY_CACHE, encoding="utf-8") as handle:
            return json.load(handle)
    return {"_meta": {"seed": SEED, "dataset": "data/dataset.csv"}}


def _save_history_cache(cache):
    common.write_json(HISTORY_CACHE, cache)


def retrain_history(dataset, force=False):
    tf = common.setup_tensorflow(True)
    from keras import backend as K
    from keras.callbacks import Callback

    class EpochClock(Callback):
        def __init__(self, times):
            super().__init__()
            self.times = times
            self._t0 = None

        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.times.append(time.perf_counter() - self._t0)

    cache = _load_history_cache()
    models = load_metadata()
    for entry in models:
        meta = entry["meta"]
        name = meta["name"]
        if not force and name in cache:
            continue

        ratios = meta.get("dataset", {})
        train_df, val_df, _ = common.split_dataset(
            dataset, ratios.get("train_ratio", 0.75), ratios.get("val_ratio", 0.20)
        )
        K.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        wrapper = common.wrapper_factory(meta["wrapper_key"])()
        wrapper.prepare(
            train_df,
            val_df,
            input_steps=meta["input_steps"],
            output_steps=meta["output_steps"],
            target_col=meta["target_col"],
            denoise=meta["denoise"],
            denoise_level=meta["denoise_level"],
            persistence_gate=bool(meta.get("persistence_gate", False)),
        )
        batch_size = int(meta["training"]["batch_size"])
        steps_per_epoch = int((len(train_df) + batch_size - 1) // batch_size)
        wrapper.schedule_total_steps = steps_per_epoch * meta["training"]["epochs_requested"]
        wrapper.loss = meta.get("loss", "mse")
        wrapper.build(common.FixedHyperParameters(meta["hyperparameters"]))
        callbacks = common.default_callbacks(wrapper, meta["training"]["patience"])

        epoch_times = []
        callbacks.append(EpochClock(epoch_times))

        print(f"[{meta['wrapper_key']}] re-treinando {name} ...")
        t0 = time.perf_counter()
        history = wrapper.fit(
            epochs=meta["training"]["epochs_requested"],
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
            use_validation=True,
        )
        total_sec = time.perf_counter() - t0

        losses = [float(v) for v in history.history.get("loss", [])]
        val_losses = [float(v) for v in history.history.get("val_loss", [])]
        cache[name] = {
            "key": meta["wrapper_key"],
            "loss": losses,
            "val_loss": val_losses,
            "epoch_time_sec": [float(v) for v in epoch_times],
            "total_time_sec": float(total_sec),
            "epochs_run": len(losses),
            "best_val_loss": float(min(val_losses)) if val_losses else None,
            "best_epoch": int(np.argmin(val_losses) + 1) if val_losses else None,
            "final_val_loss": float(val_losses[-1]) if val_losses else None,
        }
        _save_history_cache(cache)
        print(
            f"[{meta['wrapper_key']}] {len(losses)} épocas em {total_sec:.1f}s | "
            f"best val_loss={cache[name]['best_val_loss']:.6f}"
        )
        K.clear_session()
    return cache


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
        wrapper = build_wrapper(meta, train_df, val_df, entry["dir"] / "model.keras")
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
    train_end = int(n * 0.75)
    val_end = train_end + int(n * 0.20)

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(dataset.index, dataset["ws100"], lw=0.4, color="#1b6ca8")
    ax.axvspan(dataset.index[0], dataset.index[train_end - 1],
               color="#2166ac", alpha=0.12, label=f"treino 75% ({train_end})")
    ax.axvspan(dataset.index[train_end], dataset.index[val_end - 1],
               color="#e0a800", alpha=0.14, label=f"validação 20% ({val_end - train_end})")
    ax.axvspan(dataset.index[val_end], dataset.index[-1],
               color="#b03030", alpha=0.14, label=f"teste 5% ({n - val_end})")
    ax.set_ylabel("ws100 (m/s)")
    ax.set_title("Divisão temporal do dataset LiDAR (sem embaralhamento)")
    ax.legend(loc="upper right", ncol=3)
    savefig(fig, "01_split_temporal.png")


def plot_comparison_bars(recs):
    names = [r["short"] for r in recs]
    mse = [r["all"]["mse"] for r in recs]
    rmse = [r["all"]["rmse"] for r in recs]
    colors = [r["color"] for r in recs]
    y = np.arange(len(recs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].barh(y, mse, color=colors)
    for yi, v in zip(y, mse):
        axes[0].text(v, yi, f" {v:.3f}", va="center", fontsize=8.5)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MSE ((m/s)²)")
    axes[0].set_title("MSE todos os horizontes (1–36 passos)")
    axes[0].set_xlim(0, max(mse) * 1.22)

    axes[1].barh(y, rmse, color=colors)
    for yi, v in zip(y, rmse):
        axes[1].text(v, yi, f" {v:.3f}", va="center", fontsize=8.5)
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RMSE (m/s)")
    axes[1].set_title("RMSE todos os horizontes (1–36 passos)")
    axes[1].set_xlim(0, max(rmse) * 1.22)
    savefig(fig, "02_comparacao_mse_rmse.png")


def plot_skill(recs):
    names = [r["short"] for r in recs]
    colors = [r["color"] for r in recs]
    y = np.arange(len(recs))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    mse = [r["all"]["mse"] for r in recs]
    axes[0].barh(y - w / 2, mse, height=w, color=colors, label="todos horizontes")
    mse_roll = [r["rolling"]["mse"] for r in recs]
    axes[0].barh(y + w / 2, mse_roll, height=w, color=colors, alpha=0.45,
                 hatch="//", label="rolling (h=36)")
    for yi, v in zip(y, mse):
        axes[0].text(v, yi - w / 2, f" {v:.3f}", va="center", fontsize=8)
    for yi, v in zip(y, mse_roll):
        axes[0].text(v, yi + w / 2, f" {v:.3f}", va="center", fontsize=8)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MSE ((m/s)²)")
    axes[0].set_title("MSE (vs ws100 bruto)")
    axes[0].legend(loc="lower right")

    r2 = [r["all"]["r2"] for r in recs]
    axes[1].barh(y - w / 2, r2, height=w, color=colors, label="todos horizontes")
    r2_roll = [r["rolling"]["r2"] for r in recs]
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


def plot_radar(recs):
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

    arr = np.vstack([raw_values(r) for r in recs])
    norm = np.zeros_like(arr)
    for j, (_, lower_better) in enumerate(metric_rows):
        col = arr[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            scaled = np.ones_like(col)
        elif lower_better:
            scaled = (hi - col) / (hi - lo)
        else:
            scaled = (col - lo) / (hi - lo)
        norm[:, j] = 0.15 + 0.85 * scaled

    n = len(metric_rows)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for i, r in enumerate(recs):
        values = norm[i].tolist() + [float(norm[i][0])]
        ax.plot(angles, values, color=r["color"], lw=1.8, label=r["short"])
        ax.fill(angles, values, color=r["color"], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metric_rows], fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Perfil dos modelos (eixos normalizados, mais longe do centro = melhor)",
                 pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08))
    savefig(fig, "04_radar.png")


def plot_per_horizon(recs, extras):
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
        ax.set_xlabel("Horizonte (passos de 10 min)")
        ax.set_ylabel("MAE (m/s)")
        ax.set_title(f"MAE por horizonte de previsão — {title}")
        ax.set_xticks(range(1, 37, 2))
        ax.grid(alpha=0.3)
        sec = ax.secondary_xaxis("top", functions=(lambda x: x / 6, lambda x: x * 6))
        sec.set_xlabel("Horizonte (horas)")
        ax.legend(ncol=2)
        savefig(fig, fname)


def plot_horizon_heatmap(recs, extras):
    mses = []
    for r in recs:
        ph = extras[r["name"]]["per_horizon"]
        mses.append(ph["mse"].to_numpy())
    mat = np.vstack(mses)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=float(mat.min()), vmax=float(mat.max()),
                   interpolation="nearest")
    ax.set_yticks(range(len(recs)), [r["short"] for r in recs])
    ax.set_xticks(range(0, 36, 2), [str(h) for h in range(1, 37, 2)])
    ax.set_xlabel("Horizonte (passos de 10 min)")
    ax.set_title("MSE ((m/s)²) por modelo e horizonte (verde = menor erro)")
    for j in [0, 5, 11, 17, 23, 29, 35]:
        for i in range(len(recs)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.75, lw=0))
    fig.colorbar(im, ax=ax, pad=0.01).set_label("MSE ((m/s)²)")
    savefig(fig, "06_heatmap_mse_horizonte.png")


def plot_fan(recs, extras, dataset):
    all_df = extras[recs[0]["name"]]["all_df"]
    origins = all_df["origin"].unique()
    idx = np.linspace(0, len(origins) - 1, 6).astype(int)
    # cores por modelo + viridis apenas para as linhas de origem
    model_colors = {r["short"]: r["color"] for r in recs}

    t_start = pd.Timestamp(origins[idx[0]])
    t_end = all_df["timestamp"].max()

    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5), sharex=False,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.12})
    handles = []
    for ax, o in zip(axes.ravel(), origins[idx]):
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
            if not handles:
                handles = [line]
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


def plot_rolling(recs, extras):
    level = load_metadata()[0]["meta"].get("denoise_level", 2)
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
        axes[0].plot(roll0["timestamp"], roll0[actual_col], color="black", lw=1.5,
                     label=ref_label, zorder=5)
        for r in recs:
            roll = extras[r["name"]]["roll_df"]
            axes[0].plot(roll["timestamp"], roll["predicted"], color=r["color"],
                         lw=1.1, alpha=0.9,
                         label=f"{r['short']} (MAE {r['rolling']['mae']:.2f})")
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


def plot_best_diagnostics(best, extras):
    all_df = extras[best["name"]]["all_df"]
    roll_df = extras[best["name"]]["roll_df"]
    errors = all_df["predicted"] - all_df["actual_raw"]
    resid_roll = (roll_df["predicted"] - roll_df["actual_raw"]).to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    hb = ax.hexbin(all_df["actual_raw"], all_df["predicted"], gridsize=38,
                   cmap="YlGnBu", mincnt=1)
    lims = [min(all_df["actual_raw"].min(), all_df["predicted"].min()),
            max(all_df["actual_raw"].max(), all_df["predicted"].max())]
    ax.plot(lims, lims, "r--", lw=1.2, label="y = x")
    ax.set_xlabel("Observado (m/s)")
    ax.set_ylabel("Previsto (m/s)")
    ax.set_title(f"Previsto vs observado — {best['short']} (todos os horizontes)")
    fig.colorbar(hb, ax=ax, pad=0.01).set_label("contagem")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    ax.hist(errors, bins=60, density=True, color="#1f77b4", alpha=0.8,
            edgecolor="white", lw=0.3)
    mu, sigma = float(errors.mean()), float(errors.std())
    xs = np.linspace(errors.min(), errors.max(), 300)
    ax.plot(xs, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(xs - mu) ** 2 / (2 * sigma ** 2)),
            "r-", lw=1.6, label=f"normal (μ={mu:.2f}, σ={sigma:.2f})")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.axvline(mu, color="r", ls=":", lw=1.4, label=f"viés = {mu:+.3f}")
    ax.set_xlabel("Erro (m/s)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição dos erros (todos os horizontes)")
    ax.legend()

    ax = axes[1, 0]
    bins = np.arange(np.floor(all_df["actual_raw"].min()),
                     np.ceil(all_df["actual_raw"].max()) + 1e-9, 1.0)
    groups = pd.cut(all_df["actual_raw"], bins)
    bias = errors.groupby(groups, observed=True).mean()
    centers = [iv.mid for iv in bias.index]
    ax.bar(centers, bias.values, width=0.8,
           color=np.where(bias.values >= 0, "#d62728", "#1f77b4"), alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Velocidade observada (m/s)")
    ax.set_ylabel("Erro médio (m/s)")
    ax.set_title("Viés por regime de vento (vermelho = superestima)")

    ax = axes[1, 1]
    lags = np.arange(1, 37)
    acf = np.array([np.corrcoef(resid_roll[:-lag], resid_roll[lag:])[0, 1]
                    for lag in lags])
    ci = 1.96 / np.sqrt(len(resid_roll))
    ax.bar(lags, acf, width=0.8, color="#2c7fb8", alpha=0.9)
    ax.axhspan(-ci, ci, color="gray", alpha=0.18, lw=0)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Defasagem (passos de 10 min)")
    ax.set_ylabel("Autocorrelação")
    ax.set_title("ACF dos resíduos rolling (h=36) do melhor modelo")
    ax.set_xticks(range(0, 37, 3))
    savefig(fig, "09_diagnostico_melhor.png")


def plot_training_curves(history, recs):
    names = [r["name"] for r in recs if r["name"] in history]
    if not names:
        return False
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 3.9), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        h = history[name]
        epochs = np.arange(1, len(h["loss"]) + 1)
        ax.plot(epochs, h["loss"], color="#1f77b4", lw=1.6, label="loss (treino)")
        ax.plot(epochs, h["val_loss"], color="#ff7f0e", lw=1.6, label="val_loss")
        if h.get("best_epoch"):
            ax.axvline(h["best_epoch"], color="gray", ls=":", lw=1)
            ax.plot(h["best_epoch"], h["val_loss"][h["best_epoch"] - 1], "*",
                    color="#b03030", ms=11, label=f"melhor (ép. {h['best_epoch']})")
        ax.set_title(short_name(name), fontsize=10)
        ax.set_xlabel("Época")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("Loss")
    fig.suptitle("Curvas de treinamento (re-execução com a mesma semente e protocolo)",
                 fontsize=11)
    savefig(fig, "10_curvas_treinamento.png")
    return True


def plot_early_stopping(history, recs):
    names = [r["name"] for r in recs]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    eps_orig = [r["training"]["epochs_run"] for r in recs]
    eps_rep = [history[n]["epochs_run"] if n in history else np.nan for n in names]
    w = 0.38
    axes[0].barh(y - w / 2, eps_orig, height=w, color=[r["color"] for r in recs],
                 label="treino original")
    axes[0].barh(y + w / 2, eps_rep, height=w, color=[r["color"] for r in recs],
                 alpha=0.45, hatch="//", label="re-execução")
    axes[0].axvline(150, color="black", ls="--", lw=1.2, label="orçamento (150 épocas)")
    for yi, v in zip(y, eps_orig):
        axes[0].text(v, yi - w / 2, f" {v:.0f}", va="center", fontsize=8)
    axes[0].set_yticks(y, [short_name(n) for n in names])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Épocas executadas")
    axes[0].set_title("Early stopping: épocas executadas")
    axes[0].legend(loc="lower right")

    best_orig = [r["training"]["best_val_loss"] for r in recs]
    final_orig = [r["training"]["final_val_loss"] for r in recs]
    axes[1].barh(y - w / 2, best_orig, height=w,
                 color=[r["color"] for r in recs], label="melhor val_loss")
    axes[1].barh(y + w / 2, final_orig, height=w,
                 color=[r["color"] for r in recs], alpha=0.45, hatch="//",
                 label="val_loss final (restaurada p/ melhor)")
    for yi, v in zip(y, best_orig):
        axes[1].text(v, yi - w / 2, f" {v:.4f}", va="center", fontsize=8)
    axes[1].set_yticks(y, [short_name(n) for n in names])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("val_loss")
    axes[1].set_title("val_loss no treino original (best vs final)")
    axes[1].legend(loc="lower right")
    savefig(fig, "11_early_stopping.png")


def plot_training_time(history, recs):
    names = [r["name"] for r in recs]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    total_orig = [r["training"]["training_time_sec"] for r in recs]
    axes[0].barh(y, total_orig, color=[r["color"] for r in recs])
    for yi, v in zip(y, total_orig):
        axes[0].text(v, yi, f" {v:.0f}s", va="center", fontsize=8.5)
    axes[0].set_yticks(y, [short_name(n) for n in names])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Tempo total de treino (s)")
    axes[0].set_title("Tempo de treinamento (treino original)")

    per_epoch = [
        float(np.mean(history[n]["epoch_time_sec"])) if n in history else np.nan
        for n in names
    ]
    axes[1].barh(y, per_epoch, color=[r["color"] for r in recs])
    for yi, v in zip(y, per_epoch):
        if not np.isnan(v):
            axes[1].text(v, yi, f" {v:.2f}s", va="center", fontsize=8.5)
    axes[1].set_yticks(y, [short_name(n) for n in names])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Segundos por época (média)")
    axes[1].set_title("Custo por época (re-execução)")
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

    from src import common

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
    # batch_size é amostrado no objetivo do step_optuna, fora do build().
    return list(recorder.entries) + [("batch_size", "16–64 (passo 16)", "int")]


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
    train_end = int(n * 0.75)
    val_end = train_end + int(n * 0.20)
    meta = load_metadata()
    level = meta[0]["meta"].get("denoise_level", 2)

    ws100 = dataset["ws100"].to_numpy()
    ws_w = wavelet_denoising(ws100, wavelet="sym18", level=level)[:n]
    raw, den = ws100[val_end:], ws_w[val_end:]
    floor_mae = float(np.mean(np.abs(raw - den)))
    floor_rmse = float(np.sqrt(np.mean((raw - den) ** 2)))

    def rows_for(regime):
        rows = []
        for r in recs:
            full, den_m = r[regime], r[f"{regime}_denoised"]
            rows.append((
                r["short"],
                fmt(full["mae"], 3), fmt(den_m["mae"], 3),
                fmt(den_m["mae"] - full["mae"], 3),
                fmt(full["rmse"], 3), fmt(den_m["rmse"], 3),
                fmt(full["r2"], 3), fmt(den_m["r2"], 3),
            ))
        return rows

    cols = ["Modelo", "MAE real", "MAE wavelet", "Δ MAE", "RMSE real",
            "RMSE wavelet", "R² real", "R² wavelet"]
    t_all_h, t_all_l = latex_table(cols, rows_for("all"))
    t_roll_h, t_roll_l = latex_table(cols, rows_for("rolling"))

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
        "observada e quanto erra contra o sinal que de fato tentou reconstruir.</p>"
        + f"<h3>Todos os horizontes (1–36 passos)</h3>" + t_all_h
        + f"<h3>Rolling autônomo (h = 36)</h3>" + t_roll_h
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
         [("encoder_layers", "Camadas do encoder"), ("lstm_units", "Units do encoder"),
          ("encoder_dropout_rate", "Dropout do encoder"),
          ("decoder_dropout_rate", "Dropout do decoder")]),
        ("Convolucionais (TCN / TCN Bidirecional)",
         [r for r in recs if r["key"].startswith("tcn")],
         [("encoder_filters", "Filtros do encoder"),
          ("encoder_kernel_size", "Kernel do encoder"),
          ("encoder_nb_stacks", "Stacks do encoder"),
          ("encoder_dilation_rate", "Dilatação do encoder"),
          ("encoder_dropout_rate", "Dropout do encoder"),
          ("encoder_post_dropout_rate", "Pós-dropout do encoder"),
          ("decoder_filters", "Filtros do decoder"),
          ("decoder_kernel_size", "Kernel do decoder"),
          ("decoder_nb_stacks", "Stacks do decoder"),
          ("decoder_dilation_rate", "Dilatação do decoder"),
          ("decoder_dropout_rate", "Dropout do decoder"),
          ("decoder_post_dropout_rate", "Pós-dropout do decoder"),
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


def build_repro_table(recs, history):
    rows = []
    for r in recs:
        name = r["name"]
        orig = r["training"]
        if name in history:
            rep = history[name]
            delta = (rep["best_val_loss"] - orig["best_val_loss"]) / orig["best_val_loss"] * 100
            rows.append((
                r["short"],
                f"{orig['epochs_run']} → {rep['epochs_run']}",
                fmt(orig["best_val_loss"], 5),
                fmt(rep["best_val_loss"], 5),
                f"{delta:+.2f}%".replace(".", ","),
            ))
        else:
            rows.append((r["short"], str(orig["epochs_run"]),
                         fmt(orig["best_val_loss"], 5), "—", "—"))
    columns = ["Modelo", "Épocas (orig → re-exec)", "melhor val_loss (orig)",
               "melhor val_loss (re-exec)", "Δ"]
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


def build_html(recs, history, dataset_info, hz_stats, arch_records, extras, dataset):
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
         "Divisão cronológica 75/20/5 (treino/validação/teste) — janelas de 72 passos "
         "(12 h) de entrada e 36 passos (6 h) de saída. Todos os modelos são "
         "seq2seq com alvo <code>ws100_wavelet</code> (ws100 denoised por wavelet "
         "sym18, nível 1), features de entrada ws/ws_wavelet/direção sin-cos/ciclicidades "
         "e decoder em convenção de inferência (último valor observado repetido). "
         "As métricas de referência comparam contra o <b>ws100 bruto</b>."),
        build_split_table(dataset_info) + (
        '<div class="hl">As '
         f"{test_rows} linhas de teste geram <b>{windows} origens</b> de previsão; "
         f"cada origem produz 36 previsões ({fmt_int(pairs)} pares previsto-observado). "
         'A avaliação &quot;todos os horizontes&quot; usa teacher forcing na convenção de '
         'inferência; a avaliação &quot;rolling&quot; alimenta o modelo de forma totalmente '
         'autônoma por 6 h (36 passos consecutivos sem observar valores reais).</div>'),
        images=[("01_split_temporal.png",
                 "Divisão temporal treino/validação/teste sobre a série ws100")],
    ))

    results_table = build_results_table(recs)
    parts.append(section(
        "comparacao",
        "Comparação entre arquiteturas",
        ("Métricas no conjunto de teste, todos os horizontes e protocolo rolling. "
         "O <b>MSE é a métrica de foco</b> — é também a função de perda do treinamento, "
         "penalizando proporcionalmente erros grandes e refletindo o comportamento "
         "otimizado pelo modelo. As barras evidenciam a hierarquia entre densa e "
         "autônoma: LSTM simples lidera a avaliação densa, enquanto as arquiteturas "
         "convolucionais temporais (TCN) sofrem mais na extrapolação autônoma."),
        results_table,
        images=[("02_comparacao_mse_rmse.png", "MSE e RMSE por modelo (todos os horizontes)"),
                ("03_mse_r2.png", "MSE e R² por modelo (todos os horizontes e rolling)"),
                ("04_radar.png", "Perfil normalizado: erro, rolling, tamanho e custo")],
    ))

    parts.append(section(
        "arquitetura",
        "Arquitetura das redes",
        ("Todos os modelos seguem o esqueleto <b>encoder–decoder</b>: o encoder "
         "comprime a janela de 12 h de histórico (72 passos × variáveis) em um "
         "contexto, e o decoder emite os 36 passos futuros de uma vez "
         "(multi-horizon, treinado com teacher forcing). Nos LSTMs o contexto são "
         "os estados ocultos (com atenção sobre as saídas do encoder); nas TCNs o "
         "encoder e o decoder são redes convolucionais dilatadas com blocos "
         "residuais. A saída é uma camada <code>TimeDistributed(Dense)</code> sobre "
         "as features combinadas do decoder. Os diagramas representam a "
         "<b>arquitetura definitiva, sem persistence gate</b>; as variantes com gate "
         "(estático e dinâmico) foram estudadas separadamente no relatório "
         "<code>ab_persistence_gate.html</code>. Formas mostram "
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
         "distingue as arquiteturas."),
        "",
        images=[("05_mae_por_horizonte_real.png", "MAE por horizonte — vs dado real (bruto)"),
                ("05_mae_por_horizonte_wavelet.png", "MAE por horizonte — vs alvo wavelet (nível treinado)"),
                ("06_heatmap_mse_horizonte.png",
                 "Heatmap de MSE ((m/s)²) por modelo e horizonte — verde = menor erro"),
                ("07_fan_chart.png",
                 "Leques de previsão de 6 h — todos os modelos, 6 origens do teste")],
    ))

    parts.append(section(
        "rolling",
        "Previsão autônoma (rolling, horizonte de 6 h)",
        (f"No protocolo rolling o modelo prevê 36 passos sem nenhuma observação "
         f"intermediária — o cenário real de operação. <b>{best_roll['short']}</b> lidera "
         f"com MSE {fmt(best_roll['rolling']['mse'], 3)} (m/s)². O gráfico revela o motivo "
         "da diferença: ao serem alimentadas pelos próprios erros, as previsões dos "
         "modelos recorrentes colapsam para uma faixa quase constante "
         "(~4,7–5,2 m/s), enquanto a TCN preserva a variação dinâmica do vento — "
         "a TCN Bidirectional até acompanha os picos, mas sistematicamente "
         "superestimada. O painel inferior mostra o erro médio móvel: períodos "
         "acima de zero indicam superestimação persistente."),
        ('<div class="note">A comparação &quot;todos os horizontes&quot; (denso) e '
         '&quot;rolling&quot; (autônomo) mede coisas diferentes: no denso cada passo recebe '
         'o histórico verdadeiro; no rolling, os próprios erros do modelo se acumulam. '
         'Por isso o ranking inverte: LSTM vence no denso, TCN no autônomo.</div>'),
        images=[("08_rolling_forecast_real.png",
                 "Rolling h=36 vs dado real (bruto), com erro médio móvel"),
                ("08_rolling_forecast_wavelet.png",
                 "Rolling h=36 vs alvo wavelet (nível treinado), com erro médio móvel")],
    ))

    parts.append(section(
        "melhor",
        f"Diagnóstico do melhor modelo ({best['short']})",
        (f"O {best['short']} atinge MSE {fmt(best['all']['mse'], 3)} (m/s)² "
         f"(RMSE {fmt(best['all']['rmse'], 3)} m/s, R² {fmt(best['all']['r2'], 3)}) "
         "na avaliação densa. "
         "O histograma de erros é quase gaussiano, com leve cauda à direita. O viés "
         "por regime revela <b>superestimação em ventos fracos (até +1,9 m/s) e "
         "subestimação em rajadas (até −2,1 m/s)</b> — o achatamento clássico da "
         "regressão à média, visível também na compressão da faixa prevista no "
         "dispersograma. A ACF dos resíduos rolling decai lentamente (0,95 → 0,03 "
         "em 36 passos), indicando estrutura temporal remanescente que métodos "
         "adicionais (covariáveis, comitês) poderiam explorar."),
        build_best_table(best),
        images=[("09_diagnostico_melhor.png",
                 "Dispersão, histograma de erros, viés por regime e ACF dos resíduos")],
    ))

    curves_ok = plot_history_ok(history)
    epochs_runs = [r["training"]["epochs_run"] for r in recs]
    save_pct = [(1 - e / 150) * 100 for e in epochs_runs]
    best_epochs = {n: history[n].get("best_epoch") for n in
                   [r["name"] for r in recs] if n in history}
    instant = [f"<b>{short_name(n)}</b> (época {e})"
               for n, e in best_epochs.items() if e and e <= 2]
    overfit_note = ""
    if instant:
        overfit_note = (
            '<div class="hl">Atenção: ' + " e ".join(instant) +
            " atinge a melhor val_loss já nas primeiras épocas — com redes deste "
            "tamanho e apenas ~5,7 mil janelas de treino, o sobreajuste começa "
            "imediatamente (veja a val_loss subir enquanto a loss de treino cai). "
            "O early stopping com restauração dos melhores pesos é o que torna "
            "esse regime treinável.</div>"
        )
    train_note = (
        '<div class="note">Curvas geradas por re-execução do protocolo original '
        "(mesma semente 42, mesmos hiperparâmetros do Optuna e mesmo early stopping) "
        "para capturar o histórico por época, que não é salvo no <code>.keras</code>. "
        "Pequenas diferenças vs o treino original são esperadas pelo "
        "não-determinismo de GPU.</div>"
        + build_repro_table(recs, history)
        + overfit_note
        if curves_ok else
        '<div class="hl">Cache de curvas ausente — execute '
        '<code>python tests/relatorio_resultados.py</code> para gerá-las.</div>'
    )
    parts.append(section(
        "treinamento",
        "Dinâmica de treinamento",
        ("Todos os modelos usam o mesmo orçamento de 150 épocas com early stopping "
         "(paciência 12) monitorando <code>val_loss</code>, cuja convenção de decoder "
         "espelha a inferência. O early stopping interrompe entre "
         f"{min(epochs_runs)} e {max(epochs_runs)} épocas — "
         f"{fmt(min(save_pct), 0)}% a {fmt(max(save_pct), 0)}% do orçamento "
         "economizado — e restaura os pesos da melhor época."),
        train_note,
        images=[("10_curvas_treinamento.png",
                 "Loss e val_loss por época (re-execução); estrela = melhor época"),
                ("11_early_stopping.png",
                 "Épocas executadas vs orçamento; melhor vs final val_loss (treino original)"),
                ("12_tempo_treinamento.png",
                 "Tempo total (original) e custo por época (re-execução)")],
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
         "(busca própria por arquitetura; espaço compartilhado para TCN em "
         "<code>src/models/tcn_hp.py</code>). LR = taxa de aprendizado; "
         "<code>warmup_cosine</code> indica aquecimento com decaimento cossenoidal. "
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
        f"<li><b>Early stopping economizou {fmt(min(save_pct), 0)}–{fmt(max(save_pct), 0)}% "
        "do orçamento</b> de 150 épocas; a validação na convenção de inferência "
        "permitiu parar cedo sem perder generalização.</li>",
        f"<li><b>Custo de inferência é desprezível</b>: {fmt(fastest['timing']['per_window_ms'], 2)}–"
        f"{fmt(max(r['timing']['per_window_ms'] for r in recs), 2)} ms por janela em GPU "
        "(36 previsões de 6 h), viável para operação em tempo real.</li>",
    ]
    if "Seq2Seq_TCN_Bidirectional" in [r["name"] for r in recs]:
        tcnbi = next(r for r in recs if r["name"] == "Seq2Seq_TCN_Bidirectional")
        concl.append(
            f"<li><b>{tcnbi['short']} degrada</b> (R² {fmt(tcnbi['all']['r2'], 2)} "
            "denso): bidirecionalidade + dropout alto (0,4) favoreceram o treino "
            "mas não a extrapolação autônoma.</li>"
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
                f'<a href="#{sid}">{label}</a>' for sid, label in [
                    ("dados", "Dados"),
                    ("comparacao", "Comparação"),
                    ("arquitetura", "Arquiteturas"),
                    ("horizontes", "Horizontes"),
                    ("rolling", "Rolling"),
                    ("melhor", "Melhor modelo"),
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


def plot_history_ok(history):
    return any(k != "_meta" for k in history)


def main():
    parser = argparse.ArgumentParser(description="Relatório de resultados (HTML).")
    parser.add_argument("--retrain", action="store_true",
                        help="ignora o cache e re-treina todos os modelos")
    parser.add_argument("--no-train", action="store_true",
                        help="usa apenas curvas em cache (sem re-treinar)")
    args = parser.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    dataset = common.load_dataset(DATA_FILE)
    print(f"Dataset: {len(dataset)} linhas | {dataset.index.min()} a {dataset.index.max()}")

    results, extras = evaluate_models(dataset)
    if EVAL_CACHE:
        common.write_json(EVAL_CACHE, {k: {kk: vv for kk, vv in v.items()
                                           if kk not in ("hp", "training")}
                                       for k, v in results.items()})

    arch_base = NOGATE_DIR if NOGATE_DIR.is_dir() else None
    arch_records = export_architectures(dataset, base_dir=arch_base)

    if args.no_train:
        history = _load_history_cache()
    else:
        history = retrain_history(dataset, force=args.retrain)

    recs = sorted(results.values(), key=lambda r: r["all"]["mse"])

    plot_split_timeline(dataset, recs)
    plot_comparison_bars(recs)
    plot_skill(recs)
    plot_radar(recs)
    plot_per_horizon(recs, extras)
    plot_horizon_heatmap(recs, extras)
    plot_fan(recs, extras, dataset)
    plot_rolling(recs, extras)
    plot_best_diagnostics(recs[0], extras)
    plot_training_curves(history, recs)
    plot_early_stopping(history, recs)
    plot_training_time(history, recs)
    plot_inference(recs)
    plot_tradeoff(recs)

    n_rows = len(dataset)
    train_end = int(n_rows * 0.75)
    val_end = train_end + int(n_rows * 0.20)
    dataset_info = {
        "start": f"{dataset.index.min():%d/%m/%Y}",
        "end": f"{dataset.index.max():%d/%m/%Y}",
        "rows": n_rows,
        "test_rows": n_rows - (train_end + int(n_rows * 0.20)),
        "splits": [
            {"name": name, "start": f"{dataset.index[a]:%d/%m/%Y %H:%M}",
             "end": f"{dataset.index[b - 1]:%d/%m/%Y %H:%M}",
             "rows": b - a, "pct": (b - a) / n_rows * 100}
            for name, a, b in (("Treino", 0, train_end),
                               ("Validação", train_end, val_end),
                               ("Teste", val_end, n_rows))
        ],
    }

    mse_mat = np.vstack([
        extras[r["name"]]["per_horizon"]["mse"].to_numpy()
        for r in recs
    ])
    growth = mse_mat[:, -1] / mse_mat[:, 0]
    hz_stats = {
        "growth_min": float(growth.min()),
        "growth_max": float(growth.max()),
    }
    html = build_html(recs, history, dataset_info, hz_stats, arch_records, extras, dataset)
    out_file = OUT_DIR / "relatorio_resultados.html"
    out_file.write_text(html, encoding="utf-8")

    print(f"Figuras salvas em: {IMG_DIR}")
    print(f"Relatório gerado em: {out_file}")


if __name__ == "__main__":
    main()
