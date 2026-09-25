"""A/B/C: efeito do persistence gate (estático vs dinâmico vs nenhum).

Compara as avaliações honestas (protocolo step_evaluate) dos wrappers focais
em três variantes de arquitetura:
  - sem gate            pipeline/tmp/trained_compare_nogate
  - gate estático       pipeline/tmp/trained_compare           (logit por horizonte)
  - gate dinâmico       pipeline/tmp/trained_compare_dynagate  (condicional ao decoder)

A única diferença entre os treinos é a variante do gate (mesma semente, mesmos
hiperparâmetros Optuna, mesmos dados). O relatório inclui um diagnóstico
exclusivo do gate dinâmico: o valor médio de sigmoid(g[t,h]) por horizonte no
conjunto de teste, extraído dos pesos da camada.

Executar a partir da raiz do projeto:
    python tests/ab_persistence_gate.py

Saída:
    data/results/report/images/ab_*.png
    data/results/report/ab_persistence_gate.html
"""

import argparse
import json
import sys
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
from relatorio_resultados import build_wrapper, load_metadata

GATED_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "trained_compare"
NOGATE_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "trained_compare_nogate"
DYN_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "trained_compare_dynagate"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "report"
IMG_DIR = OUT_DIR / "images"
DATA_FILE = PROJECT_ROOT / "data" / "dataset.csv"
FOCUS = ("lstm", "lstm_bi", "tcn", "tcn_bi")

COLORS = {
    "Seq2Seq_LSTM": "#1f77b4",
    "Seq2Seq_LSTM_Bidirectional": "#9467bd",
    "Seq2Seq_TCN": "#ff7f0e",
    "Seq2Seq_TCN_Bidirectional": "#d62728",
}
SIDE_STYLES = {
    "sem": {"label": "sem gate", "color": "#9e9e9e", "ls": "--"},
    "estático": {"label": "gate estático", "color": "#1b6ca8", "ls": "-"},
    "dinâmico": {"label": "gate dinâmico", "color": "#2ca02c", "ls": "-"},
}
SIDES = [
    ("sem", "sem gate", NOGATE_DIR),
    ("estático", "gate estático", GATED_DIR),
    ("dinâmico", "gate dinâmico", DYN_DIR),
]

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


def latex_table(columns, rows):
    head_html = "".join(f"<th>{c}</th>" for c in columns)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    html_table = f"<table><tr>{head_html}</tr>{body_html}</table>"

    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\begin{tabular}{l" + "c" * (len(columns) - 1) + "}\n"
    latex += " & ".join(columns) + " \\\\ \\hline\n"
    for row in rows:
        latex += " & ".join(str(cell) for cell in row) + " \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\caption{...}\n\\end{table}"
    return html_table, latex


def section(sec_id, title, sub, body, images=()):
    imgs = "".join(fig_html(p, t) for p, t in images if (IMG_DIR / p).is_file())
    anchor = f' id="{sec_id}"' if sec_id else ""
    return (f"<section{anchor}><h2>{title}</h2><div class='sub'>{sub}</div>"
            f"{body}{imgs}</section>")


def load_side(base_dir):
    models = {}
    for model_dir in sorted(base_dir.iterdir()):
        metrics_path = model_dir / "evaluate" / "metrics.json"
        roll_path = model_dir / "evaluate" / "predictions_rolling.csv"
        if not metrics_path.is_file():
            continue
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        key = metrics["model"]["wrapper_key"]
        if key not in FOCUS:
            continue
        entry = {
            "key": key,
            "name": metrics["model"]["name"],
            "short": short_name(metrics["model"]["name"]),
            "color": COLORS[metrics["model"]["name"]],
            "all": metrics["all_horizons"],
            "rolling": metrics["rolling"],
            "persistence": metrics["persistence"],
            "training": metrics["model"]["training"],
            "by_horizon": pd.DataFrame(metrics["by_horizon"]),
        }
        if roll_path.is_file():
            entry["roll_df"] = pd.read_csv(roll_path, parse_dates=["origin", "timestamp"])
        models[key] = entry
    return models


def common_keys(sides):
    keys = None
    for data in sides.values():
        present = set(data)
        keys = present if keys is None else (keys & present)
    return [k for k in FOCUS if k in (keys or set())]


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def plot_metric_bars(sides, keys, metric, agg, title, xlabel, fname, persistence=None):
    n_sides = len(sides)
    first = list(sides)[0]
    w = 0.8 / n_sides
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9 + 1.1 * n_sides, 4.6))
    for si, sid in enumerate(sides):
        offset = (si - (n_sides - 1) / 2) * w
        values = [sides[sid][k][agg][metric] for k in keys]
        hatch = {"sem": "//", "estático": None, "dinâmico": "xx"}[sid]
        ax.barh(y + offset, values, height=w * 0.94,
                color=[sides[sid][k]["color"] for k in keys],
                alpha=0.5 if sid == "sem" else 0.9,
                hatch=hatch, edgecolor="white", lw=0.3,
                label=SIDE_STYLES[sid]["label"])
        for yi, v in zip(y + offset, values):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=7.5)
    if persistence is not None:
        ax.axvline(persistence, color="black", ls="--", lw=1.3,
                   label=f"persistência ({persistence:.3f})")
    ax.set_yticks(y, [sides[first][k]["short"] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7.5)
    savefig(fig, fname)


def plot_horizon_panels(sides, keys):
    first = list(sides)[0]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.3 * len(keys), 3.9), sharey=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        ref = sides[first][k]["by_horizon"].sort_values("horizon")
        ax.plot(ref["horizon"], ref["persistence_mae"], color="black", lw=1.8,
                ls=":", label="persistência", zorder=5)
        for sid in sides:
            ph = sides[sid][k]["by_horizon"].sort_values("horizon")
            style = SIDE_STYLES[sid]
            ax.plot(ph["horizon"], ph["mae"], color=style["color"], ls=style["ls"],
                    lw=1.6, label=style["label"])
        ax.set_title(sides[first][k]["short"], fontsize=10)
        ax.set_xlabel("Horizonte (passos de 10 min)")
        ax.set_xticks(range(1, 37, 6))
    axes[0].set_ylabel("MAE (m/s)")
    axes[-1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("MAE por horizonte nas três variantes (vs persistência)", fontsize=11)
    savefig(fig, "ab_03_mae_horizontes.png")


def plot_delta_dyn_static(sides, keys):
    if "dinâmico" not in sides or "estático" not in sides:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    for k in keys:
        est = sides["estático"][k]["by_horizon"].sort_values("horizon")
        dyn = sides["dinâmico"][k]["by_horizon"].sort_values("horizon")
        delta = (dyn["mae"] - est["mae"]).to_numpy()
        ax.plot(est["horizon"], delta, color=sides["estático"][k]["color"],
                lw=1.8, marker="o", ms=2.6, label=sides["estático"][k]["short"])
    ax.axhline(0, color="black", lw=1.1)
    ax.set_xlabel("Horizonte (passos de 10 min)")
    ax.set_ylabel("MAE(dinâmico) − MAE(estático)  (m/s)")
    ax.set_title("Gate dinâmico vs estático por horizonte (abaixo de zero = dinâmico melhor)")
    ax.set_xticks(range(1, 37, 2))
    ax.legend(ncol=2)
    savefig(fig, "ab_04_delta_dyn_estatico.png")


def plot_rolling(sides, keys):
    for k in keys:
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.08})
        ref = None
        for sid in sides:
            roll = sides[sid][k].get("roll_df")
            if roll is None:
                continue
            ref = roll if ref is None else ref
        if ref is None:
            plt.close(fig)
            continue
        axes[0].plot(ref["timestamp"], ref["actual_raw"], color="black", lw=1.6,
                     label="observado", zorder=5)
        for sid in sides:
            roll = sides[sid][k].get("roll_df")
            if roll is None:
                continue
            style = SIDE_STYLES[sid]
            axes[0].plot(roll["timestamp"], roll["predicted"], color=style["color"],
                         ls=style["ls"], lw=1.2, alpha=0.9,
                         label=f"{style['label']} (MAE {sides[sid][k]['rolling']['mae']:.2f})")
        axes[0].set_ylabel("Velocidade (m/s)")
        axes[0].set_title(f"Previsão autônoma (rolling h=36) — {sides[list(sides)[0]][k]['short']}")
        axes[0].legend(ncol=2, fontsize=8)

        for sid in sides:
            roll = sides[sid][k].get("roll_df")
            if roll is None:
                continue
            style = SIDE_STYLES[sid]
            resid = (roll["predicted"] - roll["actual_raw"]).rolling(12, min_periods=1).mean()
            axes[1].plot(roll["timestamp"], resid, color=style["color"], ls=style["ls"],
                         lw=1.3, label=style["label"])
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_ylabel("Erro médio\n(média móvel 2 h)")
        axes[1].set_xlabel("Timestamp")
        savefig(fig, f"ab_05_rolling_{k}.png")


def collect_gate_curves(dataset, dyn_dir):
    common.setup_tensorflow(True)
    from keras import backend as K
    from src.models.layers import make_gate_extractor

    curves = {}
    for entry in load_metadata(dyn_dir):
        meta = entry["meta"]
        ratios = meta.get("dataset", {})
        train_df, val_df, test_df = common.split_dataset(
            dataset, ratios.get("train_ratio", 0.75), ratios.get("val_ratio", 0.20)
        )
        print(f"[{meta['wrapper_key']}] extraindo curva de gate ...")
        wrapper = build_wrapper(meta, train_df, val_df, entry["dir"] / "model.keras")
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
        extract = make_gate_extractor(wrapper.model)
        gates = extract([prepared["X_encoder"], prepared["X_decoder"]])
        curves[meta["name"]] = {
            "short": short_name(meta["name"]),
            "color": COLORS[meta["name"]],
            "mean": gates.mean(axis=0).ravel(),
            "sd": gates.std(axis=0).ravel(),
        }
        K.clear_session()
    return curves


def plot_gate_behavior(curves):
    if not curves:
        return False
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, c in curves.items():
        horizons = np.arange(1, len(c["mean"]) + 1)
        ax.plot(horizons, c["mean"], color=c["color"], lw=1.9, label=c["short"])
        ax.fill_between(horizons, c["mean"] - c["sd"], c["mean"] + c["sd"],
                        color=c["color"], alpha=0.12)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="black", lw=0.8)
    ax.set_xlabel("Horizonte (passos de 10 min)")
    ax.set_ylabel("σ(gate) médio no teste  (0 = persistência, 1 = rede)")
    ax.set_title("Comportamento aprendido do gate dinâmico (média ± 1σ entre amostras)")
    ax.set_xticks(range(1, 37, 2))
    ax.legend(ncol=2)
    savefig(fig, "ab_06_gate_behavior.png")
    return True


# ----------------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------------
def build_results_table(sides, keys):
    side_ids = list(sides)
    rows = []
    for k in keys:
        row = [sides[side_ids[0]][k]["short"]]
        for metric, agg in [("mae", "all"), ("r2", "all"), ("mae", "rolling"), ("rmse", "rolling")]:
            for sid in side_ids:
                row.append(fmt(sides[sid][k][agg][metric], 3))
        rows.append(tuple(row))
    columns = ["Modelo"]
    labels = {"sem": "sem", "estático": "est.", "dinâmico": "din."}
    for metric in ["MAE denso", "R² denso", "MAE rolling", "RMSE rolling"]:
        columns += [f"{metric} ({labels[s]})" for s in side_ids]
    html_table, latex = latex_table(columns, rows)
    return latex_button(latex) + html_table


def build_horizon_tables(sides, keys):
    side_ids = list(sides)
    n_h = len(sides[side_ids[0]][keys[0]]["by_horizon"])

    mae_rows = []
    r2_rows = []
    for i in range(n_h):
        g0 = sides[side_ids[0]][keys[0]]["by_horizon"].iloc[i]
        h = int(g0["horizon"])
        mae_row = [str(h), fmt(g0["persistence_mae"], 3)]
        r2_row = [str(h)]
        for k in keys:
            for sid in side_ids:
                cell = sides[sid][k]["by_horizon"].iloc[i]
                mae_row.append(fmt(cell["mae"], 3))
                r2_row.append(fmt(cell["r2"], 3))
            if "dinâmico" in sides and "estático" in sides:
                dyn = sides["dinâmico"][k]["by_horizon"].iloc[i]["mae"]
                est = sides["estático"][k]["by_horizon"].iloc[i]["mae"]
                mae_row.append(f"{dyn - est:+.3f}")
        mae_rows.append(tuple(mae_row))
        r2_rows.append(tuple(r2_row))

    mae_cols = ["h", "persist."]
    r2_cols = ["h"]
    labels = {"sem": "sem", "estático": "est.", "dinâmico": "din."}
    for k in keys:
        s = sides[side_ids[0]][k]["short"]
        mae_cols += [f"{s} {labels[x]}" for x in side_ids]
        r2_cols += [f"{s} {labels[x]}" for x in side_ids]
        if "dinâmico" in sides and "estático" in sides:
            mae_cols.append(f"{s} Δd−e")

    mae_html, mae_latex = latex_table(mae_cols, mae_rows)
    r2_html, r2_latex = latex_table(r2_cols, r2_rows)
    return {"mae": latex_button(mae_latex) + mae_html,
            "r2": latex_button(r2_latex) + r2_html}


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
CSS = """
:root{--accent:#1b6ca8;--bg:#f4f6f9;--card:#ffffff;}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI',system-ui,Arial,sans-serif;background:var(--bg);color:#222}
header{background:linear-gradient(135deg,#0f3d5e,#1b6ca8);color:#fff;padding:28px 32px}
header h1{margin:0 0 6px;font-size:24px}
header p{margin:0;opacity:.9;font-size:14px}
.container{max-width:1100px;margin:0 auto;padding:24px 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid #e3e8ee;border-radius:10px;padding:14px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card .num{font-size:20px;font-weight:700;color:var(--accent)}
.card .lab{font-size:12px;color:#555;margin-top:2px}
section{background:var(--card);border:1px solid #e3e8ee;border-radius:10px;margin:22px 0;padding:20px 24px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
section h2{margin:0 0 4px;font-size:18px;color:#0f3d5e;border-bottom:2px solid #e8eef4;padding-bottom:8px}
section .sub{color:#666;font-size:13px;margin:8px 0 14px;line-height:1.5}
figure{margin:14px auto;text-align:center}
figure img{max-width:100%;height:auto;border:1px solid #e3e8ee;border-radius:8px}
figcaption{font-size:12px;color:#666;margin-top:5px}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}
th,td{border:1px solid #dfe5ec;padding:4px 7px;text-align:right}
th{background:#eef3f8;color:#0f3d5e}
td:first-child,th:first-child{text-align:left;font-weight:600}
.btn-copy{background:#1b6ca8;color:#fff;border:none;border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer;margin:8px 0 0;font-family:inherit}
.btn-copy:hover{background:#0f3d5e}
.hl{background:#fff7e0;padding:10px 14px;border-left:4px solid #e0a800;border-radius:0 6px 6px 0;font-size:13px;line-height:1.5;margin:10px 0}
.note{background:#eef6ee;padding:10px 14px;border-left:4px solid #3d8b57;border-radius:0 6px 6px 0;font-size:13px;line-height:1.5;margin:10px 0}
section ul{margin:8px 0;padding-left:20px}
section li{margin:7px 0;font-size:13px;line-height:1.55;color:#333}
section li b{color:#0f3d5e}
section p{font-size:13px;line-height:1.6;color:#333}
code{background:#eef3f8;border:1px solid #dfe5ec;border-radius:4px;padding:1px 5px;font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#0f3d5e}
footer{color:#888;text-align:center;font-size:12px;padding:18px}
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


def build_html(sides, keys, cards, body_parts):
    side_names = " · ".join(SIDE_STYLES[s]["label"] for s in sides)
    html = (f'<!DOCTYPE html>\n<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            "<title>A/B/C — Persistence Gate estático vs dinâmico</title>\n"
            f"<style>{CSS}</style>\n</head>\n<body>\n<header>\n"
            "<h1>A/B/C — Persistence Gate: nenhum · estático · dinâmico</h1>\n"
            f"<p>{side_names} · LSTM · LSTM Bidirectional · TCN · TCN Bidirectional · "
            "série LiDAR (dataset.csv) · protocolo step_evaluate · Keras/TensorFlow</p>\n"
            "</header>\n"
            f'<div class="container">\n<div class="cards">{cards}</div>\n'
            f"{body_parts}\n</div>\n"
            "<footer>Relatório gerado automaticamente por tests/ab_persistence_gate.py · "
            "mesma semente (42) e hiperparâmetros Optuna nas três variantes · "
            "métricas contra ws100 bruto</footer>\n"
            f"<script>{JS}</script>\n</body>\n</html>")
    return html


def main():
    parser = argparse.ArgumentParser(description="A/B/C do persistence gate (HTML).")
    parser.add_argument("--nogate-dir", type=Path, default=NOGATE_DIR)
    parser.add_argument("--gated-dir", type=Path, default=GATED_DIR)
    parser.add_argument("--dyn-dir", type=Path, default=DYN_DIR)
    parser.add_argument("--skip-gates", action="store_true",
                        help="pula o diagnóstico keras do gate dinâmico")
    args = parser.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [("sem", "sem gate", args.nogate_dir),
                  ("estático", "gate estático", args.gated_dir),
                  ("dinâmico", "gate dinâmico", args.dyn_dir)]
    sides = {}
    for sid, label, base in candidates:
        data = load_side(base)
        if data:
            sides[sid] = data
    if len(sides) < 2:
        raise SystemExit(
            "Avaliações ausentes. Rode tests/train_compare_tcn.py nas variantes "
            "(--out-base distintos) antes deste relatório."
        )
    keys = common_keys(sides)
    side_names = [s for s in sides]
    print("Variantes:", side_names, "| Modelos:", [sides[side_names[0]][k]["short"] for k in keys])

    pers = sides[list(sides)[0]][keys[0]]["persistence"]["mae"]

    plot_metric_bars(sides, keys, "mae", "all",
                     "Todos os horizontes (1–36)", "MAE (m/s)",
                     "ab_01_mae_gate.png", persistence=pers)
    plot_metric_bars(sides, keys, "mae", "rolling",
                     "Rolling autônomo (h = 36, 6 h)", "MAE (m/s)",
                     "ab_02_mae_rolling_gate.png")
    plot_horizon_panels(sides, keys)
    plot_delta_dyn_static(sides, keys)
    plot_rolling(sides, keys)

    gate_curves = {}
    if "dinâmico" in sides and not args.skip_gates:
        dataset = common.load_dataset(DATA_FILE)
        dyn_dir = next(base for sid, _, base in candidates if sid == "dinâmico")
        gate_curves = collect_gate_curves(dataset, dyn_dir)
        plot_gate_behavior(gate_curves)

    side_ids = list(sides)
    best = {sid: min(keys, key=lambda k: sides[sid][k]["all"]["mae"]) for sid in side_ids}
    mean_effects = {}
    for pair in [("dinâmico", "estático"), ("dinâmico", "sem"), ("estático", "sem")]:
        if pair[0] in sides and pair[1] in sides:
            mean_effects[pair] = (
                float(np.mean([(sides[pair[0]][k]["all"]["mae"]
                                / sides[pair[1]][k]["all"]["mae"] - 1) * 100 for k in keys])),
                float(np.mean([(sides[pair[0]][k]["rolling"]["mae"]
                                / sides[pair[1]][k]["rolling"]["mae"] - 1) * 100 for k in keys])),
            )

    cards = "".join(
        [card(f"{fmt(sides[s][best[s]]['all']['mae'], 3)} m/s",
              f"melhor denso {SIDE_STYLES[s]['label']} — {sides[s][best[s]]['short']}")
         for s in side_ids]
        + [card(f"{v[0]:+.1f}".replace(".", ",") + "%", "din vs estático (denso)")
           for p, v in mean_effects.items() if p == ("dinâmico", "estático")]
        + [card(f"{v[1]:+.1f}".replace(".", ",") + "%", "din vs estático (rolling)")
           for p, v in mean_effects.items() if p == ("dinâmico", "estático")]
    )
    if gate_curves:
        g1 = float(np.mean([c["mean"][0] for c in gate_curves.values()]))
        g36 = float(np.mean([c["mean"][-1] for c in gate_curves.values()]))
        cards += card(f"{g1:.2f} → {g36:.2f}".replace(".", ","),
                      "σ(gate) médio: h1 → h36 (dinâmico)")

    table = build_results_table(sides, keys)
    hz = build_horizon_tables(sides, keys)
    parts = []

    parts.append(section(
        "resumo",
        "Protocolo do experimento",
        ("Mesma semente (42), mesmos hiperparâmetros da melhor tentativa Optuna, "
         "mesmos dados e split — muda apenas a variante da camada final: "
         "<b>sem gate</b> (saída pura do TimeDistributed Dense), "
         "<b>gate estático</b> (logit treinável por horizonte, "
         "<code>g_h = σ(ℓ_h)</code>) e <b>gate dinâmico</b> "
         "(<code>g[t,h] = σ(W·f[t,h] + b_h)</code>, condicional às features do "
         "decoder com bias por horizonte). Cada variante foi treinada do zero e "
         "avaliada com o protocolo honesto (step_evaluate)."),
        ('<div class="note">O gate mistura a previsão da rede com o último valor '
         'observado (<code>g·previsto + (1−g)·persistência</code>). O estático fixa '
         'um peso por horizonte para todas as amostras; o dinâmico deixa a rede '
         'decidir o peso em cada amostra e horizonte, dado o estado do decoder — '
         'e subsume o estático quando W = 0.</div>'),
        images=[("ab_01_mae_gate.png", "MAE todos os horizontes: três variantes"),
                ("ab_02_mae_rolling_gate.png", "MAE rolling autônomo (h=36): três variantes")],
    ))

    parts.append(section(
        "tabela",
        "Resultados gerais",
        ("Denso = todas as 272 origens × 36 horizontes do teste; rolling = previsão "
         "autônoma no horizonte de 6 h."),
        table,
    ))

    parts.append(section(
        "horizontes",
        "Desempenho por horizonte",
        ("Painéis por modelo com as três variantes contra a persistência, seguidos "
         "do delta direto entre gate dinâmico e estático (abaixo de zero = o "
         "dinâmico melhora o estático naquele horizonte). As tabelas trazem MAE e "
         "R² de cada horizonte (272 amostras por horizonte)."),
        '<div class="hl"><b>MAE por horizonte</b> (persist.; variantes; Δ = '
        "dinâmico − estático):</div>" + hz["mae"] +
        '<div class="hl"><b>R² por horizonte</b> (variantes):</div>' + hz["r2"],
        images=[("ab_03_mae_horizontes.png",
                 "MAE por horizonte nas três variantes, painel por modelo"),
                ("ab_04_delta_dyn_estatico.png",
                 "MAE(dinâmico) − MAE(estático) por horizonte")],
    ))

    roll_imgs = [(f"ab_05_rolling_{k}.png",
                  f"Rolling h=36 — {sides[side_ids[0]][k]['short']}: três variantes")
                 for k in keys]
    parts.append(section(
        "rolling",
        "Previsão autônoma (rolling h=36)",
        ("No regime autônomo de 6 h, o gate estático havia piorado todos os "
         "modelos no A/B anterior (âncora velha). O gate dinâmico pode aprender a "
         "soltar a âncora nos horizontes longos — os gráficos mostram se essa "
         "liberdade se materializa em ganho."),
        "",
        images=roll_imgs,
    ))

    if gate_curves:
        parts.append(section(
            "gate",
            "Comportamento aprendido do gate dinâmico",
            ("Média de σ(g[t,h]) no conjunto de teste por horizonte (faixa = ±1σ "
             "entre amostras). Valores próximos de 0 indicam confiança na "
             "persistência; próximos de 1, na previsão da rede. Este diagnóstico "
             "só existe no gate dinâmico — o estático não varia entre amostras."),
            "",
            images=[("ab_06_gate_behavior.png",
                     "σ(gate) médio por horizonte e modelo (gate dinâmico)")],
        ))

    concl = ["<li><b>Melhor denso por variante:</b> "
             + "; ".join(f"{SIDE_STYLES[s]['label']}: {sides[s][best[s]]['short']} "
                         f"({fmt(sides[s][best[s]]['all']['mae'], 3)} m/s)"
                         for s in side_ids) + ".</li>"]
    for (a, b), (d_all, d_roll) in mean_effects.items():
        concl.append(
            f"<li><b>{SIDE_STYLES[a]['label']} vs {SIDE_STYLES[b]['label']}:</b> "
            f"efeito médio {fmt(d_all, 1)}% no denso e {fmt(d_roll, 1)}% no rolling.</li>"
        )
    if gate_curves:
        concl.append(
            "<li><b>Assinatura do gate dinâmico:</b> ver seção anterior — a "
            "progressão de σ(g) com o horizonte mostra onde a rede prefere a "
            "persistência e onde confia na própria previsão.</li>"
        )
    parts.append(section(
        "conclusoes",
        "Conclusões",
        "Síntese do experimento A/B/C do persistence gate.",
        f"<ul>{''.join(concl)}</ul>",
    ))

    html = build_html(sides, keys, cards, "".join(parts))
    out_file = OUT_DIR / "ab_persistence_gate.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Figuras salvas em: {IMG_DIR}")
    print(f"Relatório gerado em: {out_file}")


if __name__ == "__main__":
    main()
