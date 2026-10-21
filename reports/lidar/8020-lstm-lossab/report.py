"""Relatório — A/B controlado de loss nas LSTMs (80/20).

Compara ``horizon_weighted_mse`` × ``itilde_q`` mantendo TUDO igual
(HP canônicos, 60 épocas, seed 42, split 80/20, janela 72→36). Isola o
efeito da função de perda da rodada mista — onde o HP do HPO entrava junto
e confundia o diagnóstico.

Fontes:
    data/results/lstm_lossab_8020/lossab_8020.json + images/*.png
Gerador: scripts/lidar_8020_lstm_lossab_train.py (GPU) +
         tests/lstm_lossab_8020.py (análise, CPU).

    .venv/bin/python reports/lidar/8020-lstm-lossab/report.py
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "results" / "lstm_lossab_8020"
META_JSON = DATA_DIR / "lossab_8020.json"
IMG_DIR = DATA_DIR / "images"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIGS = ("fig_overall.png", "fig_mse_horizon.png", "fig_rolling.png")


def fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:
        return "—"
    return f"{f:.{nd}f}".replace(".", ",")


def p_txt(p):
    if p is None or p != p:
        return "—"
    return "&lt; 0,001" if p < 0.001 else fmt(p, 4)


def copy_figures():
    if IMG_DIR.is_dir():
        FIG_DIR.mkdir(exist_ok=True)
        for name in FIGS:
            src = IMG_DIR / name
            if src.is_file():
                shutil.copy2(src, FIG_DIR / name)


def build(ctx):  # noqa: ARG001 (contrato do builder)
    copy_figures()
    if not META_JSON.is_file():
        return C.page_header(
            "A/B de loss nas LSTMs (80/20)",
            "Análise ainda não executada.") + C.note(
            "Rode <code>scripts/lidar_8020_lstm_lossab_train.py</code> e "
            "<code>tests/lstm_lossab_8020.py</code>.")

    d = json.loads(META_JSON.read_text(encoding="utf-8"))
    alpha = d.get("alpha", 0.05)
    variants = d.get("variants", {})
    dm = d.get("dm", [])

    order = ("lstm_hw", "lstm_itq", "lstm_bi_hw", "lstm_bi_itq")
    rows = []
    for key in order:
        m = variants.get(key)
        if not m:
            continue
        rows.append((m["label"], fmt(m["all"]["mse"], 4),
                     fmt(m["all"]["rmse"], 4), fmt(m["all"]["mae"], 4),
                     fmt(m["all"]["r2"], 3), fmt(m["rolling"]["mse"], 4),
                     fmt(m["skill"], 3)))

    dm_rows = []
    for r in dm:
        dm_rows.append((
            r["arq"], fmt(r["mse_hw"], 4), fmt(r["mse_itq"], 4),
            fmt(r["delta"], 4), fmt(r["dm"], 2), p_txt(r["p"]),
            r["melhor"], "sim" if r.get("sig") else "não",
            f"{fmt(r.get('dm_h6'), 2)} / {fmt(r.get('dm_h12'), 2)} / "
            f"{fmt(r.get('dm_h24'), 2)} / {fmt(r.get('dm_h36'), 2)}",
        ))

    parts = [C.page_header(
        "A/B de loss nas LSTMs (80/20)",
        "Mesma arquitetura, mesmos hiperparâmetros canônicos, mesma seed e "
        "mesmo protocolo 80/20 — a única variável é a função de perda: "
        "horizon-weighted MSE × ITILDE-Q.",
        badges=("A/B controlado", "só a loss muda", "HPs canônicos",
                "DM pareado"))]

    parts.append(C.cards([
        ("2", "arquiteturas (LSTM / LSTM-Bi)"),
        ("2", "losses por arquitetura"),
        ("60", "épocas fixas · seed 42"),
        ("72→36", "passos (12 h → 6 h) de 10 min"),
    ]))

    parts.append(C.section(
        "Desenho do experimento",
        "<p>Na rodada mista anterior o HP do HPO entrou <b>junto</b> com a "
        "loss ITILDE-Q, então o resultado ruim das LSTMs não podia ser "
        "atribuído à perda. Aqui os dois braços usam os HPs canônicos de "
        "<code>pipeline/tmp/trained_8020</code>, 60 épocas fixas, seed 42 e "
        "split 80/20 sem validação. A comparação é pareada por "
        "<code>(origem, horizonte)</code> e o teste de Diebold–Mariano "
        "(HAC até h−1 + correção de Harvey) é aplicado à diferença "
        "ITILDE-Q − horizon-weighted (<b>DM &lt; 0 ⇒ ITILDE-Q melhor</b>).</p>"
        + C.table(["Variante", "MSE", "RMSE", "MAE", "R²",
                   "MSE rolling", "Skill vs persist."], rows,
                  "Métricas densas (todos os horizontes) e rolling autônomo "
                  "(h=36), contra o ws100 bruto.", label="tab:lossab_metrics",
                  note="Skill = 1 − MSE_modelo/MSE_persistência; negativo = "
                       "pior que o ingênuo.")
        + C.figure("figures/fig_overall.png",
                   "MSE, RMSE e R² por variante (todos os horizontes); "
                   "barras hachuradas = horizon-weighted.",
                   label="fig:lossab_overall", width=1.0),
        sub="Resultado: ITILDE-Q melhora a LSTM em ambas as métricas; na "
            "LSTM-Bi o ganho é menor.", sec_id="desenho"))

    parts.append(C.section(
        "Resultado por horizonte e teste estatístico",
        "<p>O ganho da ITILDE-Q não é uniforme: ela atua na forma da "
        "trajetória (amplitude/fase), o que tende a render mais nos "
        "horizontes longos — exatamente onde a LSTM canônica degrada. A "
        "tabela do DM mostra o agregado e a estatística nos horizontes "
        "h6/h12/h24/h36.</p>"
        + C.figure("figures/fig_mse_horizon.png",
                   "MSE por horizonte — LSTM e LSTM-Bi, horizon-weighted "
                   "(linha cheia) × ITILDE-Q (tracejada).",
                   label="fig:lossab_horizon", width=1.0)
        + C.table(["Arquitetura", "MSE hw", "MSE ITILDE-Q", "ΔMSE",
                   "DM agregado", "p", "Melhor", "Sig.?",
                   "DM em h6/h12/h24/h36"], dm_rows,
                  "Diebold–Mariano pareado entre as duas losses "
                  "(ΔMSE = ITILDE-Q − hweighted; DM negativo ⇒ ITILDE-Q "
                  "melhor).", label="tab:lossab_dm",
                  note="HAC até h−1 + correção de Harvey–Leybourne–Newbold "
                       "(1997). α = 0,05.")
        + C.figure("figures/fig_rolling.png",
                   "Rolling autônomo (h=36) nas últimas 3 semanas do teste: "
                   "observado × horizon-weighted × ITILDE-Q.",
                   label="fig:lossab_rolling", width=1.0),
        sub=f"α = {alpha:.2f}. O DM agregado usa a média da perda "
            "quadrática nos 36 horizontes por origem.", sec_id="resultado"))

    parts.append(C.section(
        "Leitura",
        "<div class='markdown'><ul>"
        "<li><b>LSTM:</b> ITILDE-Q reduz o MSE de "
        f"{fmt(variants['lstm_hw']['all']['mse'], 3)} para "
        f"{fmt(variants['lstm_itq']['all']['mse'], 3)} — ganho "
        f"<b>significativo</b> no DM (p = "
        f"{p_txt(next(r['p'] for r in dm if r['arq'] == 'LSTM'))}).</li>"
        "<li><b>LSTM-Bi:</b> ganho pequeno e <b>não significativo</b> — a "
        "loss shape-aware não compensa a bidirecionalidade nesta rodada.</li>"
        "<li><b>Diagnóstico da rodada mista:</b> sob os mesmos HPs "
        "canônicos, a ITILDE-Q é a melhor das duas losses. Na rodada mista "
        "o HP do HPO entrou junto (ITILDE-Q com HP do HPO ≈ 1,84 MSE vs "
        "1,70 com HP canônico), então aquela rodada não isola a perda — e o "
        "HP do HPO também não ajudou.</li>"
        "<li><b>Ainda abaixo das TCNs:</b> mesmo a melhor LSTM "
        f"({fmt(variants['lstm_itq']['all']['mse'], 3)} MSE) fica longe da "
        "TCN (~1,3 MSE) — o gargalo é a arquitetura/HPs da LSTM no sítio, "
        "não apenas a loss.</li>"
        "</ul></div>",
        sec_id="leitura"))

    return "".join(parts)


REPORT = {
    "title": "A/B de loss nas LSTMs (80/20)",
    "description": "horizon-weighted MSE × ITILDE-Q, mesmos HPs canônicos, "
                   "seed e protocolo; métricas densas/rolling e DM pareado.",
    "regenerate": [".venv/bin/python", "tests/lstm_lossab_8020.py",
                   "--models-dir", "pipeline/tmp/trained_8020_lstm_lossab",
                   "--out-dir", "data/results/lstm_lossab_8020"],
    "retrain": [".venv/bin/python",
                "scripts/lidar_8020_lstm_lossab_train.py"],
    "source_dir": "data/results/lstm_lossab_8020",
    "info": [
        ("Rodada", "trained_8020_lstm_lossab (4 modelos)"),
        ("Protocolo", "80/20 sem validação · 60 épocas · seed 42"),
        ("Variável", "apenas a loss (hweighted × ITILDE-Q)"),
        ("Teste", "DM HAC+HLN pareado, α=0,05"),
    ],
}


if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
