"""Relatório — A/B alturas do perfil LiDAR (reval 72,5/12,5/15).

Consolida as 4 rodadas do estudo de alcance vertical do perfil na
entrada dos modelos, todas no protocolo reval (split 72,5/12,5/15 com
ES/RLRP, HPs canônicos, MSE pura, seed 42), avaliação oficial na grade
trim 72:

- canônica: TCN 48 col (perfil 40–140 + meteo) / LSTM 67 (featurex);
- 40–160 m: TCN 56 / LSTM 68 (+ cis1..12);
- 40–260 m: TCN 84 / LSTM 103 (+ cis1..19);
- cis≤140 m: LSTM 58 (featurex com cis1..10) — só LSTMs.

Questão: o corte canônico em 140 m (`src/common.py:CANONICAL_HEIGHTS`)
custa ou ganha performance? Fontes: `data/results/report_reval_{sbsr,
h160,allh,cisltd}/evaluation.json` + `pipeline/tmp/trained_reval_*/
<wrapper>/{model.json,history.json}`.

    python reports/lidar/ab-alturas/report.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402

RES = PROJECT_ROOT / "data" / "results"
TMP = PROJECT_ROOT / "pipeline" / "tmp"
FIG_DIR = Path(__file__).resolve().parent / "figures"

ROUNDS = [
    ("canonica", "Canônica (40–140 m)", "report_reval_sbsr",
     "trained_reval_sbsr", "48 / 67 col"),
    ("h160", "Perfil 40–160 m", "report_reval_h160",
     "trained_reval_h160", "56 / 68 col"),
    ("allh", "Perfil 40–260 m", "report_reval_allh",
     "trained_reval_allh", "84 / 103 col"),
]
WRAPPERS = [
    ("Seq2Seq_LSTM", "LSTM"),
    ("Seq2Seq_LSTM_Bidirectional", "LSTM-Bi"),
    ("Seq2Seq_TCN", "TCN"),
    ("Seq2Seq_TCN_Bidirectional", "TCN-Bi"),
]
SHORT = {"Seq2Seq_LSTM": "LSTM",
         "Seq2Seq_LSTM_Bidirectional": "LSTM-Bi",
         "Seq2Seq_TCN": "TCN",
         "Seq2Seq_TCN_Bidirectional": "TCN-Bi"}


def load_round(eval_dir, models_dir):
    models_dir = PROJECT_ROOT / models_dir
    ev = json.loads((RES / eval_dir / "evaluation.json").read_text(
        encoding="utf-8"))
    out = {}
    for key, _ in WRAPPERS:
        meta_p = models_dir / key / "model.json"
        hist_p = models_dir / key / "history.json"
        m = ev.get(key, {})
        ph = m.get("per_horizon") or {}
        h36 = ph.get("36") if isinstance(ph, dict) else (
            ph[35] if isinstance(ph, list) and len(ph) >= 36 else {})
        entry = {
            "all": m.get("all") or {},
            "h36": h36 or {},
        }
        if meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            entry["train"] = meta.get("training") or {}
            entry["inputs"] = meta.get("hyperparameters", {})
        if hist_p.is_file():
            h = json.loads(hist_p.read_text(encoding="utf-8"))
            entry["history"] = h
        out[key] = entry
    return out


def build(ctx):  # noqa: ARG001
    rounds = {rid: load_round(eval_dir, models_dir)
              for rid, _, eval_dir, models_dir, _ in ROUNDS}
    cisltd = load_round("report_reval_cisltd", TMP / "trained_reval_cisltd")

    # --- tabela principal: R² all / MSE / R² h36 por rodada --------------
    header = ["Wrapper"] + [lbl for _, lbl, _, _, _ in ROUNDS] + ["cis≤140*"]
    rows = []
    for key, name in WRAPPERS:
        row = [name]
        for rid in dict.fromkeys(r[0] for r in ROUNDS):
            e = rounds[rid][key]
            a, h36 = e.get("all") or {}, e.get("h36") or {}
            if a:
                row.append(f"{a['r2']:.3f} / {a['mse']:.3f}"
                           + (f" / {h36['r2']:.3f}" if h36 else ""))
            else:
                row.append("—")
        e = cisltd.get(key)
        a, h36 = (e.get("all"), e.get("h36")) if e else (None, None)
        row.append(f"{a['r2']:.3f} / {a['mse']:.3f}"
                   + (f" / {h36['r2']:.3f}" if h36 else "") if a else "—")
        rows.append(row)

    # --- tabela de treino (history) --------------------------------------
    trows = []
    for rid, lbl, _, models_dir, _ in ROUNDS:
        for key, name in WRAPPERS:
            t = rounds[rid][key].get("train") or {}
            if not t:
                continue
            trows.append((lbl.split(" (")[0], name,
                          f"{t.get('epochs_run', '—')}/{t.get('epochs_requested', 60)}",
                          f"{t.get('best_val_loss', float('nan')):.5f}",
                          f"{t.get('training_time_sec', 0):.0f}s"))
    for key, name in WRAPPERS[:2]:
        t = cisltd[key].get("train") or {}
        if t:
            trows.append(("cis≤140", name,
                          f"{t.get('epochs_run', '—')}/{t.get('epochs_requested', 60)}",
                          f"{t.get('best_val_loss', float('nan')):.5f}",
                          f"{t.get('training_time_sec', 0):.0f}s"))

    # --- figura: barras R² all por wrapper × rodada ----------------------
    FIG_DIR.mkdir(exist_ok=True)
    x = np.arange(len(WRAPPERS))
    width = 0.8 / (len(ROUNDS) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = [(lbl, [rounds[rid][key].get("all", {}).get("r2", np.nan)
                   for key, _ in WRAPPERS])
            for rid, lbl, _, _, _ in ROUNDS]
    bars.append(("cis≤140 (LSTMs)",
                 [cisltd[key].get("all", {}).get("r2", np.nan)
                  if i < 2 else np.nan
                  for i, (key, _) in enumerate(WRAPPERS)]))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for i, (lbl, vals) in enumerate(bars):
        ax.bar(x + (i - 1.5) * width, vals, width * 0.92, label=lbl,
               color=colors[i % len(colors)])
        for xi, v in zip(x + (i - 1.5) * width, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.008, f"{v:.3f}".replace(".", ","),
                        ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [n for _, n in WRAPPERS])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("R² (all, grade trim 72)")
    ax.set_title("A/B alturas do perfil — R² agregado por wrapper")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ab_alturas_r2.png", dpi=140)
    plt.close(fig)

    return "".join([
        C.page_header(
            "A/B alturas do perfil LiDAR (reval)",
            "Quatro rodadas no mesmo protocolo reval (72,5/12,5/15 com "
            "ES/RLRP, HPs canônicos, MSE pura, seed 42) variando SÓ o "
            "alcance vertical da entrada: 40–140 (canônica), 40–160, "
            "40–260 e o cisalhamento limitado a 140 m. Avaliação oficial "
            "na grade trim 72, sem baselines.",
            badges=("4 rodadas", "reval 72,5/12,5/15", "seed 42",
                    "grade trim 72")),
        C.cards([
            ("140 m", "corte canônico (11 alturas)"),
            ("46→103", "canais por wrapper no estudo"),
            ("cis1..19", "camadas Δws/Δh (adjacentes)"),
            ("±0,04", "piso de ruído de retreino (R²)"),
        ]),
        C.section(
            "Comparação oficial (R² all / MSE / R² h36)",
            C.figure("figures/ab_alturas_r2.png",
                     "R² agregado por wrapper em cada alcance do perfil.",
                     label="fig:ab_alturas_r2", width=1.0)
            + C.table(header, rows,
                      "Métricas oficiais (trim 72, filtro de observadas). "
                      "Colunas: R² all / MSE / R² h36.",
                      label="tab:ab_alturas_oficial",
                      note="* cis≤140 só existe para as LSTMs (58 col: "
                           "featurex com cis1..10). TCNs canônicas: 48 col "
                           "(perfil+meteo); 160 m: 56; 260 m: 84. LSTMs: "
                           "67 / 68 / 103 / 58."),
            sub="Leitura: a **canônica 40–140 m empata ou vence tudo** "
                "dentro do piso de ruído. Subir o perfil até 160 m não "
                "compra nada (TCNs empatam; LSTM +0,010 = ruído; LSTM-Bi "
                "cai 0,065). Descer o cis para ≤140 (58 col) custa "
                "~0,06–0,07 nas DUAS LSTMs — as camadas cis11..19 (140→260) "
                "têm sinal real. E os canais crus acima de 140 (103 col) "
                "são tóxicos: a LSTM-Bi **colapsa** (0,561→0,312) com "
                "val_loss MELHOR — overfit de largura/memorização.",
            sec_id="comparacao"),
        C.section(
            "Treino (history por rodada)",
            C.table(["Rodada", "Wrapper", "Épocas (ES)", "Melhor val_loss",
                     "Tempo"], trows,
                    "Épocas rodadas até o EarlyStopping (patience 12) e "
                    "melhor val_loss de cada treino.",
                    label="tab:ab_alturas_history",
                    note="history.json completo por wrapper em "
                         "pipeline/tmp/trained_reval_{h160,cisltd}/; as "
                         "demais rodadas têm curvas em "
                         "pipeline/tmp/train_dashboard/."),
            sub="Assinatura de overfit da LSTM-Bi no 40–260: melhor "
                "val_loss da série (0,00851) com o PIOR teste (0,312) — a "
                "validação de 12,5% não vira generalização quando a "
                "entrada dobra de largura.",
            sec_id="history"),
    ])


REPORT = {
    "title": "A/B alturas do perfil LiDAR (reval)",
    "description": "Alcance vertical da entrada: 40–140 (canônica) × "
                   "40–160 × 40–260 × cis≤140, protocolo reval, seed 42.",
    "source_dir": "data/results/report_reval_allh",
    "info": [
        ("Protocolo", "reval 72,5/12,5/15 + ES/RLRP, MSE, seed 42"),
        ("Variação", "somente o alcance vertical das features"),
        ("Avaliação", "oficial trim 72, sem baselines"),
        ("Rodadas", "report_reval_{sbsr,h160,allh,cisltd}"),
    ],
}

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
