"""Relatório de resultados — 80/20 com os hiperparâmetros do Optuna round 7.

Mesmo protocolo do lidar/8020 (80/20 sem validação, 60 épocas fixas, seed 42),
com os hiperparâmetros da rodada **Optuna round 7** — objetivo skill vs ARIMA
com termo de estabilidade (λ=0,3), SuccessiveHalvingPruner (rungs 12/36)
podando pelo objetivo real (decoder inferência), espaço com cantos agressivos
cortados (lr ≤ 2–3e-3, dropout ≥ 0,05, wd fixo 1e-4), clipnorm 1,0 e
**seleção 1-SE com desempate conservador** (`scripts/experiments/optuna_select_1se.py`).

Pipeline de dados (rodar nesta ordem quando a rodada encerrar):

    python scripts/dist/colab_round2.py collect --kaggle
    python scripts/experiments/optuna_select_1se.py --storage pipeline/tmp/optuna_round7 --study round7
    # promote: copiar pipeline/tmp/optuna_round7/<Wrapper>/best_trial.json
    #          para pipeline/tmp/optuna/<Wrapper>/ (com backup)
    python scripts/experiments/lidar_7525_train.py --train-ratio 0.80 \
        --output-dir pipeline/tmp/trained_8020_optuna7 --static-dilations
    python reports/src/relatorio_resultados.py \
        --models-dir pipeline/tmp/trained_8020_optuna7 \
        --out-dir data/results/report_8020_optuna7 --no-baselines

    python reports/lidar/8020-optuna7/report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reports.src import components as C  # noqa: E402
from reports.src.resultados import report_module  # noqa: E402

SWEEP_DIR = PROJECT_ROOT / "data" / "results" / "lidar_8020_optuna7_sweep"
SWEEP_FIG_DIR = Path(__file__).resolve().parent / "figures"
# Fallback se sweep_meta.json ainda não existir (escrito pelo sweep a cada
# rodada futura): tamanhos do conjunto 1-SE da round 7 atual.
EXPECTED_TOTALS = {"lstm": 25, "lstm_bi": 21}

plt.rcParams.update({
    "figure.dpi": 130, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.25,
})


def _num(x, dec=4):
    return f"{float(x):.{dec}f}".replace(".", ",")


def _sweep_data():
    if not (SWEEP_DIR / "sweep.json").is_file():
        return None, {}
    rows = json.loads((SWEEP_DIR / "sweep.json").read_text(encoding="utf-8"))
    meta_path = SWEEP_DIR / "sweep_meta.json"
    meta = (json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file() else {})
    return rows, meta


def _sweep_figure(rows):
    SWEEP_FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    by_wrapper = {}
    for r in rows:
        by_wrapper.setdefault(r["wrapper"], []).append(r)
    colors = {"lstm": "#1f77b4", "lstm_bi": "#d62728"}
    for wrapper, items in by_wrapper.items():
        xs = [r["val_objective"] for r in items]
        ys = [r["test"]["mse"] for r in items]
        ax.scatter(xs, ys, s=28, alpha=0.75, color=colors.get(wrapper),
                   label=f"{wrapper} ({len(items)} trials)")
        for r in items:
            if r["is_1se_chosen"]:
                ax.scatter([r["val_objective"]], [r["test"]["mse"]],
                           marker="*", s=180, color=colors.get(wrapper),
                           edgecolors="k", linewidths=0.6, zorder=5)
            if r["is_argmin"]:
                ax.scatter([r["val_objective"]], [r["test"]["mse"]],
                           marker="X", s=90, color=colors.get(wrapper),
                           edgecolors="k", linewidths=0.6, zorder=5)
    pers = [r["persistence_mse"] for r in rows if r.get("persistence_mse")]
    if pers:
        ax.axhline(float(pers[0]), color="k", ls="--", lw=1,
                   label=f"persistência (MSE {_num(pers[0])})")
    art = SWEEP_DIR / "lstm_artigo_controle.json"
    if art.is_file():
        am = json.loads(art.read_text(encoding="utf-8"))["mesma_grade"]
        ax.axhline(float(am["mse"]), color="#2ca02c", ls=":", lw=1.4,
                   label=f"LSTM do artigo (MSE {_num(am['mse'])})")
    ax.set_xlabel("objetivo na validação 72/8 (−[skill − 0,3·std]; menor = melhor)")
    ax.set_ylabel("MSE no teste 80/20 (menor = melhor)")
    ax.set_title("Sweep 1-SE round 7 — validação vs teste (★ 1-SE, ✕ argmin)")
    ax.legend()
    fig.tight_layout()
    out = SWEEP_FIG_DIR / "sweep_1se.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _control_rows(rows):
    """Linhas de controle na MESMA grade do teste: LSTM do artigo + persistência."""
    out = []
    path = SWEEP_DIR / "lstm_artigo_controle.json"
    if path.is_file():
        d = json.loads(path.read_text(encoding="utf-8"))
        m = d["mesma_grade"]
        out.append(("LSTM do artigo (controle)",
                    "featurex 67 col · MSE · seed 42",
                    "—", "—", _num(m["mse"]), _num(m["rmse"]),
                    _num(m["mae"]), _num(m["skill_persistence"], 3),
                    "controle"))
    path = SWEEP_DIR / "persistencia_controle.json"
    if path.is_file():
        d = json.loads(path.read_text(encoding="utf-8"))["persistence"]
        out.append(("Persistência (controle)", "último valor observado",
                    "—", "—", _num(d["mse"]), _num(d["rmse"]), _num(d["mae"]),
                    "0,000 (def.)", "controle"))
        return out
    pers = next((r.get("persistence_mse") for r in rows
                 if r.get("persistence_mse")), None)
    if pers is not None:
        out.append(("Persistência (controle)", "último valor observado",
                    "—", "—", _num(pers), _num(pers ** 0.5), "—",
                    "0,000 (def.)", "controle"))
    return out


def _sweep_section():
    rows, meta = _sweep_data()
    if not rows:
        return C.section(
            "Sweep do conjunto 1-SE (ao vivo)",
            C.note("Sem resultados ainda — a seção popula a cada trial "
                   "concluído por <code>scripts/experiments/lidar_8020_optuna7_sweep.py</code> "
                   "(resumo em <code>data/results/lidar_8020_optuna7_sweep/</code>)."),
            sub="O vigia <code>watch_8020_optuna7_report.sh</code> reconstrói "
                "o site a cada atualização do resumo.",
            sec_id="sweep-1se")
    totals = {k: v.get("n_total") for k, v in meta.items()} or EXPECTED_TOTALS
    total = sum(v for v in totals.values() if v)
    order = {"lstm": 0, "lstm_bi": 1}
    rows_sorted = sorted(rows, key=lambda r: (order.get(r["wrapper"], 9),
                                              r["val_objective"]))
    table_rows, highlight = [], []
    controls = _control_rows(rows)
    table_rows.extend(controls)
    ctrl_offset = len(controls)
    for i, r in enumerate(rows_sorted):
        p = r["params"]
        marks = []
        if r["is_1se_chosen"]:
            marks.append("1-SE")
            highlight.append(i + ctrl_offset)
        if r["is_argmin"]:
            marks.append("argmin")
            if (i + ctrl_offset) not in highlight:
                highlight.append(i + ctrl_offset)
        table_rows.append((
            f"{r['wrapper']} #{r['trial']}",
            f"lr {_num(p['learning_rate'], 5)} · "
            f"{int(p.get('lstm_units', p.get('filters', 0)))}u · "
            f"drop {_num(p.get('encoder_dropout_rate', 0), 2)}/"
            f"{_num(p.get('decoder_dropout_rate', 0), 2)}",
            _num(r["val_objective"]), _num(r["val_skill"], 3),
            _num(r["test"]["mse"]), _num(r["test"]["rmse"]),
            _num(r["test"]["mae"]),
            _num(r["test_skill_persistence"], 3),
            "+".join(marks) or "—"))
    best_row = min(rows, key=lambda r: r["test"]["mse"])
    body = C.figure("figures/sweep_1se.png",
                    "Objetivo na validação (72/8) vs MSE no teste (80/20) "
                    "para os trials já treinados do conjunto 1-SE; linha "
                    "tracejada = persistência na mesma grade do teste.",
                    label="fig:sweep_1se_val_teste")
    body += C.table(
        ["Trial", "Config (lr · units · drop enc/dec)", "Obj. val",
         "Skill val", "MSE teste", "RMSE", "MAE", "Skill pers.", "Marca"],
        table_rows,
        f"Sweep do conjunto 1-SE ({len(rows)}/{total} trials treinados) "
        f"vs controles na MESMA grade de origens do teste (1.416 origens × "
        f"36 passos): LSTM do artigo SBSR (run 5, featurex 67 col — o "
        f"artigo publica MSE 1,3928 na grade com trim 72) e persistência. "
        f"Cada trial da banda 1-SE retreinado com o protocolo da rodada "
        f"(80/20, 60 épocas, seed 42, MSE). Melhor MSE até agora: "
        f"{_num(best_row['test']['mse'])} "
        f"({best_row['wrapper']} #{best_row['trial']}).",
        formats=("sbsr", "steam", "generic"), label="tab:sweep_1se",
        highlight_rows=highlight,
        note="Controles: LSTM do artigo (trained_8020_sbsr, re-avaliada na "
             "grade do sweep) e persistência (último valor observado). "
             "Diagnóstico de estabilidade da região 1-SE — NÃO é seleção: "
             "escolher trial pelo teste seria vazamento. A promoção continua "
             "segundo a regra 1-SE na validação.",
    )
    return C.section(
        "Sweep do conjunto 1-SE (ao vivo)", body,
        sub=f"Atualiza a cada run: {len(rows)} de {total} trials "
            f"(lstm {sum(1 for r in rows if r['wrapper'] == 'lstm')}"
            f"/{totals.get('lstm', '?')}, lstm_bi "
            f"{sum(1 for r in rows if r['wrapper'] == 'lstm_bi')}"
            f"/{totals.get('lstm_bi', '?')}). Fonte: sweep.json, lido a "
            "cada build da página.",
        sec_id="sweep-1se")


REPORT, _base_build = report_module(
    name="8020-optuna7",
    title="LiDAR 80/20 — HP do Optuna (round 7)",
    description="Modelos 80/20 treinados com os hiperparâmetros da rodada "
                "Optuna 7 (skill estável λ=0,3, SHP, seleção 1-SE "
                "conservadora, clipnorm 1,0, wd fixo 1e-4, MSE em todos).",
    source_dir="data/results/report_8020_optuna7",
    subtitle="Mesma reamostragem 80/20 do lidar/8020 (60 épocas, seed 42), "
             "agora com os hiperparâmetros do Optuna round 7 — objetivo "
             "skill−0,3·std(blocos) na validação (72/8/20), pruner "
             "SuccessiveHalving (rungs 12/36) sobre o objetivo real, espaço "
             "sem cantos agressivos e seleção 1-SE com desempate "
             "conservador antes da promoção.",
    badges=("80/20", "Optuna round 7", "λ=0,3", "1-SE", "MSE"),
    regenerate=[".venv/bin/python", "reports/src/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_8020_optuna7",
                "--out-dir", "data/results/report_8020_optuna7",
                "--no-baselines"],
    retrain=[".venv/bin/python", "scripts/experiments/lidar_8020_optuna7_train.py",
             "--static-dilations"],
    info=[("Protocolo", "80/20 sem validação"),
          ("Hiperparâmetros", "Optuna round 7 (1-SE conservador)"),
          ("Objetivo HPO", "skill − 0,3·std(blocos) vs ARIMA (72/8/20)"),
          ("Pruner", "SuccessiveHalving min_resource=12 η=3 (rungs 12/36)"),
          ("Loss", "MSE em todos os wrappers"),
          ("Regularização", "clipnorm 1,0 · wd fixo 1e-4 · dropout ≥ 0,05"),
          ("Dilatações", "estáticas C′ (encoder [1,2,4,8,16,32,36])"),
          ("Épocas", "60 fixas"), ("Seed", "42"),
          ("Modelos", "trained_8020_optuna7")],
)


def build(ctx):
    return _base_build(ctx) + _sweep_section()


if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
