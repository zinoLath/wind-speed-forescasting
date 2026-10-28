"""Relatório de resultados — Optuna Round 4 (skill vs ARIMA).

Rodada distribuída no Kaggle (2 kernels) sob o protocolo 80/20 do artigo:
split 70/10/20 (validação de HPO = 10% entre treino e teste), objetivo
skill vs ARIMA na validação (cache anti-vazamento), loss por modelo
(LSTM→ITILDE-Q, TCN→MSE) e dilatações estáticas C′. Os melhores trials são
promovidos para pipeline/tmp/optuna/ e alimentam a nova versão do lidar 8020.

Fontes:
    pipeline/tmp/optuna_round4/<Wrapper>/{trials.csv,best_trial.json,optuna.db}

    python reports/resultados/optuna-round4/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from reports.src import components as C  # noqa: E402

STUDY_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "optuna_round4b"
FIG_DIR = Path(__file__).resolve().parent / "figures"
WRAPPERS = {"Seq2Seq_TCN": "TCN"}
STUDY_NAME = "round4b"

plt.rcParams.update({
    "figure.dpi": 130, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.25,
})


def load_trials():
    out = {}
    for name, label in WRAPPERS.items():
        csv = STUDY_DIR / name / "trials.csv"
        if csv.is_file():
            out[name] = {"label": label, "df": pd.read_csv(csv)}
    return out


def best_table(data):
    rows = []
    for name, d in data.items():
        df = d["df"]
        ok = df[df["state"] == "COMPLETE"].sort_values("value")
        if not len(ok):
            continue
        best = ok.iloc[0]
        skill = -float(best["value"])
        params = {k[len("params_"):]: v for k, v in best.items()
                  if k.startswith("params_")}
        rows.append((d["label"], skill, params, int(len(df)), best))
    return rows


def fmt_params(params):
    keep = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, float):
            v = round(v, 5)
        keep.append(f"{k}={v}")
    return ", ".join(keep)


def fig_running_best(data):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for name, d in data.items():
        df = d["df"]
        ok = df[df["state"] == "COMPLETE"].sort_values("number")
        if not len(ok):
            continue
        running = (-ok["value"]).cummax()  # skill acumulado do melhor
        ax.plot(ok["number"], running, lw=1.8, marker="o", ms=2,
                label=d["label"])
    ax.set_xlabel("trial")
    ax.set_ylabel("melhor skill acumulado (vs ARIMA; maior = melhor)")
    ax.set_title("Evolução do melhor skill por wrapper (round 4)")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "round4_running_best.png")
    plt.close(fig)


def fig_importance():
    """Importância dos hiperparâmetros por estudo (sqlite local)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (name, label) in zip(axes.ravel(), WRAPPERS.items()):
        db = STUDY_DIR / name / "optuna.db"
        if not db.is_file():
            ax.axis("off")
            continue
        try:
            study = optuna.load_study(
                study_name=STUDY_NAME, storage=f"sqlite:///{db}")
            imp = optuna.importance.get_param_importances(study)
        except Exception as error:  # noqa: BLE001
            ax.axis("off")
            ax.set_title(f"{label}: importância indisponível\n{error}",
                         fontsize=8)
            continue
        imp = dict(sorted(imp.items(), key=lambda kv: kv[1])[-8:])
        ax.barh(list(imp.keys()), list(imp.values()), color="#00587a")
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("Importância dos hiperparâmetros (fANOVA/TPE) — round 4",
                 fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "round4_importance.png")
    plt.close(fig)


def build(ctx):  # noqa: ARG001
    data = load_trials()
    if not data:
        return C.page_header(
            "Optuna Round 4b — TCN (skill vs ARIMA)",
            "Sem dados locais.") + C.note(
            "Rode <code>scripts/dist/colab_round2.py collect --kaggle</code> "
            "para trazer os resultados dos kernels.")

    rows = best_table(data)
    best_skill = max((r[1] for r in rows), default=0.0)
    total_trials = sum(len(d["df"]) for d in data.values())
    fig_running_best(data)
    fig_importance()

    parts = [C.page_header(
        "Optuna Round 4b — TCN (skill vs ARIMA)",
        "Rodada de hiperparâmetros sob o protocolo 80/20 do artigo: "
        "validação de 10% (split 70/10/20), objetivo skill vs ARIMA com "
        "cache anti-vazamento, loss por modelo e dilatações estáticas.",
        badges=("Kaggle ×2", "120 trials/wrapper", "70/10/20",
                "skill vs ARIMA")),
        C.cards([
            (len(data), "wrappers otimizados"),
            (total_trials, "trials totais"),
            (f"+{best_skill:.3f}", "melhor skill (val)"),
            ("2", "kernels Kaggle"),
        ])]

    # ---- protocolo
    parts.append(C.section(
        "Protocolo e espaço de busca",
        "<div class='markdown'>"
        "<p><strong>Split 70/10/20</strong>: teste de 20% intocado; a "
        "validação do HPO são os 10% entre treino e teste (mimica o regime "
        "do teste). <strong>Objetivo</strong>: média em 4 blocos de "
        "<code>1 − MSE_modelo/MSE_ARIMA</code> (peso de horizonte) na "
        "validação, com as predições do ARIMA pré-computadas "
        "(<code>arima_val_cache.csv</code>, ordem por AIC só no treino). "
        "<strong>Loss por modelo</strong>: LSTM/LSTM-Bi → "
        "<code>ITILDE-Q</code>; TCN/TCN-Bi → <code>MSE</code>. "
        "<strong>Dilatações estáticas C′</strong>: encoder "
        "<code>[1,2,4,8,16,32,36]</code>, decoder "
        "<code>[1,2,4,8,16]</code>; o HP é a profundidade (prefixo).</p>"
        "<p><strong>Espaço</strong>: <code>learning_rate</code> (log, "
        "1e-4–2e-2), <code>weight_decay</code> {0·1e-5·1e-4·3e-4·1e-3}, "
        "units/filters 48–252 (passo 4), dropouts 0–0,4, e (TCN) "
        "<code>kernel_size</code> {2,3}, <code>nb_stacks</code> {1,2}, "
        "profundidade de dilatação {4..7}/{3..5}.</p></div>",
        sub="Fontes verificadas: skill score (FPP3 §5.9); teste DM "
            "(Diebold–Mariano 1995; Harvey–Leybourne–Newbold 1997); "
            "ITILDE-Q (Lee et al. 2022, arXiv:2210.15050; DONG Sheng et al. "
            "2026).",
        sec_id="protocolo"))

    # ---- melhores trials
    skill_rows = [(r[0], f"+{r[1]:.4f}", str(r[3])) for r in rows]
    param_rows = [(r[0], fmt_params(r[2])) for r in rows]
    parts.append(C.section(
        "Melhores trials (promovidos)",
        C.table(["Wrapper", "Skill (val)", "Trials"],
                skill_rows,
                "Melhor skill vs ARIMA na validação e nº de trials por "
                "wrapper.",
                label="tab:round4_best_skill",
                highlight_rows=[max(range(len(rows)),
                                    key=lambda i: rows[i][1])]
                if rows else None)
        + C.table(["Wrapper", "Melhores hiperparâmetros"], param_rows,
                  "Hiperparâmetros do melhor trial por wrapper (promovidos "
                  "para pipeline/tmp/optuna/).",
                  label="tab:round4_best_params"),
        sub="Estes parâmetros alimentam a nova versão do treino lidar 8020 "
            "(pipeline/tmp/optuna/ → estágio train).",
        sec_id="melhores"))

    # ---- evolução + importância
    parts.append(C.section(
        "Evolução e importância",
        C.figure("figures/round4_running_best.png",
                 "Melhor skill acumulado ao longo dos trials por wrapper.",
                 label="fig:round4_running_best", width=1.0)
        + C.figure("figures/round4_importance.png",
                   "Importância dos hiperparâmetros (fANOVA/TPE) por "
                   "wrapper.",
                   label="fig:round4_importance", width=1.0),
        sub="O running-best mostra a convergência do TPE; a importância "
            "revela quais dimensões moveram o skill (tamanho, lr, "
            "profundidade de dilatação…).",
        sec_id="evolucao"))

    return "".join(parts)


REPORT = {
    "title": "Optuna Round 4b — TCN (skill vs ARIMA)",
    "description": "Rodada distribuída no Kaggle (2 kernels) com objetivo "
                   "skill vs ARIMA, split 70/10/20, loss por modelo e "
                   "dilatações estáticas.",
    "regenerate": None,  # a página lê trials.csv/optuna.db locais
    "retrain": None,
    "source_dir": "pipeline/tmp/optuna_round4b",
    "info": [
        ("Objetivo", "skill vs ARIMA (4 blocos, peso de horizonte)"),
        ("Split", "70/10/20 (teste intocado)"),
        ("Loss", "LSTM→ITILDE-Q · TCN→MSE"),
        ("Dilatações", "estáticas C′ + profundidade"),
        ("Coleta", "scripts/dist/colab_round2.py collect --kaggle --promote"),
    ],
}

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
