"""Relatório de treinamento — reamostragem 80/20 (LiDAR).

Documenta o treino dos 4 wrappers sob o protocolo de reamostragem 80/20
(sem validação, 60 épocas fixas, seed 42): hiperparâmetros efetivos, curvas
de loss por época (train dashboard), tempo/época, learning rate e avaliação
inline do treino (MSE/skill vs persistência).

Fontes:
    pipeline/tmp/trained_8020/<wrapper>/model.json
    pipeline/tmp/train_dashboard/<run>.jsonl   (último run por wrapper)
    data/results/lidar_8020_features/all.json  (avaliação inline do treino)

    python reports/treinamento/8020/report.py
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

MODELS_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "trained_8020"
DASH_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "train_dashboard"
EVAL_JSON = PROJECT_ROOT / "data" / "results" / "lidar_8020_features" / "all.json"
FIG_DIR = Path(__file__).resolve().parent / "figures"

WRAPPERS = ("lstm", "lstm_bi", "tcn", "tcn_bi")
LABELS = {"lstm": "LSTM", "lstm_bi": "LSTM bidirecional",
          "tcn": "TCN", "tcn_bi": "TCN bidirecional"}
COLORS = {"lstm": "#1f77b4", "lstm_bi": "#9467bd",
          "tcn": "#ff7f0e", "tcn_bi": "#d62728"}

plt.rcParams.update({
    "figure.dpi": 130, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.25,
})


def load_models() -> dict:
    out = {}
    for key in WRAPPERS:
        meta_path = MODELS_DIR / key / "model.json"
        if meta_path.is_file():
            out[key] = json.loads(meta_path.read_text(encoding="utf-8"))
    return out


def load_curves() -> dict:
    """Último run por wrapper no dashboard de treino (jsonl por época)."""
    runs = {}
    if not DASH_DIR.is_dir():
        return runs
    for path in DASH_DIR.glob("*.jsonl"):
        try:
            first = json.loads(path.read_text(encoding="utf-8")
                               .splitlines()[0])
        except Exception:  # noqa: BLE001
            continue
        wrapper = first.get("wrapper", "")
        key = wrapper.replace("Seq2Seq_", "").replace("_", "_").lower()
        # chave canonica: lstm, lstm_bi, tcn, tcn_bi
        for cand in WRAPPERS:
            if wrapper.lower().endswith(cand) or \
                    wrapper.replace("Seq2Seq_", "").lower() == cand:
                key = cand
                break
        mtime = path.stat().st_mtime
        if key not in runs or mtime > runs[key][0]:
            runs[key] = (mtime, path)
    curves = {}
    for key, (_, path) in runs.items():
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if rec.get("status") == "epoch":
                rows.append(rec)
        curves[key] = rows
    return curves


def _fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def make_figures(curves: dict) -> list:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    if curves:
        # loss por época
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for key in WRAPPERS:
            rows = curves.get(key) or []
            if not rows:
                continue
            epochs = [r["epoch"] for r in rows]
            axes[0].plot(epochs, [r["loss"] for r in rows],
                         color=COLORS[key], lw=1.6, label=LABELS[key])
            sec = [r.get("epoch_sec") for r in rows if r.get("epoch_sec")]
            if sec:
                axes[1].plot(range(1, len(sec) + 1), sec, color=COLORS[key],
                             lw=1.4, label=LABELS[key])
        axes[0].set_xlabel("época")
        axes[0].set_ylabel("loss de treino (escala normalizada)")
        axes[0].set_title("Curva de loss por época — reamostragem 80/20")
        axes[0].legend()
        axes[1].set_xlabel("época")
        axes[1].set_ylabel("segundos por época")
        axes[1].set_title("Custo por época")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "loss_curvas.png", bbox_inches="tight")
        plt.close(fig)
        made.append("loss_curvas.png")

        # learning rate
        fig, ax = plt.subplots(figsize=(13, 3.6))
        for key in WRAPPERS:
            rows = [r for r in (curves.get(key) or [])
                    if r.get("lr") is not None]
            if not rows:
                continue
            ax.plot([r["epoch"] for r in rows], [r["lr"] for r in rows],
                    color=COLORS[key], lw=1.5, label=LABELS[key])
        ax.set_xlabel("época")
        ax.set_ylabel("learning rate")
        ax.set_title("Agenda de learning rate (por wrapper)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "learning_rate.png", bbox_inches="tight")
        plt.close(fig)
        made.append("learning_rate.png")
    return made


def build(ctx):
    models = load_models()
    curves = load_curves()
    figs = make_figures(curves)
    evals = {}
    if EVAL_JSON.is_file():
        evals = json.loads(EVAL_JSON.read_text(encoding="utf-8"))

    first = next(iter(models.values()), None)
    ds = (first or {}).get("dataset", {})
    total_time = sum((m.get("training") or {}).get("training_time_sec", 0)
                     for m in models.values())

    parts = [C.page_header(
        "Treinamento — reamostragem 80/20",
        "Treino dos wrappers seq2seq sob o protocolo de reamostragem "
        "(80% treino / 20% teste, sem validação, 60 épocas fixas, seed 42), "
        "com curvas de loss por época e avaliação inline do treino.",
        badges=("80/20", "60 épocas", "seed 42",
                ds.get("split", "split ?"))),
        C.cards([
            (f"{len(models)}", "wrappers treinados"),
            (f"{ds.get('train_rows', '—'):,}".replace(",", "."),
             "linhas de treino"),
            (f"{ds.get('test_rows', '—'):,}".replace(",", "."),
             "linhas de teste"),
            (f"{total_time / 60:.0f} min", "tempo total de GPU"),
        ])]

    parts.append(C.section(
        "Protocolo de reamostragem",
        "<div class='markdown'><p>Split temporal <strong>80/20 sem "
        "partição de validação</strong>: 60 épocas de orçamento fixo, seed "
        "42, sem early stopping. Janela de entrada de 72 passos (12 h) e "
        "horizonte de 36 passos (6 h) em resolução de 10 min, alvo "
        "<code>ws100_wavelet</code> (wavelet sym18 nível 2) com métricas "
        "contra o <code>ws100</code> bruto. Comando: <code>python "
        "scripts/lidar_7525_train.py --train-ratio 0.80</code> → modelos em "
        "<code>pipeline/tmp/trained_8020</code>. Avaliação completa: "
        "<a href='../../lidar/8020/'>relatório LiDAR 80/20</a>.</p></div>",
        sec_id="protocolo"))

    # ---- hiperparâmetros efetivos
    if models:
        hp_keys = ["batch_size", "learning_rate", "units", "filters",
                   "kernel_size", "dropout", "dilation_rates", "layers",
                   "num_layers", "lr", "epochs"]
        present = []
        for k in hp_keys:
            if any(k in (m.get("hyperparameters") or {}) for m
                   in models.values()):
                present.append(k)
        rows = []
        for key in WRAPPERS:
            if key not in models:
                continue
            m = models[key]
            hp = m.get("hyperparameters") or {}
            tr = m.get("training") or {}
            row = [LABELS[key], m.get("loss", "—"),
                   f"{tr.get('epochs_run', '—')} ép",
                   f"{(tr.get('training_time_sec') or 0) / 60:.1f} min"
                   .replace(".", ","),
                   _fmt(tr.get("final_loss"))]
            row += [str(hp.get(k, "—")) for k in present]
            rows.append(tuple(row))
        cols = (["Wrapper", "Loss", "Épocas", "Tempo", "Loss final"]
                + present)
        parts.append(C.section(
            "Hiperparâmetros efetivos do treino",
            C.table(cols, rows,
                    "Hiperparâmetros e custos do treino por wrapper sob a "
                    "reamostragem 80/20.",
                    label="tab:hp_treinamento",
                    note=f"fonte: model.json de "
                         f"<code>{MODELS_DIR.relative_to(PROJECT_ROOT)}"
                         f"</code>; params_source="
                         f"{(first or {}).get('params_source', '?')}."),
            sub="Todos os wrappers partem dos hiperparâmetros vencedores "
                "do Optuna canônico (params_source optuna) e da mesma loss "
                "horizon-weighted, salvo ablação indicada — o que muda entre "
                "linhas é só a arquitetura. \"Loss final\" está na escala "
                "normalizada do treino, então comparável apenas entre "
                "modelos com a mesma função de perda; o julgamento de "
                "qualidade fica com a tabela de avaliação abaixo (escala "
                "física, m/s).",
            sec_id="hiperparametros"))

    # ---- curvas
    if figs:
        items = ""
        for f in figs:
            caption = ("Curvas de loss e custo por época — treino 80/20"
                       if f == "loss_curvas.png"
                       else "Agenda de learning rate por wrapper")
            items += C.figure(f"figures/{f}", caption,
                              label=f"fig:{f.replace('.png', '')}")
        parts.append(C.section("Curvas de treino (dashboard por época)",
                               C.grid(items),
                               sub="Cada linha é uma época registrada pelo "
                                   "dashboard (<code>pipeline/tmp/"
                                   "train_dashboard/*.jsonl</code>, último "
                                   "run completo por wrapper). Na curva de "
                                   "loss, o formato importa mais que o "
                                   "valor absoluto: queda rápida e platô "
                                   "precoce indica convergência fácil "
                                   "(risco de capacidade ociosa), enquanto "
                                   "degraus sugerem a agenda de learning "
                                   "rate agindo. No painel de custo por "
                                   "época, TCNs são mais baratas por época "
                                   "que as LSTMs bidirecionais (convolução "
                                   "paralelizável vs dois passes "
                                   "recorrentes) — some ao número de "
                                   "épocas para o custo total.",
                               sec_id="curvas"))

    # ---- avaliacao inline
    if evals:
        rows = []
        for key in WRAPPERS:
            ev = evals.get(key)
            if not ev:
                continue
            rows.append((LABELS.get(key, key),
                         _fmt(ev.get("mse"), 3), _fmt(ev.get("rmse"), 3),
                         _fmt(ev.get("mae"), 3), _fmt(ev.get("r2"), 3),
                         _fmt(ev.get("persistence_mse"), 3),
                         _fmt(ev.get("skill"), 3)))
        best = min((float(r[1].replace(",", ".")) for r in rows), default=0)
        highlight = [i for i, r in enumerate(rows)
                     if float(r[1].replace(",", ".")) == best]
        parts.append(C.section(
            "Avaliação inline do treino (vs ws100 bruto)",
            C.table(["Wrapper", "MSE (m/s)²", "RMSE (m/s)", "MAE (m/s)",
                     "R²", "MSE persistência", "Skill vs persistência"],
                    rows, "Avaliação inline do treino sob a reamostragem "
                    "80/20: métricas no conjunto de teste e skill vs "
                    "baseline de persistência.",
                    label="tab:avaliacao_treino", highlight_rows=highlight,
                    note="skill > 0 = melhor que persistência; "
                         "fonte: <code>data/results/"
                         "lidar_8020_features/all.json</code>."),
            sub="O <strong>skill</strong> é o termômetro honesto: 0 significa "
                "empate com a persistência (repetir a última observação) e "
                "valores negativos indicam que o modelo perde do ingênuo — "
                "comum em treinos curtos sem early stopping, pois a "
                "persistência é forte em séries de 10 min. Compare com "
                "runs anteriores pela variância de treino (seed/GPU), não "
                "como regressão determinística. A avaliação completa, com "
                "rolling, DM e diagnósticos, está no "
                "<a href='../../lidar/8020/'>relatório LiDAR 80/20</a>.",
            sec_id="avaliacao"))
    return "\n".join(parts)


REPORT = {
    "title": "Treinamento 80/20 (reamostragem)",
    "description": "Curvas de treino, hiperparâmetros e avaliação inline do "
                   "protocolo de reamostragem 80/20.",
    "regenerate": None,  # reconstruir página já re-gera as curvas
    "retrain": [".venv/bin/python", "scripts/lidar_7525_train.py",
                "--train-ratio", "0.80"],
    "source_dir": "pipeline/tmp/train_dashboard",
    "info": [("Protocolo", "80/20 sem validação"), ("Épocas", "60 fixas"),
             ("Seed", "42"), ("Janela", "72 → 36 passos (10 min)"),
             ("Loss", "horizon-weighted"), ("Modelos", "trained_8020")],
}

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
