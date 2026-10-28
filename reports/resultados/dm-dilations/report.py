"""Relatório — DM por horizonte: dilatações TCN (legado × C′).

Isola o efeito das dilatações estáticas C′ nos TCN, mantendo TODOS os demais
fatores do treino canônico (HPs do round 1, denoise ws100 nível 2, loss
horizon-weighted, 60 épocas, seed 42). Compara, horizonte a horizonte, os
erros de dil. legado (trained_8020) × dil. C′ (trained_8020_tcn_cprime) via
Diebold–Mariano.

Fonte: data/results/dm_dilations/ (studies/dm_vs_hpo.py).

    python reports/resultados/dm-dilations/report.py
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402

SRC = PROJECT_ROOT / "data" / "results" / "dm_dilations"
FIG_DIR = Path(__file__).resolve().parent / "figures"


def build(ctx):  # noqa: ARG001
    data_path = SRC / "dm_matrix.json"
    if not data_path.is_file():
        return C.page_header(
            "DM por horizonte — dilatações TCN (legado × C′)",
            "Sem dados.") + C.note(
            "Rode <code>python scripts/experiments/tcn_cprime_train.py</code> e "
            "<code>python studies/dm_vs_hpo.py --a pipeline/tmp/trained_8020 "
            "--b pipeline/tmp/trained_8020_tcn_cprime --wrappers tcn tcn_bi "
            "--out data/results/dm_dilations --a-label \"dil. legado\" "
            "--b-label \"dil. C′\"</code>.")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    a_label = payload.get("a_label", "legado")
    b_label = payload.get("b_label", "C′")
    n = next(iter(payload["pairs"].values()))["n_origins"]

    FIG_DIR.mkdir(exist_ok=True)
    shutil.copy2(SRC / "figures" / "dm_heatmap.png",
                 FIG_DIR / "dm_heatmap.png")

    rows = []
    for label, pair in payload["pairs"].items():
        dm = pair["dm"]
        h = payload["horizons"]
        early = sum(v for v, hh in zip(dm, h) if hh <= 12) / 12
        late = sum(v for v, hh in zip(dm, h) if hh > 12) / 24
        mean = sum(dm) / len(dm)
        verdict = f"{b_label} melhor" if mean > 0 else f"{a_label} melhor"
        rows.append((label, str(pair["n_origins"]), str(pair["sig_count"]),
                     f"{early:+.2f}", f"{late:+.2f}", f"{mean:+.2f}", verdict))

    return "".join([C.page_header(
        "DM por horizonte — dilatações TCN (legado × C′)",
        "Efeito ISOLADO das dilatações estáticas C′ nos TCN: HPs do round 1, "
        "mesmo pipeline (denoise ws100 nível 2), mesma loss (hwmse) e mesmo "
        "split 80/20 — só muda a sequência de dilatações do bloco temporal.",
        badges=(f"{n} origens", "DM + HLN", "só dilatação")),
        C.cards([
            (len(payload["pairs"]), "models TCN comparados"),
            (f"{n}", "origens (mesma grade)"),
            (str(sum(p["sig_count"] for p in payload["pairs"].values())),
             "horizontes significativos (Σ)"),
            ("C′", "encoder [1,2,4,8,16,32,36]"),
        ]),
        C.section(
            "Heatmap DM(legado, C′) por horizonte",
            C.figure("figures/dm_heatmap.png",
                     f"DM por horizonte: vermelho (+) ⇒ {a_label} pior; azul "
                     f"(−) ⇒ {b_label} pior; * ⇒ p<0,05.",
                     label="fig:dm_dilations_heatmap", width=1.0)
            + C.table(["Wrapper", "Origens", "Horiz. sig.",
                       "DM médio h1–12", "DM médio h13–36", "DM médio",
                       "Veredito"], rows,
                      "Resumo do DM por horizonte da mudança de dilatações.",
                      label="tab:dm_dilations_summary",
                      note="Convenção: DM(loss_legado − loss_C′); positivo = "
                           "perda do legado maior. HAC h−1 + HLN "
                           "(src/utils)."),
            sub="Resultado: a TCN **unidirecional** fica levemente pior com "
                "C′ no longo horizonte (4/36 sig.); a **TCN bidirecional** "
                "melhora em TODOS os horizontes (31/36 sig.) — o ganho da "
                "dilatação estática é específico da variante bidirecional.",
            sec_id="heatmap")])


REPORT = {
    "title": "DM por horizonte — dilatações TCN (legado × C′)",
    "description": "Efeito isolado das dilatações estáticas C′ nos TCN "
                   "(HPs antigos, mesmo pipeline e loss).",
    "regenerate": [".venv/bin/python", "studies/dm_vs_hpo.py",
                   "--a", "pipeline/tmp/trained_8020",
                   "--b", "pipeline/tmp/trained_8020_tcn_cprime",
                   "--wrappers", "tcn", "tcn_bi",
                   "--out", "data/results/dm_dilations",
                   "--a-label", "dil. legado", "--b-label", "dil. C'"],
    "retrain": [".venv/bin/python", "scripts/experiments/tcn_cprime_train.py"],
    "source_dir": "data/results/dm_dilations",
    "info": [
        ("Comparação", "dil. legado × dil. C′ (TCN, TCN-Bi)"),
        ("Controle", "HPs round 1, hwmse, ws100 nível 2, seed 42"),
        ("Métrica", "DM por horizonte (HAC h−1 + HLN)"),
        ("Origens", "mesma grade alinhada (1203)"),
    ],
}

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
