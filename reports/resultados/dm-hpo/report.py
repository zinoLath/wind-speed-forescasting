"""Relatório — Diebold–Mariano por horizonte: HP antigos × HP do Optuna.

Compara, horizonte a horizonte, os erros dos modelos treinados com os HPs
ANTIGOS (trained_8020) contra os do Optuna round 4 (trained_8020_optuna),
na mesma grade de origens (1203) do split 80/20. O heatmap é o análogo do
"erro por horizonte", mas mostrando DM(modelo_antigo, modelo_hpo).

Fonte: data/results/dm_vs_hpo/ (gerado por tests/dm_vs_hpo.py).

    python reports/resultados/dm-hpo/report.py
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import components as C  # noqa: E402

SRC = PROJECT_ROOT / "data" / "results" / "dm_vs_hpo"
FIG_DIR = Path(__file__).resolve().parent / "figures"


def build(ctx):  # noqa: ARG001
    data_path = SRC / "dm_matrix.json"
    if not data_path.is_file():
        return C.page_header(
            "DM por horizonte — antigo × Optuna",
            "Sem dados.") + C.note(
            "Rode <code>python tests/dm_vs_hpo.py</code> (GPU, ~10 min) "
            "para gerar <code>dm_matrix.json</code> e o heatmap.")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    n = next(iter(payload["pairs"].values()))["n_origins"]

    FIG_DIR.mkdir(exist_ok=True)
    shutil.copy2(SRC / "figures" / "dm_vs_hpo_heatmap.png",
                 FIG_DIR / "dm_vs_hpo_heatmap.png")

    rows = []
    for label, pair in payload["pairs"].items():
        dm = pair["dm"]
        early = [v for v, h in zip(dm, payload["horizons"]) if h <= 12]
        late = [v for v, h in zip(dm, payload["horizons"]) if h > 12]
        mean = sum(dm) / len(dm)
        verdict = "HPO melhor" if mean > 0 else "antigo melhor"
        rows.append((label, str(pair["n_origins"]), str(pair["sig_count"]),
                     f"{sum(early)/len(early):+.2f}",
                     f"{sum(late)/len(late):+.2f}", f"{mean:+.2f}", verdict))

    parts = [C.page_header(
        "Diebold–Mariano por horizonte — antigo × Optuna",
        "Cada célula é a estatística DM entre o modelo de HP antigo e o do "
        "Optuna round 4, no mesmo horizonte, sobre a mesma grade de origens "
        "do split 80/20. Substitui o heatmap de erro por um heatmap de "
        "significância da diferença.",
        badges=(f"{n} origens", "DM + HLN", "80/20")),
        C.cards([
            (len(payload["pairs"]), "wrappers comparados"),
            (f"{n}", "origens (mesma grade)"),
            (str(sum(p["sig_count"] for p in payload["pairs"].values())),
             "horizontes significativos (Σ)"),
            ("36", "horizontes (6 h)"),
        ])]

    parts.append(C.section(
        "Heatmap DM(modelo_antigo, modelo_hpo) por horizonte",
        C.figure("figures/dm_vs_hpo_heatmap.png",
                 "DM por horizonte: vermelho (+) ⇒ HP antigos piores; azul "
                 "(−) ⇒ HPs do Optuna piores; * ⇒ p<0,05.",
                 label="fig:dm_hpo_heatmap", width=1.0)
        + C.table(["Wrapper", "Origens", "Horiz. sig.", "DM médio h1–12",
                   "DM médio h13–36", "DM médio", "Veredito"],
                  rows,
                  "Resumo do DM por horizonte entre as duas configurações de "
                  "hiperparâmetros.",
                  label="tab:dm_hpo_summary",
                  note="Convenção: DM(loss_antigo − loss_hpo); positivo = "
                       "perda antiga maior. HAC com defasagem h−1 e correção "
                       "de Harvey–Leybourne–Newbold (src/utils)."),
        sub="Vermelho ⇒ os HPs antigos são piores naquele horizonte; azul ⇒ "
            "os HPs do Optuna são piores. O asterisco marca p<0,05 de que o "
            "sinal é detectável. Use para ver ONDE (curto vs longo) cada "
            "conjunto de HPs ganha — o HPO mira o fim da janela.",
        sec_id="heatmap"))

    return "".join(parts)


REPORT = {
    "title": "DM por horizonte — antigo × Optuna",
    "description": "Heatmap Diebold–Mariano por horizonte comparando os HP "
                   "antigos (trained_8020) com os do Optuna round 4 "
                   "(trained_8020_optuna).",
    "regenerate": [".venv/bin/python", "tests/dm_vs_hpo.py"],
    "retrain": None,
    "source_dir": "data/results/dm_vs_hpo",
    "info": [
        ("Antigo", "trained_8020 (HP round 1, hwmse)"),
        ("Optuna", "trained_8020_optuna (HP round 4)"),
        ("Métrica", "DM por horizonte (HAC h−1 + HLN)"),
        ("Origens", "mesma grade alinhada (1203)"),
    ],
}

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
