"""Relatório homologado — resultados canônicos do paper (75/20/5).

Protocolo original com early stopping na validação (20%) e teste nos 5%
finais; modelos em pipeline/tmp/trained_compare_homologado. É a fonte das
figuras do paper (scripts/paper_sync.py).

    python reports/lidar/homologado/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="homologado",
    title="LiDAR homologado (75/20/5)",
    description="Protocolo canônico com early stopping — fonte das figuras "
                "do paper (paper_sync).",
    source_dir="data/results/report_homologado",
    subtitle="Protocolo homologado: 75% treino / 20% validação / 5% teste, "
             "com early stopping e restauração do melhor val_loss. Alimenta "
             "as figuras do artigo via <code>scripts/paper_sync.py</code>.",
    badges=("75/20/5", "early stopping", "paper"),
    regenerate=[".venv/bin/python", "tests/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_compare_homologado",
                "--out-dir", "data/results/report_homologado",
                "--no-baselines"],
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
