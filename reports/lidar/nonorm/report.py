"""Relatório de resultados — ablação sem normalização (75/20/5).

Modelos em pipeline/tmp/trained_compare_nonorm (mesmo protocolo homologado,
sem normalização das entradas).

    python reports/lidar/nonorm/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="nonorm",
    title="LiDAR sem normalização",
    description="Ablação: protocolo homologado sem normalizar as entradas.",
    source_dir="data/results/report_nonorm",
    subtitle="Ablação do protocolo homologado (75/20/5) removendo a "
             "normalização das features — mede o quanto o escalonamento "
             "contribui.",
    badges=("75/20/5", "ablação", "sem normalização"),
    regenerate=[".venv/bin/python", "tests/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_compare_nonorm",
                "--out-dir", "data/results/report_nonorm",
                "--no-baselines"],
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
