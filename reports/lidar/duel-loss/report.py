"""Relatório de resultados — duelos de loss (fan vs per-horizon).

Experimentos de perda alternativas (pipeline/tmp/fan_duel_loss.py e
perhorizon_duel.py); sem retreino automático configurado — regenere os
dados pelos scripts originais e use "Reconstruir página".

    python reports/lidar/duel-loss/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="duel-loss",
    title="LiDAR duelos de loss",
    description="Comparação de funções de perda alternativas (fan/per-horizon).",
    source_dir="data/results/report_duel_loss",
    subtitle="Duelos de função de perda gerados por <code>pipeline/tmp/"
             "fan_duel_loss.py</code> e <code>perhorizon_duel.py</code> — "
             "sem retreino automático no hub.",
    badges=("duelo loss", "80/20"),
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
