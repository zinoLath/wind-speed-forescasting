"""Relatório de resultados — 80/20 com loss MSE pura (ablação da loss).

Mesmo protocolo 80/20 da reamostragem canônica, mas com MSE pura em vez da
horizon-weighted (scripts/lidar_8020_mseloss_train.py -> trained_8020_mseloss).

    python reports/lidar/8020-mseloss/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="8020-mseloss",
    title="LiDAR 80/20 — loss MSE",
    description="Ablação da loss: MSE pura vs horizon-weighted no split 80/20.",
    source_dir="data/results/report_8020_mseloss",
    subtitle="Ablação: mesma reamostragem 80/20 (60 épocas, seed 42), com MSE "
             "pura em vez da horizon-weighted — isola o efeito da função de "
             "perda.",
    badges=("80/20", "ablação loss", "MSE pura"),
    regenerate=[".venv/bin/python", "tests/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_8020_mseloss",
                "--out-dir", "data/results/report_8020_mseloss",
                "--no-baselines"],
    retrain=[".venv/bin/python", "scripts/lidar_8020_mseloss_train.py"],
    info=[("Protocolo", "80/20 sem validação"), ("Loss", "MSE pura (ablação)"),
          ("Épocas", "60 fixas"), ("Seed", "42"),
          ("Modelos", "trained_8020_mseloss")],
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
