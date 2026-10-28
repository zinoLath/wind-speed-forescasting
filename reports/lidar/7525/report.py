"""Relatório de resultados — LiDAR 75/25 (split parametrizável).

75% treino / 25% teste sem validação (scripts/experiments/lidar_7525_train.py
--train-ratio 0.75 -> pipeline/tmp/trained_7525).

    python reports/lidar/7525/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="7525",
    title="LiDAR 75/25",
    description="Split 75/25 sem validação, 60 épocas, seed 42.",
    source_dir="data/results/report_7525",
    subtitle="Protocolo: 75% treino / 25% teste, sem validação, 60 épocas "
             "fixas, seed 42. Mesma janela (72 → 36 passos de 10 min).",
    badges=("75/25", "60 épocas", "seed 42"),
    regenerate=[".venv/bin/python", "reports/src/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_7525",
                "--out-dir", "data/results/report_7525", "--no-baselines"],
    retrain=[".venv/bin/python", "scripts/experiments/lidar_7525_train.py",
             "--train-ratio", "0.75"],
    info=[("Protocolo", "75/25 sem validação"), ("Épocas", "60 fixas"),
          ("Seed", "42"), ("Janela", "72 → 36 passos (10 min)"),
          ("Modelos", "trained_7525")],
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
