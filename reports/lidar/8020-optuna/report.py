"""Relatório de resultados — 80/20 com os melhores hiperparâmetros do Optuna.

Mesmo protocolo do lidar/8020 (80/20 sem validação, 60 épocas fixas, seed 42),
mas com os hiperparâmetros vencedores da rodada **Optuna round 4**
(objetivo skill vs ARIMA; promovidos para pipeline/tmp/optuna/) e com a
política de loss por modelo (LSTM→ITILDE-Q, TCN→MSE) e dilatações estáticas
C′ nos TCN.

    python reports/lidar/8020-optuna/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="8020-optuna",
    title="LiDAR 80/20 — HP do Optuna (round 4)",
    description="Modelos 80/20 treinados com os melhores hiperparâmetros da "
                "rodada Optuna 4 (skill vs ARIMA), loss por modelo e "
                "dilatações estáticas C′.",
    source_dir="data/results/report_8020_optuna",
    subtitle="Mesma reamostragem 80/20 do lidar/8020 (60 épocas, seed 42), "
             "agora com os hiperparâmetros do Optuna round 4 — objetivo "
             "skill vs ARIMA na validação (70/10/20), loss por modelo e "
             "dilatações estáticas C′ nos TCN.",
    badges=("80/20", "Optuna round 4", "skill vs ARIMA"),
    regenerate=[".venv/bin/python", "tests/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_8020_optuna",
                "--out-dir", "data/results/report_8020_optuna",
                "--no-baselines"],
    retrain=[".venv/bin/python", "scripts/lidar_7525_train.py",
             "--train-ratio", "0.80",
             "--output-dir", "pipeline/tmp/trained_8020_optuna",
             "--static-dilations"],
    info=[("Protocolo", "80/20 sem validação"),
          ("Hiperparâmetros", "Optuna round 4 (skill vs ARIMA)"),
          ("Loss", "LSTM→ITILDE-Q · TCN→MSE"),
          ("Dilatações", "estáticas C′ (encoder [1,2,4,8,16,32,36])"),
          ("Épocas", "60 fixas"), ("Seed", "42"),
          ("Modelos", "trained_8020_optuna")],
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
