"""Relatório de resultados — origens dos hiperparâmetros da LSTM-Bi.

Rastreia de onde vieram os hp da lstm_bi (treinos lstm_bi_* em
pipeline/tmp) e o impacto de cada origem nas métricas.

    python reports/lidar/lstm-bi-origins/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="lstm-bi-origins",
    title="LiDAR LSTM-Bi — origens dos hp",
    description="Rastreio das origens dos hiperparâmetros da LSTM "
                "bidirecional.",
    source_dir="data/results/report_lstm_bi_origins",
    subtitle="Prova de origem dos hiperparâmetros da LSTM bidirecional "
             "(treinos <code>pipeline/tmp/trained_8020_lstm_bi*</code>).",
    badges=("lstm_bi", "hiperparâmetros", "80/20"),
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
