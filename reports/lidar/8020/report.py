"""Relatório de resultados — protocolo de reamostragem 80/20 (LiDAR).

Split 80/20 sem validação, 60 épocas fixas, seed 42 (scripts/lidar_7525_train.py
--train-ratio 0.80 -> pipeline/tmp/trained_8020). Página padronizada a partir
de data/results/report_8020 (evaluation.json + images/).

    python reports/lidar/8020/report.py   # reconstrói a página (todo o site)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="8020",
    title="LiDAR 80/20 (reamostragem)",
    description="Split 80/20 sem validação, 60 épocas, seed 42 — lstm, "
                "lstm_bi, tcn, tcn_bi.",
    source_dir="data/results/report_8020",
    subtitle="Protocolo de reamostragem: 80% treino / 20% teste, sem partição "
             "de validação, orçamento fixo de 60 épocas, seed 42. Entrada de "
             "72 passos (12 h) → horizonte de 36 passos (6 h), resolução de "
             "10 min.",
    badges=("80/20", "60 épocas", "seed 42", "sem validação"),
    protocol_note="Modelos treinados em <code>pipeline/tmp/trained_8020</code> "
                  "por <code>scripts/lidar_7525_train.py --train-ratio 0.80</"
                  "code>. Avaliação gerada por "
                  "<code>tests/relatorio_resultados.py</code>.",
    regenerate=[".venv/bin/python", "tests/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_8020",
                "--out-dir", "data/results/report_8020", "--no-baselines"],
    retrain=[".venv/bin/python", "scripts/lidar_7525_train.py",
             "--train-ratio", "0.80"],
    info=[("Protocolo", "80/20 sem validação"), ("Épocas", "60 fixas"),
          ("Seed", "42"), ("Janela", "72 → 36 passos (10 min)"),
          ("Modelos", "trained_8020")],
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
