"""Relatório da rodada FINAL do artigo SBSR — LiDAR 80/20.

Lineup canônico unificado (finding `rodada-final-sbsr`): TCN/TCN-Bi com
46 canônicas + humid/temp (48 col), LSTM/LSTM-Bi com featurex (67 col),
MSE pura em todos, HPs canônicos, seed 42, 60 épocas, baselines na mesma
grade (scripts/lidar_8020_sbsr_train.py → pipeline/tmp/trained_8020_sbsr).
Página padronizada a partir de data/results/report_8020_sbsr.

    python reports/lidar/8020-sbsr/report.py   # reconstrói a página
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="8020-sbsr",
    title="LiDAR 80/20 — rodada final SBSR",
    description="Lineup do artigo: TCN/TCN-Bi 46+meteo (48 col), "
                "LSTM/LSTM-Bi featurex (67 col), MSE pura em todos, "
                "seed 42, com baselines na mesma grade.",
    source_dir="data/results/report_8020_sbsr",
    subtitle="Rodada canônica do artigo SBSR: split 80/20 sem validação, "
             "60 épocas fixas, seed 42, perda MSE pura em todos os "
             "modelos (ITILDE-Q descartada), alvo ws100_wavelet. TCNs "
             "consumem o perfil LiDAR + meteorologia (48 col); LSTMs "
             "acrescentam cis1..19 (featurex, 67 col). Entrada de 72 "
             "passos (12 h) → horizonte de 36 passos (6 h), 10 min.",
    badges=("80/20", "60 épocas", "seed 42", "MSE pura", "lineup misto",
            "com baselines", "sem origens de fronteira"),
    protocol_note="Modelos treinados em <code>pipeline/tmp/"
                  "trained_8020_sbsr</code> por <code>scripts/"
                  "lidar_8020_sbsr_train.py</code> (feature_keep por "
                  "wrapper). Avaliação com baselines na mesma grade por "
                  "<code>tests/relatorio_resultados.py</code>; as "
                  "primeiras 72 origens do teste (janela de entrada "
                  "cruzando a fronteira treino/teste) são descartadas "
                  "para todos os métodos.",
    regenerate=[".venv/bin/python", "tests/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_8020_sbsr",
                "--out-dir", "data/results/report_8020_sbsr",
                "--trim-boundary-origins", "72",
                "--run-title", "LiDAR 80/20 — rodada final SBSR"],
    retrain=[".venv/bin/python", "scripts/lidar_8020_sbsr_train.py"],
    info=[("Protocolo", "80/20 sem validação"), ("Épocas", "60 fixas"),
          ("Seed", "42"), ("Perda", "MSE pura (todos)"),
          ("TCN/TCN-Bi", "46 + humid/temp (48 col)"),
          ("LSTM/LSTM-Bi", "46 + cis1..19 + humid/temp (67 col)"),
          ("Baselines", "ARIMA, RF, persistência (mesma grade)"),
          ("Janela", "72 → 36 passos (10 min)"),
          ("Modelos", "trained_8020_sbsr")],
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
