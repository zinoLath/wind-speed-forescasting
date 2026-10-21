"""Relatório de resultados — 80/20 rodada mista (ITILDE-Q + MSE/C′).

Mesmo protocolo 80/20 da reamostragem canônica, mas com duas famílias
treinadas de formas distintas (scripts/lidar_8020_itildeq_train.py →
trained_8020_itildeq):

- LSTM / LSTM-Bi: HPs do HPO 80/20 + loss ITILDE-Q (shape-aware:
  MSE + α·(β·L_a.shift + (1−β)·L_phase + γ·L_amp), Lee et al. 2022,
  arXiv:2210.15050; variante Eq. 15 de DONG Sheng et al. 2026);
- TCN / TCN-Bi: HPs canônicos + loss MSE pura + dilatações estáticas C′
  (encoder até 36 passos = 6 h).

Baselines ARIMA, Random Forest e persistência entram na mesma grade de
origens, e o Model Confidence Set inclui todos os modelos.

    python reports/lidar/8020-itildeq/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="8020-itildeq",
    title="LiDAR 80/20 — rodada mista (ITILDE-Q + MSE/C′)",
    description="LSTM/LSTM-Bi com HPs do HPO 80/20 e loss ITILDE-Q; "
                "TCN/TCN-Bi com HPs canônicos, loss MSE e dilatações até 36; "
                "baselines ARIMA/RF/persistência na mesma grade.",
    source_dir="data/results/report_8020_itildeq",
    subtitle="Rodada mista sob a reamostragem 80/20 (60 épocas, seed 42): as "
             "LSTMs usam os HPs do HPO 80/20 com a loss shape-aware "
             "ITILDE-Q; as TCNs usam os HPs canônicos com MSE pura e "
             "dilatações estáticas até 36 passos (6 h). Baselines ARIMA, "
             "Random Forest e persistência entram na comparação e no MCS.",
    badges=("80/20", "rodada mista", "ITILDE-Q", "MSE + C′", "baselines"),
    regenerate=[".venv/bin/python", "tests/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_8020_itildeq",
                "--out-dir", "data/results/report_8020_itildeq"],
    retrain=[".venv/bin/python", "scripts/lidar_8020_itildeq_train.py",
             "--alpha", "0.3", "--beta", "0.5", "--gamma", "0.1"],
    info=[("Protocolo", "80/20 sem validação"),
          ("LSTM / LSTM-Bi", "HPs do HPO 80/20 + ITILDE-Q (α=0,3 · β=0,5 · γ=0,1)"),
          ("TCN / TCN-Bi", "HPs canônicos + MSE pura + dilatações C′ até 36"),
          ("Baselines", "ARIMA, Random Forest, persistência (80/20)"),
          ("Épocas", "60 fixas"), ("Seed", "42"),
          ("Modelos", "trained_8020_itildeq"),
          ("Referências", "lee2022tildeq; sheng2026shape")],
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
