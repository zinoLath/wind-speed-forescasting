"""Relatório de resultados — 80/20 com os melhores hiperparâmetros do
Optuna Round 5 (rodada de referência; NÃO é a canônica do artigo).

Mesmo protocolo do lidar/8020 (80/20 sem validação, 60 épocas fixas,
seed 42), mas com os hiperparâmetros vencedores da **Optuna round 5**
(objetivo skill em bloco k=4 na validação, split 72/8/20; promovidos
para pipeline/tmp/optuna/) e a política de loss por modelo
(LSTM→ITILDE-Q, TCN→MSE) com dilatações estáticas C′ nos TCN.

**Nota de transferência (21/10/2026):** os HPs da round 5, otimizados no
split 72/8/20, NÃO transferiram bem para o retreino 80/20 — os TCNs
caíram (TCN 0,29 vs 0,52 da itildeq canônica), em parte porque campos
estruturais fora do espaço de busca (`context_pooling`,
`encoder_dilation_rate`) reverteram aos defaults. A rodada canônica do
artigo segue sendo **lidar/8020-itildeq** (HPs HPO 80/20 + canônicos);
esta página existe como referência comparativa.

    python reports/lidar/8020-round5/report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.src.resultados import report_module  # noqa: E402

REPORT, build = report_module(
    name="8020-round5",
    title="LiDAR 80/20 — HP do Optuna (round 5)",
    description="Modelos 80/20 com os hiperparâmetros vencedores da "
                "Optuna round 5 (skill em bloco k=4, 72/8/20), loss por "
                "modelo e dilatações C′ — rodada de referência: a "
                "transferência dos HPs degradou os TCN; a canônica do "
                "artigo segue lidar/8020-itildeq.",
    source_dir="data/results/report_8020_round5",
    subtitle="Mesma reamostragem 80/20 do lidar/8020 (60 épocas, seed 42) "
             "com os hiperparâmetros do Optuna round 5 — objetivo skill "
             "em bloco k=4 na validação (72/8/20), loss por modelo e "
             "dilatações estáticas C′ nos TCN (profundidade buscada).",
    badges=("80/20", "Optuna round 5", "skill bloco k=4",
            "referência (não-canônica)"),
    regenerate=[".venv/bin/python", "tests/relatorio_resultados.py",
                "--models-dir", "pipeline/tmp/trained_8020_round5",
                "--out-dir", "data/results/report_8020_round5"],
    retrain=[".venv/bin/python", "scripts/lidar_8020_itildeq_train.py",
             "--output-dir", "pipeline/tmp/trained_8020_round5"],
    info=[("Protocolo", "80/20 sem validação"),
          ("Hiperparâmetros", "Optuna round 5 (skill bloco k=4, 72/8/20)"),
          ("Loss", "LSTM→ITILDE-Q · TCN→MSE"),
          ("Dilatações", "estáticas C′; profundidade = HP da round 5"),
          ("Épocas", "60 fixas"), ("Seed", "42"),
          ("Modelos", "trained_8020_round5"),
          ("Status", "referência — canônica do artigo: lidar/8020-itildeq")],
)

if __name__ == "__main__":
    from reports.src import builder
    builder.run_standalone(__file__)
