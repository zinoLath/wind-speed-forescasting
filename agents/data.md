# agents/data.md — dados e artefatos

> Base canônica: `findings/knowledge/dados-map.md` (skill `canonical-findings`).

- Entradas: `data/dataset.csv` (série de previsão) e `data/wind_data.csv`
  (série longa para imputação/walk-forward) — gitignored
- Saídas do pipeline: `pipeline/tmp/` (sobrescrito a cada run) e
  `data/impute/<method>/`
- Resultados: `data/results/{optuna,wfo_optuna,benchmark*}/`
- **Banco de runs de predição: `data/predictions.db`** (SQLite,
  `src/predictions_db.py`) — runs/modelos/hp/features/stats por horizonte
  (0 = total) e predições linha a linha; gravado por
  `reports/src/relatorio_resultados.py` (uma run canônica por `report_id`),
  lido pelos relatórios do hub antes do `evaluation.json`. Controle via
  `python -m src.predictions_db list|show|categories|homologar|prune`;
  detalhes no finding `predicoes-db`.
- Fontes dos relatórios LiDAR: `data/results/report*/` (evaluation.json +
  images/ + mcs.json — SÓ artefatos de dados) — **as métricas vêm primeiro
  do banco `data/predictions.db`** (finding `predicoes-db`); o
  evaluation.json é fallback/cache; os modelos ficam em
  `pipeline/tmp/trained_*/`. Os HTML/MD standalone que existiam aqui
  (`relatorio_resultados.html`, `relatorio_metodos.*`, `relatorio.html`,
  `investigacao/`, `features_study/`, `context_mode_ablation.md`,
  `ab_persistence_gate.html`) foram migrados para `reports/` e os geradores
  (`reports/src/relatorio_resultados.py`, `studies/metodos_complementares.py`,
  `studies/ab_persistence_gate.py`, `studies/build_investigacao.py`,
  `studies/build_feature_study_report.py`, `studies/feature_study.py`,
  `scripts/experiments/wfo_lightgbm_report.py`) agora escrevem direto em `reports/`.
  **Nunca criar relatório em `data/results/`** — só artefatos de dados.
- Artefatos arquivados/órfãos: `data/archive/` (não consumir)
- Fatias de período completo: `data/series_list/` (regenerado por
  `src/periods.py`)
- Dashboard de treino por época: `pipeline/tmp/train_dashboard/<run>.jsonl`
  (uma linha JSON por época; fonte das curvas do relatório de treinamento)
- Avaliação inline do treino 80/20: `data/results/lidar_<tag>_features/all.json`

Arquivos de dados (`/data/**/*.csv`, `*.h5`, `*.keras`) são gitignored;
`/models/`, `/OLD/`, `/images/` e `ptxas` são gitignored e fora do git
(des-rastreados em 28/10/2026).
