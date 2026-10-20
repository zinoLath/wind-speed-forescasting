# agents/data.md — dados e artefatos

> Base canônica: `findings/knowledge/dados-map.md` (skill `canonical-findings`).

- Entradas: `data/dataset.csv` (série de previsão) e `data/wind_data.csv`
  (série longa para imputação/walk-forward) — gitignored
- Saídas do pipeline: `pipeline/tmp/` (sobrescrito a cada run) e
  `data/impute/<method>/`
- Resultados: `data/results/{optuna,wfo_optuna,benchmark*}/`
- Fontes dos relatórios LiDAR: `data/results/report*/` (evaluation.json +
  images/ + mcs.json — SÓ artefatos de dados) — consumidos pelos relatórios
  padronizados em `reports/` (ver agents/reports.md); os modelos ficam em
  `pipeline/tmp/trained_*/`. Os HTML/MD standalone que existiam aqui
  (`relatorio_resultados.html`, `relatorio_metodos.*`, `relatorio.html`,
  `investigacao/`, `features_study/`, `context_mode_ablation.md`,
  `ab_persistence_gate.html`) foram migrados para `reports/` e os geradores
  (`tests/relatorio_resultados.py`, `tests/metodos_complementares.py`,
  `tests/ab_persistence_gate.py`, `tests/build_investigacao.py`,
  `tests/build_feature_study_report.py`, `tests/feature_study.py`,
  `scripts/wfo_lightgbm_report.py`) agora escrevem direto em `reports/`.
  **Nunca criar relatório em `data/results/`** — só artefatos de dados.
- Artefatos arquivados/órfãos: `data/archive/` (não consumir)
- Fatias de período completo: `data/series_list/` (regenerado por
  `src/periods.py`)
- Dashboard de treino por época: `pipeline/tmp/train_dashboard/<run>.jsonl`
  (uma linha JSON por época; fonte das curvas do relatório de treinamento)
- Avaliação inline do treino 80/20: `data/results/lidar_<tag>_features/all.json`

Arquivos de dados (`/data/**/*.csv`, `*.h5`, `*.keras`) são gitignored;
`/models/`, `/OLD/`, `/images/` são gitignored mas alguns arquivos internos
são rastreados.
