---
title: Imputação da série longa
area: metodos
updated: 2026-10-19
---

# Imputação da série longa (walk-forward)

Reconstrução da série longa com lacunas (`data/wind_data.csv`) para os
protocolos walk-forward e WFO.

- Métodos (`src/impute/`): **RF** (`rf.py`, IterativeImputer `max_iter=10`,
  ajustado em amostra de 3.000 linhas), **LightGBM** (`lightgbm.py`, idem),
  **KNN** (`knn.py`), **Kalman** (`kalman.py`); base comum em `base.py`
  (loaders, conversão de direção 0–360° em **seno/cosseno** com conversão
  de volta, validação, métricas, plots). Registrados em
  `pipeline/step_impute.py:METHODS`.
- Validação de qualidade: **injeção de lacunas artificiais** (comprimentos
  1–36 amostras, 30 repetições por comprimento, semente fixa) num segmento
  completo; métricas MAE/RMSE/viés/R² por coluna e por comprimento.
- Dataset imputado carrega coluna flag `imputed` → métricas podem
  decompor-se em amostras observadas vs imputadas.
- Linha LightGBM em produção (WFO): `scripts/experiments/wfo_lightgbm_*.py` +
  relatórios `reports/imputacao/wfo-lightgbm{,-2m}` (2 m = série de 2 m).
- Comparativos detalhados: `studies/compare_rf_lgbm_impute_detailed.py --optuna`
  (imputer tunado por Optuna); relatórios `reports/imputacao/wfo-benchmarks`
  e `wfo-estudo-detalhado`.

## Fontes

- `docs/metodologia.md` (seção 2); `agents/architecture.md`;
  `pipeline/step_impute.py`; `src/impute/base.py`.
