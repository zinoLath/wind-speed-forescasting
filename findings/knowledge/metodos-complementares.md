---
title: Métodos complementares e testes estatísticos
area: metodos
updated: 2026-10-22
---

# Métodos complementares e testes estatísticos

Baselines e significância estatística que acompanham os modelos de deep
learning.

- **Baselines**: persistência (último observado para todos os horizontes),
  **ARIMA** (statsmodels) e **Random Forest**; WFO com ARIMA tem cache de
  validação (`scripts/arima_val_cache.py`); RF baseline e RF+imputação em
  `scripts/wfo_rf_baseline.py` / `wfo_rf_impute.py`.
- Gerador: `tests/metodos_complementares.py` → artefatos em
  `data/results/metodos_complementares/` (`metodos.json` + imagens) e
  relatório `reports/protocolo/metodos-complementares/`.
- **Estatísticas** em `metodos.json`: DM (Diebold-Mariano) e KSPA com
  bootstrap (parâmetros `alpha`, `n_boot`, `mean_block`, `seed`),
  `loss_ab` (ablação de loss), `residuals`, `breakpoints`/`near_break`
  (quebras estruturais) e estratos `regime`/`banda` (skill por horizonte
  em `skill_por_horizonte.csv`, caudas em `caudas_por_horizonte.csv`).
- **Persistence gate A/B**: `tests/ab_persistence_gate.py` →
  `reports/protocolo/persistence-gate-ab` (gate estático vs dinâmico;
  configs `pipeline.train_compare_*`).
- Relatórios LiDAR usam `mcs.json` (model confidence set) gerado junto do
  `evaluation.json`.
- **Bateria seeds + Friedman + KSPA**: `tests/seeds_significance_8020.py`
  → `data/results/itildeq_seeds_8020/significance/` (Wilcoxon pareado por
  seed n=10, Friedman por dia com pós-hoc Holm entre as 4 canônicas do
  predictions.db, KSPA com bootstrap em blocos de origens). Sem GPU.

## Fontes

- `tests/metodos_complementares.py`; `data/results/metodos_complementares/metodos.json`;
  `agents/data.md`; `reports/lidar/8020/report.py`.
