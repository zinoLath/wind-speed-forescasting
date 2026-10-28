---
title: Mapa de dados e artefatos
area: dados
updated: 2026-10-22
---

# Mapa de dados e artefatos

Onde vivem entradas, saídas e artefatos — e o que é gitignored.

## Entradas (gitignored)

- `data/dataset.csv` — série de previsão (treino/avaliação, 10 min).
  **Verificado (20/10/2026): 7.561 linhas, 16/09 18:00 → 08/11/2021 — a
  série NÃO é completa** (grade do período daria ~7.776; 7.632 = 53 dias
  cheios): há linhas inteiras ausentes (sem NaN) e timestamps fora da
  grade (ex.: minutos :29). Confirmar se é pós-imputação ou bruto antes
  de afirmar "série sem valores ausentes" (thread Overleaf `6ace7f52` —
  o 03 atual afirma o contrário).
- `data/wind_data.csv` — série longa com lacunas (imputação/walk-forward;
  cobre período diferente, 2021-11 → 2022-10).
- `data/siga-empreendimentos-geracao-diario.csv` — SIGA/ANEEL base
  diária (snapshot 20-10-2026; fonte do dado de capacidade da
  Introdução do SBSR).

## Saídas e artefatos

- `pipeline/tmp/` — sobrescrito a cada run do pipeline; modelos LiDAR em
  `pipeline/tmp/trained_8020/` (backups em `trained_8020_backup_*`).
- `data/results/{optuna,wfo_optuna,benchmark*}/` — resultados gerais;
  rodadas Optuna em `data/results/optuna_round{2,4,4b,4c,5}`.
- `data/results/report*/` — **fontes de dados dos relatórios LiDAR**
  (`evaluation.json` + `images/` + `mcs.json`); report_8020 é a rodada
  canônica atual (ver finding `resultados-canonicos`). Métricas agora têm
  fonte primária no banco `data/predictions.db` (finding `predicoes-db`) —
  o evaluation.json vira fallback/cache.
- `data/predictions.db` — banco SQLite das runs de predição
  (`src/predictions_db.py`): runs por `report_id`, modelos, hp, features,
  stats por horizonte (0 = total) e predições linha a linha. Gravado por
  `reports/src/relatorio_resultados.py`; CLI
  `python -m src.predictions_db list|show|homologar|prune`.
- `data/results/lidar_<tag>_features/all.json` — avaliação inline do treino.
- `data/results/metodos_complementares/` — ARIMA/RF + estatísticas DM/KSPA
  (`metodos.json`).
- `pipeline/tmp/train_dashboard/<run>.jsonl` — métricas por época (fonte
  das curvas do relatório de treinamento).
- `data/impute/<method>/` — saídas de imputação.
- `data/series_list/` — fatias de período completo (regenerado por
  `src/periods.py --min-length 108`).
- `data/archive/` — arquivados/órfãos, **não consumir**.

## Regras

- `/data/**/*.csv`, `*.h5`, `*.keras` gitignored; `/models/`, `/OLD/`,
  `/images/` gitignored com arquivos internos rastreados.
- `data/results/` guarda **só artefatos de dados** — relatório nenhum
  (nunca criar HTML/MD lá; ver finding `relatorios-framework`).
- Arquivos `.keras` + `model.json` (classe, hp, origem, modo decoder/alvo,
  loss, histórico, dimensões) permitem recarregar e reavaliar bit a bit.

## Fontes

- `agents/data.md`; `docs/metodologia.md` (seção 5, rastreabilidade).
