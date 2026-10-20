---
title: Mapa de dados e artefatos
area: dados
updated: 2026-10-19
---

# Mapa de dados e artefatos

Onde vivem entradas, saídas e artefatos — e o que é gitignored.

## Entradas (gitignored)

- `data/dataset.csv` — série de previsão (treino/avaliação, 10 min).
- `data/wind_data.csv` — série longa com lacunas (imputação/walk-forward).

## Saídas e artefatos

- `pipeline/tmp/` — sobrescrito a cada run do pipeline; modelos LiDAR em
  `pipeline/tmp/trained_8020/` (backups em `trained_8020_backup_*`).
- `data/results/{optuna,wfo_optuna,benchmark*}/` — resultados gerais;
  rodadas Optuna em `data/results/optuna_round{2,4,4b,4c,5}`.
- `data/results/report*/` — **fontes de dados dos relatórios LiDAR**
  (`evaluation.json` + `images/` + `mcs.json`); report_8020 é a rodada
  canônica atual (ver finding `resultados-canonicos`).
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
