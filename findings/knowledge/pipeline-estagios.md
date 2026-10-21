---
title: Pipeline em estágios e configs
area: infra
updated: 2026-10-19
---

# Pipeline em estágios e configs

Entrada principal do repositório.

- Estágios: `optuna` → `train` (salva `models/best_model/model.keras` +
  `model.json`) → `evaluate` → `impute` → `walkforward`.
- `python pipeline/pipeline.py [--stage ...]`; cada estágio isolado:
  `pipeline/step_<estágio>.py`. Estágios aceitam `--config <arquivo>`
  (mesclado sobre `pipeline/pipeline.default.json`; seção `common` para
  valores compartilhados; valores de estágio vencem — ver
  `pipeline/README.md`).
- Configs especiais existentes: `optuna_round{2..5}` (HPO distribuído),
  `train_compare_*`, `train_round3_compare*` (gate/no-gate/nonorm),
  `train_tcn_*` (compare, ctxhorizon, ctxnone).
- `src/common.py` é a **fonte única** de setup de GPU, carga/split do
  dataset, adaptadores de hp, métricas e `predict_all_horizons`; lógica
  compartilhada nova vai lá — **sem efeitos colaterais de import** (TF é
  configurado por `common.setup_tensorflow()` antes dos imports lazy de
  modelo).
- **Armadilha de import**: arquivos sob `pipeline/` (e `reports/*/report.py`)
  precisam do bootstrap `sys.path` para a raiz do projeto antes de
  `from pipeline import ...` — senão `pipeline/pipeline.py` sombreia o
  pacote.

## Fontes

- `pipeline/README.md`; `agents/setup.md`; `agents/gotchas.md`;
  `src/common.py`.
