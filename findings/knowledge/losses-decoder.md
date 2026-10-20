---
title: Loss ponderada por horizonte e decoder direto
area: arquitetura
updated: 2026-10-19
---

# Loss ponderada por horizonte e decoder direto (sem vazamento)

Duas decisões centrais que definem o comportamento honesto dos modelos.

## Loss

- Treino usa **MSE ponderada por horizonte** (`src/models/losses.py`,
  `horizon_weighted_mse`), favorecendo horizontes longos. A chave `loss` do
  config é mantida por compatibilidade, mas **os wrappers fixam a loss**.
- Ablações canônicas: MSE pura (`scripts/lidar_8020_mseloss_train.py` →
  `reports/lidar/8020-mseloss`) e loss invertida
  (`scripts/lidar_8020_startweight_train.py`).

## Convenção do decoder

- **Treino**: teacher forcing (`X_decoder[:, 0]` = último observado;
  `X_decoder[:, 1:]` = `y[t+1..t+35]` reais).
- **Validação e inferência**: decoder direto — último alvo observado +
  fração de horizonte; **nunca vê valores futuros**. Portanto `val_loss`,
  early stopping e o objetivo do Optuna medem o comportamento real de
  deployment (não o proxy teacher-forced).
- Modo alternativo `target_mode="residual"`: rede prevê o desvio vs
  persistência, somado de volta na inversão de escala.

## Fontes

- `agents/architecture.md`; `docs/metodologia.md` (seção 1, decisão 1).
- `src/models/losses.py`, `src/models/seq2seq_wrapper.py`.
