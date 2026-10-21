---
title: Registry de wrappers e arquiteturas
area: arquitetura
updated: 2026-10-19
---

# Registry de wrappers e arquiteturas

Todos os modelos herdam de `Seq2SeqWrapper`
(`src/models/seq2seq_wrapper.py`), que padroniza `prepare()`, `build()`,
`fit()`, `predict()` e `rolling_forecast()` **em lote** (janelas são funções
puras do histórico conhecido; um único `model.predict`). O registro é
`src/common.py:WRAPPERS` com imports lazy.

## Chaves do registry (nomes viram diretórios de resultados)

`lstm`, `lstm_bi`, `gru`, `gru_bi`, `lstm_cnn`, `tcn`, `tcn_bi`,
`tcn_lstm`, `transformer` (Pre-LN). `validate_wrapper_names()` falha rápido
em `.name` duplicados.

## Particularidades por família

- **LSTM/LSTM-Bi + atenção** (estilo Luong, aditiva): estados do encoder
  inicializam o decoder; BiLSTM concatena forward/backward e o decoder tem
  o dobro de unidades.
- **TCN/TCN-Bi**: blocos causais dilatados (biblioteca `keras-tcn`,
  LayerNorm); encoder resumido por mean-pooling; espaço de hp
  **compartilhado e reduzido** via `src/models/tcn_hp.py:tcn_hyperparameters()`
  (um único `filters` para encoder/decoder, kernels 2–3, dilatações rasas);
  TCN-Bi usa mapeamento direto encoder→decoder (sem pooling de contexto).
- **TCN-LSTM (híbrido)**: encoder TCN; último passo projetado por
  `Dense(tanh)` nos estados h/c do decoder LSTM.
- **Transformer Pre-LN**: positional encoding **aprendido**, cross-attention
  com máscara causal, opcional AdamW + warmup 10% e decaimento cossenoidal;
  variante Post-LN senoidal mantida como referência
  (`s2s_transformer_wrapper.py`, legado/comparação).
- Cabeça comum: `TimeDistributed(Dense(1, linear))`.
- Wrappers aposentados vivem em `OLD/models/`.

## Fontes

- `agents/architecture.md`, `docs/metodologia.md` (seção 3).
- `src/common.py` (registry + adaptadores `FixedHyperParameters`/
  `OptunaHyperParameters`), `src/models/*.py`.
