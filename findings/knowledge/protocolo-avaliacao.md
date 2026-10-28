---
title: Protocolo de avaliação (80/20, métricas, rolling)
area: protocolo
updated: 2026-10-28
---

# Protocolo de avaliação

Três protocolos coexistem — saber qual está em jogo é essencial:

## 1) Pipeline padrão = protocolo reval (default desde 28/10/2026)

**O `python pipeline/pipeline.py` (estágio train) roda por default o
protocolo reval**: split cronológico **72,5/12,5/15** (treino/validação/
teste), **EarlyStopping(val_loss, patience 12, restore_best) +
ReduceLROnPlateau** (via `common.default_callbacks`), **60 épocas**, seed
42, MSE pura, alvo `ws100_wavelet` (denoise `["ws100"]` nível 2). Valores
em `pipeline/pipeline.default.json` E `pipeline/pipeline.json` (o arquivo
de usuário sobrescreve o default — os dois precisam estar alinhados).
Antes de 28/10 o default era 75/20/5 com patience 8/épocas 100 e o
`pipeline.json` antigo usava denoise `["ws100","ws40","v40"]` nível 1
(desalinhado do canônico). O treinador standalone equivalente é
`scripts/lidar_reval_sbsr_train.py` (lineup feature_keep por wrapper);
o estágio train treina 1 wrapper (default `lstm_bi`) — o lineup completo
continua nos scripts.

## 2) Protocolo LiDAR 80/20 (rodada oficial do paper `8020-sbsr`)

- Split temporal **80/20 sem validação**, orçamento fixo de **60 épocas**,
  seed 42; janela 72→36 passos; alvo `ws100_wavelet`.
- Gerado por `scripts/experiments/lidar_7525_train.py --train-ratio 0.80` →
  `pipeline/tmp/trained_8020`; variantes 75/25, MSE pura, loss invertida,
  seeds múltiplas (`itildeq`).
- **Métricas headline sempre contra `ws100` BRUTO**; contra o alvo denoised
  ficam em colunas secundárias `*_denoised`.
- Avaliação dupla: (i) **todas as origens** do teste geram 36 passos
  (`predict_all_horizons`) → métricas agregadas e por horizonte; (ii)
  **rolling forecast** contínuo no horizonte máximo, janelas função só do
  histórico conhecido, uma chamada batched — sem vazamento.
- Métricas reportadas: MAE, MSE, RMSE, NMSE, NRMSE, NMAE, R²; baseline de
  referência: **persistência** (último observado para todos os horizontes).
- Resultados **variam entre execuções** (sem early stopping): diferenças de
  ±0.3 MSE não são regressão; comparar com a rodada anterior e comentar
  variância.

## HPO e vazamento

Otimizar hiperparâmetros **contra o teste** é vazamento de seleção: o
conjunto de teste influencia a escolha do modelo, e o melhor de N trials
fica otimista pelo máximo do ruído (viés de seleção) — o número deixa de
ser estimativa honesta. Regra: objetivo/pruning **só** na validação
(holdout do treino ou split temporal dedicado); teste intocado até a
avaliação final. HPO em 80/20 sem validação exige segurar uma fatia do
treino para objetivo. Exceção consciente: estudo exploratório (ex.:
round6), explicitamente não reportável.

## Fontes

- `docs/metodologia.md` (seções 1, 5, 6); `README.md` (convenções).
- Skill `lidar-retrain`; `agents/architecture.md`.
