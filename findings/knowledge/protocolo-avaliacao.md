---
title: Protocolo de avaliação (80/20, métricas, rolling)
area: protocolo
updated: 2026-10-19
---

# Protocolo de avaliação

Dois protocolos coexistem — saber qual está em jogo é essencial:

## 1) Pipeline padrão (legado do TCC)

Split cronológico **75/20/5** (treino/validação/teste, sem embaralhamento);
`MinMaxScaler` (alvo e covariáveis separados) ajustado **só no treino**;
early stopping na val_loss (decoder em modo inferência); walk-forward de
60 d treino / 30 d teste / avanço 30 d com retreino do zero (partição
interna 80/20).

## 2) Protocolo LiDAR de reamostragem (canônico dos relatórios/artigos)

- Split temporal **80/20 sem validação**, orçamento fixo de **60 épocas**,
  seed 42; janela 72→36 passos; alvo `ws100_wavelet`.
- Gerado por `scripts/lidar_7525_train.py --train-ratio 0.80` →
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

## Fontes

- `docs/metodologia.md` (seções 1, 5, 6); `README.md` (convenções).
- Skill `lidar-retrain`; `agents/architecture.md`.
