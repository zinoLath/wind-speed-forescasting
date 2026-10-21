---
title: Visão geral do projeto
area: arquitetura
updated: 2026-10-19
---

# Visão geral do projeto

Previsão de velocidade do vento multi-horizonte (6 h = 36 passos de 10 min)
a partir de 12 h de histórico (72 passos), com arquiteturas seq2seq com
atenção, treinadas com dados de LiDAR do **Parque Eólico Delta do
Maranhão**. Projeto de TCC (licença MIT); o artigo em andamento é o SBSR
(comparação TCN vs LSTM, impacto da bidirecionalidade).

## Fatos estáveis

- Tarefa: mapeamento seq2seq **direct multi-step** (não recursivo),
  `(X_encoder, X_decoder, y_decoder)` por janelas deslizantes vetorizadas.
- Alvo: `ws100` (velocidade a 100 m); treino sobre `ws100_wavelet`
  (denoising), métricas headline contra o ws100 **bruto**.
- Framework: Keras/TensorFlow (>= 2.21), Optuna para HPO, baselines
  ARIMA/Random Forest/persistência.
- Entrada principal: `pipeline/pipeline.py` (estágios
  optuna→train→evaluate→impute→walkforward).
- Há dois "mundos" de protocolo: o pipeline padrão (split 75/20/5 com
  validação) e o protocolo LiDAR de reamostragem **80/20 sem validação**
  (orçamento fixo 60 épocas, seed 42) — o protocolo dos relatórios/artigos.

## Fontes

- `README.md` — visão, comandos, convenções de métricas.
- `docs/metodologia.md` — formulação completa (decisões 1–3, seções 1–6).
- `articles/sbr/LaTeX_Template_SBSR_Portugues/01_Resumo.tex` — escopo do
  artigo SBSR (TCN vs LSTM, bidirecionalidade, R² 0.502/0.416 citados).
