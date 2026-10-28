---
title: Decoder esparso stride-6 (h=6..36): só ajuda a família LSTM, piora TCN
area: resultados
updated: 2026-10-23
---

# Decoder esparso stride-6 (h=6..36): só ajuda a família LSTM, piora TCN

Experimento de isolamento de variável sobre a rodada canônica mista
8020-itildeq: mesmo protocolo (80/20 sem validação, 60 épocas, seed 42,
HPs Round 5, mesma loss por família — TCN=MSE+C′, LSTM=ITILDE-Q), só muda
a grade de saída — o decoder emite 6 passos amostrados a stride 6
(horizontes 6, 12, 18, 24, 30, 36 = 1 h em 1 h até 6 h). **Resultado
dependente da família**: stride-6 vence a canônica (36 passos) nos MESMOS
horizontes na família LSTM (LSTM R² 0,363 vs 0,273; LSTM-Bi 0,350 vs
0,299) e perde com folga nas TCN (TCN R² 0,287 vs 0,493; TCN-Bi 0,273 vs
0,459). Não vira canônico; utilidade = evidência de que o treino denso em
36 passos funciona como multi-tarefa que beneficia a TCN nos horizontes
hourly, enquanto o ITILDE-Q do LSTM se beneficia do objetivo esparso.

## Detalhes

- **Implementação**: `scripts/experiments/lidar_8020_stride6_train.py` — subclasse
  dinâmica do wrapper canônico cujo `create_sequences` delega ao método
  base com janela cheia de 36 passos e fatia X_decoder/y a passo 6
  (precedente: `s2s_lstm_mean_wrapper`). O canal "fração de horizonte" do
  decoder direto preserva a semântica real (k/6 = 6k/36); covariadas
  futuras ficam nos timestamps-alvo. Avaliação via
  `predict_all_horizons_stride` (labels/posições de horizonte em stride;
  persistence = último observado, válido em h≠1).
- **Artefatos**: modelos em `pipeline/tmp/trained_8020_stride6/`
  (model.json documenta `output_stride: 6`, `horizons: [6..36]`);
  `data/results/lidar_8020_stride6/evaluation.json` (tabela por horizonte
  stride-6 vs canônica vs persistence). Remount do `--eval-only` validado:
  tabela idêntica à avaliação inline.
- **Números (MAE | R², agregado nos 6 horizontes, vs ws100 bruto,
  comparado à canônica report_8020_itildeq no mesmo subconjunto)**:
  - LSTM: 1,017 | 0,363 vs canônica 1,058 | 0,273 — **stride-6 vence em
    todos os 6 horizontes** (ex. h=12: 0,919|0,481 vs 1,042|0,301).
  - LSTM-Bi: 1,028 | 0,350 vs 1,041 | 0,299 — vitória marginal.
  - TCN: 1,080 | 0,287 vs 0,877 | 0,493 — **stride-6 perde feio**; em
    h=36, R² 0,101 vs 0,433 (a TCN densa transfere melhor).
  - TCN-Bi: 1,101 | 0,273 vs 0,910 | 0,459 — perde.
  - Persistence (mesma grade): MAE 0,689 (h6) → 1,172 (h36).
- **Ressalvas**: 1 seed (42), sem teste de significância — ruído de
  retreino ≈ ±0,04 MSE (finding estudos-ablacoes) não explica a diferença
  das TCN (ΔR² ~0,2). O agregado "canônica nos 6 horizontes" usa as linhas
  `per_horizon` de `report_8020_itildeq/evaluation.json` (mesma grade de
  origens do protocolo 80/20).
- **Leitura**: a TCN canônica trata os 36 passos como multi-tarefa; restringir
  o objetivo aos 6 horizontes hourly a empobrece. O LSTM ITILDE-Q (shape-aware)
  se beneficia de prever só os alvos hourly — consistente com o teste
  multi-resolução `lstm-mean` (médias horárias), onde a representação
  hourly também foi competitiva. Não muda a rodada canônica do artigo.

## Fontes

- `scripts/experiments/lidar_8020_stride6_train.py` — gerador do experimento (treino +
  avaliação + comparação com a canônica).
- `data/results/lidar_8020_stride6/evaluation.json` — números por horizonte.
- `data/results/report_8020_itildeq/evaluation.json` — referência canônica
  (`per_horizon` h=6..36).
- `pipeline/tmp/lidar_8020_stride6_train.log` — log do treino completo
  (210/218/180/477 s por wrapper, 60 épocas, seed 42).
