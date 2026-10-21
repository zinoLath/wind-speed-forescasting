# Comparativo LSTM, LSTM-bi, TCN e TCN-bi — hiperparâmetros Optuna otimizados

Data: 2026-09-18. Modelos treinados do zero com os `best_trial.json` da busca
agressiva de Optuna (400 trials por wrapper, espaço de busca do TCN expandido
para cobrir o receptive field de 72 passos). Avaliação no mesmo split
(treino 75% / val 20% / teste 5%) contra o **ws100 cru** (coluna
`*_denoised` = métricas contra o alvo suavizado, secundárias).

Artefatos: `pipeline/tmp/trained_compare/{Seq2Seq_*}/` + `summary.json`.

## Tabela resumo

| modelo | loss | epochs | treino(s) | all-h MAE | all-h RMSE | r² | roll MAE | roll RMSE |
|---|---|---|---|---|---|---|---|---|
| **Seq2Seq_LSTM** | mae | 13 | 24 | **0.9033** | **1.1692** | **0.2693** | 2.2688 | 2.5624 |
| Seq2Seq_LSTM_Bidirectional | huber | 19 | 61 | 0.9231 | 1.1888 | 0.2446 | 2.2320 | 2.5205 |
| Seq2Seq_TCN | huber | 37 | 230 | 0.9537 | 1.2494 | 0.1655 | **1.1417** | **1.4213** |
| Seq2Seq_TCN_Bidirectional | huber | 30 | 257 | 1.1339 | 1.4083 | -0.0602 | 1.3667 | 1.7313 |
| persistência (baseline) | — | — | — | 0.9752 | 1.3093 | 0.0836 | — | — |

- `all-h` = todas as origens × todos os 36 horizontes (9 792 amostras).
- `roll` = rolling forecast, horizonte **h=36** em cada origem (janelas
  consecutivas de história conhecida; 344 amostras).

## Leitura dos resultados

### 1. Objetivo do Optuna ≠ desempenho no teste
O TCN teve o **melhor objetivo** da busca (val_block_mse 1.003 vs 1.025 do
TCN-bi), mas no teste all-horizons o LSTM simples é melhor (MAE 0.903 vs
0.954). Três causas prováveis:

- Os `best_trial.json` do LSTM/LSTM-bi são de uma **era anterior de objetivo**
  (o valor 0.795 do LSTM foi medido com outra convenção), então o LSTM não é
  o melhor da busca atual — mas ainda assim generaliza bem no teste.
- O objetivo Optuna usa `val_block_mse(k=4)` com o decoder de inferência;
  o teste all-horizons mistura os 36 horizontes, diluindo o erro de longo
  prazo onde o TCN é mais forte.
- O TCN-bi, apesar da busca expandida, piorou no teste (r² negativo).

### 2. Curto vs longo prazo: o TCN é o rei do horizonte longo
Por horizonte (teste, contra ws100 cru):

| modelo | h=1 MAE | h=19 MAE | h=36 MAE |
|---|---|---|---|
| LSTM | **0.476** | 0.888 | 1.190 |
| LSTM-bi | 0.701 | 0.926 | 1.190 |
| TCN | 0.668 | 0.988 | **1.147** |
| TCN-bi | 0.780 | 1.148 | 1.374 |

- **LSTM** domina o curto prazo (h=1: 0.476 vs 0.668 do TCN), mas degrada
  no longo.
- **TCN** é o melhor no horizonte 36 (MAE 1.147) e no rolling forecast
  (MAE 1.142 vs 2.27 do LSTM) — comportamento esperado de um TCN, cuja
  vantagem é exatamente capturar dependências longas.

### 3. O rolling forecast expõe o problema do LSTM
O rolling avalia só h=36 com janelas consecutivas. O LSTM despenca
(MAE 2.27 vs 1.14 do TCN), mesmo tendo o melhor all-horizons. Isso sugere:

- O bom all-horizons do LSTM vem dos **horizontes curtos/médios**; no
  horizonte 36 o erro é alto.
- O TCN é **muito mais estável no longo prazo**, com erro de rolling quase
  igual ao seu erro all-horizons (1.14 vs 0.95), enquanto o LSTM degrada
  muito (2.27 vs 0.90).

### 4. TCN-bi não se beneficiou da busca agressiva
- Pior MAE all-horizons (1.134), r² negativo (-0.06).
- Em h=1 (0.780) e h=36 (1.374), é o pior entre os quatro.
- Possíveis causas: overfitting (encoder_filters 160 + dropout 0.4 com
  lr 0.0026 alta), ou o Bidirectional não agregar valor com o TCN no
  encoder já causal. É o candidato a **não usar** no pipeline final.

### 5. Persistência ainda é um baseline forte
Todos os modelos vencem a persistência no all-horizons (MAE 0.975) exceto o
TCN-bi (1.134). No rolling (h=36), a comparação direta com persistência
precisa ser medida, mas o TCN (1.142) está próximo do erro de longo prazo
esperado — vale verificar se o TCN supera a persistência de h=36.

## Pontos bons para investigar

1. **Ensemble / interpolação de previsões:** LSTM é melhor no curto prazo,
   TCN no longo. Um blend (ex.: pesos por horizonte, ou média) pode vencer
   os dois em todas as métricas. **Prioridade alta.**
2. **Métricas de longo prazo como objetivo:** como o objetivo Optuna não
   premiou a vantagem do TCN em h=36, testar objetivos que pesem mais os
   horizontes longos (ou `rolling h=36` no val) pode alinhar busca ↔ teste.
3. **TCN com validação/lr mais conservadora:** o TCN usa lr 0.00042 com
   warmup_cosine; tentar `ReduceLROnPlateau` adicional ou mais épocas pode
   reduzir o gap all-horizons (0.954 vs 0.903 do LSTM).
4. **Descartar TCN-bi** no pipeline final, ou re-buscá-lo com espaço menor
   (ele não convergiu para um bom ponto).
5. **Verificar o rolling do LSTM:** a queda de 0.90 → 2.27 merece análise
   (possível efeito de janelas consecutivas com muita sobreposição de
   regime de vento). Medir rolling por horizonte e comparar com persistência.
6. **Re-treinar LSTM/LSTM-bi com a busca atual** (objetivo val_block_mse)
   para uma comparação justa de arquiteturas no mesmo padrão de avaliação.

## Próximo passo sugerido

1. Medir persistência de h=36 para os quatro (baseline de rolling).
2. Implementar blend LSTM+TCN por horizonte e avaliar no teste.
3. Rodar o walk-forward com os dois vencedores (LSTM e TCN) para validar
   a estabilidade no tempo.