# Relatório: Otimização de Hiperparâmetros com Optuna

## Objetivo

Encontrar, para cada arquitetura Seq2Seq (`Seq2Seq_LSTM`, `Seq2Seq_LSTM_Bidirectional`, `Seq2Seq_TCN`, `Seq2Seq_TCN_Bidirectional` e `Seq2Seq_TCN_LSTM`), o conjunto de hiperparâmetros que minimiza o `val_loss` (MSE) na previsão da velocidade do vento a 100 m (`ws100_wavelet`), usando como entrada uma janela de 72 passos (12 h) e horizonte de saída de 36 passos (6 h).

## Configuração da busca

- **Algoritmo:** Optuna com `TPESampler` (Tree-structured Parzen Estimator), em modo multivariado com agrupamento de parâmetros relacionados (`multivariate=True`, `group=True`), `n_startup_trials=20` e `n_ei_candidates=64`.
- **Podas (pruning):** `MedianPruner` com `n_startup_trials=10`, `n_warmup_steps=8` e `interval_steps=2`; cada trial reporta o `val_loss` ao final de cada época e é interrompido se ficar acima da mediana histórica.
- **Callbacks de treinamento:** `EarlyStopping` (`patience=8`, restaurando os melhores pesos), `ReduceLROnPlateau` (fator 0.5, `patience=4`, `min_lr=1e-6`).
- **Divisão dos dados:** 75% treino / 20% validação (sequencial, sem embaralhamento), com `denoise` wavelet (`sym18`, nível 2) aplicado sobre `ws100`.
- **Parâmetros da busca:** 100 trials por modelo, 60 épocas máximas por trial, `batch_size=32`, otimizador Adam com loss MSE, seed 42, executado em GPU.

## Hiperparâmetros otimizados

| Modelo | Hiperparâmetros |
|---|---|
| LSTM | `learning_rate` (log, 1e-5–1e-2), `lstm_units` (64–512, passo 32), `encoder_dropout_rate` (0–0.5, passo 0.05), `decoder_dropout_rate` (0–0.5, passo 0.05) |
| LSTM Bidirecional | idem, com `lstm_units` 64–384 |
| TCN | `learning_rate` (log, 1e-4–1e-2), e para encoder/decoder: `filters` (32–256 / 32–128, passo 16), `kernel_size` (2–3), `nb_stacks` (1–2), `dropout_rate` (0–0.5), `dilation_rate` (1–5 / 1–4) |
| TCN Bidirecional | mesmo espaço do TCN |
| TCN-LSTM | `learning_rate`, `encoder_filters` (32–256), `encoder_kernel_size`, `encoder_nb_stacks`, `encoder_dropout_rate`, `encoder_dilation_rate`, `lstm_units` (64–256), `decoder_dropout_rate` |

## Resultados

| Modelo | Trials completos | Trials podados | Melhor val_loss | Tempo total |
|---|---|---|---|---|
| Seq2Seq_LSTM | 58 | 42 | 1.66e-05 | ~75 min |
| Seq2Seq_LSTM_Bidirectional | 49 | 51 | 1.96e-05 | ~116 min |
| Seq2Seq_TCN | 67 | 33 | 4.08e-05 | ~97 min |
| Seq2Seq_TCN_Bidirectional | 66 | 34 | 4.98e-05 | ~116 min |
| Seq2Seq_TCN_LSTM | 2 | 0 | 3.59e-03 | ~1 min |

## Observações

- Os modelos baseados em **LSTM superaram os TCN** em `val_loss`, com o LSTM unidirecional obtendo o melhor resultado geral (1.66e-05).
- O pruner foi eficaz: ~40% dos trials dos modelos LSTM foram podados, reduzindo o custo computacional total.
- Os melhores trials tendem a favorecer `lstm_units=64`, `learning_rate` em torno de 6e-3–7e-3 e `decoder_dropout_rate` baixo (0–0.15).
- O `Seq2Seq_TCN_LSTM` executou apenas 2 trials (a busca foi interrompida), portanto seu resultado não é conclusivo.

Os melhores parâmetros de cada modelo foram salvos em `data/results/optuna/<WrapperName>/best_trial.json` e o histórico completo em `trials.csv`, sendo consumidos posteriormente pela avaliação walk-forward em `tests/wfo_optuna.py`.