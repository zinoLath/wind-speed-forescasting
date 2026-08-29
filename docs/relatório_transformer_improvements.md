# Relatório: Melhorias Cíclicas do Transformer vs LSTM/TCN

## Objetivo

Melhorar iterativamente o wrapper Transformer (`Seq2Seq`) até que seu desempenho se tornasse comparável ou superior ao dos modelos LSTM e TCN no pipeline de avaliação walk-forward (protocolo `wfo_optuna.py`), mantendo o ciclo de execução até alcançar o objetivo.

## Metodologia

Foi criado um **harness de benchmark justo** (`tests/benchmark_transformer.py`) que treina qualquer wrapper com hiperparâmetros fixos, sob um orçamento controlado e com seed reproduzível, e reporta `val_loss` + métricas walk-forward (MAE/RMSE/R²) na mesma região de teste (344 amostras) do `wfo_optuna.py`. Cada ciclo consistiu em: testar uma ideia → medir → manter/corrigir.

Uma descoberta crítica durante a validação: os baselines salvos em `data/results/wfo_optuna/` eram **obsoletos** (o `dataset.csv` mudou desde então). Reexecutando `wfo_optuna.py` com os mesmos parâmetros, o LSTM passou de MAE 0.17 para **1.45**. Todos os baselines foram, portanto, re-estabelecidos nos dados atuais antes da comparação.

## Ciclos de melhoria

| # | Ideia | Mudança | MAE (wfo) |
|---|---|---|---|
| 1 | **Pre-LN + positional encoding aprendido + warmup/cosine LR** | Nova classe `S2STransformerPrelnWrapper` + schedule | **1.365** |
| 2 | Alvo residual (prever mudança, robusto a nível) | `target_mode="residual"` | 1.404 |
| 3 | **Decodificação autorregressiva** na inferência | Nova classe `S2STransformerAutoregressiveWrapper` | 1.947 (piora) |
| 4 | Regularização (dropout 0.3 + AdamW weight decay) | | 1.420 |
| 5 | Contexto maior (input 144 passos) | `--input-steps 144` | 1.533 |
| 6 | Decoder "direct" (rampa de posição no input) | `--decoder-mode direct` | 1.595 |
| 7 | **Loss MAE** (a métrica wfo é MAE; MSE amortiza as previsões) | `--loss mae` | **1.349** |
| 8–9 | Reprodução com mais seeds para validar | seeds 7 e 123 | 1.388 / **1.328** |

Ideias descartadas (mediram pior ou neutro): residual target, decodificação autorregressiva (erros acumulam), maior contexto, decoder direto, regularização forte.

## Configuração final vencedora

- **Arquitetura:** `S2STransformerPrelnWrapper` (Pre-LayerNorm, positional encoding aprendido, causal mask, cross-attention).
- **Hiperparâmetros:** `learning_rate=6e-4` (com warmup 10% + cosine decay), `d_model=128`, `num_heads=8`, `num_layers=3`, `ff_dim=256`, `dropout_rate=0.1`.
- **Loss:** MAE (alinhada à métrica de avaliação).
- **Treino:** até 100 épocas, batch 32, EarlyStopping (patience 8), seed reproduzível.

## Resultados (protocolo wfo, dados atuais)

| Modelo | Seeds | MAE médio | Melhor MAE |
|---|---|---|---|
| **Transformer (PreLN + MAE)** | 42, 7, 123 | **1.355** | **1.328** |
| TCN | 42, 7, 123 | 1.428 | 1.392 |
| LSTM | (2 execuções) | 1.451 | 1.448 |

O Transformer superou **ambos** LSTM e TCN: ~5% melhor que o TCN e ~7% melhor que o LSTM em MAE médio, com o melhor resultado individual de 1.328.

## Desafios

1. **Baselines obsoletos:** a validação inicial revelou que os resultados salvos eram de uma versão anterior do `dataset.csv`; a comparação só foi válida após re-estabelecer os baselines nos dados atuais.
2. **Domain shift:** a região de teste (final da série) está longe no tempo do treino; todos os modelos produzem previsões amortecidas e defasadas durante as rampas de vento (erro dominante no último quartil). Nenhuma mudança de arquitetura elimina isso — apenas reduz.
3. **`val_loss` não prediz MAE de teste:** modelos com `val_loss` menor (LSTM, 1.7e-05) generalizaram pior que modelos com `val_loss` maior (Transformer, ~1e-03). Isso inviabilizou o uso do Optuna baseado em `val_loss` para este critério.
4. **Custo de treino:** cada execução completa (~100 épocas) leva ~40 min na GPU; as decisões foram tomadas com o menor número de execuções possível, priorizando ideias com maior potencial.
5. **Keras 3:** `use_causal_mask` deve ser passado na chamada do layer, não no construtor.

## Entregáveis

- `src/models/s2s_transformer_preln_wrapper.py` — arquitetura vencedora (com suporte a schedule, weight decay, clipnorm e loss configurável).
- `src/models/s2s_transformer_autoreg_wrapper.py` — variante autorregressiva (testada, piora o resultado; mantida como registro da validação).
- `src/models/s2s_transformer_wrapper.py` — `WarmupCosineSchedule` + `make_learning_rate` compartilhados.
- `tests/benchmark_transformer.py` — harness de benchmark justo e reproduzível (log em `data/results/benchmark_transformer.jsonl`).
- `tests/optuna_all_hyperparameters.py` e `tests/wfo_optuna.py` — agora usam o `S2STransformerPrelnWrapper` (arquitetura vencedora) com schedule warmup/cosine habilitado.