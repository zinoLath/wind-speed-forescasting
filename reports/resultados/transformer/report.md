# Relatório: Wrapper Transformer (Seq2Seq) + Otimização com Optuna + Walk-Forward

## O que foi feito

Foi criado um novo wrapper `S2STransformerWrapper` em `src/models/s2s_transformer_wrapper.py`, seguindo a mesma estrutura dos demais wrappers (`Seq2SeqWrapper` base, método `build(hp)`, teacher forcing, `prepare()`/`fit()`/`predict()`/`rolling_forecast()` herdados). Em vez de LSTM/TCN, a arquitetura usa **Transformers** (encoder-decoder com atenção multi-cabeça).

Em seguida, o modelo foi integrado aos dois testes canônicos do projeto:

1. **`tests/optuna_all_hyperparameters.py`** — adicionado o wrapper ao dicionário `wrapper_factories` e à lista `--wrappers` (`transformer`).
2. **`tests/wfo_optuna.py`** — adicionado o wrapper ao dicionário `WRAPPERS`, permitindo a avaliação walk-forward com os melhores hiperparâmetros salvos pela busca.

## Configuração

### Arquitetura do wrapper
- **Encoder:** projeção das features para `d_model` → *positional encoding* senoidal (Vaswani et al., 2017) → dropout → N blocos de self-attention + FFN (com residual e LayerNorm).
- **Decoder:** projeção do input teacher-forced (1 feature, o alvo) para `d_model` → *positional encoding* → dropout → N blocos com **self-attention causal** (`use_causal_mask=True`) + **cross-attention** sobre a saída do encoder + FFN.
- **Saída:** `TimeDistributed(Dense(1, linear))`, compilado com Adam, loss MSE e métrica MAE.

### Hiperparâmetros otimizados pelo Optuna
| Parâmetro | Espaço |
|---|---|
| `learning_rate` | log uniforme, 1e-4 a 1e-2 |
| `d_model` | 32 a 128, passo 16 |
| `num_heads` | 1 a 8 |
| `num_layers` | 1 a 3 |
| `ff_dim` | 64 a 512, passo 32 |
| `dropout_rate` | 0 a 0.3, passo 0.05 |

### Busca (Optuna)
- **Sampler:** `TPESampler` multivariado com agrupamento, seed 42.
- **Pruner:** `MedianPruner` (interrompe trials com `val_loss` acima da mediana).
- **Treino por trial:** batch 32, até 60 épocas, `EarlyStopping` (patience 8) + `ReduceLROnPlateau`.
- **Dados:** 75% treino / 20% validação, target `ws100_wavelet` com denoising wavelet `sym18` nível 2, janela de entrada 72 passos e saída 36 passos.

## Desafios enfrentados

1. **Mudança de API no Keras 3:** a versão instalada (TensorFlow 2.21 / Keras 3) não aceita `use_causal_mask` como argumento do construtor de `MultiHeadAttention` — o primeiro build falhou. A correção foi passar `use_causal_mask=True` **no momento da chamada** do layer.
2. **Verificação de consistência dimensional:** durante a auditoria foi validado empiricamente que o `MultiHeadAttention` do Keras 3 projeta a saída sempre para `d_model` (dimensão do query), portanto não é necessário restringir que `num_heads` divida `d_model` — todas as combinações do espaço de busca são válidas.
3. **Custo computacional:** o Transformer é mais pesado que LSTM/TCN na GTX 1660; cada trial completo levou ~1 minuto já com apenas 2 épocas, o que torna a busca completa (100 trials) demorada.
4. **Auditoria do código:** encontrado e corrigido um import não utilizado (`backend as K`) e linhas longas; todos os arquivos foram validados com `py_compile`, build de borda (ex.: `d_model=96, num_heads=5, num_layers=1`) e testes ponta-a-ponta.

## Resultados dos testes

### Busca Optuna (smoke test, 2 trials, 2 épocas)
- Melhor `val_loss` = **0.006808**
- Melhores parâmetros: `learning_rate=5.6e-4`, `d_model=128`, `num_heads=6`, `num_layers=2`, `ff_dim=128`, `dropout_rate=0.05`
- Resultados salvos em `data/results/optuna/Seq2Seq_Transformer/`.

### Walk-forward (wfo, smoke test, 2 épocas, 344 amostras de teste)
- MAE = **1.346**, RMSE = **1.723**, MSE = 2.968, R² = -0.535, tempo de predição ≈ 27 s.
- Valores indicativos (modelo treinado com apenas 2 épocas e a mini-busca); para resultados conclusivos executar a busca completa (`python tests/optuna_all_hyperparameters.py --wrappers transformer --n-trials 100 --epochs 60`) e depois `python tests/wfo_optuna.py --wrappers transformer`.

## Validação da auditoria

A auditoria foi considerada **bem-sucedida** após as correções: nenhum problema remanescente foi identificado. Todos os entregáveis compilam e foram executados ponta-a-ponta (wrapper, trial Optuna e walk-forward), além do build com combinação de borda de hiperparâmetros.