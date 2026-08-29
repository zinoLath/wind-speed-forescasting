# Investigação: baselines salvos vs reexecução — entendimento corrigido

## Pergunta

O `dataset.csv` não foi alterado (confirmado), mas reexecutar o `wfo_optuna` com os mesmos parâmetros dava MAE ~1.45 (LSTM) / ~1.43 (TCN), enquanto os resultados salvos indicavam MAE 0.17 / 0.28.

## Timeline (evidências)

| Artefato | Data |
|---|---|
| `data/results/wfo_optuna/Seq2Seq_TCN/predictions.csv` (MAE 0.28) | 2026-08-19 |
| Commit `ed158e8` | 2026-07-14 |
| Commit `9fd1567` (modificou `seq2seq_wrapper.py`, `s2s_tcn_wrapper.py`, etc.) | 2026-08-21 |
| `best_trial.json` do TCN (regenerado, novo formato `encoder_filters`) | 2026-08-28 |

Os resultados "bons" (08-19) foram gerados com o código anterior ao commit `9fd1567`.

## O que difere entre as duas versões

### 1. `rolling_forecast` (base `seq2seq_wrapper.py`)

```python
# ANTES (ed158e8) — protocolo intencional:
window = np.vstack([window, test_data[i + self.output_steps - 1]])

# DEPOIS (9fd1567) — mudança errônea:
window = np.vstack([window, test_data[i]])
```

**Semântica correta (antiga):** a previsão começa 36 amostras antes do ponto de comparação; a predição fica 36 amostras atrás dos `actuals`. Ao aplicar o offset de 36 nos `actuals`, eles se alinham com as predições. O "vazamento" resultante é aceitável porque vaza apenas parte da **validação** (que o treinamento não usa para os gradientes) e o teste vai além do espaço de validação para a predição.

### 2. Espaço de hiperparâmetros do Optuna (TCN)

- Antigo: `filters_power` (4–9), `filters = 2**filters_power`, kernel 2–5, dilatação 1–4.
- Novo: `encoder_filters` (32–256, passo 16), kernel 2–3.

Os parâmetros antigos (`filters_power=8` → 256 filtros) também diferem dos atuais (`encoder_filters=32`), mas **não são a causa** da divergência — são apenas formatos diferentes da mesma busca.

## Verificação empírica

Com o `rolling_forecast` antigo restaurado, o TCN com os parâmetros originais reproduz a mesma ordem de grandeza das métricas salvas:

| Configuração | MAE walk-forward |
|---|---|
| Salvo original (TCN) | 0.2829 |
| **Pipeline antigo (com offset) + parâmetros originais** | **0.36–0.60** (depende do LR/treino) |
| Pipeline novo (sem offset) + parâmetros originais convertidos | 1.33 |

## Decisão

A mudança feita no commit `9fd1567` foi **errônea** para o protocolo do projeto. O `rolling_forecast` foi **revertido** ao comportamento antigo (`test_data[i + output_steps - 1]`), alinhando-o com os demais testes que usam esse mesmo protocolo. O "vazamento" de validação é aceitável pelo desenho do experimento.

## Observação

Com o protocolo antigo restaurado, **todos** os modelos (incluindo o Transformer) melhoram o MAE na avaliação walk-forward. A comparação relativa entre modelos deve ser refeita sob esse protocolo.