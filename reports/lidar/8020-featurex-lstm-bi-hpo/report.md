# HPO LSTM-Bi featurex — 67 canais, MSE, dropout ≥ 0,20, wd fixo

Estudo Optuna local (21/10/2026) sobre a **LSTM-Bi** com o conjunto de
atributos estendido da rodada featurex (`cis1..19 + humid + temp` ⇒ 67
colunas) e perda **MSE pura**. Motivação: a LSTM-Bi memoriza o treino em
qualquer capacidade testada (regime de interpolação) — a busca restringiu
o espaço para regularização alta: **encoder/decoder dropout em
[0,20; 0,45]** e **weight_decay fixo em 1e-3** (AdamW, fora da busca).

Receita: batch 32 travado, split interno 90/10 (teste de 20% intocado),
MedianPruner, TPE seed 42, 60 épocas, trial âncora com HPs canônicos
clampeados ao novo espaço.

Trials: **40** (37 completos, 3 podados).
Top-5 por val_loss:

| # | val_loss | units | lr | enc drop | dec drop |
|---|---|---|---|---|---|
| 21 | 0.01029 | 160 | 0.0092 | 0.36 | 0.44 |
| 7 | 0.01061 | 160 | 0.0051 | 0.35 | 0.43 |
| 24 | 0.01079 | 160 | 0.0074 | 0.42 | 0.44 |
| 9 | 0.01130 | 256 | 0.0023 | 0.39 | 0.25 |
| 25 | 0.01136 | 160 | 0.0115 | 0.42 | 0.45 |

## Melhor trial

```
{
 "learning_rate": 0.009210061561670112,
 "lstm_units": 160,
 "encoder_dropout_rate": 0.36037018703202256,
 "decoder_dropout_rate": 0.4423174630121795,
 "batch_size": 32
}
```

val_loss = **0.01029**

## Resultado no teste (treino completo, 60 épocas)

| Métrica | LSTM-Bi HPO featurex |
|---|---|
| MSE | 1.7555 |
| MAE | 1.0053 |
| R² (all) | **0.3659** |
| skill vs persistência | -18.8% |
| rolling MSE | 1.7272 |

## Contexto (linha LSTM-Bi)

| Variante | R² teste |
|---|---|
| Canônica 16/10 (ITILDE-Q, 46 feat) | 0,329 |
| Repro42 (mesmo setup) | 0,340 |
| Featurex U=256 (HPs canônicos) | 0,386 |
| Featurex U=128 | 0,211 |
| Featurex enc-drop 0,25 | 0,313 |
| **HPO featurex (esta rodada)** | **0.3659** |

## Artefatos

- Modelos: `pipeline/tmp/trained_8020_lstm_bi_hpo_featurex/`
- Estudo: `pipeline/tmp/optuna_8020_lstm_bi_featurex/optuna.db`
- Resumo: `data/results/lstm_bi_hpo_featurex_8020/summary.json`
- Log: `pipeline/tmp/lstm_bi_hpo_featurex.log`
