---
title: Teste multi-resolução — médias horárias (lstm_mean) e cascata coarse-to-fine (lstm_detail)
area: resultados
updated: 2026-10-23
---

# Cascata multi-resolução: lstm_mean (head 1) + lstm_detail (head 2)

Cascata coarse-to-fine completa: a head 1 prevê a **média de cada hora
futura** (6 saídas, cada uma = média das 6 amostras de 10 min), e a head
2 prevê o **detalhe** de 10 min (resíduo `ws100 − broadcast(médias)`)
condicionada no hint. Predição final = broadcast(médias) + detalhe.
Protocolo canônico 80/20 (seed 42, 60 épocas, HPs da LSTM canônica, loss
MSE, decoder direto); avaliação headline vs `ws100` bruto.

## Wrappers

- `src/models/s2s_lstm_mean_wrapper.py` (`Seq2SeqLSTMMean`, chave
  `lstm_mean`) — saída agrupada em blocos de `POOL=6` (blocos alinhados à
  origem); `create_sequences` próprio; `rolling_forecast` com guard.
- `src/models/s2s_lstm_detail_wrapper.py` (`Seq2SeqLSTMDetail`, chave
  `lstm_detail`) — head 2; sem head anexada degrada para LSTM padrão no
  `target_col` (é o **controle** do experimento). Hint pré-computado pela
  própria pipeline da head (`_head_hint_units`) e stashado por
  `prepare_data`; canal de decoder LEVEL-FREE `(hint − última observação)`;
  `hint_noise` (σ em unidades) para ruído estilo DART **só no treino**.
- Treinos: `scripts/experiments/lidar_8020_hmean_train.py` (head 1) e
  `scripts/experiments/lidar_8020_cascade_train.py` (controle + cascata) →
  `pipeline/tmp/trained_8020_hmean/` e `pipeline/tmp/trained_8020_cascade/`.
  Head 1 custa 83 s; cada head 2, ~115 s.

## Resultados (23/10/2026, 1.416 origens × 36 passos, vs ws100 bruto)

| modelo | MSE | R² |
|---|---|---|
| **CASCATA (head1 + detalhe, σ=0.6)** | **1,552** | **0,440** |
| persistência broadcast | 1,625 | 0,413 |
| controle LSTM raw (sem dica) | 1,652 | 0,403 |
| head1 broadcast (médias) | 1,811 | 0,346 |
| LSTM canônica 36 passos (ITILDE-Q) | — | 0,338 |

- **Por horizonte**: a cascata vence o controle de h1 a h15 com folga
  (h1: **0,871** vs 0,735; h6: 0,633 vs 0,496; h12: 0,495 vs 0,438) e
  empata na cauda (h36: 0,218 vs 0,211). O ganho vem do curto/médio
  prazo, onde o hint (trajetória suave de 6 h) informa.
- `std_ratio` da cascata 0,850 vs controle 0,813 (menos sub-suavização).
- Head 2 isolada: R² do resíduo 0,048 — a habilidade própria é modesta,
  mas a composição com a head 1 ganha no agregado.
- Head 1 (médias horárias) isolada: broadcast R² 0,346 ≈ LSTM canônica
  (0,338) e ≫ LSTM-mse 36 passos mesmos HPs (0,053, run degenerada);
  teto da representação por médias horárias = 0,916 (vs 0,932 do teto
  wavelet) — agregar em horas custa ~1,6 ponto de R².
- Run única; σ do ruído varrido {0, 0.3, 0.6, 0.8, 1.0, 1.35} — pico em
  0.6–0.8; diferenças da ordem do ruído de seed (±0,03–0,07).

## Cascata com TCN (24/10) — não transfere para o modelo forte

Mesmo desenho com os wrappers TCN (`tcn_mean` + `tcn_detail`, criados
via mixins em `src/models/s2s_cascade_mixins.py`; HPs canônicos da TCN
da rodada SBSR; σ=0.6; `scripts/experiments/lidar_8020_tcn_cascade_train.py` →
`pipeline/tmp/trained_8020_tcn_cascade/`). Resultado (1.416 origens,
vs bruto):

| modelo | MSE | R² |
|---|---|---|
| controle TCN raw (sem dica) | 1,556 | **0,438** |
| head1 tcn_mean broadcast | 1,667 | 0,398 |
| CASCATA TCN | 1,689 | 0,390 |

- **A cascata não ajuda a TCN**: detalhe OOS sem habilidade (R² do
  resíduo −0,154); só ganho marginal em h1 (0,650 vs 0,636) e perda de
  h2 a h36. Padrão de boosting: correção residual ajuda o modelo fraco
  (LSTM 0,403 → 0,440) e não o forte (TCN 0,438 → 0,390) — o direto
  TCN já captura a estrutura que o detalhe carregaria.
- Curioso: `tcn_mean` broadcast (0,398) > `lstm_mean` broadcast
  (0,346) — a head 1 TCN é melhor; o gargalo é a head 2.
- TCN treinou em ~176 s (head 1) por modelo; mesmos HPs canônicos SBSR.

## Armadilhas encontradas (todas oneraram o resultado até corrigir)

1. **Hint in-sample vaza**: com a head 1 prevendo o próprio treino, a
   head 2 aprende correções que não transferem (ruído σ≈erro OOS da head
   resolve — DART-style; ruído na avaliação deve ficar DESLIGADO).
2. **Atalho de nível**: `y = raw − hint` anticorrelaciona com o canal de
   hint; com queda do nível médio (treino 9,5 → teste 8,2 m/s) o atalho
   `ŷ ≈ α − β·hint` gera viés de +2,5. Correção: canal LEVEL-FREE
   `(hint − última observação)`.
3. **Semântica do MinMax**: `inverse(s) = s·rng + data_min` — a
   composição `inverse(ŷ_s + m_s) = ŷ_u + m̂_u − data_min_` (NÃO `min_`);
   errar isso adiciona viés ≈ +2,5 e derruba o R² para −1,4.
4. **Compartilhar scalers entre head e cascade NÃO funciona**: os alvos
   diferem (ws100 vs ws100_wavelet) ⇒ as listas de colunas escaladas
   trocam uma coluna e o `scaler_other` compartilhado desalinha todas as
   colunas após a posição do alvo (diff até 1,46 no espaço escalado).
5. **Two-fold honesto cross-regime falha**: heads parciais treinadas em
   regimes diferentes (9,5 vs 6,1 m/s) geram hints encolhidos/lixo.
6. `align_decoder_channels` corta canais do decoder quando o wrapper foi
   preparado com `create_sequences=False` (`future_indices` indefinido) —
   reconstruir wrappers salvos SEMPRE com `create_sequences=True`.

## Fontes

- `scripts/experiments/lidar_8020_hmean_train.py`; `scripts/experiments/lidar_8020_cascade_train.py`;
  `pipeline/tmp/trained_8020_cascade/evaluation.json`.
- Contexto: findings `wavelet-denoise`, `predicoes-diagnostico`,
  `estudos-ablacoes` (ablação mseloss), `rodada-final-sbsr` (detalhe
  previsível, R²=0,455 no D1+D2).
