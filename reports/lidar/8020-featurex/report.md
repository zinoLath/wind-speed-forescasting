# Rodada 8020-featurex — conjunto de atributos estendido (67 canais, MSE puro)

Rodada de **estudo** (não canônica) de 21/10/2026, respondendo à demanda da
Vanessa (`V-6ace8ce9` — "tabela com o dataset completo, o atributo, se
coletado ou computado e seu significado") e às decisões do autor tomadas ao
longo da análise (kanban c-13/c-34 da sessão-comentarios-sbsr-1).

**Pergunta:** o que acontece se os modelos consumirem também o
cisalhamento (`cis1..19`) e as meteorológicas `humid`/`temp` — hoje
descartados pelo `load_dataset` — com perda **MSE pura em todos os
wrappers** (ITILDE-Q fora)?

## 1. Decisões que moldaram a rodada (21/10)

| Decisão | Status |
|---|---|
| `dir{h}` limitada a 40–140 m | **já era assim** na entrada dos modelos (150–260 m descartados) — nada a fazer |
| Incluir `cis1..19` | ✓ entrou (`keep_raw`) |
| Incluir `humid`, `temp` | ✓ entraram (`keep_raw`); `press` ficou fora |
| `disp`/`vdisp` | continuam fora (não pedidos) |
| Perda | **MSE puro em todos** (LSTM/LSTM-Bi saem da ITILDE-Q) |
| HPs | copiados da canônica 16/10 (`trained_8020_itildeq/*/model.json`) — **não** da Round 5 (revertida por transferência negativa) |

Mudança de infra necessária: `common.forecast_redundant_columns` passou a
respeitar `keep` também para `cis*` e metadados (antes era descarte
incondicional — o primeiro disparo da rodada treinou sem os atributos novos
por causa disso e virou a rodada de reprodutibilidade abaixo). O avaliador
(`reports/src/relatorio_resultados.py`) agora lê `feature_keep` do `model.json`.

## 2. Dicionário de atributos (fonte: sessão-comentarios-sbsr-1 §1.3.1)

**Entra nos modelos — 46 canais + alvo (canônica):**

| Atributo | Qtd | Origem | Significado |
|---|---|---|---|
| `ws{40..140}` | 11 | Coletado (LiDAR) | Velocidade horizontal (m/s) na altura h; passo 10 m |
| `v{40..140}` | 11 | Coletado (LiDAR) | Componente vertical (m/s) |
| `dir{h}_sin/cos` | 22 | Computado | Direção no círculo unitário (evita 359°→1°) |
| `hour_sin/cos` | 2 | Computado | Hora do dia cíclica |
| `ws100_wavelet` (alvo) | 1 | Computado | ws100 suavizada (sym18, nível 2, limiar soft) |

**+21 na featurex:** `cis{1..19}` (computado — cisalhamento, derivada da
velocidade na dimensão altura), `humid` (coletado, % ), `temp` (coletado, °C)
⇒ **67 colunas / 69 canais de encoder**.

**Seguem descartados:** `ws/v/dir{150..260}` (27), `disp{}` (20),
`vdisp{}` (20), `dir{h}` em graus, `press`.

Por que `cis` tinha sido tirado: redundância estrutural com o perfil `ws`
(é combinação linear das alturas vizinhas). O estudo univariado anterior
(`reports/resultados/features_study`) testou uma formulação de shear
(ws260−ws40): MI 0,118/0,057/0,089 em h1/h12/h36 — fraco, mas nunca testado
como os 19 `cis` locais. A rodada fecha essa lacuna empiricamente.

## 3. Resultados (quick-eval de treino) (quick-eval de treino — MESMO protocolo entre as rodadas)

R² no teste (janela 72→36, split 80/20, seed 42). **Não** é o número do
avaliador completo (sem rolling, sem filtro de amostras observadas).

| Modelo | Canônica 16/10 | Repro42 (piso de ruído) | **Featurex** | Δ vs canônica |
|---|---|---|---|---|
| LSTM | 0,318 (ITILDE-Q) | 0,332 (ITILDE-Q) | **0,498** (MSE) | **+0,180** |
| LSTM-Bi | 0,329 (ITILDE-Q) | 0,340 (ITILDE-Q) | 0,386 (MSE) | +0,057 |
| TCN | 0,519 | 0,510 | 0,512 | −0,007 |
| TCN-Bi | 0,490 | 0,533 | 0,416 | −0,117 |

MSE absolutos (featurex): LSTM 1,389 · LSTM-Bi 1,699 · TCN 1,352 · TCN-Bi 1,617.

### 3.1 Piso de ruído de retreino (rodada `trained_8020_repro42`)

O primeiro disparo da featurex saiu **sem** os atributos novos (bug do
`keep_raw`, corrigido) — e virou medição de variância: mesmas features,
HPs, seed e perda da canônica, resultado:

TCN-Bi **+0,043**, LSTM +0,014, LSTM-Bi +0,011, TCN −0,009.
**Diferenças abaixo de ≈±0,04 entre rodadas são ruído.** É a régua para
todas as leituras acima.

### 3.2 Variantes da LSTM-Bi (67 feat, MSE)

| Variante | encoder_dropout | Train loss final | R² teste |
|---|---|---|---|
| U=256 (featurex) | 0,083 | 0,00090 | **0,386** |
| U=128 (`trained_8020_featurex_lbi128`) | 0,083 | 0,00090 | 0,211 |
| enc-drop 0,25 (`..._lbi_ed025`) | **0,25** | 0,00127 | 0,313 |

## 4. Leitura

1. **A ITILDE-Q era o gargalo das LSTMs.** LSTM com MSE salta
   0,32 → 0,50 (bem acima do ruído) e passa a competir com a TCN
   (0,51). Para as LSTMs a rodada muda loss **e** features juntos — o
   crédito exato entre os dois exige o A/B "MSE com 46 features"
   (pendente, §6).
2. **TCN é insensível aos atributos novos** (−0,007 = ruído). TCN-Bi
   **caiu** 0,117 — bem fora do ruído; atributos extras parecem ter
   atrapalhado a bidirecional (ou o modelo é intrinsecamente instável:
   a própria repro dela oscilou +0,043).
3. **LSTM-Bi está em regime de interpolação**: memoriza o treino
   (loss ≈ 0,0009) com 256 **ou** 128 unidades; as alavancas clássicas
   pioram (−capacidade: −0,18; +dropout: −0,07). Os HPs canônicos do HPO
   (0,083/256) resistem a qualquer ajuste manual. Tuning adicional sem
   validação no protocolo seria seleção no conjunto de teste — parado aqui.
4. **Sobre o decoder 2×** (descoberta de implementação): o decoder do
   LSTM-Bi é `2×lstm_units` por compatibilidade dimensional com o estado
   concatenado forward‖backward — herdado **hardcoded** da implementação
   original (14/07, `LSTM(256*2)`) e jamais revisitado. Toda `lstm_units`
   escolhida pela HPO chega com decoder em dobro embutido. Corrigir isso
   (projeção do estado ou decoder independente) é mudança de arquitetura —
   rodada nova com HPO próprio.
5. **Contexto da literatura** (coleção local): Lee et al. (Shape-Aware
   Seq2Seq, origem da ITILDE-Q) usa **BiGRU no decoder** (direto); na
   coleção, BiLSTM/BiGRU fora do encoder aparecem como preditor standalone
   ou membro de ensemble. A implementação local (decoder unidirecional)
   segue a vertente conservadora e é fiel à alegação do artigo
   ("bidirecionalidade no encoder").

## 5. Avaliação completa (padrão artigo) e heatmap

Regeneração via `reports/src/relatorio_resultados.py` (rolling sobre o teste,
**filtro de amostras observadas** — os 51 slots interpolados nunca são
pontuados, per-horizonte, DM/MCS com baselines na mesma grade):

| Modelo | R² (all, avaliado) | MSE |
|---|---|---|
| TCN | **0,494** | 1,367 |
| LSTM | 0,485 | 1,393 |
| LSTM-Bi | 0,379 | 1,679 |
| TCN-Bi | 0,401 | 1,618 |
| ARIMA *(baseline)* | 0,508 | 1,329 |
| Random Forest *(baseline)* | 0,418 | 1,572 |
| Persistência *(baseline)* | 0,388 | 1,653 |

Mesma grade de origens para todos (4 DL + 3 baselines); métricas dos
baselines de `data/results/baselines/arima_rf/metrics.json`
(`all_horizons`, 53208 pontos — mesmo protocolo da rodada canônica).
No agregado o ARIMA segue à frente dos 4 (0,508), com TCN e LSTM
empatadas logo atrás; **MCS (α=0,10) mantém só a TCN**; **DM: 10 pares,
0 significativos** no agregado (p<0,05) — nenhuma diferença entre os 7 é
estatisticamente distinguível, consistente com o piso de ruído (§3.1).

**Heatmap de MSE por modelo × horizonte** (mesmo recorte dos heatmaps do
artigo; fontes grandes já na versão nova do gerador):

![Heatmap de MSE por modelo e horizonte](figures/06_heatmap_mse_horizonte.png)

**Heatmap de R² por modelo × horizonte:**

![Heatmap de R² por modelo e horizonte](figures/06_heatmap_r2_horizonte.png)

Leitura do heatmap: TCN e LSTM disputam a liderança até ~h18 (ARIMA à
frente no início, padrão conhecido); nos horizontes longos (24/30/36) as
redes se destacam — com a LSTM-MSE de 67 canais colada na TCN em
praticamente toda a extensão.

## 6. Estado e promoção

- **Status: rodada de estudo.** Nada foi promovido; a canônica do artigo
  segue sendo a mista 16/10 (`trained_8020_itildeq`).
- Critério de promoção (se o autor decidir): regeneração completa em
  padrão artigo (abaixo) com rolling + filtro de amostras observadas,
  var. implícita única e DM; ganho da LSTM mantido acima do piso de
  ruído; decisão de narrativa do artigo (hoje: "TCN vs LSTM" ficaria
  equilibrado — LSTM-MSE 0,50 vs TCN 0,51 — e a ITILDE-Q viraria ablação).

## 7. Próximos passos

1. **Regeneração completa** da featurex (~5 min):
   `python reports/src/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020_featurex --out-dir data/results/report_8020_featurex`
2. **A/B que separa loss de features**: LSTM unidirecional, MSE,
   **46 features** canônicas (responde: o ganho de 0,32→0,50 foi da MSE
   ou dos 21 atributos?).
3. Decisão do autor sobre o c-34 (texto do `03` vs pipeline) à luz dos
   resultados: os atributos extras não ajudaram TCN e LSTM-Bi — a frase
   da Vanessa sobre "dispersão/cisalhamento incluídos" continua sem base
   no pipeline canônico.

## 8. Artefatos

| Item | Caminho |
|---|---|
| Script da rodada | `scripts/experiments/lidar_8020_featurex_train.py` (flags `--units`, `--enc-dropout`, `--wrappers`) |
| Modelos featurex | `pipeline/tmp/trained_8020_featurex/` |
| Piso de ruído | `pipeline/tmp/trained_8020_repro42/` (log `lidar_8020_repro42_train.log`) |
| Variantes LSTM-Bi | `pipeline/tmp/trained_8020_featurex_lbi128/`, `..._lbi_ed025/` |
| Logs | `pipeline/tmp/lidar_8020_featurex_train.log`, `pipeline/tmp/featurex_lbi128.log`, `pipeline/tmp/featurex_lbi_ed025.log` |
| Patches de infra | `src/common.py` (`keep` em `forecast_redundant_columns`), `reports/src/relatorio_resultados.py` (`feature_keep` do `model.json`) |
| Dicionário canônico | `reports/protocolo/sessao-comentarios-sbsr-1/` §1.3.1 |
