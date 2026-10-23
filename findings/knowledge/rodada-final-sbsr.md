---
title: Rodada final do paper — unificação MSE e report lidar_8020_sbsr
area: artigos
updated: 2026-10-22
---

# Rodada final do paper — unificação MSE e report `lidar_8020_sbsr`

Decisões do autor (22/10/2026) que regem a última rodada antes da
integração no artigo SBSR.

## Decisões registradas

1. **Loss canônica = MSE pura para TODOS os wrappers** (inclui LSTM e
   LSTM-Bi). **ITILDE-Q está descartada** do paper. Fundamentação já
   coletada: terms search (α=0 venceu todos os combos, 0,422 vs 0,26–0,37
   em 1 seed), ablação de coeficientes (não confirmou ganho), seeds
   ITILDE-Q fracos vs persistência em `all` (Wilcoxon 0–10, p=0,002).
2. **Narrativa do ARIMA/persistência em horizontes curtos** (vence por
   correlação temporal no curto, perde nos horizontes futuros) será
   **citada no artigo** — o autor redige; candidatos na coleção:
   `shengxiang2022deep`, `nascimentotransformer` (SARIMA > LSTM/GRU em
   vento offshore horário), contexto em `metodos-complementares.md`.
3. **A narrativa de superioridade da TCN** (rank Friedman, sem pós-hoc
   Holm TCN×LSTM) será redigida **pelo autor no artigo** — não cabe ao
   repo escolher a prosa.
4. Depois da unificação completa da metodologia (grill-me), haverá **um
   último treino** produzindo o relatório **`reports/lidar/8020-sbsr`**
   (nome reservado; modelos em `pipeline/tmp/trained_8020_sbsr`) — essa
   será a rodada que alimenta `tab:error_all`/`tab:error_h36`, Resumo e
   figuras do SBSR (via sbsr-integrate).

## Estado do grill-me (metodologia unificada)

Decidido pelo autor (22/10/2026):

- **Loss**: MSE pura em todos (acima).
- **Seeds**: **1 seed (42)** na rodada final — testes estatísticos do
  paper ficam nos blocos de tempo (DM ajustado/Friedman/KSPA), não por
  seed.
- **HPs**: canônicos (TCNs 16/10 + dilatações C′; LSTMs do HPO 80/20).
- **Baselines**: sim, avaliados na mesma grade (`sem --no-baselines`).
- **Saída**: modelos em `pipeline/tmp/trained_8020_sbsr`, relatório
  `reports/lidar/8020-sbsr`.

**Clarificação featurex (verificado no código + `dataset.csv`)**: a
canônica 46 = ws/v/dir_sin/cos × 11 alturas (40–140) + hora cíclica;
`press`/`disp*`/`vdisp*` **nunca entram** (só via `keep_raw`). Logo a
rodada featurex 67 **já é** "featurex sem disp/press" e já foi treinada
para os 4 wrappers (treino: TCN 0,512 / LSTM 0,498 / TCN-Bi 0,416 /
LSTM-Bi 0,386; avaliação completa: 0,494/0,485/0,401/0,379 vs ARIMA
0,508 · RF 0,418 · Pers 0,388). Toxicidade de press/disp foi medida nas
ablações h3cis/h3cisvdisp (adicionadas como extras).

- [ ] **Feature set pendente de decisão**: 46 único vs 67 único vs
  misto (ex.: TCNs 46 + LSTMs 67) — tensão: TCN-Bi prefere 46 (0,490 vs
  0,416); LSTMs preferem 67; TCN empatada (0,519 vs 0,512).
  **Resultado do teste 46+meteo (22/10 17h)**: TCN **piorou**
  (MSE 1,4343/R² 0,482 vs 1,2773 canônica) — TCN fica no 46;
  TCN-Bi **melhorou** (MSE 1,3071/R² **0,528** vs 1,3628/~0,50) —
  melhor número dela em todas as rodadas, mas ganho (+0,03) dentro do
  piso de ruído de retreino ±0,04 ("empate a favor"). Models:
  `pipeline/tmp/trained_8020_46met`; log `pipeline/tmp/46met.log`.
- **Feature set dos LSTMs testado (22/10 ~17h30)**: 46+meteo nos LSTMs
  (`trained_8020_46met_lstm`, log `46met_lstm.log`): LSTM R² 0,440 /
  MSE 1,550; LSTM-Bi 0,312/1,906 — **não supera o 67** (0,498/0,386);
  o diferencial das recorrentes é o cisalhamento cis1..19, não a meteo.
- **GRILL-ME FECHADO (22/10 ~18h) — metodologia unificada da rodada
  final**: lineup decidido pelo autor:
  **TCN 48 (46+meteo) · TCN-Bi 48 · LSTM 67 (featurex) · LSTM-Bi 67**.
  Justificativa do autor: a queda da TCN no 48 (0,519→0,482) parece
  ruído de seed e a inclusão de meteo faz mais sentido para a narrativa
  do paper; para as LSTMs fica featurex (cis1..19 é o diferencial).
  Grade completa de evidência no INDEX de `estudos-ablacoes.md`.
  Protocolo: alvo `ws100_wavelet`, 60 épocas, seed 42, split 80/20 sem
  validação, HPs canônicos (TCN + dilatações C′), MSE pura em todos,
  baselines na mesma grade. Saída: `pipeline/tmp/trained_8020_sbsr` →
  `data/results/report_8020_sbsr` → `reports/lidar/8020-sbsr`.
  Treinador: `scripts/lidar_8020_sbsr_train.py` (feature_keep por
  wrapper).

## Resultado da rodada final (22/10/2026) — PROTOCOLO ATUAL: -72 origens

**Decisão do autor (22/10)**: descartar as primeiras 72 origens do
teste (entrada cruzava treino) para TODOS os métodos — evita
argumentação de vazamento. Implementado como `--trim-boundary-origins
72` em `tests/relatorio_resultados.py` (corta dense, rolling e
baselines; recalcula as métricas ARIMA/RF/persistência na grade
cortada). Run oficial #5, relatório `reports/lidar/8020-sbsr`:

| Método | R² all | MSE | rolling R² |
|---|---|---|---|
| **TCN-Bi** | **0,525** | 1,322 | +0,377 |
| **ARIMA** | **0,525** | **1,320** | — |
| LSTM | 0,499 | 1,393 | +0,427 |
| TCN | 0,472 | 1,470 | +0,322 |
| RF | 0,433 | 1,576 | — |
| LSTM-Bi | 0,388 | 1,702 | +0,347 |
| Persistência | 0,417 | 1,621 | — |

MCS α=0,10: 7/7 no conjunto (LSTM-Bi com p exclusão 0,112, no limiar);
DM 10 pares, 0 significativos. **Empate técnico no topo: TCN-Bi ×
ARIMA** (ΔMSE 0,002; as origens de fronteira favoreciam as DLs vs o
ARIMA). Narrativa do paper segue por horizonte: dominância DL nos
h≥18–36 (ver tabela de cruzamento abaixo). Rolling melhorou para todos
com o corte (LSTM +0,427). Sem corte (run #4, referência): TCN-Bi
0,501 / ARIMA 0,508 / LSTM 0,485 / TCN 0,452 / LSTM-Bi 0,379 /
RF 0,418 / Pers 0,388.

## Variância de seed da LSTM-Bi (sbsr)

`scripts/lidar_8020_sbsr_seeds_train.py` →
`pipeline/tmp/trained_8020_sbsr_seeds/lstm_bi/` (seeds 43–46, mesmo
protocolo oficial com filtro de observadas): R² all
**0,312 ± 0,009** (0,302–0,323), rolling 0,195 ± 0,016. **A seed 42
canônica (0,379) é outlier favorável**: +0,067 acima da média das 4 ≈
7σ; com as 5 seeds (42–46): 0,326 ± 0,031. Consequência: o número
0,379 da tabela do SBSR é de uma seed sortuda — a tendência central da
LSTM-Bi featurex+MSE é ~0,31–0,33. Estabilidade da arquitetura em si
melhorou muito vs o estudo ITILDE-Q (±0,009 vs ±0,045).

## Reversão da LSTM-Bi (decisão final do lineup)

Autor reverteu para HPs canônicos (run #4, oficial, **estado atual**):
LSTM-Bi 0,379/1,679 de volta; **MCS α=0,10 = 7/7** (todas no conjunto,
incl. persistência); DM 10 pares 0 significativos. HPO wd1e-3
descartado (0,357, fora do MCS). Backups em
`trained_8020_sbsr_backup_20261022/{lstm_bi_canhp42,lstm_bi_hpowd1e3}`.

## Dynagate nos 4 canônicos (22/10) — resultado NEGATIVO

Teste do autor ("só pra ver"): 4 canônicas com persistence gate
**dinâmico** (`scripts/lidar_8020_sbsr_train.py --gate dynamic` →
`trained_8020_sbsr_dynagate`; mesma seed/HPs/features; camada
`output_dynamic_gate` verificada nos 4 .keras). Métrica log rápida
(pooled dense, sem filtro):

| Modelo | sem gate | dynagate | Δ |
|---|---|---|---|
| LSTM | 0,498 | 0,362 | −0,137 |
| TCN-Bi | 0,525 | 0,459 | −0,066 |
| LSTM-Bi | 0,386 | 0,331 | −0,055 |
| TCN | 0,472 | 0,466 | −0,005 |

**Dynagate piora todos** — misturar de volta à persistência custa nos
horizontes onde as DLs ganham. Fora da canônica; sem avaliação oficial.
Gotcha de implementação (custou 1 retreino): `Seq2SeqWrapper.prepare()`
SOBRESCREVE `self.persistence_gate` com o kwarg default `False`
(seq2seq_wrapper.py:254) — o flag precisa ser passado **como argumento
do prepare()**, não como atributo prévio (o `gate_mode` pode ir após,
prepare não o toca).

## Persistência no MCS — leitura por horizonte (argumento do paper)

(números na grade completa pré-corte; a conclusão qualitativa —
cruzamento h≈6–12 e dominância DL nos horizontes finais — mantém-se na
grade -72 origens)

Persistência no MCS agregado ≠ derrota: decompondo (persistência =
último observado, teste 80/20, R² por horizonte):

| h | Pers. | LSTM | LSTM-Bi | TCN | TCN-Bi |
|---|---|---|---|---|---|
| 1 | **0,870** | 0,818 | 0,822 | 0,763 | 0,836 |
| 6 | **0,610** | 0,513 | 0,298 | 0,546 | 0,626 |
| 12 | 0,457 | 0,474 | 0,366 | 0,461 | **0,516** |
| 24 | 0,243 | 0,443 | 0,338 | 0,407 | **0,447** |
| 36 | 0,112 | **0,417** | 0,339 | 0,308 | 0,357 |

- Persistência imbatível em h=1–2 (correlação temporal de 10 min);
  **cruzamento h≈6–12**; de h≈18–36 os modelos dão **2–4× o R²** dela
  (pers. decai ~8×, modelos ~2×). R² agregado: TCN-Bi 0,501 vs pers.
  0,388.
- MCS agregado mantém a persistência por **variância entre blocos
  diários** (~11 dias), não por média — narrativa do paper: MCS/DM
  honestos no agregado + decomposição por horizonte = exatamente a
  citação planejada ("baseline ganha no curto por correlação temporal,
  perde nos horizontes futuros" — candidatos: `shengxiang2022deep`,
  `nascimentotransformer`).
  `nascimentotransformer`).
- [ ] Alvo `ws100_wavelet` e 60 épocas fixas mantidos (protocolo).

## Fontes

- Decisões: sessão com o autor 22/10/2026.
- Evidências: `estudos-ablacoes.md`; `metodos-complementares.md`;
  `seeds-literatura.md`.
