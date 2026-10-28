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
72` em `reports/src/relatorio_resultados.py` (corta dense, rolling e
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

## Potência vs empate real (23/10) — diagnóstico d de Cohen

Com efeito padronizado por par (agregado h1–35, run #5): TCN-Bi×ARIMA
**d=0,02** (precisaria de ~169 dias de teste p/ "detectar"); LSTM×ARIMA
d=0,07 (~16 dias) — **empates reais, não falta de amostra**. Onde o
efeito é grande, os testes acharam: TCN-Bi×TCN d=0,18 (CW p<0,0001),
LSTM×LSTM-Bi d=0,35 (IC fechado h6–h24), TCN-Bi×Pers h24–35 d=0,20
(IC fechado) — ou seja, **o aparato tem potência para efeitos ≥~0,2 e
os encontrou**. Piso independente de amostra: diferenças entre wrappers
< ±0,04 R² (ruído de seed) são indecidíveis a 1 seed. Correção de
rota: DM/HLN TCN×ARIMA corretamente alinhado dá p=0,58 (não
significativo) — o z≈−5 ingênuo era sem o HAC de 35 lags.

## TCN h3 (14col) × 5 seeds (23/10) — o "0,514" era seed sortuda

`trained_8020_sbsr_h3seeds` (script com `--extras none --heights 40 100
140`, receita original da ablação de alturas): R² **0,417 ± 0,072**
(0,339–0,480), rolling 0,300±0,071. Comparação multi-seed das receitas
da TCN-uni: **h3 14col 0,417±0,072 ≈ 48col 0,400±0,031** (nenhuma
diferença significativa) — e o **0,514 do h3 era seed 42 sortuda** (a
seed 43 da ablação original já dava 0,441). A TCN-uni, em qualquer
receita, tem centro ~0,40–0,42 com desvio 0,03–0,07; os 0,51–0,52
históricos eram sorteios. Campo `feature_keep` do meta do h3seeds está
rotulado errado (gravou meteo; modelo real tem 16 features de entrada =
14 col + 2 canais de alvo). Script de seeds agora suporta
`--extras {none,meteo,featurex}` + `--heights`.

## O detalhe fino da wavelet NÃO é ruído (23/10) — revisão do "piso de ruído"

Teste proposto pelo autor (validar H0 "o detalhe é imprevisível"):
**REFUTADO.** O detalhe D1+D2 (20–40 min, 6,6% da variância) tem
estrutura oscilatória: Ljung–Box rejeita ruído branco nos lags 6/12/36
(p≈0,0000; lag 1 branco) e um **ridge com lags 1–6 prevê o detalhe com
R²=0,455** (IC95% blocos diários [0,426; 0,482]; r=0,68). Implicações:
(1) o denoise é escolha pragmática de regularização de treino (A/B a
favor), NÃO porque o detalhe seja imprevisível; (2) o teto R² vs bruto
sobe de 0,932 para ≈0,96 se o detalhe for explorado; (3) ~3% da
variância total é sinal previsível descartado pelo VisuShrink —
trabalho futuro natural: decoder multi-task (wavelet + detalhe) ou
threshold menor; a TCN (dilatações 1–4 = 10–40 min) é a arquitetura
natural para essa oscilação. Artefato: análise inline (sem script
salvo ainda); números no finding. A narrativa "piso de ruído de 6,8%"
NÃO deve ser citada como "irredutível" — é 6,8% dos quais ~45%
previsíveis (irredutível real ≈ 3,6% da variância total).

## Testes de hipótese na rodada 8020-sbsr (23/10)

Bateria sobre perdas por origem (MSE médio nos 36 passos, 1.405 origens,
7 modelos), DM com HLN+HAC lag 35, α=0.05. Artefato:
`data/results/report_8020_sbsr/testes_hipotese.json`.

- **DM pareado (21 pares): NENHUM rejeita H0** após Holm; o único p<0.05
  bruto é LSTM-Bi vs LSTM (p=0.0138, LSTM melhor) — p_Holm=0.29.
  TCN-Bi vs ARIMA p=0.994 — empate técnico do TCN-Bi com o ARIMA
  (consistente com a narrativa do paper).
- **Friedman (origens como blocos): REJEITA** (χ²=222.6, p=2.9e-45) —
  "pelo menos um modelo difere"; ranks: TCN-Bi 3.44 < ARIMA 3.86 ≈ LSTM
  3.89 < TCN 4.01 ≈ P 4.03 < RF 4.19 < LSTM-Bi 4.59.
- **Wilcoxon pareado + Holm: 12/21 pares rejeitam** — separações claras:
  TCN-Bi > LSTM-Bi/RF/P/TCN/LSTM (p<1e-4); LSTM-Bi < todos (p<1e-9);
  LSTM > RF; ARIMA > RF. NS após Holm: TCN vs ARIMA/RF/LSTM/P,
  ARIMA vs P, LSTM vs P, LSTM-Bi vs RF.
- **DM por horizonte (23/10, cálculo do banco, 756 células = 21 pares ×
  36 passos): só 12 rejeitam H0 (1,6%) — e TODAS são ARIMA > RF nos
  passos 1–12**. Nenhum par entre redes, nem vs persistência, rejeita
  com DM(HLN+HAC) a 5%. Potência, não ausência de diferença: ~11 dias
  efetivos + HAC lag h−1. A evidência horizonte-wise do paper fica no
  bootstrap em blocos (IC fecha em h24/h35 vs persistência) e o DM
  serve como confirmação conservadora de empate de perdas médias.
- Leitura: DM (conservador, HAC) não distingue perdas médias; Wilcoxon
  capta diferenças distribucionais — usar os dois no paper (DM como
  "sem diferença de perda média comprovada" + Wilcoxon/Holm para ordem).
- **No report do hub**: `reports/lidar/8020-sbsr/report.py` ganhou a
  seção "Testes de hipótese" (tabelas DM 21 pares, Friedman/ranks,
  Wilcoxon+Holm e MCS multi-α, todas com popup LaTeX tab:dm_pares /
  tab:friedman_ranks / tab:wilcoxon_pares / tab:mcs_multialpha), lida de
  `testes_hipotese.json` + `mcs_multialpha.json` no source_dir.

## MCS: sem valor de comentário no paper (decisão do autor, 23/10)


MCS agregado = 7/7 e **MCS horizonte a horizonte (23/10,
`report_8020_sbsr/mcs_horizon.json`, 1.000 boots/bloco 36) = 7/7 em
todos os 35 horizontes** — nem a persistência sai em h35. Motivo:
correção de comparação múltipla (estatística de máximo vs o melhor do
conjunto) + ~11 dias efetivos ⇒ potência zero para excluir qualquer um.
**Multi-α (23/10, `report_8020_sbsr/mcs_multialpha.json`, 2.000 boots):
7/7 até α=0.01; só α=0.25 exclui LSTM-Bi (p=0.111)** — único p de
exclusão não-censurado; TCN-Bi/TCN/LSTM/ARIMA/RF/P nunca entraram em
rodada de eliminação.
Decisão: **MCS fora do texto do artigo** (ou uma linha de honestidade);
o suporte estatístico do paper passa por Clark & West aninhado (TCN-Bi >
TCN, p<0,0001), heatmap de bootstrap em blocos (TCN-Bi/LSTM > Pers. com
IC fechado em h24/h35) e Friedman global. Nota técnica: o IC par a par
do heatmap fecha em h35 onde o MCS não exclui — MCS é mais conservador
(máximo + correção múltipla) que o IC par a par.

## Friedman global na rodada canônica (23/10) — marginal, não citar a 5%

Reavaliado na run #5 (11 dias, MSE diário, 4 canônicas):
**χ²=6,82, p=0,078 — NÃO significativo a 5%** (o χ²=11,4/p=0,0098
citado antes vinha da run #1, rodada ponderada antiga). Ranks médios:
TCN-Bi 2,0 < LSTM 2,09 < TCN 2,64 < LSTM-Bi 3,27. Suporte estatístico
do paper fica reduzido a: **Clark & West aninhado** (p<0,0001) + **ICs
par a par do bootstrap em blocos** (heatmap) — Friedman só como
"marginal a 10%" ou fora do texto. Força a tese de que 11 dias de teste
não sustentam claims de superioridade no agregado; a narrativa
descritiva por horizonte + CW é o que resta.

## Clark & West nos pares aninhados (23/10/2026)

CW MSPE-ajustado (`src/utils.py:clark_west`), agregado por origem,
run #5 (grade −72), restrito = uni, irrestrito = Bi:

- **TCN × TCN-Bi: CW = +4,25, p < 0,0001 → bidirecionalidade ajuda
  SIGNIFICATIVAMENTE na TCN** — primeiro resultado significativo a
  favor de qualquer modelo na rodada; o DM não captura porque é
  inválido/subdimensionado em comparações aninhadas (exatamente a
  crítica de Clark & West 2007).
- **LSTM × LSTM-Bi: CW = −0,49, p = 0,62** → Bi não supera a uni
  (nada significativo, nem por horizonte).
- Por horizonte: 0 significativos em ambos os pares (o sinal do TCN
  par é agregado).

Narrativa do paper: a alegação de bidirecionalidade se sustenta
estatisticamente **na TCN** com o teste apropriado a modelos
aninhados (cite Clark & West 2007); nos LSTMs, não.

## CW TCN-Bi × ARIMA (23/10) — NÃO USAR (par não-aninhado)

Rodado a pedido, mas **inválido como evidência**: ARIMA não está
aninhado na TCN-Bi — fora do aninhamento o termo de correção
(A−T)² do CW sempre soma a favor do modelo irrestrito, e o resultado
denuncia o artefato: "significativo a favor da TCN-Bi em h1–18"
exatamente onde as predições mais divergem, **inclusive nos horizontes
h1–17 em que o MSPE cru do ARIMA é MELHOR** (R² h1: ARIMA 0,888 vs
TCN-Bi 0,844). CW agregado +2,09/p=0,037 = viés, não achado. Para
DL × ARIMA o instrumento continua sendo DM/HLN (0 significativos no
agregado) + comparação descritiva por horizonte. Nota de alinhamento:
baseline (origin, h=1) mira a própria origem — o par com o DL exige
shift de +10 min na origem do baseline.

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

## Rodada de 10 seeds (1–10) no Kaggle (23/10, em andamento)

`scripts/sbsr_seeds_kaggle.py` + `kaggle_seeds_dispatch.py`: 4 kernels
(1 por wrapper), seeds **1–10** (convenção nova a pedido do autor),
grade 80/20 com filtro de observadas. Kernels leves (lstm, tcn) rodando
na conta joaohenriqueflauzino (~0,9h da cota); pesados (lstm_bi,
tcn_bi) preparados em `dist/kaggle/kernel-sbsr10-*`, pendentes de
envio (cota). Saída: `pipeline/tmp/sbsr10/gpu{0,1}/<wrapper>/` →
`collect` consolida mean±std. Motivação: 5 seeds mostraram que
TCN-uni honesta = 0,400±0,031 (48col) / 0,417±0,072 (h3) e que os
0,51–0,52 históricos eram seeds sortudas; 10 seeds permitem Wilcoxon
pareado (n=5 nunca alcança p<0,05 bilateral) e médias com IC útil.
Nota: com seeds 1–10, lembrar que as seeds 42–51 dos estudos antigos
são INDEPENDENTES desta rodada (não somar amostras).

## Fontes

- Decisões: sessão com o autor 22/10/2026.
- Evidências: `estudos-ablacoes.md`; `metodos-complementares.md`;
  `seeds-literatura.md`.
