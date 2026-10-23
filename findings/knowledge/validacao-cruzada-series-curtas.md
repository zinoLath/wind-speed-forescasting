---
title: CV em série curta (53 dias): viável com purge, inviável como k-fold aleatório
area: protocolo
updated: 2026-10-22
---

# CV em série curta (53 dias): viável com purge, inviável como k-fold aleatório

> Atualizado 22/10/2026 com a escolha de K (recomendação K=5).

Resumo: a série LiDAR (7.561 amostras de 10 min ≈ 53 dias) cobre bem a
sazonalidade **diária** (~53 ciclos completos; período diurno = 144 passos) —
o que falta é escala sinótica/sazonal anual, limitação de **qualquer**
protocolo (holdout incluído). K-fold aleatório é inviável não por falta de
ciclos, mas por **sobreposição de janelas**; a forma correta é CV em blocos
cronológicos com purge ≥ 108 passos. Precedente no catálogo: TSCV com gap em
dataset de 10 min de 8.640 pontos (mesmo porte).

## Detalhes

- **Cobertura de escala diária**: 7.561 × 10 min ≈ 52,5 dias = ~53 ciclos
  diurnos completos (144 passos/dia). Qualquer bloco de teste de um k-fold
  (~10 dias) contém ~10 ciclos completos ⇒ a escala diária é amostrada em
  treino e teste de todo fold. A limitação real é a escala **maior** que
  diária: variabilidade sinótica (dias), mensal/anual, interanual — nada
  interno (CV ou holdout) demonstra generalização para regimes ausentes.
- **Por que k-fold aleatório falha**: janela 72→36 com passo 1 ⇒ cada
  amostra cobre 108 passos (18 h) e vizinhas compartilham ~97% do conteúdo;
  k-fold aleatório põe quase-duplicatas em treino e teste ⇒ estimativa
  otimista. Some-se a não-estacionariedade (treinar no futuro para prever o
  passado quebra a premissa de deploy).
- **Antídoto (purge/embargo)**: gap mínimo entre bloco de treino e bloco de
  teste = janela + horizonte = **108 passos (18 h)**. Com purge, o número
  efetivo de blocos semi-independentes é ~7.561/108 ≈ **70** (não 7.5k).
  53 dias comportam 3–5 folds em blocos (teste de ~10 dias/fold) preservando
  a ordem temporal; rolling-origin (expanding window) com poucas origens é
  a variante equivalente.
- **Precedente no catálogo (mesmo porte)**: HTD-MOBBSA-Seq2Seq (Energy,
  2025; `articles/knowledge/markdown/Deep learning combined...md`) usa
  **TSCV** com validação não-adjacente ao treino (gap de um bloco,
  Appendix B2) como função objetivo em dataset de 10 min com **8.640
  pontos (~2 meses)** — praticamente o porte da nossa série. Cita Bergmeir,
  Hyndman & Koo (2018, Comput. Stat. Data Anal.) sobre a validade de CV em
  séries autorregressivas; usa grid search 10-fold CV para SVR.
- **Contexto da literatura**: a maioria dos trabalhos de vento com 10 min
  usa ~1 ano (~52k pontos) com split cronológico simples; parte usa k-fold
  aleatório indevido (com vazamento). CV purgada em 53 dias é mais honesta
  que a prática comum, não impossível.
- **Escolha de K (folds)**: não há número mágico (o 5/10 vem de CV iid —
  Kohavi 1995); limitam-no blocos independentes e comprimento do teste.
  Neste dataset (7.561 amostras, purge 108/fold): K=3 ⇒ teste ~2.520
  (~17,5 d, desperdício ~2,9%); K=5 ⇒ ~1.512 (~10,5 d, ~5,7%); K=10 ⇒
  ~756 (~5,2 d, ~12,9%). Piso K=3 (variância entre folds sem sentido
  abaixo disso); teto prático ~5–8 (bloco de teste encolhe à escala
  sinótica e o desperdício da purge cresce). Recomendação: **K=5** como
  escolha prática equilibrada (K=3 aceitável só para HPO); cada teste de
  1.512 amostras contém ~14 blocos semi-independentes. Alternativa
  equivalente: rolling-origin com 3–5 origens em janela expansiva.
- **Decisão para este projeto**: números headline permanecem no protocolo
  canônico (80/20 + todas as origens rolling no teste, sem vazamento — ver
  `protocolo-avaliacao.md`). CV em blocos com purge de 108 passos é opção
  legítima para HPO/ablações, não para substituir o holdout canônico. Na
  seção de limitações do SBSR, declarar a falta de cobertura de escalas
  > diária (sem ciclo anual), e não "ausência de CV".

## Fontes

- `findings/knowledge/dados-map.md` — 7.561 linhas, 16/09→08/11/2021 (~53
  dias), 10 min.
- `findings/knowledge/protocolo-avaliacao.md` — protocolos canônicos
  (80/20 sem validação; rolling sem vazamento) e regra de HPO/leakage.
- `articles/knowledge/markdown/Deep learning combined wind speed
  forecasting with hybrid time series decomposition and multi-objective
  parameter optimization - ScienceDirect.md` — dataset 10-min de 8.640
  pontos, TSCV com gap (Appendix B2), referência Bergmeir et al. 2018 [75].
- Aritmética própria (conversa 22/10/2026): 144 passos/dia ⇒ ~53 ciclos
  diurnos; purge mínima 72+36 = 108 passos; ~70 blocos semi-independentes.
