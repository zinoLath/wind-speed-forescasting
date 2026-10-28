---
title: Testes de hipótese na literatura da base — H0, pares e vereditos
area: artigos
updated: 2026-10-23
---

# H0, modelos comparados e vereditos nos artigos da base

Varredura completa em `articles/knowledge/markdown/` (kb-scout, 23/10/2026;
consolida `docs/revisions/sbr/kbscout/result-5.md`). Para cada artigo com
teste de hipótese: H0 formal, pares comparados e veredito.

## H0 por família de teste

| Teste | H0 medida |
|---|---|
| Diebold–Mariano (DM) | E(d_t)=0, d_t = L(e₁)−L(e₂) com L quadrática — **mesma acurácia preditiva esperada** entre dois modelos |
| Friedman | ranks médios dos modelos iguais (sem diferença entre medianas de desempenho) |
| Wilcoxon pareado | as duas amostras pareadas vêm da mesma distribuição |
| t-Student pareado | diferença média de erros nula entre dois modelos |
| Giacomini–White (GW) | perda esperada **condicional** (dado information set) igual entre dois modelos |
| Kruskal–Wallis (KW) | todos os grupos têm a mesma distribuição/mediana |

## Vereditos por artigo

**Rejeita H0 limpo (modelo proposto > comparados):**
- `wei2026secondary` (SD-TCGRU) — DM: proposto vs CNN-LSTM-AM, CNN-ATTEEMD, VMD-CNN-GRU, EMD-KM-SXL; "DM negative across all datasets, p < 0.05".
- `12026s2b` (DWT+1D-CNN+Bi-LSTM+ARSO) — DM por horizonte (30/40/60 min, 10 runs) vs LSTM, 1D-CNN, GRU, Bi-LSTM, 1D-CNN+Bi-LSTM; "clear rejection of the null".
- `leongomez2026multi` (RFF-RNN) — Friedman com horizontes como blocos vs SRNN/GRU/LSTM/Transformer/Autoformer/FEDformer, 7 datasets; p ≪ 0.05 em todos.
- `zhao2027mcbformer` (MCBformer) — Friedman em 24 blocos temporais vs 15 métodos; "Reject H0" nos 15 pares (p ≤ 0.00763).
- `nascimentotransformer` (Transformer+wavelet, Bahia) — Wilcoxon vs LSTM fine-tuned, 3 torres; "H0 was rejected for all towers" (p de 4,5e-35 a 1,85e-13).
- `aslam2026physics` (VORTEX) — Wilcoxon vs baselines, 4 datasets × 3 métricas; p < 0.01 em tudo.
- `cao2026probabilistic` — t pareado em 12 sazonalidades, saída paralela vs recursiva; p < 0.05.
- `sridhar2026predictive` (BiPE-LSTM) — t pareado vs RF p<0.001, demais p<0.05.
- `kandemir2026explainable` — t pareado, 10 inits, BiLSTM vs LSTM: **p=0.0475** (rejeição rasa).
- `wan2026multiscale` (VMD-MSI) — DM de ablação (ganho do módulo GTTS).

**Veredito MISTO (rejeita na maioria, casos NS reportados):**
- `binrong2026whole` (ensemble Applied Energy) — DM do ensemble vs 12 modelos × 4 estações: rejeita vs ML/estatísticos; **TCN-BiLSTM outono p=0.128 NS; Informer-BiGRU verão DM negativo (p=0.036, ensemble PIOR), outono p=0.213 NS, inverno p=0.814 NS**. Friedman: ensemble com menor rank nas 4 estações.
- `gomez2026predicting` (TCN-sLSTM-MHA) — DM (Harvey+Newey–West) + KSPA vs CatBoost, TFT, CNN-LSTMAE, CNN-Trans-MLP, BOGP-RF: significativo vs a maioria; **CNN-LSTMAE e CNN-Trans-MLP com DM negativo em um cenário TL**; No-TL rejeita contra todos.
- `bendes2026bio` (DWT–BiLSTM, 30 runs pareados por seed) — Friedman entre 8 metaheurísticas rejeita (χ²=49.76); ablação DWT rejeita (p=1.5e-11); **ablação bidirecional NÃO REJEITA (p=0.674, "completely insignificant")**.
- `jie2026designing` (SKLD-RR-VMD, GW condicional) — rejeita em 238/240 instâncias; **2 NS** (C-LSTM 12-passos não-extremo, 1-passos extremo).
- `shengxiang2022deep` (HTD+MOBBSA+Seq2Seq) — DM vs 18 benchmarks: "rejected for most"; **GRU e BiGRU superiores ao Seq2Seq em passos curtos (10-min)**; KW: módulo de previsão NÃO significativo no dataset 5-min.
- `yesudhas2026offshore` (VMD-Swin-LSTM) — t pareado (n=5 runs) por horizonte: 12 h vs VMD-LSTM MAE **p=0.126 NS**; demais rejeita.

**Não rejeita como RESULTADO DESEJADO (uso invertido):**
- `chen2024research` (MSFSC) — KW das previsões de 6 modelos híbridos vs dados reais: **p > 0.05 em todos** ("previsão ≈ observação") — não-rejeição usada como argumento de adequação.

**Sem veredito reportado:** `data2026driven` (matriz DM no suplementar), `neshat2026physics` (só ranks, sem p-valor).
**Sem teste de hipótese:** `aodi2026optimized`, `xiang2026seqbinder`.

## Observações

- H0 do DM com perda quadrática é textual em `wei2026secondary` e
  `shengxiang2022deep` — formulação adotável no SBSR (já planejada com
  bootstrap em blocos).
- Honestidade de NS: 6 artigos reportam não-significativos com p exato —
  precedente para reportar casos mistos por horizonte no SBSR.
- Blocos para Friedman na base: estações, horizontes, subtarefas
  temporais, sazonalidades — todos encaixam num split temporal único.

## Fontes

- `docs/revisions/sbr/kbscout/result-5.md` (varredura de padrões);
  extrações em `articles/knowledge/markdown/`.
