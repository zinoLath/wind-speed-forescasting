---
title: Diagnóstico de predições (DTW, suavidade, viés)
area: metodos
updated: 2026-10-21
---

# Diagnóstico de predições (DTW, suavidade, viés)

Painel para responder **por que** uma perda converge melhor — mede o
comportamento da série prevista, não só o erro agregado. Script:
`scripts/prediction_diagnostics.py` (carrega os pesos, gera as predições
multi-horizonte e mede por origem; salva
`data/results/prediction_diagnostics/diagnostics.json`).

## Métricas do painel

| Métrica | O que mede | Como |
|---|---|---|
| MSE/R²/MAE | erro ponto a ponto | sobre origens completas (36 passos) |
| viés | erro assinado médio (pred − actual) | sub/superestimação |
| `std_ratio` | encolhimento de amplitude | std(pred)/std(actual) |
| `hf_ratio` | **suavidade** (energia de alta frequência) | rfft, bins ≥ 4 / total |
| `ac1` | autocorrelação lag-1 (persistência interna) | média por origem |
| `corr_mean` | correlação de Pearson por origem | — |
| `dtw_mean` | **similaridade de forma** (Dynamic Time Warping) | custo \|a−b\|, caminho ótimo, normalizado |
| `lag_mean_steps` | defasagem média | pico da correlação cruzada |

DTW mede forma/evolução (invariância a deslocamento temporal), ao
contrário de MSE/R² que comparam ponto a ponto — é a métrica natural para
julgar uma perda "shape-aware" (o Dong Sheng usa MDTW como similaridade).

## Resultado (21/10/2026, LSTM, mesmas origens)

| variante | MSE | R² | viés | std_ratio | hf_pred | ac1 | corr | DTW | lag |
|---|---|---|---|---|---|---|---|---|---|
| MSE vencedor (busca) | **1,740** | **0,414** | −0,256 | 0,704 | 0,000185 | 0,911 | **0,172** | 0,399 | ~10 |
| ITILDE-Q (canônica) | 1,967 | 0,338 | −0,150 | 0,669 | 0,000343 | 0,865 | 0,151 | **0,395** | ~9 |
| hweight | 2,023 | 0,319 | −0,179 | 0,813 | 0,000172 | 0,916 | 0,145 | 0,460 | ~10 |
| MSE antigo (rodada ruim) | 2,705 | 0,089 | +0,571 | 0,700 | 0,000266 | 0,852 | 0,105 | 0,500 | ~10 |
| *real (referência)* | — | — | — | 1,0 | **0,002** | **0,632** | — | — | — |

**Leituras estáveis:**
- Todos os modelos **suavizam demais**: energia de HF ~0,0002–0,0003 vs
  0,002 do real (~10×) e autocorrelação 0,85–0,92 vs 0,63 — regressão à
  média, comum às três perdas;
- **A ITILDE-Q não piora a forma**: DTW praticamente empata com a MSE
  (0,395 × 0,399) e a hweight é a pior (0,460) — a perda shape-aware não
  trouxe ganho de forma mensurável neste dataset;
- A MSE vencedora ganha em **correlação** e **menos encolhimento**, com
  viés negativo maior; a diferença de MSE não vem de um artefato de
  suavização;
- **Fase**: todos defasam ~9–10 passos (~1,5 h) — o termo de fase da
  perda não corrigiu a defasagem.

## Cuidados

- Os três modelos comparados têm **fontes de HP diferentes** (HPO 80/20,
  canônica, round5) — painel descritivo, não experimento controlado. Para
  atribuir à perda, variar só a perda com HPs/seed fixos;
- Runs únicos: o ruído de seed (±0,045–0,07) é da ordem das diferenças.

## Fontes

- `scripts/prediction_diagnostics.py`;
  `data/results/prediction_diagnostics/diagnostics.json`.
- Report `reports/protocolo/itildeq-termos/` (tabela de perdas + DM).
- DTW: definição em `dtw_distance` (custo |a−b|, caminho ótimo,
  normalizado por n+m); MDTW no Dong Sheng 2026 (`sheng2026shape`).
