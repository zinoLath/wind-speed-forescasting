---
title: Variância implícita R²×MSE — diagnóstico de consistência
area: protocolo
updated: 2026-10-23
---

# Variância implícita R²×MSE — diagnóstico de consistência

Diagnóstico originado do comentário da Vanessa no artigo SBSR ("inconsistência
sutil entre R² e MSE — variância implícita ~2,70 vs ~2,78"; thread
`6ace944b`), usado para **provar** que baselines e modelos DL da Tabela 1
foram avaliados sobre conjuntos de origens diferentes.

## Como se calcula

Da definição do R² usada no artigo (eq. 1 do `04_Resultados.tex`):

```
R² = 1 − MSE(ŷ, y) / Var(y)   ⇒   Var(y) implícita = MSE / (1 − R²)
```

- Aplica-se a **cada linha** de uma tabela de métricas (R² e MSE do mesmo
  modelo/avaliação).
- **Critério:** se todos os modelos foram avaliados sobre os mesmos
  valores reais, `Var(y)` é única ⇒ **todas as linhas produzem a mesma
  variância implícita**. Valores em grupos diferentes ⇒ conjuntos de
  avaliação diferentes ⇒ os R² não são comparáveis entre si.

## O que o diagnóstico revelou (Tabela 1 do artigo, rodada antiga)

| Grupo | Conta (MSE/(1−R²)) | Var implícita | Origens |
|---|---|---|---|
| Baselines (ARIMA 0,508/1,329; RF 0,418/1,572; Persistência 0,388/1,653) | 1,329/0,492 | **≈ 2,702** | 1478 × 36 = 53.208 amostras (`data/results/baselines/arima_rf/metrics.json`) |
| DL (TCN-Bi 0,502/1,384; TCN 0,486/1,428; LSTM-Bi 0,460/1,501; LSTM 0,406/1,651) | 1,384/0,498 | **≈ 2,779** | ≈1406 × 36 = 50.616 (1513 − 72 − 36 + 1) |

**Causa estrutural:** o rolling forecast dos DL consome a janela de
entrada (72) + saída (36) por origem; os baselines avaliavam todas as
origens possíveis (1513 − 36 + 1 = 1478). Janelas de valores reais
diferentes ⇒ Var(y) diferente ⇒ R² não comparáveis — mesmo com a mesma
série de teste.

## O fix (como garantir variância implícita única)

`reports/src/relatorio_resultados.py` (~linha 360) alinha o DL à **mesma grade
de origens dos baselines**: primeira origem = primeira linha do teste, com
o encoder consumindo o fim da fronteira treino/teste. Regenerar a rodada
**SEM `--no-baselines`** produz DL + baselines comparáveis.

**Critério de pronto** para qualquer tabela do artigo: `MSE/(1−R²)` igual
(± arredondamento) para **todas as linhas**. Este critério é verificação
obrigatória da skill **sbsr-integrate** ("variância implícita uniforme")
e do relatório `reports/protocolo/sessao-comentarios-sbsr-1/` (§1.1/§4.1).

## Não confundir

- Não é a variância do erro nem o denominador escolhido à mão — é o valor
  de `Var(y)` que o par (R², MSE) **implica**, dado que ambos vêm da
  mesma avaliação.
- Diferenças pequenas (±0,001) são arredondamento das métricas publicadas
  (3 casas); grupos com Δ ≥ ~0,05 indicam conjuntos diferentes.

## Fontes

- Cálculo verificado: `data/results/report_8020/evaluation.json` (DL) e
  `data/results/baselines/arima_rf/metrics.json` (baselines, config
  `n_origins 1478`).
- Alinhamento de origens: `reports/src/relatorio_resultados.py` (~linha 360).
- Contexto completo: `docs/revisions/sbr/comentarios_revisao_sbsr.md` §4.2;
  bloco `% [V-6ace944b]` em `articles/sbr/.../04_Resultados.tex`.
- Skills: `sbsr-integrate` (critério), `lidar-retrain` (regeneração
  canônica sem `--no-baselines`).
