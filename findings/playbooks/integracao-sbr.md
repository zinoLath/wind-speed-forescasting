---
title: Integrar rodada de resultados no artigo SBR
area: playbook
updated: 2026-10-19
---

# Playbook: integração no artigo SBSR

Fonte: skill **sbsr-integrate** (autoritativa). Situação atual (2026-10-19):
tabelas do artigo estão numa **rodada antiga** (TCN-Bi 0.502/0.416,
rolling 1406) enquanto as figuras vêm do **report_8020** (TCN 0.514 all /
0.403 rolling, rolling 1477) — ver finding `artigo-sbr`.

## Passos

1. Identificar a rodada fonte (ex.: `data/results/report_8020`) e conferir
   que `evaluation.json` + `images/` + `mcs.json` são da MESMA rodada;
2. Gerar as tabelas prontas (caption+label) com
   `python scripts/sbsr_tables.py --round-dir data/results/report_<rodada>`
   (finding `tabelas-sbsr-geradas`) e substituir `tab:error_all`/
   `tab:error_h36` em `articles/sbr/LaTeX_Template_SBSR_Portugues/` —
   com baselines ARIMA/RF/persistência da mesma rodada;
   - números citados no texto e no Resumo (R² all e h36);
   - descrição da função de perda (MSE ponderada por horizonte);
   - figuras sincronizadas (`scripts/articles_sync.py sync sbr` —
     heatmap MSE, heatmap R², divisão do dataset, via `sync.json`);
3. Verificações de consistência: variância implícita uniforme nas tabelas;
   figura = tabela da mesma rodada; Resumo = tabela;
4. Commit no repo principal — o hook post-commit empurra figuras ao
   Overleaf (best-effort).

## Não esquecer

- Push Overleaf só leva FIGURAS; o .tex é editado aqui e vai ao Overleaf
  por fora (colar manualmente) — ver finding `overleaf-sync`.
- Após integrar, atualizar o finding `knowledge/artigo-sbr.md` (novo
  estado) e rodar `python findings/scripts/findings.py check`.
