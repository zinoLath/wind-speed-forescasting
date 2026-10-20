---
title: Artigo SBR (SBSR) — estado e pendências
area: artigos
updated: 2026-10-20
---

# Artigo SBR (SBSR) — estado e pendências

`articles/sbr/LaTeX_Template_SBSR_Portugues/` — clone git do Overleaf
(remote `origin` = git bridge). Estrutura: `00_Template_SBSR.tex` +
`01_Resumo` … `06_Conclusoes` + `07_SBSR.bib` + `figuras/` + `sbsr.sty`.

## Escopo do artigo

Comparação **TCN vs LSTM** em seq2seq com atenção + impacto da
bidirecionalidade no encoder, com dados LiDAR do litoral maranhense.
Resumo cita R² **0.502** (all) e **0.416** no horizonte de 6 h, ambos
TCN-Bi.

## Estado das tabelas/figuras (checado em 2026-10-19)

- **Tabelas** (`tab:error_all`, `tab:error_h36` em `04_Resultados.tex`)
  trazem números de uma **rodada anterior**: TCN-Bi 0.502/0.416, TCN
  0.486/0.415, LSTM-Bi 0.460/0.388, LSTM 0.406/0.359, ARIMA 0.508/0.316,
  RF 0.418/0.290, persistência 0.388/0.122; texto fala de 1513 amostras de
  teste e 1406 rolling predictions.
- **Figuras** já vêm da rodada canônica `report_8020` (via
  `articles/sbr/sync.json`: `fig_mse_heatmap.png`, `fig_r2_heatmap.png`,
  `fig_dataset_division.png` ← `data/results/report_8020/images/`).
- **Inconsistência conhecida**: tabela (rodada antiga) ≠ figuras/números
  atuais do report_8020 (TCN 0.514 all / 0.403 rolling, TCN-Bi 0.403,
  rolling 1477 amostras). Corrigir via skill **sbsr-integrate** (atualiza
  as duas tabelas, números do Resumo/texto, descrição da loss e figuras,
  com verificação de consistência); as tabelas novas já podem ser geradas
  prontas com `scripts/sbsr_tables.py` (finding `tabelas-sbsr-geradas`;
  artefato em `reports/lidar/8020/tabelas_sbsr.tex`).

## Regras

- Sync de figuras: `python scripts/articles_sync.py sync sbr` — **só
  comita figuras, nunca .tex do autor** (ver finding `overleaf-sync`).
- Ao trocar a rodada que alimenta o paper: usar a skill sbsr-integrate
  (variância implícita uniforme; figura = tabela da mesma rodada).

## Sessão de revisão 20/10/2026

- **Dado ANEEL atualizado** (thread `6ace4915`): Introdução agora cita o
  **SIGA/ANEEL base diária** (`@misc{aneelSiga}` no bib; snapshot
  20-10-2026; arquivo `data/siga-empreendimentos-geracao-diario.csv`):
  eólica **35 GW em operação (outorgada), 2ª fonte nacional, 15,8%**
  (hídrica 110,3 GW); ranking global 2023 (6º) segue com
  `arcilanVentoSul`. O texto antigo ("22 GW na geração, 3º lugar")
  misturava capacidade com geração.
- **Comentários dos revisores nos `.tex`**: rodada Overleaf inserida
  como blocos `% COMENTÁRIOS DE REVISÃO` em `02/03/04/06` (marcadores
  `[V-…]/[F-…]` + ID do thread; remover antes de submeter).
- **Consolidação da sessão**: relatório favoritado
  `reports/protocolo/sessao-comentarios-sbsr-1/` (3 seções: aberto /
  decisão humana / resolvido + roteiro de estudo DM/Theil U/Friedman);
  varredura de testes estatísticos da coleção em
  `docs/revisions/sbr/kbscout/result-5.md`.

## Fontes

- `articles/sbr/LaTeX_Template_SBSR_Portugues/{01_Resumo,04_Resultados}.tex`
- `articles/sbr/sync.json`; skill **sbsr-integrate**;
  finding `resultados-canonicos`.
