---
title: Resultados canônicos das rodadas
area: resultados
updated: 2026-10-19
---

# Resultados canônicos das rodadas

Números de referência (R² agregado `all`, MSE em (m/s)²) — sempre checar o
`evaluation.json` da rodada antes de citar qualquer número.

## report_8020 (rodada canônica atual, loss ponderada)

`data/results/report_8020/evaluation.json` — 4 modelos, rolling com 1477
amostras:

| Modelo | R² (all) | MSE (all) | R² (rolling) |
|---|---|---|---|
| TCN | **0.514** | 1.314 | 0.403 |
| TCN-Bi | 0.403 | 1.615 | 0.320 |
| LSTM | 0.279 | 1.948 | 0.146 |
| LSTM-Bi | 0.225 | 2.094 | 0.135 |

- `mcs.json` complementa com testes estatísticos (model confidence set).

## report_homologado (rodada antiga, figura do paper/steam)

`data/results/report_homologado/evaluation.json` — rolling com 344
amostras; LSTM R² all 0.141, demais modelos **negativos**. Alimenta as
figuras de rolling/inferência do artigo steam e do paper — **não citar
esses números como resultado atual do SBSR**.

## Artigo SBSR (rodada ANTERIOR à report_8020)

As tabelas `tab:error_all`/`tab:error_h36` do artigo (TCN-Bi 0.502 all /
0.416 h36; ARIMA 0.508; RF 0.418; persistência 0.388; 1406 rolling
predictions de 1513 amostras de teste) **não batem** com a report_8020
atual (TCN 0.514 / TCN-Bi 0.403, 1477 rolling). Figuras já vêm do
report_8020 (ver `articles/sbr/sync.json`) — **integração/consistência
pendente**; fluxo de correção no playbook
`playbooks/integracao-sbr.md` e skill `sbsr-integrate`.

## Métricas históricas inválidas

MAE ≈ 0.1589/0.16 (TCC/docs antigos) vem de dataset antigo + protocolo
rolling **com vazamento** — nunca usar. Números honestos: `reports/` e `docs/`.

## Fontes

- `data/results/report_8020/evaluation.json`, `data/results/report_homologado/evaluation.json`.
- `articles/sbr/LaTeX_Template_SBSR_Portugues/04_Resultados.tex` (tabelas atuais do artigo).
- `agents/gotchas.md` (métricas históricas); skill `sbsr-integrate`.
