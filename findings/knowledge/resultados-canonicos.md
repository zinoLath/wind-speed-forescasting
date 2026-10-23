---
title: Resultados canônicos das rodadas
area: resultados
updated: 2026-10-22
---

# Resultados canônicos das rodadas

Números de referência (R² agregado `all`) — sempre checar o
`evaluation.json` da rodada antes de citar qualquer número.

## RODADA CANÔNICA ESCOLHIDA (20/10/2026): `report_8020_itildeq` (mista)

TCN/TCN-Bi com HPs canônicos + dilatações estáticas C′ + perda **MSE**;
LSTM/LSTM-Bi com HPs do HPO 80/20 + perda **ITILDE-Q**. **Treino executado em 21/10 com HPs da round5 e REVERTIDO**: a transferência falhou nos TCNs (0,52→0,29; campos estruturais fora do espaço da busca — `context_pooling`/`dilation_rate` — caíram p/ defaults + split de HPO 72/8/20 ≠ 80/20). Canônica em produção = execução 16/10 restaurada (backup `trained_8020_itildeq_backup_202610211004`; a rodada com HPs round5 NÃO substitui a canônica: virou rodada própria de referência `report_8020_round5` (modelos `trained_8020_round5`, relatório `reports/lidar/8020-round5/`, status não-canônico)). Registro original: treino pendente — retreino via `scripts/lidar_8020_itildeq_train.py`
(fluxo completo no playbook `retreino-lidar.md` / skill `lidar-retrain`;
regeneração SEM `--no-baselines` ⇒ DL e baselines na mesma grade de
origens, var implícita única). Números da execução anterior (16/10):

| Modelo | R² (all) | Loss |
|---|---|---|
| TCN | **0.519** | mse |
| TCN-Bi | 0.490 | mse |
| LSTM | 0.318 | itilde_q |
| LSTM-Bi | 0.329 | itilde_q |

- Complementos da rodada: `reports/lidar/8020-itildeq-metodos/` (DM
  ajustado HLN+HAC+Holm+Clark–West) e `reports/lidar/8020-itildeq-seeds/`.
- Tabelas do artigo: `scripts/sbsr_tables.py --round-dir
  data/results/report_8020_itildeq`.
- Figuras do artigo: `articles/sbr/sync.json` já aponta para
  `data/results/report_8020_itildeq/images/`.

## report_8020_featurex (rodada de ESTUDO — 67 canais, MSE puro — 21/10/2026)

Avaliação completa (rolling + filtro de observadas, mesma grade dos
baselines, `data/results/report_8020_featurex/evaluation.json`):
TCN **0,494** / LSTM **0,485** / TCN-Bi 0,401 / LSTM-Bi 0,379 — vs
ARIMA 0,508 / RF 0,418 / Pers 0,388. DM 10 pares 0 significativos;
MCS α=0,10 só a TCN. **NÃO canônica** (não promoveu) — análise no
relatório `reports/lidar/8020-featurex`. Lineup híbrido
(`report_8020_hybrid`: TCN/TCN-Bi canônicas + LSTMs featurex): MSE TCN
1,301 < ARIMA 1,329 < TCN-Bi 1,379 < LSTM 1,393; DM 0 significativos,
MCS manteve todos os 7.

## report_8020 (rodada ponderada — ANTERIOR, mantida como ablação)

`data/results/report_8020/evaluation.json` — 4 modelos com
`horizon_weighted_mse`, rolling com 1477 amostras: TCN **0.514** (MSE
1.314) / TCN-Bi 0.403 / LSTM 0.279 / LSTM-Bi 0.225 (R² rolling: 0.403
TCN). `mcs.json` com testes estatísticos.

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
