---
title: HPO com Optuna e rodadas distribuídas
area: metodos
updated: 2026-10-21
---

# HPO com Optuna e rodadas distribuídas

Um estudo por modelo; protocolo metodológico:

- **TPESampler multivariado com agrupamento** (`group=True`), 20 ensaios
  iniciais aleatórios, 64 candidatos por iteração; **MedianPruner** (10
  ensaios de aquecimento, poda a partir da época 8) via callback Keras.
- **Objetivo: RMSE em validação com decoder em modo inferência** (não a
  val_loss teacher-forced); RMSE preferido ao MAE para penalizar erros
  grandes.
- Espaços: lr log-uniforme [1e-5, 1e-2] (LSTMs) ou [1e-4, 1e-2] (demais);
  unidades LSTM 64–512 (passo 32); dropout 0–0.5; TCN espaço compartilhado
  (filtros 32–256, kernel 2–3, 1–2 stacks, dilatação 1–5 → potências de 2);
  Transformer d_model 32–256, cabeças 1–8, camadas 1–4, ff_dim 64–512,
  dropout 0–0.3.
- Reprodutibilidade: seed 42, `K.clear_session()` entre ensaios, dados
  preparados uma única vez, contrato `hp.*` abstraído por adaptadores
  (`OptunaHyperParameters`/`FixedHyperParameters`).
- **Storage sqlite retomável**: `pipeline/tmp/optuna/<name>/optuna.db`,
  `load_if_exists=True` — apagar o `.db` para recomeçar.

## Rodadas distribuídas (round2+)

- `pipeline/pipeline.optuna_round*.json` define estudos (round2: lstm,
  lstm_bi, tcn, tcn_bi; um sqlite por wrapper em
  `pipeline/tmp/optuna_round2/`).
- `scripts/colab_round2.py` planeja/empacota (pesados → notebooks Colab e
  kernels Kaggle; leves → watchdog local `scripts/optuna_watchdog.sh`);
  notebooks em `notebooks/colab_round2/` — **nunca editar à mão, regenerar**.
- **Round 5 (COLETADA E PROMOVIDA em 21/10/2026)**: 2 kernels Kaggle,
  404 trials (LSTM 104), split 72/8/20, objetivo = skill bloco k=4 na
  validação (571 origens), combinação canônica (LSTM→ITILDE-Q,
  TCN→MSE) + dilatações C′. Best skill (val): LSTM-Bi **+0,378** >
  LSTM +0,348 > TCN +0,320 > TCN-Bi +0,312. Promovida para
  `pipeline/tmp/optuna/` (backups: `optuna_backup_*`,
  `Seq2Seq_TCN.bak_r4`); relatório `reports/resultados/optuna-round5/`.
  Coletor automático: `scripts/watch_round5_collect.sh`.
- Heartbeats/snapshots no Drive (`MyDrive/wind-speed-colab/round2/`);
  Kaggle via `scripts/kaggle_round2_worker.py` + `kaggle_runner.py`
  (snapshot a cada 10 min); `scripts/colab_monitor.py` consolida
  (exit 3 = erro/stale). RAM: watchdog calibrado por ambiente (local 85%
  do MemTotal; Colab/Kaggle 82%).
- Runbook completo: skill **optuna-colab**.

## Composição do espaço: merge, nunca substituição

O espaço de busca da round 5 **não é superset** dos HPs canônicos: campos
estruturais (`context_pooling`, `encoder/decoder_dilation_rate`,
`batch_size`) não estão no espaço — e no retreino 80/20 eles **caíram
para os defaults do wrapper** em vez de herdar os canônicos, junto com um
split de HPO diferente (72/8/20 × 80/20). Resultado: TCN 0,52 → 0,29
(round canônica restaurada; rodada round5 mantida como referência em
`report_8020_round5`). **Regra**: ao treinar com HPs de uma rodada,
mesclar `base canônica + override apenas dos campos buscados` (ou
garantir que o espaço inclua todos os campos estruturais).

## Fontes

- `docs/metodologia.md` (seção 4); `agents/optuna-colab.md`;
  `pipeline/step_optuna.py`; `pipeline/pipeline.optuna_round2.json`.
