---
title: Retreino LiDAR 80/20 e variantes
area: playbook
updated: 2026-10-20
---

# Playbook: retreino LiDAR sob reamostragem

Fonte: skill **lidar-retrain** (runbook autoritativo). Resumo da
**rodada canônica (mista `8020-itildeq`, decidida 20/10/2026)**:

```bash
source .venv/bin/activate && source scripts/tf_gpu_env.sh

# 1) BACKUP (treino sobrescreve; sem undo)
cp -r pipeline/tmp/trained_8020_itildeq \
      pipeline/tmp/trained_8020_itildeq_backup_$(date +%Y%m%d%H%M)

# 2) treino da rodada mista (TCN MSE + HPs canônicos/dilatações C′;
#    LSTM ITILDE-Q + HPs do HPO — tudo embutido no script)
python scripts/experiments/lidar_8020_itildeq_train.py

# 3) regenerar figuras + evaluation.json + DM/MCS (~5 min GPU)
#    SEM --no-baselines: DL e baselines na MESMA grade de origens
#    GRAVA a run no banco data/predictions.db (sempre; substitui a
#    anterior do report_id) — relatórios leem do banco
python reports/src/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020_itildeq \
    --out-dir data/results/report_8020_itildeq

# 4) tabelas LaTeX do artigo (agregado + h36) da rodada canônica
python scripts/sbsr_tables.py --round-dir data/results/report_8020_itildeq

# 5) reconstruir o site
python reports/src/builder.py build
```

- Depois: `python scripts/articles_sync.py sync sbr` (sync.json já aponta
  para `report_8020_itildeq`) e integração no texto pela skill
  **sbsr-integrate**.
- Rodada ponderada (legado): `scripts/experiments/lidar_7525_train.py --train-ratio 0.80`
  → `trained_8020` → `report_8020` (com `--no-baselines`); ablações:
  `lidar_8020_mseloss_train.py` | `lidar_8020_startweight_train.py`.
- Atalho 1–5 automático: servidor do hub + botão "Regenerar com retreino"
  em `reports/lidar/8020-itildeq/`.
- Sessão armada (vigia): `setsid bash scripts/dist/watch_retreino_itildeq.sh 240
  < /dev/null >> pipeline/tmp/watch_retreino_itildeq.log 2>&1 & disown` —
  espera o treino começar/terminar e calcula os pontos sozinho (regen com
  baselines → tabelas → build → notificação). Não sobrevive a reboot.
- Sem early stopping/validação: variância entre execuções é esperada
  (±0.3 MSE ≠ regressão).
- Curvas por época: `pipeline/tmp/train_dashboard/<run>.jsonl`; avaliação
  inline em `data/results/lidar_8020_features/all.json`.
- Banco de runs: a regeneração grava/substitui a run em
  `data/predictions.db` (finding `predicoes-db`); conferir com
  `python -m src.predictions_db show <report>` e homologar com
  `python -m src.predictions_db homologar <report> --model <nome>`.
- Background: `setsid ... < /dev/null & disown`; pgrep com colunas.
- Se os números forem para o artigo SBSR: seguir o playbook
  `integracao-sbr.md`.
