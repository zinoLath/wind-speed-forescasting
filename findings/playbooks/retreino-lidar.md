---
title: Retreino LiDAR 80/20 e variantes
area: playbook
updated: 2026-10-19
---

# Playbook: retreino LiDAR sob reamostragem

Fonte: skill **lidar-retrain** (runbook autoritativo). Resumo:

```bash
source .venv/bin/activate && source scripts/tf_gpu_env.sh

# 1) BACKUP (treino sobrescreve trained_8020; sem undo)
cp -r pipeline/tmp/trained_8020 pipeline/tmp/trained_8020_backup_$(date +%Y%m%d%H%M)

# 2) treino (~17 min GPU: lstm 2, lstm_bi 5, tcn 3, tcn_bi 7)
python scripts/lidar_7525_train.py --train-ratio 0.80   # 75/25: --train-ratio 0.75
# ablações: scripts/lidar_8020_mseloss_train.py | lidar_8020_startweight_train.py

# 3) regenerar figuras + evaluation.json (~5 min GPU)
python tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020 \
    --out-dir data/results/report_8020 --no-baselines

# 4) reconstruir o site
python reports/src/builder.py build
```

- Atalho 1–4 automático: servidor do hub + botão "Regenerar com retreino"
  em `reports/lidar/8020/` ou `reports/treinamento/8020/`.
- Sem early stopping/validação: variância entre execuções é esperada
  (±0.3 MSE ≠ regressão).
- Curvas por época: `pipeline/tmp/train_dashboard/<run>.jsonl`; avaliação
  inline em `data/results/lidar_8020_features/all.json`.
- Background: `setsid ... < /dev/null & disown`; pgrep com colunas.
- Se os números forem para o artigo SBSR: seguir o playbook
  `integracao-sbr.md`.
