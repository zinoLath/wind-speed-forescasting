#!/usr/bin/env bash
# Vigia do treino 80/20 com HP do Optuna sob o PIPELINE do HPO (denoise
# 3 colunas, nível 1): ao terminar, gera o report e reconstrói o hub.
ROOT="/home/zino/projects/wind-speed-forescasting"
exec "$ROOT/scripts/dist/watch_exp_notify.sh" optuna8020hpo "[l]idar_7525_train" \
  "source $ROOT/.venv/bin/activate && source $ROOT/scripts/tf_gpu_env.sh && cd $ROOT && python reports/src/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020_optuna_hpo --out-dir data/results/report_8020_optuna_hpo --no-baselines && python reports/src/builder.py build"
