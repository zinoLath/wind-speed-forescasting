#!/usr/bin/env bash
# Vigia do treino lidar 80/20 com HP do Optuna round 4: ao terminar, gera o
# report e reconstrói o hub, depois notifica (ntfy + desktop).
ROOT="/home/zino/projects/wind-speed-forescasting"
exec "$ROOT/scripts/watch_exp_notify.sh" optuna8020 "[l]idar_7525_train" \
  "source $ROOT/.venv/bin/activate && source $ROOT/scripts/tf_gpu_env.sh && cd $ROOT && python tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020_optuna --out-dir data/results/report_8020_optuna --no-baselines && python reports/src/builder.py build"
