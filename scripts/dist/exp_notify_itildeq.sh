#!/usr/bin/env bash
# Executa o vigia para o treino ITILDE-Q: ao terminar, regenera o report e
# reconstrói o hub, depois notifica (ntfy + desktop) com o resumo.
ROOT="/home/zino/projects/wind-speed-forescasting"
exec "$ROOT/scripts/dist/watch_exp_notify.sh" itildeq "[l]idar_8020_itildeq_train" \
  "source $ROOT/.venv/bin/activate && source $ROOT/scripts/tf_gpu_env.sh && cd $ROOT && python reports/src/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020_itildeq --out-dir data/results/report_8020_itildeq --no-baselines && python reports/src/builder.py build"
