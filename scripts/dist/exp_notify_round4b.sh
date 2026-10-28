#!/usr/bin/env bash
# Vigia do estudo Optuna round4b (TCN, local via watchdog): notifica ao terminar.
ROOT="/home/zino/projects/wind-speed-forescasting"
exec "$ROOT/scripts/dist/watch_exp_notify.sh" round4b "[o]ptuna_watchdog" \
  "echo 'estudo round4b concluido; trials em pipeline/tmp/optuna_round4b'"
