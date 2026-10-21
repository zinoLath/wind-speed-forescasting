#!/usr/bin/env bash
# Aggressive Optuna run with a RAM watchdog.
#
# Runs `pipeline/step_optuna.py --config <cfg> [--wrappers ...]` and restarts
# it whenever the total used RAM exceeds MAX_USED_GB (default 40). Because
# Optuna studies persist to sqlite with load_if_exists=True, each restart
# resumes exactly where the previous run stopped. Logs to $LOG.
#
# FAIL_ORPHANS=0 skips marking RUNNING trials as FAIL on restart (by default
# they are failed so the n_remaining budget is not spent on dead trials).
#
# Usage:
#   bash scripts/optuna_watchdog.sh [config_path] [wrapper ...]

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-pipeline/pipeline.optuna_tcn_aggressive.json}"
[ $# -gt 0 ] && shift
WRAPPER_ARGS=("$@")
POLL_SECS="${POLL_SECS:-5}"
LOG="${LOG:-pipeline/tmp/optuna_watchdog.log}"
FAIL_ORPHANS="${FAIL_ORPHANS:-1}"
RAM_FRAC="${RAM_FRAC:-0.85}"

# Limite de RAM calibrado pelo ambiente de execucao: 85% da MemTotal quando
# MAX_USED_GB nao vem do ambiente (Colab/Kaggle usam fracoes proprias; o
# worker Kaggle calibra em scripts/kaggle_round2_worker.py).
if [ -z "${MAX_USED_GB:-}" ]; then
    MAX_USED_GB=$(awk '/^MemTotal/ {printf "%.1f", $2 / 1024 / 1024 * '"$RAM_FRAC"'}' /proc/meminfo)
fi

mkdir -p "$(dirname "$LOG")"
source .venv/bin/activate
if [ -f scripts/tf_gpu_env.sh ]; then
    # shellcheck disable=SC1091
    source scripts/tf_gpu_env.sh
fi

# Ask glibc to return freed host memory to the OS more eagerly; TensorFlow
# tends to hoard freed blocks, which is what drives the slow RAM creep that
# the watchdog has to police.
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-134217728}"

fail_orphan_trials() {
    # Trials left in RUNNING by a killed process waste trial budget on every
    # restart (n_remaining counts them); mark them FAIL so the budget is only
    # spent on real work.
    python - "$CONFIG" <<'PY'
import glob
import json
import sys

import optuna
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)
config = json.load(open(sys.argv[1]))
tmp_dir = config.get("paths", {}).get("tmp_dir", "pipeline/tmp")
subdir = config.get("optuna", {}).get("storage_subdir", "optuna")
for db in glob.glob(f"{tmp_dir}/{subdir}/*/optuna.db"):
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{db}")
    for summary in storage.get_all_studies():
        for trial in storage.get_all_trials(
            summary._study_id, states=(TrialState.RUNNING,), deepcopy=False
        ):
            storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
PY
}

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

used_ram_gb() {
    awk '/MemTotal|MemAvailable/ {print $2, $1}' /proc/meminfo \
        | awk '{ if ($2 ~ /^MemTotal/) total=$1; else avail=$1 }
               END { printf "%.1f", (total - avail) / 1024 / 1024 }'
}

restarts=0
while :; do
    log "Starting optuna run (config=$CONFIG, wrappers=${WRAPPER_ARGS[*]:-all}, restarts=$restarts, fail_orphans=$FAIL_ORPHANS)..."
    if [ ${#WRAPPER_ARGS[@]} -gt 0 ]; then
        python pipeline/step_optuna.py --config "$CONFIG" --wrappers "${WRAPPER_ARGS[@]}" 2>&1 | tee -a "$LOG" &
    else
        python pipeline/step_optuna.py --config "$CONFIG" 2>&1 | tee -a "$LOG" &
    fi
    pid=$!

    killed=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL_SECS"
        used=$(used_ram_gb)
        if awk "BEGIN{ exit !($used > $MAX_USED_GB) }"; then
            log "RAM ${used}GB > ${MAX_USED_GB}GB -> killing PID $pid"
            kill -TERM "$pid" 2>/dev/null
            sleep 3
            kill -KILL "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            killed=1
            break
        fi
    done

    if [ "$killed" -eq 1 ]; then
        log "Run killed by watchdog; restarting from checkpoint."
        [ "$FAIL_ORPHANS" = "1" ] && fail_orphan_trials
        restarts=$((restarts + 1))
        sleep 5
        continue
    fi

    wait "$pid" 2>/dev/null
    code=$?
    if [ "$code" -eq 0 ]; then
        log "Optuna finished successfully after $restarts restart(s)."
        break
    fi
    log "Run exited with code $code; restarting (restarts=$((restarts + 1)))."
    [ "$FAIL_ORPHANS" = "1" ] && fail_orphan_trials
    restarts=$((restarts + 1))
    sleep 5
done