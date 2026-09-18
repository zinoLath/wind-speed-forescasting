#!/usr/bin/env bash
# Aggressive Optuna run with a RAM watchdog.
#
# Runs `pipeline/step_optuna.py --config <cfg>` and restarts it whenever the
# total used RAM exceeds MAX_USED_GB (default 40). Because Optuna studies
# persist to sqlite with load_if_exists=True, each restart resumes exactly
# where the previous run stopped. Logs to $LOG.
#
# Usage:
#   bash scripts/optuna_watchdog.sh [config_path]

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-pipeline/pipeline.optuna_tcn_aggressive.json}"
MAX_USED_GB="${MAX_USED_GB:-40}"
POLL_SECS="${POLL_SECS:-5}"
LOG="${LOG:-pipeline/tmp/optuna_watchdog.log}"

mkdir -p "$(dirname "$LOG")"
source .venv/bin/activate
if [ -f scripts/tf_gpu_env.sh ]; then
    # shellcheck disable=SC1091
    source scripts/tf_gpu_env.sh
fi

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
    log "Starting optuna run (config=$CONFIG, restarts=$restarts)..."
    python pipeline/step_optuna.py --config "$CONFIG" 2>&1 | tee -a "$LOG" &
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
    restarts=$((restarts + 1))
    sleep 5
done