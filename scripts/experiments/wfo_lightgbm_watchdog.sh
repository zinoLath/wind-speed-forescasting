#!/usr/bin/env bash
# Watchdog do walk-forward LightGBM-imputado (GPU local).
#
# Roda scripts/experiments/wfo_lightgbm_run.py para os wrappers locais e reinicia sempre
# que a RAM usada exceder MAX_USED_GB (padrao 44, requisito do experimento)
# ou o processo sair com erro. Cada janela concluida fica em disco
# (predictions_window_*.csv + window_metrics_*.json), entao cada reinicio
# retoma exatamente de onde parou.
#
# Usage:
#   bash scripts/experiments/wfo_lightgbm_watchdog.sh [wrapper ...]   # default: lstm lstm_bi
#
# Variaveis: MAX_USED_GB (44), POLL_SECS (5), LOG, EPOCHS, WRAPPERS.

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WRAPPERS=("$@")
[ ${#WRAPPERS[@]} -eq 0 ] && WRAPPERS=(${WRAPPERS_LIST:-lstm lstm_bi})
EPOCHS_ARG="${EPOCHS:-}"
POLL_SECS="${POLL_SECS:-5}"
LOG="${LOG:-pipeline/tmp/wfo_lightgbm_watchdog.log}"
MAX_USED_GB="${MAX_USED_GB:-44}"

mkdir -p "$(dirname "$LOG")"
source .venv/bin/activate
if [ -f scripts/tf_gpu_env.sh ]; then
    # shellcheck disable=SC1091
    source scripts/tf_gpu_env.sh
fi

# glibc devolve memoria liberada ao SO mais agressivamente (TF retém blocos
# e causa o RAM creep que o watchdog policia).
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-134217728}"
# avisa o runner que ele esta supervisionado (crash -> "watchdog reinicia")
export WFO_SUPERVISED=1

NTFY_URL="https://ntfy.sh/zino-imputeoptuna"
RESTART_LOG="pipeline/tmp/wfo_lightgbm_watchdog_restarts.log"
LOOP_ALERT_STATE="pipeline/tmp/.wfo_loop_alert"
LOOP_WINDOW_SECS=1800
LOOP_THRESHOLD=5

# notificacao fire-and-forget (falha de rede nunca derruba o watchdog)
ntfy() {
    # ntfy <title> <priority> <tags> <body...>
    local title="$1" priority="$2" tags="$3"
    shift 3
    curl -s -m 10 -H "Title: $title" -H "Priority: $priority" -H "Tags: $tags" \
        --data-binary "$*" "$NTFY_URL" >/dev/null 2>&1 || true
}

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

register_restart() {
    # registra o restart e alerta se cair num loop de crashes
    mkdir -p "$(dirname "$RESTART_LOG")"
    date +%s >> "$RESTART_LOG"
    local recent now last_alert
    recent=$(awk -v cut=$(( $(date +%s) - LOOP_WINDOW_SECS )) '$1 >= cut' "$RESTART_LOG" | wc -l)
    if [ "$recent" -ge "$LOOP_THRESHOLD" ]; then
        now=$(date +%s)
        last_alert=$(cat "$LOOP_ALERT_STATE" 2>/dev/null || echo 0)
        if [ $((now - last_alert)) -ge "$LOOP_WINDOW_SECS" ]; then
            ntfy "WFO watchdog LOOP" urgent "rotating_light" \
                "🔄 Watchdog preso em loop: ${recent} reinicios em 30 min. Verifique ${LOG} — provavel crash de causa fixa."
            echo "$now" > "$LOOP_ALERT_STATE"
        fi
    fi
}

used_ram_gb() {
    awk '/MemTotal|MemAvailable/ {print $2, $1}' /proc/meminfo \
        | awk '{ if ($2 ~ /^MemTotal/) total=$1; else avail=$1 }
               END { printf "%.1f", (total - avail) / 1024 / 1024 }'
}

CMD="python scripts/experiments/wfo_lightgbm_run.py --wrappers ${WRAPPERS[*]}"
if [ -n "$EPOCHS_ARG" ]; then
    CMD="$CMD --epochs $EPOCHS_ARG"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    CMD="$CMD ${EXTRA_ARGS}"
fi

restarts=0
while :; do
    log "Iniciando walk-forward ($CMD) | restarts=$restarts | MAX_USED_GB=$MAX_USED_GB"
    $CMD 2>&1 | tee -a "$LOG" &
    pid=$!

    killed=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL_SECS"
        used=$(used_ram_gb)
        if awk "BEGIN{ exit !($used > $MAX_USED_GB) }"; then
            log "RAM ${used}GB > ${MAX_USED_GB}GB -> matando PID $pid (retomara do checkpoint)"
            kill -TERM "$pid" 2>/dev/null
            sleep 3
            kill -KILL "$pid" 2>/dev/null
            # o pipeline ($CMD | tee) pode deixar o python orfao; garante que
            # nenhum runner fique rodando em paralelo ao reinicio
            pkill -TERM -f "scripts/experiments/wfo_lightgbm_run.py" 2>/dev/null
            sleep 2
            pkill -KILL -f "scripts/experiments/wfo_lightgbm_run.py" 2>/dev/null
            wait "$pid" 2>/dev/null
            killed=1
            break
        fi
    done

    if [ "$killed" -eq 1 ]; then
        log "Morto pelo watchdog; reiniciando do checkpoint."
        ntfy "WFO watchdog RAM" high "thermometer" \
            "🌡 RAM ${used}GB > ${MAX_USED_GB}GB — runner morto e reiniciando do checkpoint (restart $((restarts + 1)))."
        register_restart
        restarts=$((restarts + 1))
        sleep 5
        continue
    fi

    wait "$pid" 2>/dev/null
    code=$?
    if [ "$code" -eq 0 ]; then
        log "Walk-forward concluido com sucesso apos $restarts reinicio(s)."
        ntfy "WFO watchdog OK" default "white_check_mark" \
            "🏁 Watchdog local concluido com sucesso (${WRAPPERS[*]}) apos $restarts reinicio(s)."
        break
    fi
    log "Processo saiu com codigo $code; reiniciando do checkpoint (restarts=$((restarts + 1)))."
    ntfy "WFO watchdog restart" high "warning" \
        "⚠ ${WRAPPERS[*]} saiu com codigo ${code} — reiniciando do checkpoint (restart $((restarts + 1)))."
    register_restart
    restarts=$((restarts + 1))
    sleep 5
done
