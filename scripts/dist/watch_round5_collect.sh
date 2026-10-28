#!/usr/bin/env bash
# Coletor automático da Optuna Round 5 (Kaggle): espera os 2 kernels
# ficarem COMPLETE, baixa os outputs (collect --kaggle), backup do
# optuna antigo, promove para pipeline/tmp/optuna/ e reconstrói o hub.
#
# uso:  watch_round5_collect.sh [MAX_WAIT_HORAS]   (padrão: 48)
# rodar desligado: setsid ... < /dev/null >> log 2>&1 & disown
set -u

ROOT="/home/zino/projects/wind-speed-forescasting"
LOG="$ROOT/pipeline/tmp/watch_round5_collect.log"
MAX_WAIT_H="${1:-48}"
POLL=300

log() { printf '%s %s\n' "$(date +%F\ %T)" "$*" >> "$LOG"; }

export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token" 2>/dev/null || true)"
export KAGGLE_USERNAME=joaohenriqueflauzino
export KAGGLE_KEY=unused-with-api-token

status() { kaggle kernels status "joaohenriqueflauzino/optuna-round5-kaggle-$1" 2>&1 | tail -1; }

log "coletor armado (espera máx ${MAX_WAIT_H}h pelos kernels round5)"

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while :; do
    S1="$(status gpu1)"; S2="$(status gpu2)"
    log "gpu1: $S1 | gpu2: $S2"
    case "$S1$S2" in
        *COMPLETE*COMPLETE*) break ;;
    esac
    case "$S1$S2" in
        *ERROR*|*CANCEL*) log "kernel em erro — coletor encerra sem coletar"; \
            "$ROOT/.venv/bin/python" "$ROOT/scripts/experiments/wfo_notify.py" \
              --title "Wind: round5" --tags warning --desktop \
              "kernel Kaggle em erro: $S1 | $S2" >/dev/null 2>&1; exit 1 ;;
    esac
    [ "$(date +%s)" -ge "$deadline" ] && { log "timeout ${MAX_WAIT_H}h — encerrado"; exit 0; }
    sleep "$POLL"
done

log "ambos COMPLETE — coletando"
OUT=$(bash -lc "
    source '$ROOT/.venv/bin/activate' && cd '$ROOT' &&
    cp -r pipeline/tmp/optuna pipeline/tmp/optuna_backup_\$(date +%Y%m%d%H%M) 2>/dev/null;
    python scripts/dist/colab_round2.py collect --kaggle --promote &&
    python reports/src/builder.py build
" 2>&1)
RC=$?
printf '%s\n' "$OUT" >> "$LOG"

RESUME=$(printf '%s\n' "$OUT" | grep -vE '^\s*$' | tail -3 | tr '\n' ' ')
if [ "$RC" -eq 0 ]; then
    log "coleta + promoção OK (backup do optuna anterior em pipeline/tmp/optuna_backup_*)"
    MSG="round5 coletada e promovida | $RESUME"; TAGS="checkered_flag"
else
    log "ERRO na coleta (rc=$RC) — ver log"; MSG="round5: FALHA na coleta (rc=$RC)"; TAGS="warning"
fi
"$ROOT/.venv/bin/python" "$ROOT/scripts/experiments/wfo_notify.py" \
    --title "Wind: optuna round5" --priority high --tags "$TAGS" --desktop \
    "$MSG" >/dev/null 2>&1
exit "$RC"
