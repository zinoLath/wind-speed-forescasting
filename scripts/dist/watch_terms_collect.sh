#!/usr/bin/env bash
# Coletor da busca de termos ITILDE-Q: espera os 2 kernels COMPLETE,
# baixa os outputs, mescla/ranka os combos e notifica.
# uso: watch_terms_collect.sh [MAX_WAIT_HORAS]   (padrão: 12)
set -u
ROOT="/home/zino/projects/wind-speed-forescasting"
LOG="$ROOT/pipeline/tmp/watch_terms_collect.log"
MAX_WAIT_H="${1:-12}"
POLL=300
log() { printf '%s %s\n' "$(date +%F\ %T)" "$*" >> "$LOG"; }

export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token" 2>/dev/null || true)"
export KAGGLE_USERNAME=joaohenriqueflauzino
export KAGGLE_KEY=unused-with-api-token

status() { kaggle kernels status "joaohenriqueflauzino/itildeq-terms-$1" 2>&1 | tail -1; }
log "coletor armado (espera máx ${MAX_WAIT_H}h)"

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while :; do
    S1="$(status a)"; S2="$(status b)"
    log "a: $S1 | b: $S2"
    case "$S1$S2" in
        *COMPLETE*COMPLETE*) break ;;
    esac
    case "$S1$S2" in
        *ERROR*|*CANCEL*) log "kernel em erro — sem coletar"; \
            "$ROOT/.venv/bin/python" "$ROOT/scripts/experiments/wfo_notify.py" \
              --title "Wind: termos ITILDE-Q" --tags warning --desktop \
              "kernel em erro: $S1 | $S2" >/dev/null 2>&1; exit 1 ;;
    esac
    [ "$(date +%s)" -ge "$deadline" ] && { log "timeout"; exit 0; }
    sleep "$POLL"
done

log "ambos COMPLETE — coletando"
OUT=$(bash -lc "source '$ROOT/.venv/bin/activate' && cd '$ROOT' && \
    python scripts/experiments/itildeq_terms_kaggle.py collect" 2>&1)
RC=$?
printf '%s\n' "$OUT" >> "$LOG"
RESUME=$(printf '%s\n' "$OUT" | grep -vE '^\s*$' | tail -2 | tr '\n' ' ')
[ "$RC" -eq 0 ] && { log "coleta OK"; MSG="busca ITILDE-Q coletada | $RESUME"; TAGS="checkered_flag"; } || \
    { log "ERRO (rc=$RC)"; MSG="busca ITILDE-Q: FALHA na coleta"; TAGS="warning"; }
"$ROOT/.venv/bin/python" "$ROOT/scripts/experiments/wfo_notify.py" \
    --title "Wind: termos ITILDE-Q" --priority high --tags "$TAGS" --desktop \
    "$MSG" >/dev/null 2>&1
exit "$RC"
