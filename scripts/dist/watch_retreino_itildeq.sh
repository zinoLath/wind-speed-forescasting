#!/usr/bin/env bash
# Sessão de pós-re treino da rodada canônica (8020-itildeq).
#
# Arma um vigia em 2 fases: (1) espera o TREINO `lidar_8020_itildeq_train`
# começar e terminar; (2) quando acaba, calcula os pontos da rodada em um
# resultado novo — métricas + DM/MCS com baselines na MESMA grade de
# origens (sem --no-baselines), tabelas LaTeX do artigo e rebuild do hub —
# e notifica (ntfy + desktop).
#
# uso:  watch_retreino_itildeq.sh [MAX_WAIT_HORAS]
#   MAX_WAIT_HORAS: quanto tempo esperar o treino começar (padrão: 72).
#   Rodar desligado (background): setsid ... < /dev/null & disown
#   Log: pipeline/tmp/watch_retreino_itildeq.log
#
# Se o treino rodar no Kaggle: traga os modelos para
# pipeline/tmp/trained_8020_itildeq e execute a cadeia do PASSO 2 abaixo
# manualmente (o vigia local não enxerga kernel remoto).
set -u

ROOT="/home/zino/projects/wind-speed-forescasting"
PATTERN="[l]idar_8020_itildeq_train"
MAX_WAIT_H="${1:-72}"
POLL=30

log() { printf '%s %s\n' "$(date +%F\ %T)" "$*" >> "$ROOT/pipeline/tmp/watch_retreino_itildeq.log"; }

log "sessão armada (espera máx ${MAX_WAIT_H}h pelo treino)"

# ---- fase 1: esperar o treino APARECER --------------------------------
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while ! pgrep -f "$PATTERN" >/dev/null 2>&1; do
    [ "$(date +%s)" -ge "$deadline" ] && {
        log "timeout: treino não começou em ${MAX_WAIT_H}h — sessão encerrada sem calcular"
        "$ROOT/.venv/bin/python" "$ROOT/scripts/experiments/wfo_notify.py" \
            --title "Wind: vigia itildeq" --tags sleeping_zzz --desktop \
            "vigia encerrado: retreino não começou em ${MAX_WAIT_H}h" >/dev/null 2>&1
        exit 0
    }
    sleep "$POLL"
done
log "treino detectado (pid $(pgrep -f "$PATTERN" | head -1))"

# ---- fase 2: esperar o treino TERMINAR --------------------------------
while pgrep -f "$PATTERN" >/dev/null 2>&1; do sleep "$POLL"; done
log "treino terminou — calculando pontos da rodada"

# ---- fase 3: calcular os pontos em um resultado novo ------------------
OUT=$(bash -lc "
    source '$ROOT/.venv/bin/activate' && source '$ROOT/scripts/tf_gpu_env.sh' && cd '$ROOT' &&
    python reports/src/relatorio_resultados.py \
        --models-dir pipeline/tmp/trained_8020_itildeq \
        --out-dir data/results/report_8020_itildeq &&
    python scripts/sbsr_tables.py --round-dir data/results/report_8020_itildeq \
        --out reports/lidar/8020-itildeq/tabelas_sbsr.tex &&
    python reports/src/builder.py build
" 2>&1)
RC=$?
printf '%s\n' "$OUT" >> "$ROOT/pipeline/tmp/watch_retreino_itildeq.log"

RESUME=$(printf '%s\n' "$OUT" | grep -vE '^\s*$' | tail -3 | tr '\n' ' ')
if [ "$RC" -eq 0 ]; then
    log "cadeia completa OK — resultado novo em data/results/report_8020_itildeq"
    MSG="retreino itildeq: pontos calculados (métricas + DM/MCS, tabelas e hub) | $RESUME"
    TAGS="checkered_flag"
else
    log "ERRO na cadeia (rc=$RC) — ver log"
    MSG="retreino itildeq: FALHA ao calcular pontos (rc=$RC) | $RESUME"
    TAGS="warning"
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/experiments/wfo_notify.py" \
    --title "Wind: retreino itildeq" --priority high --tags "$TAGS" --desktop \
    "$MSG" >/dev/null 2>&1
exit "$RC"
