#!/usr/bin/env bash
# Vigia de experimento: notifica (ntfy + desktop) quando um processo termina
# e, opcionalmente, roda um comando final (ex.: gerar relatório + rebuild).
# Idempotente: notifica uma única vez por TAG (marca em $XDG_RUNTIME_DIR).
#
# uso:  watch_exp_notify.sh TAG PROCESS_PATTERN [ON_DONE_CMD]
#   - PROCESS_PATTERN deve usar o truque do colchete p/ não casar consigo
#     mesmo, ex.: "[l]idar_8020_itildeq_train"
#   - ON_DONE_CMD roda via `bash -lc` quando o processo sumir; as 3 últimas
#     linhas da saída entram na notificação
#
# pense para uso em cron/systemd: não depende de $PATH do usuário nem de env
# de display (o wfo_notify resolve o DBUS sozinho).

set -u
TAG="${1:?uso: watch_exp_notify.sh TAG PATTERN [ON_DONE_CMD]}"
PATTERN="${2:?faltou PROCESS_PATTERN}"
ON_DONE="${3:-}"

ROOT="/home/zino/projects/wind-speed-forescasting"
STATE_DIR="${XDG_RUNTIME_DIR:-/tmp}/exp-notify"
mkdir -p "$STATE_DIR"
STATE="$STATE_DIR/$TAG.done"

# já notificado nesta sessão de login? sai.
[ -f "$STATE" ] && exit 0

# ainda rodando? sai (padrão com colchete evita auto-match do próprio argv).
if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    exit 0
fi

# terminou: roda o comando final (se houver) e captura o resumo.
RESULT=""
if [ -n "$ON_DONE" ]; then
    OUT=$(bash -lc "$ON_DONE" 2>&1)
    RESULT=$(printf '%s\n' "$OUT" | grep -vE '^\s*$' | tail -3 | tr '\n' ' ')
fi

MSG="experimento $TAG finalizado"
[ -n "$RESULT" ] && MSG="$MSG | $RESULT"

"$ROOT/.venv/bin/python" "$ROOT/scripts/experiments/wfo_notify.py" \
    --title "Wind: $TAG" --priority high --tags checkered_flag --desktop \
    "$MSG" >/dev/null 2>&1

touch "$STATE"
exit 0
