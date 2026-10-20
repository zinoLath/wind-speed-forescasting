#!/usr/bin/env bash
# Launcher do hub de relatórios (entrada drun/rofi): sobe o servidor se
# necessário e abre o navegador em http://127.0.0.1:8765/
set -u
repo="/home/zino/projects/wind-speed-forescasting"
port="${HUB_PORT:-8765}"

if ! ss -tln 2>/dev/null | grep -q ":${port} "; then
    cd "$repo" || exit 1
    setsid "$repo/.venv/bin/python" reports/src/server.py --port "$port" \
        >> /tmp/opencode/hub_desktop.log 2>&1 < /dev/null &
    disown
    for _ in $(seq 1 30); do
        ss -tln 2>/dev/null | grep -q ":${port} " && break
        sleep 0.3
    done
fi

exec xdg-open "http://127.0.0.1:${port}/"
