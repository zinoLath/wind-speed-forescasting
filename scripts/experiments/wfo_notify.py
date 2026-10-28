"""Notificacoes ntfy.sh + desktop do experimento walk-forward LightGBM.

Envia push para o topico https://ntfy.sh/zino-imputeoptuna (fire-and-forget:
falha de rede nunca derruba a corrida) e, opcionalmente, notificacao de
desktop via notify-send (mesmo estilo fire-and-forget). Emojis vao no CORPO
da mensagem (headers HTTP sao ASCII-only).

Uso como biblioteca:
    from wfo_notify import notify
    notify("WFO lstm OK", "✅ lstm concluido | MAE=0.30", priority="default",
           desktop=True)

Uso como CLI:
    python scripts/experiments/wfo_notify.py --desktop --title "WFO teste" \
        --priority high "mensagem"
"""

import argparse
import os
import shutil
import subprocess

import urllib.request

NTFY_URL = "https://ntfy.sh/zino-imputeoptuna"

_URGENCY = {"min": "low", "low": "low", "default": "normal",
            "high": "critical", "urgent": "critical"}


def desktop_notify(title, message, urgency="normal"):
    """Notificacao de desktop (notify-send); retorna o código ou None."""
    if not shutil.which("notify-send"):
        return None
    env = dict(os.environ)
    # sessões iniciadas por cron/systemd sem env de display
    env.setdefault("DBUS_SESSION_BUS_ADDRESS",
                   f"unix:path=/run/user/{os.getuid()}/bus")
    env.setdefault("DISPLAY", ":0")
    try:
        proc = subprocess.run(
            ["notify-send", "-u", urgency, title, message],
            env=env, timeout=10, capture_output=True)
        return proc.returncode
    except Exception as exc:  # nunca derruba a corrida por notificacao
        print(f"[notify-send] falhou ({title!r}): {exc}", flush=True)
        return None


def notify(title, message, priority="default", tags="satellite",
           desktop=False):
    """Publica no topico ntfy (e no desktop, se pedido); retorna HTTP status."""
    if desktop:
        desktop_notify(title, message, _URGENCY.get(priority, "normal"))
    try:
        request = urllib.request.Request(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={
                "Title": title.encode("ascii", "replace").decode("ascii"),
                "Priority": priority,
                "Tags": tags,
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except Exception as exc:  # nunca derruba a corrida por causa de notificacao
        print(f"[ntfy] falhou ({title!r}): {exc}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--title", default="WFO LightGBM")
    parser.add_argument("--priority", default="default",
                        choices=("min", "low", "default", "high", "urgent"))
    parser.add_argument("--tags", default="satellite")
    parser.add_argument("--desktop", action="store_true",
                        help="também emite notificação de desktop "
                             "(notify-send)")
    parser.add_argument("message")
    args = parser.parse_args()
    status = notify(args.title, args.message, args.priority, args.tags,
                    desktop=args.desktop)
    print("notificado" if status else "FALHOU")


if __name__ == "__main__":
    main()
