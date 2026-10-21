"""Servidor do hub de relatorios (site estatico + API de regeneracao).

Serve reports/_site/ em http://127.0.0.1:<porta>/ e expoe:

    POST /api/rebuild                    -> reconstruiu o site inteiro
    POST /api/regenerate/<cat>/<nome>    -> roda REPORT["regenerate"] e
                                            reconstrói o site (sincrono)
    POST /api/retrain/<cat>/<nome>       -> roda REPORT["retrain"] em background
                                            (log em <relatorio>/retrain.log)
    GET  /api/job/<cat>/<nome>           -> estado do job (running/ok/secs)
    GET  /api/status

Uso:
    python reports/src/server.py [--port 8765]
"""

from __future__ import annotations

import json
import shlex
import signal
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from reports.src import builder  # noqa: E402

SITE_DIR = builder.SITE_DIR
RUN_DIR = PROJECT_ROOT / "pipeline" / "tmp" / "reports_jobs"


def _report(cat: str, name: str):
    for rep in builder.discover():
        if rep.category == cat and rep.name == name:
            return rep
    return None


def _argv(meta, key) -> list[str] | None:
    value = meta.get(key)
    if not value:
        return None
    if isinstance(value, str):
        return shlex.split(value)
    return [str(v) for v in value]


# ---------------------------------------------------------------------------
# Jobs (regenerate sincrono / retrain em background)
# ---------------------------------------------------------------------------

class JobStore:
    def __init__(self):
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return RUN_DIR / f"{key}.json"

    def set(self, key: str, **state):
        state.setdefault("ts", time.time())
        self._path(key).write_text(json.dumps(state), encoding="utf-8")

    def get(self, key: str) -> dict:
        path = self._path(key)
        if not path.is_file():
            return {"running": False, "ok": None}
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("pid") and state.get("running"):
            try:
                import os
                os.kill(state["pid"], 0)
            except OSError:
                state["running"] = False
                state["ok"] = state.get("ok", False)
                self.set(key, **state)
        state["secs"] = time.time() - state.get("ts", time.time())
        return state


JOBS = JobStore()
VISITS_PATH = RUN_DIR / "last_visit.json"


def record_visit(path: str):
    """Registra a última página de relatório aberta (para o hub/API)."""
    clean = path.split("?")[0].split("#")[0]
    if clean.startswith("/api/") or clean.startswith("/assets/"):
        return
    if not (clean == "/" or clean.endswith("/") or
            clean.endswith("index.html")):
        return
    try:
        VISITS_PATH.write_text(json.dumps(
            {"path": clean, "ts": time.time()}, ensure_ascii=False),
            encoding="utf-8")
    except OSError:
        pass


def last_visit() -> dict:
    if not VISITS_PATH.is_file():
        return {"path": None, "ts": None}
    try:
        return json.loads(VISITS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"path": None, "ts": None}


def run_regenerate(rep) -> tuple[bool, str]:
    argv = _argv(rep.meta, "regenerate")
    log = rep.path / "regenerate.log"
    if argv is None:
        # sem comando dedicado: apenas reconstruir a pagina a partir dos dados
        builder.build_site()
        return True, "reconstruído (sem comando de regeneração)"
    started = time.time()
    proc = subprocess.run(
        argv, cwd=PROJECT_ROOT, capture_output=True, text=True)
    log.write_text(
        f"$ {' '.join(argv)}\n\n{proc.stdout}\n\n{proc.stderr}",
        encoding="utf-8")
    ok = proc.returncode == 0
    if ok:
        builder.state_update(rep.key, regenerated=time.time())
    builder.build_site()
    msg = f"regenerado em {time.time() - started:.0f}s" if ok else \
        f"FALHOU (rc={proc.returncode}) — log: {log}"
    return ok, msg


def start_retrain(rep, cat: str, name: str) -> dict:
    argv = _argv(rep.meta, "retrain")
    if argv is None:
        return {"error": "retreino não configurado para este relatório"}
    key = f"{cat}/{name}"
    state = JOBS.get(key)
    if state.get("running"):
        return {"async": True, "already_running": True}
    log = rep.path / "retrain.log"
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n==== {time.strftime('%F %T')} retreino iniciado: "
                 f"{' '.join(argv)}\n")
        fh.flush()
        proc = subprocess.Popen(
            argv, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True)
    JOBS.set(key, running=True, pid=proc.pid, type="retrain",
             argv=argv, ts=time.time())

    def finalize():
        rc = proc.wait()
        ok = rc == 0
        extra = ""
        if ok:
            # encadeia a regeneracao (figuras/evaluation) apos o treino
            regen_argv = _argv(rep.meta, "regenerate")
            if regen_argv:
                regen = subprocess.run(
                    regen_argv, cwd=PROJECT_ROOT, capture_output=True,
                    text=True)
                (rep.path / "regenerate.log").write_text(
                    f"$ {' '.join(regen_argv)} (pós-retreino)\n\n"
                    f"{regen.stdout}\n\n{regen.stderr}", encoding="utf-8")
                ok = regen.returncode == 0
                extra = "" if ok else " (regeneração falhou — ver log)"
        if ok:
            builder.state_update(rep.key, regenerated=time.time())
        JOBS.set(key, running=False, pid=proc.pid, type="retrain",
                 ok=ok, rc=rc, ts=time.time())
        builder.build_site()
        return extra

    import threading
    threading.Thread(target=finalize, daemon=True).start()
    return {"async": True, "pid": proc.pid}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def log_message(self, fmt, *args):  # enxuto: só /api/ e erros (404 etc.)
        try:
            line = fmt % args
        except (TypeError, ValueError):
            try:
                line = fmt % tuple(str(a) for a in args)
            except Exception:  # noqa: BLE001
                line = f"{fmt} {args!r}"
        if "/api/" in line or line.startswith("code "):
            sys.stderr.write("%s - %s\n" % (self.address_string(), line))

    def end_headers(self):
        # hub local: figura regenerada tem o MESMO nome de arquivo — sem
        # no-cache o browser serve a versão velha do cache heurístico
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    def do_GET(self):
        if self.path.startswith("/api/status"):
            return self._json({"ok": True, "site": str(SITE_DIR),
                               "time": time.strftime("%F %T")})
        if self.path.startswith("/api/state"):
            state = {k: dict(v) for k, v in builder.state_all().items()}
            return self._json({"ok": True, "reports": state})
        if self.path.startswith("/api/last"):
            visit = last_visit()
            if visit.get("ts"):
                visit["when"] = time.strftime(
                    "%d/%m/%Y %H:%M:%S", time.localtime(visit["ts"]))
            return self._json(visit)
        if self.path.startswith("/api/meta/"):
            parts = self.path.rstrip("/").split("/")[3:]
            if len(parts) != 2:
                return self._json({"error": "use /api/meta/<cat>/<nome>"},
                                  404)
            rep = _report(parts[0], parts[1])
            if rep is None:
                return self._json({"error": "relatório desconhecido"}, 404)
            key = f"{parts[0]}/{parts[1]}"
            job = JOBS.get(key)
            log_tail = []
            for logname in ("retrain.log", "regenerate.log"):
                log = rep.path / logname
                if log.is_file():
                    lines = log.read_text(encoding="utf-8",
                                          errors="replace").splitlines()
                    log_tail = [f"{logname}:"] + lines[-4:]
                    break
            page = SITE_DIR / parts[0] / parts[1] / "index.html"
            return self._json({
                "title": rep.title,
                "description": rep.description,
                "info": rep.meta.get("info", []),
                "has_regenerate": bool(_argv(rep.meta, "regenerate")),
                "has_retrain": bool(_argv(rep.meta, "retrain")),
                "job": job,
                "log_tail": log_tail,
                "favorite": rep.favorite,
                "created": rep.created,
                "regenerated": rep.regenerated,
                "generated": time.strftime(
                    "%d/%m/%Y %H:%M",
                    time.localtime(page.stat().st_mtime))
                if page.is_file() else None,
            })
        if self.path.startswith("/api/job/"):
            key = self.path.rsplit("/", 1)[-1]
            return self._json(JOBS.get(key))
        record_visit(self.path)
        return super().do_GET()

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if parts[:2] == ["api", "rebuild"] and len(parts) == 2:
            builder.build_site()
            return self._json({"ok": True, "rebuilt": True})
        if len(parts) >= 4 and parts[0] == "api":
            action, cat, name = parts[1], parts[2], parts[3]
            if action == "rebuild":
                builder.build_site()
                return self._json({"ok": True, "rebuilt": True})
            if action in ("favorite", "rename"):
                rep = _report(cat, name)
                if rep is None:
                    return self._json({"error": "relatório desconhecido"},
                                      404)
                body = self._body()
                if action == "favorite":
                    value = body.get("favorite")
                    if not isinstance(value, bool):
                        value = not rep.favorite
                    builder.state_update(rep.key, favorite=value)
                    return self._json({"ok": True, "favorite": value})
                title = str(body.get("title") or "").strip()
                if title:
                    builder.state_update(rep.key, title=title)
                else:
                    builder.state_for(rep.key).pop("title", None)
                    builder.state_save()
                return self._json({"ok": True, "title": rep.title})
            if action in ("regenerate", "retrain"):
                rep = _report(cat, name)
                if rep is None:
                    return self._json({"error": "relatório desconhecido"},
                                      404)
                if action == "regenerate":
                    ok, msg = run_regenerate(rep)
                    status = 200 if ok else 500
                    return self._json({"ok": ok, "message": msg}, status)
                result = start_retrain(rep, cat, name)
                if "error" in result:
                    return self._json(result, 400)
                return self._json({"ok": True, **result})
        return self._json({"error": "endpoint desconhecido"}, 404)


def serve(port: int = 8765) -> int:
    if not (SITE_DIR / "index.html").is_file():
        print("site inexistente — construindo...")
        builder.build_site()
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"hub: http://127.0.0.1:{port}/  (Ctrl-C para sair)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    port = 8765
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    sys.exit(serve(port))
