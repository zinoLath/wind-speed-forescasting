"""Monitor do experimento walk-forward LightGBM com notificacoes ntfy.sh.

Agenda de verificacoes: 7, 15, 30 e 45 minutos apos o inicio; depois a cada
1 hora (105, 165, ... min). Cada verificacao examina:

- progresso local (progress.json por wrapper) e processos vivos;
- watchdog local (vivo? reinicios recentes? loop de crashes?);
- kernels Kaggle (RUNNING/COMPLETE/ERROR/CANCELLED).

Avisos ntfy (topico zino-imputeoptuna) apenas para eventos importantes:
crash sem auto-restart, stall de progresso, loop de reinicios, kernel
Kaggle em erro/ concluido, corrida inteira finalizada.

Usage:
    python scripts/experiments/wfo_lightgbm_monitor.py                # agenda completo
    python scripts/experiments/wfo_lightgbm_monitor.py --once         # uma verificacao
    python scripts/experiments/wfo_lightgbm_monitor.py --max-hours 14 # teto de runtime
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import wfo_lightgbm_common as proto  # noqa: E402
import wfo_lightgbm_kaggle as kaggle_orch  # noqa: E402
from wfo_notify import notify  # noqa: E402

LOG = proto.common.resolve("pipeline/tmp/wfo_lightgbm_monitor.log")
EARLY_CHECKS_MIN = (7, 15, 30, 45)
STALL_MIN = 50          # progress.json sem update com runner vivo
DEAD_MIN = 15           # progress.json sem update e runner morto
LOOP_THRESHOLD = 5      # reinicios do watchdog em 30 min
RESTARTS_LOG = proto.common.resolve("pipeline/tmp/wfo_lightgbm_watchdog_restarts.log")
WATCHDOG_LOG = proto.common.resolve("pipeline/tmp/wfo_lightgbm_watchdog.log")


def log(message):
    line = f"[{time.strftime('%F %T')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def pgrep(pattern):
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(pid) for pid in result.stdout.split()]


def kaggle_status(slug):
    """Estado do kernel Kaggle: COMPLETE/RUNNING/ERROR/CANCELLED/UNKNOWN."""
    result = kaggle_orch._kaggle_run(
        "kernels", "status", f"{kaggle_orch._kaggle_user()}/{slug}", check=False
    )
    text = (result.stdout or "") + (result.stderr or "")
    match = re.search(r'status\s+"?([^"\n]+)"?', text)
    if not match:
        return "UNKNOWN"
    # a CLI pode devolver "KernelWorkerStatus.ERROR" — normalize
    return match.group(1).strip().split(".")[-1].upper()


def check_restarts():
    """Numero de reinicios do watchdog nos ultimos 30 min."""
    if not RESTARTS_LOG.is_file():
        return 0
    cutoff = time.time() - 1800
    try:
        return sum(
            1 for line in RESTARTS_LOG.read_text().split()
            if line.strip().isdigit() and float(line.strip()) >= cutoff
        )
    except (OSError, ValueError):
        return 0


def check_tail(path, n=3):
    if not Path(path).is_file():
        return []
    return Path(path).read_text(errors="replace").strip().splitlines()[-n:]


class Monitor:
    def __init__(self, wrappers):
        self.wrappers = wrappers                      # todos os 4 (local + kaggle)
        self.local_wrappers = proto.LOCAL_WRAPPERS
        self.kaggle_wrappers = [w for w in wrappers if w in proto.KAGGLE_WRAPPERS]
        self.notified = set()                         # chaves de alerta unico
        self.last_loop_alert = 0.0
        self.unknown_since = {}                       # slug -> timestamp
        self.last_progress_update = {}                # wrapper -> epoch do conteudo

    def once_seen(self, key):
        if key in self.notified:
            return False
        self.notified.add(key)
        return True

    def check_local(self):
        progress = {}
        if proto.PROGRESS_JSON.is_file():
            try:
                with open(proto.PROGRESS_JSON, encoding="utf-8") as handle:
                    progress = json.load(handle).get("wrappers", {})
            except json.JSONDecodeError:
                log("AVISO: progress.json corrompido")
        runners = pgrep("scripts/experiments/wfo_lightgbm_run.py")
        watchdogs = pgrep("scripts/experiments/wfo_lightgbm_watchdog.sh")
        lines = [
            f"local: runner={'vivo' if runners else 'MORTO'} "
            f"watchdog={'vivo' if watchdogs else 'MORTO'}"
        ]
        for wrapper in self.local_wrappers:
            info = progress.get(wrapper)
            if info is None:
                lines.append(f"  {wrapper}: sem progress.json ainda")
                continue
            age_min = (time.time() - time.mktime(
                time.strptime(info["updated"][:19], "%Y-%m-%dT%H:%M:%S"))) / 60
            lines.append(
                f"  {wrapper}: {info['done']}/{info['total']} [{info['status']}] "
                f"ram={info.get('ram_gb')}GB update ha {age_min:.0f} min"
            )
            status = info.get("status")
            if status == "finished":
                if self.once_seen(f"local-finished-{wrapper}"):
                    log(f"{wrapper}: FINISHED")
                continue
            if not runners and age_min > DEAD_MIN:
                if watchdogs:
                    if self.once_seen(f"local-dead-watchdog-{wrapper}"):
                        notify("WFO monitor", f"🩹 {wrapper}: runner morto, watchdog vivo — "
                               "reiniciara do checkpoint.", tags="adhesive_bandage")
                        log(f"{wrapper}: runner morto; watchdog reinicia")
                elif self.once_seen("local-dead-nowatchdog"):
                    notify("WFO monitor CRASH",
                           f"❌ {wrapper}: runner MORTO e watchdog MORTO — nada vai "
                           "reiniciar sozinho. Inicie: bash scripts/experiments/wfo_lightgbm_watchdog.sh",
                           priority="urgent", tags="rotating_light")
                    log(f"{wrapper}: CRASH sem auto-restart (runner+watchdog mortos)")
            elif runners and age_min > STALL_MIN:
                if self.once_seen(f"local-stall-{wrapper}-{info['done']}"):
                    notify("WFO monitor stall",
                           f"🐢 {wrapper}: sem progresso ha {age_min:.0f} min com runner "
                           f"vivo (ultima janela: {info['done'] + 1}).", priority="high",
                           tags="turtle")
                    log(f"{wrapper}: stall {age_min:.0f} min (done={info['done']})")
        restarts = check_restarts()
        lines.append(f"  watchdog: {restarts} reinicios em 30 min")
        if restarts >= LOOP_THRESHOLD and time.time() - self.last_loop_alert > 1800:
            self.last_loop_alert = time.time()
            notify("WFO monitor loop",
                   f"🔄 Watchdog em loop: {restarts} reinicios em 30 min. Tail: "
                   + " | ".join(check_tail(WATCHDOG_LOG, 2)),
                   priority="urgent", tags="arrows_clockwise")
            log(f"ALERTA loop: {restarts} reinicios/30min")
        return lines

    def check_kaggle(self):
        lines = []
        for wrapper in self.kaggle_wrappers:
            slug = f"{kaggle_orch.KERNEL_SLUG_PREFIX}-{wrapper.replace('_', '-')}"
            try:
                state = kaggle_status(slug)
            except SystemExit as exc:
                state = "UNKNOWN"
                log(f"kaggle status indisponivel: {exc}")
            lines.append(f"  kaggle {wrapper}: {state}")
            if state == "COMPLETE":
                if self.once_seen(f"kaggle-complete-{wrapper}"):
                    notify("WFO Kaggle OK",
                           f"📦 Kaggle {wrapper}: kernel COMPLETE — traga com "
                           "python scripts/experiments/wfo_lightgbm_kaggle.py collect")
                    log(f"kaggle {wrapper}: COMPLETE")
            elif state in ("ERROR", "CANCELLED"):
                if self.once_seen(f"kaggle-error-{wrapper}"):
                    notify("WFO Kaggle erro",
                           f"❌ Kaggle {wrapper}: kernel {state} — leia o log no site; "
                           "considere rodar localmente.",
                           priority="urgent", tags="rotating_light")
                    log(f"kaggle {wrapper}: {state}")
            elif state == "UNKNOWN":
                first = self.unknown_since.setdefault(slug, time.time())
                if time.time() - first > 3600 and self.once_seen(f"kaggle-unknown-{wrapper}"):
                    notify("WFO Kaggle status",
                           f"❓ Kaggle {wrapper}: sem status ha 1 h (kernel nao empurrado "
                           "ou auth). Rode: python scripts/experiments/wfo_lightgbm_kaggle.py push "
                           f"--wrappers {wrapper}",
                           priority="high", tags="grey_question")
                    log(f"kaggle {wrapper}: UNKNOWN > 1h")
        return lines

    def all_done(self):
        progress = {}
        if proto.PROGRESS_JSON.is_file():
            try:
                with open(proto.PROGRESS_JSON, encoding="utf-8") as handle:
                    progress = json.load(handle).get("wrappers", {})
            except json.JSONDecodeError:
                pass
        local_done = all(
            progress.get(w, {}).get("status") == "finished" for w in self.local_wrappers
        )
        kaggle_done = True
        for wrapper in self.kaggle_wrappers:
            slug = f"{kaggle_orch.KERNEL_SLUG_PREFIX}-{wrapper.replace('_', '-')}"
            try:
                state = kaggle_status(slug)
            except SystemExit:
                state = "UNKNOWN"
            if state in ("RUNNING", "QUEUED", "UNKNOWN") and state != "":
                kaggle_done = False
        return local_done and kaggle_done


def build_schedule():
    """Minutos relativos: 7, 15, 30, 45; depois horario (105, 165, ...)."""
    schedule = list(EARLY_CHECKS_MIN)
    nxt = EARLY_CHECKS_MIN[-1] + 60
    return schedule, nxt


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="uma verificacao e sai")
    parser.add_argument("--max-hours", type=float, default=14.0)
    args = parser.parse_args()

    monitor = Monitor(list(proto.WRAPPERS))
    log(f"monitor iniciado (verificacoes aos {EARLY_CHECKS_MIN} min, depois horaria; "
        f"max {args.max_hours} h)")
    if not args.once:
        notify("WFO monitor", "👀 Monitoramento iniciado: local lstm/lstm_bi + "
               "Kaggle tcn/tcn_bi. Avisos em eventos importantes.")

    start = time.time()
    if args.once:
        try:
            lines = monitor.check_local() + monitor.check_kaggle()
            log("verificacao unica:\n" + "\n".join("  " + line for line in lines))
        except Exception as exc:
            log(f"ERRO na verificacao: {exc!r}")
        return

    early, nxt_hourly = build_schedule()
    deadline = early.pop(0) * 60 if early else nxt_hourly * 60

    while True:
        elapsed = time.time() - start
        if elapsed < deadline:
            time.sleep(min(deadline - elapsed, 30))
            continue
        try:
            lines = monitor.check_local() + monitor.check_kaggle()
            log("verificacao:\n" + "\n".join("  " + line for line in lines))
        except Exception as exc:
            log(f"ERRO na verificacao (seguindo): {exc!r}")
        if args.once:
            return
        if monitor.all_done():
            log("tudo concluido!")
            notify("WFO concluido",
                   "🏁 Tudo finalizado (local + Kaggle). Rode: "
                   "python scripts/experiments/wfo_lightgbm_kaggle.py collect && "
                   "python scripts/experiments/wfo_lightgbm_report.py --open")
            return
        if (time.time() - start) / 3600 > args.max_hours:
            log("teto de runtime atingido; encerrando o monitor (a corrida continua).")
            notify("WFO monitor", f"⏱ Monitor encerrado apos {args.max_hours} h "
                   "(a corrida continua; verifique manualmente).", priority="high")
            return
        if early:
            deadline = early.pop(0) * 60
        else:
            nxt_hourly += 60
            deadline = nxt_hourly * 60


if __name__ == "__main__":
    main()
