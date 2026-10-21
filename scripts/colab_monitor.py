"""Monitor da rodada 2 distribuida do Optuna (local + workers Colab).

Consolida, numa tabela so, o progresso de cada estudo (sqlite direto, sem
optuna) e a saude de cada worker (heartbeats que os notebooks gravam no
Drive + log do watchdog local). Ideal para deixar rodando com --watch.

    python scripts/colab_monitor.py                 # um snapshot
    python scripts/colab_monitor.py --watch 60      # loop
    python scripts/colab_monitor.py --json          # saida maquina (skill)

Espelho do Drive: --drive-dir <pasta local com o conteudo de
MyDrive/wind-speed-colab/round2> (cliente desktop do Drive, rclone ou
download manual). Sem espelho, so o worker local aparece.

Codigos de saida: 0 = sem erros; 3 = ha erro/stale (usado pela skill para
disparar correcao).
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PLAN = Path("pipeline/tmp/colab_round2/plan.json")
MIRROR_DIR = Path("pipeline/tmp/colab_round2/drive_mirror")

ERROR_RE = re.compile(
    "ResourceExhaustedError|OutOfMemoryError|OOM was thrown|CUDA_ERROR|"
    "Xid|InternalError|ImportError|ModuleNotFoundError|FileNotFoundError|"
    "Traceback \\(most recent call last\\)|Killed|OperationalError",
    re.IGNORECASE,
)


def load_plan(path):
    plan_path = PROJECT_ROOT / path
    if not plan_path.is_file():
        raise SystemExit(
            f"Plano {plan_path} nao existe; rode "
            "'python scripts/colab_round2.py plan --colab-gpus N' primeiro."
        )
    with open(plan_path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_mirror(args, plan):
    """Espelho local da pasta do Drive (explicito, env, rclone ou nada)."""
    if args.drive_dir:
        return Path(args.drive_dir)
    if os.environ.get("COLAB_DRIVE_DIR"):
        return Path(os.environ["COLAB_DRIVE_DIR"])
    remote = args.rclone_remote or os.environ.get("COLAB_RCLONE_REMOTE")
    if remote and shutil.which("rclone"):
        mirror = PROJECT_ROOT / MIRROR_DIR
        remote_path = f"{remote.rstrip(':')}:{plan['drive_root']}"
        result = subprocess.run(
            ["rclone", "copy", remote_path, str(mirror), "--quiet"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"[aviso] rclone falhou: {result.stderr.strip()[:200]}")
            return None
        return mirror
    return None


def _utcnow():
    return datetime.now(timezone.utc)


def _parse_iso(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def read_study(db_path, study_name):
    """(trials, best) direto do sqlite; copia para tmp para nao travar escrita."""
    if not db_path or not Path(db_path).is_file():
        return None
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(tmp_fd)
    try:
        shutil.copy2(db_path, tmp_path)
        conn = sqlite3.connect(tmp_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM trials t JOIN studies s ON t.study_id = s.study_id "
                "WHERE s.study_name = ?",
                (study_name,),
            ).fetchone()
            trials = row[0]
            best_row = conn.execute(
                "SELECT MIN(tv.value) FROM trial_values tv "
                "JOIN trials t ON tv.trial_id = t.trial_id "
                "JOIN studies s ON t.study_id = s.study_id "
                "WHERE s.study_name = ? AND t.state = 'COMPLETE'",
                (study_name,),
            ).fetchone()
            best = best_row[0] if best_row else None
        finally:
            conn.close()
        return {"trials": trials, "best": best}
    except sqlite3.Error:
        return None
    finally:
        os.unlink(tmp_path)


def load_heartbeats(mirror):
    beats = {}
    if not mirror:
        return beats
    for path in Path(mirror).glob("heartbeats/*.json"):
        try:
            with open(path, encoding="utf-8") as handle:
                beat = json.load(handle)
            beat["_age_min"] = None
            stamp = _parse_iso(beat.get("updated_at"))
            if stamp is not None:
                delta = (_utcnow() - stamp).total_seconds() / 60
                beat["_age_min"] = round(delta, 1) if delta >= 0 else None
            beats[beat.get("worker", path.stem)] = beat
        except (OSError, ValueError):
            continue
    return beats


def local_worker_status(plan, stale_min):
    """Atividade do watchdog local pela idade do log + erros no tail."""
    local = plan.get("local")
    if not local:
        return None
    log = PROJECT_ROOT / local["log"]
    info = {
        "worker": "local",
        "log": str(log),
        "age_min": None,
        "status": "missing",
        "errors": [],
    }
    if not log.is_file():
        return info
    age = (_utcnow().timestamp() - log.stat().st_mtime) / 60
    info["age_min"] = round(age, 1)
    try:
        with open(log, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 32 * 1024))
            tail = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        tail = []
    info["errors"] = [line[:160] for line in tail if ERROR_RE.search(line)][-5:]
    finished = any("finished successfully" in line for line in tail[-5:])
    if finished:
        info["status"] = "done"
    elif age > max(stale_min, 5):
        info["status"] = "stale"
    elif info["errors"]:
        info["status"] = "running-with-errors"
    else:
        info["status"] = "running"
    return info


def kaggle_kernel_status(worker, stale_min):
    """Estado do kernel Kaggle via CLI (None se CLI/auth indisponiveis)."""
    if not shutil.which("kaggle"):
        return {"status": "UNKNOWN", "detail": "CLI kaggle ausente"}
    try:
        from scripts.colab_round2 import _kaggle_env, _kaggle_user
    except ImportError:
        return {"status": "UNKNOWN", "detail": "colab_round2 indisponivel"}
    slug = worker.get("slug")
    if not slug:
        return {"status": "UNKNOWN", "detail": "worker sem slug"}
    try:
        user = _kaggle_user()
    except SystemExit as exc:
        return {"status": "UNKNOWN", "detail": str(exc)[:120]}
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", f"{user}/{slug}"],
            capture_output=True, text=True, timeout=60, env=_kaggle_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "UNKNOWN", "detail": str(exc)[:120]}
    state = (result.stdout or result.stderr or "").strip().lower()
    if result.returncode != 0:
        # 404 de kernel ainda nao empurrado, auth falhando etc.
        return {"status": "UNKNOWN", "detail": state[:160] or "kaggle CLI falhou"}
    if "error" in state or "cancel" in state:
        return {"status": "ERROR", "detail": state[:160]}
    if "complete" in state:
        return {"status": "COMPLETE", "detail": "output pronto p/ collect --kaggle"}
    if "run" in state or "queue" in state:
        return {"status": "running", "detail": state[:160]}
    return {"status": "UNKNOWN", "detail": state[:160]}


def collect_status(args):
    plan = load_plan(args.plan)
    stale_min = args.stale_min
    mirror = resolve_mirror(args, plan)
    heartbeats = load_heartbeats(mirror)
    local_info = local_worker_status(plan, stale_min)

    owners = {}
    workers_by_name = {}
    for worker in plan.get("workers", []):
        workers_by_name[worker["worker"]] = worker
        for wrapper in worker["wrappers"]:
            owners.setdefault(wrapper, worker["worker"])
    local = plan.get("local") or {}
    for wrapper in local.get("wrappers", []):
        owners.setdefault(wrapper, "local")

    from scripts.colab_round2 import WRAPPER_DIRS  # mapa estatico, sem TF

    rows = []
    for wrapper, worker_name in owners.items():
        storage = PROJECT_ROOT / "pipeline" / "tmp" / plan["storage_subdir"]
        candidates = [
            storage / WRAPPER_DIRS[wrapper] / "optuna.db",
            Path(mirror) / "optuna" / WRAPPER_DIRS[wrapper] / "optuna.db" if mirror else None,
        ]
        study = None
        source = None
        for candidate in candidates:
            study = read_study(candidate, plan["study_name"])
            if study is not None:
                source = candidate
                break

        beat = heartbeats.get(worker_name)
        worker_def = workers_by_name.get(worker_name, {})
        trials = study["trials"] if study else None
        budget = plan["n_trials"]
        done = trials is not None and trials >= budget
        kernel_detail = None

        if done:
            status = "done"
        elif beat:
            if beat.get("phase") == "error":
                status = "ERROR"
            elif beat.get("_age_min") is None or beat["_age_min"] > stale_min:
                status = "STALE"
            else:
                status = "running"
        elif worker_def.get("backend") == "kaggle":
            kstat = kaggle_kernel_status(worker_def, stale_min)
            kernel_detail = kstat.get("detail")
            status = {"COMPLETE": "complete (colete!)", "ERROR": "ERROR",
                      "running": "running"}.get(kstat["status"],
                                                 "UNKNOWN (" + kstat["status"] + ")")
        elif worker_name == "local":
            if local_info:
                status = {"done": "done", "stale": "STALE", "missing": "MISSING",
                          "running-with-errors": "RUNNING-ERR",
                          "running": "running"}.get(local_info["status"], "unknown")
        elif study is not None:
            status = "running (sem heartbeat)"
        else:
            status = "MISSING"

        rows.append({
            "wrapper": wrapper,
            "worker": worker_name,
            "trials": trials,
            "budget": budget,
            "best": study["best"] if study else None,
            "status": status,
            "hb_age_min": beat["_age_min"] if beat else None,
            "hb_phase": beat.get("phase") if beat else None,
            "current_wrapper": beat.get("current_wrapper") if beat else None,
            "restarts": beat.get("restarts") if beat else None,
            "hb_last_error": beat.get("last_error") if beat else None,
            "kernel_detail": kernel_detail,
            "hb_errors": [e.get("line", "")[:160] for e in (beat.get("recent_errors") or [])][-3:]
                         if beat else [],
            "source_db": str(source) if source else None,
        })

    problems = [r for r in rows if r["status"] in ("ERROR", "STALE", "MISSING", "RUNNING-ERR")]
    report = {
        "generated_at": _utcnow().isoformat(),
        "plan": str(args.plan),
        "drive_mirror": str(mirror) if mirror else None,
        "workers": plan.get("workers", []),
        "local": local,
        "local_info": local_info,
        "rows": rows,
        "ok": not problems,
        "problems": [r["wrapper"] for r in problems],
        "all_done": bool(rows) and all(r["status"] == "done" for r in rows),
    }
    return report


def print_report(report):
    print(f"== optuna round2 @ {report['generated_at'][:19]} (UTC) ==")
    if not report["drive_mirror"]:
        print("[aviso] sem espelho do Drive: workers Colab sem heartbeat/progresso "
              "(use --drive-dir / COLAB_DRIVE_DIR / rclone)")
    header = f"{'wrapper':<12}{'worker':<12}{'trials':<10}{'best':<10}{'status':<18}{'hb':<7}"
    print(header)
    print("-" * len(header))
    for row in report["rows"]:
        best = f"{row['best']:.5f}" if row["best"] is not None else "-"
        trials = f"{row['trials']}/{row['budget']}" if row["trials"] is not None else "-/-"
        hb = f"{row['hb_age_min']}m" if row["hb_age_min"] is not None else "-"
        print(f"{row['wrapper']:<12}{row['worker']:<12}{trials:<10}{best:<10}"
              f"{row['status']:<18}{hb:<7}")
        if row.get("hb_last_error"):
            print(f"    !! {row['hb_last_error'][:150]}")
        if row.get("kernel_detail"):
            print(f"    k> {row['kernel_detail'][:150]}")
        for err in row["hb_errors"]:
            print(f"    ! {err}")
    local_info = report.get("local_info")
    if local_info and local_info["errors"]:
        print("\nerros recentes no log local:")
        for err in local_info["errors"]:
            print(f"  ! {err}")
    if report["all_done"]:
        print("\nTodos os estudos alcancaram o orcamento. Rode o collect:")
        print("  python scripts/colab_round2.py collect --drive-dir <mirror> --kaggle --promote")
    elif not report["ok"]:
        print(f"\nATENCAO: problemas em {', '.join(report['problems'])}")
    if any(r["status"].startswith("complete") for r in report["rows"]):
        print("\nKernel Kaggle COMPLETE: traga o progresso com "
              "'python scripts/colab_round2.py collect --kaggle'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--drive-dir", default=None,
                        help="espelho local de MyDrive/wind-speed-colab/round2")
    parser.add_argument("--rclone-remote", default=None,
                        help="remote rclone (ex.: gdrive) para sincronizar o espelho")
    parser.add_argument("--stale-min", type=float, default=12,
                        help="heartbeat mais velho que isso = STALE (min)")
    parser.add_argument("--watch", nargs="?", const=60, type=float, default=None,
                        metavar="SECS", help="loop de monitoramento")
    parser.add_argument("--json", action="store_true", help="saida JSON")
    args = parser.parse_args()

    while True:
        report = collect_status(args)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            if args.watch:
                os.system("clear 2>/dev/null || cls 2>/dev/null")
            print_report(report)
        if not args.watch:
            sys.exit(0 if report["ok"] else 3)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
