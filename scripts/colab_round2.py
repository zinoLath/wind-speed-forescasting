"""Preparo da rodada 2 do Optuna distribuida (Google Colab + GPU local).

Distribui os estudos por GPU: os wrappers pesados (transformer, TCNs) vao para
notebooks do Colab (1 notebook = 1 GPU T4/L4/A100) e os leves (LSTM/GRU) ficam
na GPU local, ambos com watchdog anti-OOM. Cada wrapper tem seu proprio sqlite
(``pipeline/tmp/optuna_round2/<WrapperName>/optuna.db``), entao workers nao
disputam escrita e tudo resume de onde parou.

Subcomandos (``all`` encadeia os primeiros):

    python scripts/colab_round2.py plan --colab-gpus 2
    python scripts/colab_round2.py package
    python scripts/colab_round2.py notebooks
    python scripts/colab_round2.py local [--start]
    python scripts/colab_round2.py collect --drive-dir <mirror> [--promote]

Fluxo Colab (uma vez): ``package`` -> subir o zip para a pasta
``MyDrive/wind-speed-colab/round2/package/`` no drive.google.com -> abrir cada
notebook gerado em ``notebooks/colab_round2/`` com Runtime GPU -> Run all. O
notebook restaura o progresso do Drive, roda o step_optuna com watchdog de RAM
e sincroniza sqlite + heartbeat de volta para o Drive a cada 10 min.

O progresso sincronizado pode voltar para a maquina local com ``collect``
(usb do Drive via cliente desktop, rclone ou download manual da pasta).
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROUND_CONFIG = "pipeline/pipeline.optuna_round2.json"
STORAGE_SUBDIR = "optuna_round2"
STUDY_NAME = "round2"
DRIVE_ROOT = "wind-speed-colab/round2"
PLAN_PATH = Path("pipeline/tmp/colab_round2/plan.json")

# .name de cada wrapper (nomes dos diretorios de resultado); manter em sync com
# o registro em src/common.py (mapa estatico para nao importar TF aqui).
WRAPPER_DIRS = {
    "lstm": "Seq2Seq_LSTM",
    "lstm_bi": "Seq2Seq_LSTM_Bidirectional",
    "gru": "Seq2Seq_GRU",
    "gru_bi": "Seq2Seq_GRU_Bidirectional",
    "lstm_cnn": "Seq2Seq_LSTM_CNN",
    "tcn": "Seq2Seq_TCN",
    "tcn_lstm": "Seq2Seq_TCN_LSTM",
    "tcn_bi": "Seq2Seq_TCN_Bidirectional",
    "transformer": "Seq2Seq_Transformer_PreLN",
}

# Pesados -> Colab (VRAM/tempo alto: filtros<=320 x2 stacks, d_model<=256 x4),
# leves -> GPU local (GTX 1660 6GB da conta).
HEAVY = ["tcn_bi", "transformer", "tcn_lstm", "tcn"]
LIGHT = ["lstm", "lstm_bi", "gru", "gru_bi", "lstm_cnn"]

# Custo relativo por trial (fallback quando nao ha trials.csv historico).
WEIGHTS = {
    "tcn_bi": 3.0, "transformer": 2.6, "tcn_lstm": 2.2, "tcn": 1.8,
    "lstm_cnn": 1.1, "lstm_bi": 1.0, "lstm": 1.0, "gru_bi": 0.95, "gru": 0.9,
}


def _round_cfg(config_path=None):
    path = PROJECT_ROOT / (config_path or ROUND_CONFIG)
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    optuna = cfg.get("optuna", {})
    study_name = optuna.get("study_name", STUDY_NAME)
    return {
        "config": str(path.relative_to(PROJECT_ROOT)),
        "storage_subdir": optuna.get("storage_subdir", STORAGE_SUBDIR),
        "study_name": study_name,
        "drive_root": optuna.get("drive_root", f"wind-speed-colab/{study_name}"),
        "assign": optuna.get("assign"),
        "n_trials": int(optuna.get("n_trials", 120)),
        "epochs": int(optuna.get("epochs", 100)),
        "wrappers": list(optuna.get("wrappers", [])),
        "patience": int(cfg.get("common", {}).get("patience", 12)),
    }


def estimate_costs():
    """Custo por trial: mediana do elapsed_sec historico quando existe."""
    costs = {}
    for key, wrapper_dir in WRAPPER_DIRS.items():
        elapsed = []
        for trials_csv in PROJECT_ROOT.glob(f"pipeline/tmp/*/{wrapper_dir}/trials.csv"):
            try:
                with open(trials_csv, encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        value = row.get("user_attrs_elapsed_sec")
                        if value not in (None, "", "nan"):
                            elapsed.append(float(value))
            except (OSError, ValueError):
                continue
        costs[key] = statistics.median(elapsed) if elapsed else WEIGHTS.get(key, 1.0)
    return costs


def cmd_plan(args):
    cfg = _round_cfg(args.config)
    wrappers = cfg["wrappers"]
    unknown = [w for w in wrappers if w not in WRAPPER_DIRS]
    if unknown:
        raise SystemExit(f"Wrappers sem diretorio mapeado em WRAPPER_DIRS: {unknown}")

    costs = estimate_costs()

    # Distribuicao explicita (config "assign") tem precedencia sobre o
    # HEAVY/LIGHT + Longest Processing Time.
    assign = cfg.get("assign")
    if assign:
        colab_list = list(assign.get("colab", []))
        kaggle_list = list(assign.get("kaggle", []))
        local_list = list(assign.get("local", []))
        assigned = set(colab_list) | set(kaggle_list) | set(local_list)
        if assigned != set(wrappers):
            raise SystemExit(
                "assign do config nao cobre os wrappers do estudo:\n"
                f"  config: {sorted(wrappers)}\n  assign: {sorted(assigned)}"
            )
        pool = []
        for i, wrapper in enumerate(colab_list):
            pool.append({"worker": f"colab-gpu{i + 1}", "backend": "colab",
                         "wrappers": [wrapper], "cost": costs.get(wrapper, 1.0)})
        for i, wrapper in enumerate(kaggle_list):
            pool.append({"worker": f"kaggle-gpu{i + 1}", "backend": "kaggle",
                         "wrappers": [wrapper], "cost": costs.get(wrapper, 1.0)})
        local_wrappers = list(local_list)
        colab_gpus, kaggle_gpus = len(colab_list), len(kaggle_list)
    else:
        heavy = [w for w in wrappers if w in HEAVY]
        light = [w for w in wrappers if w in LIGHT]
        misplaced = [w for w in wrappers if w not in HEAVY + LIGHT]
        if misplaced:
            print(f"[aviso] wrappers fora de HEAVY/LIGHT ficam locais: {misplaced}")
            light = light + misplaced

        # Pool de GPUs remotas: cada worker = 1 GPU (notebook Colab ou kernel
        # Kaggle). Longest Processing Time distribui os pesados pelo menor custo
        # acumulado; leves ficam na GPU local.
        pool = []
        for i in range(max(0, args.colab_gpus)):
            pool.append({"worker": f"colab-gpu{i + 1}", "backend": "colab",
                         "wrappers": [], "cost": 0.0})
        for i in range(max(0, args.kaggle_gpus)):
            pool.append({"worker": f"kaggle-gpu{i + 1}", "backend": "kaggle",
                         "wrappers": [], "cost": 0.0})
        if heavy and not pool:
            print("[aviso] wrappers pesados sem GPU remota no plano; ficam locais.")
            light = light + heavy
            heavy = []
        for wrapper in sorted(heavy, key=lambda w: costs.get(w, 1.0), reverse=True):
            target = min(pool, key=lambda w: w["cost"])
            target["wrappers"].append(wrapper)
            target["cost"] += costs.get(wrapper, 1.0)
        local_wrappers = light
        colab_gpus, kaggle_gpus = max(0, args.colab_gpus), max(0, args.kaggle_gpus)

    workers = []
    for worker in pool:
        big = any(w in ("tcn_bi", "transformer") for w in worker["wrappers"])
        entry = {"worker": worker["worker"], "backend": worker["backend"],
                 "wrappers": worker["wrappers"]}
        if worker["backend"] == "colab":
            entry["gpu_hint"] = ("T4 (L4/A100 se disponivel — tcn_bi/transformer pesam)"
                                 if big else "T4")
            entry["notebook"] = f"notebooks/colab_round2/{worker['worker']}.ipynb"
        else:
            entry["slug"] = f"optuna-{cfg['study_name']}-{worker['worker']}"
            entry["command"] = (
                f"python scripts/kaggle_round2_worker.py --config {cfg['config']} "
                f"--wrappers " + " ".join(worker["wrappers"])
            )
        workers.append(entry)

    local_command = (
        f"LOG=pipeline/tmp/colab_round2/logs/local-watchdog.log "
        f"bash scripts/optuna_watchdog.sh {cfg['config']} " + " ".join(local_wrappers)
        if local_wrappers else ""
    )
    if args.max_ram_gb and local_wrappers:
        local_command = f"MAX_USED_GB={args.max_ram_gb} " + local_command
    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **cfg,
        "drive_root": cfg["drive_root"],
        "colab_gpus": colab_gpus,
        "kaggle_gpus": kaggle_gpus,
        "local": {
            "wrappers": local_wrappers,
            "command": local_command,
            "log": "pipeline/tmp/colab_round2/logs/local-watchdog.log",
            "ram_note": "MAX_USED_GB auto-calibrado (85% de MemTotal) pelo proprio "
                        "watchdog; sobrescreva com MAX_USED_GB=<gb> se necessario",
        },
        "workers": workers,
        "kaggle": {
            "dataset_slug": "wind-speed-forecasting",
            "output_glob": f"pipeline/tmp/{cfg['storage_subdir']}",
        },
        "costs": {k: round(v, 2) for k, v in costs.items() if k in wrappers},
    }
    if not local_wrappers:
        plan.pop("local")

    path = PROJECT_ROOT / PLAN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)

    print(f"Plano salvo em {path}\n")
    print(f"{'destino':<14}{'backend':<10}{'wrappers':<32}custo relativo")
    if local_wrappers:
        print(f"{'local':<14}{'(1660)':<10}{', '.join(local_wrappers):<32}"
              f"{round(sum(costs.get(w, 1.0) for w in local_wrappers), 2)}")
    for worker in workers:
        print(f"{worker['worker']:<14}{worker['backend']:<10}"
              f"{', '.join(worker['wrappers']):<32}"
              f"{round(sum(costs.get(w, 1.0) for w in worker['wrappers']), 2)}")
    print(f"\nTotal: {cfg['n_trials']} trials x {cfg['epochs']} epochs por wrapper "
          f"(estudo '{cfg['study_name']}' em pipeline/tmp/{cfg['storage_subdir']}/).")
    return plan


def cmd_package(args):
    cfg = _round_cfg(args.config)
    stamp = time.strftime("%Y%m%d")
    out = PROJECT_ROOT / "dist" / f"colab_round2_package_{stamp}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        # Diretorios de codigo entram recursivamente (glob() so retorna o proprio
        # diretorio, que is_file() descartaria).
        for pattern in ("src", "scripts", "tests"):
            for path in (PROJECT_ROOT / pattern).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    zf.write(path, path.relative_to(PROJECT_ROOT))
        for pattern in ("pipeline/*.py", "pipeline/*.json",
                        "requirements.txt", "data/dataset.csv"):
            for path in PROJECT_ROOT.glob(pattern):
                if path.is_file():
                    zf.write(path, path.relative_to(PROJECT_ROOT))
        # Progresso atual da rodada entra no pacote: um notebook novo pode
        # comecar mesmo sem ainda ter sincronizado nada do Drive.
        progress = PROJECT_ROOT / "pipeline" / "tmp" / cfg["storage_subdir"]
        if progress.is_dir():
            for path in progress.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    zf.write(path, path.relative_to(PROJECT_ROOT))

    print(f"Pacote gerado: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"Suba-o para o Drive em: MyDrive/{cfg['drive_root']}/package/")
    return out


# --- Celulas do notebook -----------------------------------------------------

MD_HEADER = """\
# Optuna {study} — worker `{worker}` (Google Colab GPU)

Busca distribuída: **{wrappers}** rodam aqui; os wrappers leves ficam na GPU
local. Cada trial é gravado no sqlite (estudo `{study}`), então desconexões do
Colab **não perdem progresso**.

## Setup (uma vez)
1. No computador local: `python scripts/colab_round2.py package` e faça upload
   do `dist/colab_round2_package_*.zip` para a pasta
   `MyDrive/{drive_root}/package/` no drive.google.com.
2. **Runtime → Change runtime type → GPU** (sugestão: {gpu_hint}).
3. **Runtime → Run all**. O notebook restaura o progresso do Drive, roda o
   `step_optuna` com watchdog de RAM (reinicia o treino em caso de estouro,
   marcando trials órfãos como FAIL) e sincroniza sqlite + heartbeat de volta
   ao Drive a cada 10 min.

## Depois de uma desconexão
Rode tudo de novo — o progresso volta do Drive e o estudo continua.

O monitor local (`python scripts/colab_monitor.py --watch`) acompanha este
worker pelo heartbeat `heartbeats/{worker}.json`. Mantenha esta aba aberta.
"""

CODE_PARAMS = """\
# Parametros deste worker (gerado por scripts/colab_round2.py notebooks)
import json as _json

WORKER = _json.loads(\"\"\"
{worker_json}
\"\"\")
"""

CODE_GPU = """\
!nvidia-smi
"""

CODE_PIP = """\
%pip install -q "tensorflow[and-cuda]==2.21.0" "keras==3.12.1" "keras-tcn==3.5.6" \
    "optuna==4.9.0" "PyWavelets==1.8.0" "pandas>=2.0" "scikit-learn>=1.3" "numpy>=1.26"
import tensorflow as tf
print("TF", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
"""

CODE_RESTORE = """\
import shutil
import sqlite3
import zipfile
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive")

WORK = Path("/content")
REPO = WORK / "wind-speed-forecasting"
DRIVE = Path("/content/drive/MyDrive") / WORKER["drive_root"]
PKG_DIR = DRIVE / "package"
PKG_DIR.mkdir(parents=True, exist_ok=True)

zips = sorted(PKG_DIR.glob("colab_round2_package_*.zip"), key=lambda p: p.stat().st_mtime)
assert zips, (
    f"Nenhum colab_round2_package_*.zip em {PKG_DIR}.\\n"
    "Rode 'python scripts/colab_round2.py package' localmente e suba o zip para la."
)
pkg = zips[-1]
print("pacote:", pkg.name)
REPO.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(pkg) as z:
    z.extractall(REPO)
assert (REPO / "data" / "dataset.csv").is_file(), "dataset.csv ausente no pacote"

# Restaura o progresso ja sincronizado para o Drive. O backup API do sqlite
# garante uma copia consistente mesmo se o arquivo de origem for um snapshot
# pego no meio de uma escrita.
DEST = REPO / "pipeline" / "tmp" / WORKER["storage_subdir"]
optuna_drive = DRIVE / "optuna"
if optuna_drive.is_dir():
    for src_dir in sorted(p for p in optuna_drive.iterdir() if p.is_dir()):
        db = src_dir / "optuna.db"
        if not db.is_file():
            continue
        dst = DEST / src_dir.name / "optuna.db"
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_conn = sqlite3.connect(db)
        dst_conn = sqlite3.connect(dst)
        with dst_conn:
            src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
        for f in src_dir.glob("*.json"):
            shutil.copy2(f, dst.parent / f.name)
        print("progresso restaurado:", src_dir.name)
print("repo em:", REPO)
"""

CODE_RUNNER = '''\
# Watchdog do worker: roda o step_optuna como subprocesso, reinicia em estouro
# de RAM/crash (o estudo sempre resume do sqlite) e publica heartbeat + copia
# do progresso no Drive.
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.chdir(REPO)
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

LOG = Path("pipeline/tmp/colab_round2/logs") / (WORKER["worker"] + ".log")
LOG.parent.mkdir(parents=True, exist_ok=True)
HB_PATH = DRIVE / "heartbeats" / (WORKER["worker"] + ".json")
HB_PATH.parent.mkdir(parents=True, exist_ok=True)
STORAGE = Path("pipeline") / "tmp" / WORKER["storage_subdir"]
POLL = int(WORKER.get("poll_secs", 5))
HB_EVERY = int(WORKER.get("heartbeat_secs", 60))
SYNC_EVERY = int(WORKER.get("sync_secs", 600))
WRAPPER_DIRS_LOCAL = WORKER["wrapper_dirs"]


def _meminfo():
    info = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            info[key] = int(value.strip().split()[0])  # kB
    return info


def total_ram_gb():
    return _meminfo()["MemTotal"] / 1024 / 1024


def used_ram_gb():
    info = _meminfo()
    return (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().split(",")
        return {"name": out[0].strip(), "mem_used_mb": int(out[1]),
                "mem_total_mb": int(out[2]), "util_pct": int(out[3])}
    except Exception:
        return None


def tail_lines(path, n=25):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            window = min(end, 64 * 1024)
            handle.seek(end - window)
            return handle.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


ERROR_RE = re.compile(
    "ResourceExhaustedError|OutOfMemoryError|OOM was thrown|CUDA_ERROR|"
    "Xid|InternalError|ImportError|ModuleNotFoundError|FileNotFoundError|"
    "Traceback \\(most recent call last\\)|Killed|sqlite3\\.|OperationalError",
    re.IGNORECASE,
)
OPTIMIZING_RE = re.compile(r"Optimizing (\\w+)")

HB = {
    "worker": WORKER["worker"],
    "phase": "starting",
    "updated_at": None,
    "restarts": 0,
    "current_wrapper": None,
    "wrappers": {},
    "gpu": None,
    "ram_gb": {},
    "recent_errors": [],
    "last_log_lines": [],
    "package": Path(pkg).name,
}


def write_hb(**updates):
    HB.update(updates)
    HB["updated_at"] = datetime.now(timezone.utc).isoformat()
    HB["ram_gb"] = {"used": round(used_ram_gb(), 1), "total": round(total_ram_gb(), 1)}
    HB["gpu"] = gpu_stats()
    HB["last_log_lines"] = tail_lines(LOG)
    hits = [line for line in HB["last_log_lines"] if ERROR_RE.search(line)]
    if hits:
        stamp = HB["updated_at"]
        for line in hits[-3:]:
            HB["recent_errors"] = (
                [{"at": stamp, "line": line[:200]}] + HB["recent_errors"]
            )[:10]
    tmp = HB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(HB, indent=1), encoding="utf-8")
    tmp.replace(HB_PATH)


def sqlite_backup(src, dst):
    import sqlite3
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(src)
    dst_conn = sqlite3.connect(dst)
    with dst_conn:
        src_conn.backup(dst_conn)
    src_conn.close()
    dst_conn.close()


def sync_progress():
    if not STORAGE.is_dir():
        return
    for db in STORAGE.glob("*/optuna.db"):
        try:
            sqlite_backup(db, DRIVE / "optuna" / db.parent.name / "optuna.db")
            for f in db.parent.glob("*.json"):
                shutil.copy2(f, DRIVE / "optuna" / db.parent.name / f.name)
        except Exception as exc:  # Drive pode cair em sessao longa; nao derruba o treino
            HB["recent_errors"] = (
                [{"at": HB["updated_at"], "line": "sync_drive: " + str(exc)[:200]}]
                + HB["recent_errors"]
            )[:10]


def budget_state():
    import sqlite3
    state = {}
    for wrapper in WORKER["wrappers"]:
        wrapper_dir = WRAPPER_DIRS_LOCAL[wrapper]
        db = STORAGE / wrapper_dir / "optuna.db"
        trials = None
        if db.is_file():
            try:
                conn = sqlite3.connect(db)
                rows = conn.execute(
                    "SELECT COUNT(*) FROM trials t "
                    "JOIN studies s ON t.study_id = s.study_id WHERE s.study_name = ?",
                    (WORKER["study_name"],),
                ).fetchone()
                trials = rows[0]
                conn.close()
            except sqlite3.Error:
                trials = None
        state[wrapper] = {"trials": trials, "budget": WORKER["n_trials"],
                          "done": trials is not None and trials >= WORKER["n_trials"]}
    return state


def fail_orphan_trials():
    import sqlite3
    for db in STORAGE.glob("*/optuna.db"):
        try:
            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE trials SET state = 'FAIL' "
                "WHERE state = 'RUNNING' AND study_id IN "
                "(SELECT study_id FROM studies WHERE study_name = ?)",
                (WORKER["study_name"],),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            print("fail_orphans", db, "->", exc)


MAX_RAM_GB = WORKER.get("max_ram_gb") or round(total_ram_gb() * 0.82, 1)
print(f"worker={WORKER['worker']} wrappers={WORKER['wrappers']} "
      f"budget={WORKER['n_trials']} trials | limite RAM {MAX_RAM_GB} GB "
      f"(total {total_ram_gb():.1f} GB)")

fail_orphan_trials()
sync_progress()
restarts = 0
fast_crashes = 0
backoff = 5
while True:
    write_hb(phase="running", restarts=restarts,
             wrappers=budget_state(), current_wrapper=None)
    if all(info["done"] for info in HB["wrappers"].values()):
        print("orcamento alcancado para todos os wrappers deste worker.")
        break

    cmd = [sys.executable, "pipeline/step_optuna.py", "--config", WORKER["config_path"],
           "--wrappers", *WORKER["wrappers"]]
    print("\\n$ " + " ".join(cmd))
    log_handle = open(LOG, "ab")
    started = time.time()
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                            env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2"))
    killed = False
    last_hb = last_sync = 0.0
    while proc.poll() is None:
        time.sleep(POLL)
        now = time.time()
        if now - last_hb >= HB_EVERY:
            last_hb = now
            log_tail = tail_lines(LOG, 40)
            current = None
            for line in log_tail:
                found = OPTIMIZING_RE.search(line)
                if found:
                    current = found.group(1)
            write_hb(phase="running", restarts=restarts,
                     wrappers=budget_state(), current_wrapper=current)
            counts = ", ".join("{}:{}".format(k, v["trials"])
                               for k, v in HB["wrappers"].items())
            gpu_mb = HB["gpu"]["mem_used_mb"] if HB["gpu"] else "?"
            print("[hb] {} ram={}GB gpu={}MB wrapper={} trials=({})".format(
                datetime.now().strftime("%H:%M:%S"), HB["ram_gb"]["used"],
                gpu_mb, HB["current_wrapper"], counts), flush=True)
        if now - last_sync >= SYNC_EVERY:
            last_sync = now
            sync_progress()
        if used_ram_gb() > MAX_RAM_GB:
            print(f"RAM {used_ram_gb():.1f}GB > {MAX_RAM_GB}GB -> kill + restart")
            proc.terminate()
            time.sleep(3)
            proc.kill()
            killed = True
            break
    rc = proc.wait()
    log_handle.close()
    runtime = time.time() - started
    write_hb(phase="restarting" if rc != 0 else "finishing", restarts=restarts,
             wrappers=budget_state())

    if rc == 0 and not killed:
        # rc 0 = passo completo do step_optuna (orcamento OU timeout_sec): o
        # worker termina; um re-run do notebook retoma o que faltar.
        state = budget_state()
        write_hb(phase="done", wrappers=state)
        print("step_optuna terminou (rc=0).")
        break

    restarts += 1
    fail_orphan_trials()
    sync_progress()
    if runtime < 120:
        fast_crashes += 1
    else:
        fast_crashes = 0
        backoff = 5
    if fast_crashes >= 5:
        write_hb(phase="error",
                 last_error="5 reinicios rapidos consecutivos; veja " + str(LOG))
        raise RuntimeError("watchdog: crash loop detectado (veja o log do worker)")
    print(f"subprocesso rc={rc} killed={killed} runtime={runtime:.0f}s -> "
          f"restart #{restarts} em {backoff}s")
    time.sleep(backoff)
    backoff = min(backoff * 2, 60)

sync_progress()
state = budget_state()
write_hb(phase="done", wrappers=state)
print("\\nResumo final:")
for wrapper, info in state.items():
    print(f"  {wrapper}: {info['trials']}/{info['budget']} trials")
'''

CODE_FINAL = """\
import shutil
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
STORAGE = Path("pipeline/tmp") / WORKER["storage_subdir"]
for db in sorted(STORAGE.glob("*/optuna.db")):
    try:
        study = optuna.load_study(study_name=WORKER["study_name"], storage=f"sqlite:///{db}")
        done = [t for t in study.trials if t.state.name == "COMPLETE"]
        best = min((t.value for t in done), default=float("nan"))
        print(f"{db.parent.name}: {len(study.trials)} trials | melhor={best:.5f}")
    except Exception as exc:
        print(f"{db.parent.name}: {exc}")

# zip final tambem no Drive (copia completa, independente do sync periodico)
archive = Path("/content") / f"{WORKER['storage_subdir']}_{WORKER['worker']}"
shutil.make_archive(str(archive), "zip", ".", str(STORAGE))
shutil.copy2(str(archive) + ".zip", DRIVE / (archive.name + ".zip"))
print("zip salvo no Drive:", archive.name + ".zip")
try:
    from google.colab import files
    files.download(str(archive) + ".zip")
except Exception as exc:
    print("download automatico falhou (pegue no Drive):", exc)
"""


def _nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def cmd_notebooks(args):
    plan = _load_plan(args.plan)
    out_dir = PROJECT_ROOT / "notebooks" / "colab_round2"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for worker in plan["workers"]:
        if worker.get("backend", "colab") != "colab":
            continue  # workers Kaggle rodam como kernels, nao notebooks
        worker_cfg = {
            "worker": worker["worker"],
            "gpu_hint": worker["gpu_hint"],
            "wrappers": worker["wrappers"],
            "config_path": plan["config"],
            "storage_subdir": plan["storage_subdir"],
            "study_name": plan["study_name"],
            "n_trials": plan["n_trials"],
            "drive_root": plan["drive_root"],
            "wrapper_dirs": {w: WRAPPER_DIRS[w] for w in worker["wrappers"]},
            "poll_secs": 5,
            "heartbeat_secs": 60,
            "sync_secs": 600,
            "max_ram_gb": None,
        }
        header = MD_HEADER.format(
            worker=worker["worker"], wrappers=", ".join(worker["wrappers"]),
            study=plan["study_name"], drive_root=plan["drive_root"],
            gpu_hint=worker["gpu_hint"],
        )
        notebook = _nb([
            _cell("markdown", header),
            _cell("code", CODE_PARAMS.format(worker_json=json.dumps(worker_cfg, indent=2))),
            _cell("code", CODE_GPU),
            _cell("code", CODE_PIP),
            _cell("code", CODE_RESTORE),
            _cell("code", CODE_RUNNER),
            _cell("code", CODE_FINAL),
        ])
        path = out_dir / (worker["worker"] + ".ipynb")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(notebook, handle, indent=1)
        paths.append(path)
        print("notebook:", path.relative_to(PROJECT_ROOT))
    return paths


def _load_plan(path=None):
    plan_path = PROJECT_ROOT / (path or PLAN_PATH)
    if not plan_path.is_file():
        raise SystemExit(
            f"Plano {plan_path} nao existe. Rode primeiro: "
            "python scripts/colab_round2.py plan --colab-gpus N"
        )
    with open(plan_path, encoding="utf-8") as handle:
        return json.load(handle)


def cmd_local(args):
    plan = _load_plan(args.plan)
    local = plan.get("local")
    if not local:
        print("Plano sem wrappers locais (tudo remoto).")
        return None
    log_path = PROJECT_ROOT / local["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", "scripts/optuna_watchdog.sh", plan["config"], *local["wrappers"]]
    env = dict(os.environ)
    env["LOG"] = str(log_path)
    if args.max_ram_gb:
        env["MAX_USED_GB"] = str(args.max_ram_gb)

    if not args.start:
        prefix = "MAX_USED_GB={} ".format(args.max_ram_gb) if args.max_ram_gb else ""
        print("Comando para os wrappers leves na GPU local (RAM auto-calibrada):\n")
        print("  {}LOG={} bash scripts/optuna_watchdog.sh {} {}".format(
            prefix, log_path, plan["config"], " ".join(local["wrappers"])))
        return cmd

    out = open(log_path.with_suffix(".out"), "ab")
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env,
                            cwd=PROJECT_ROOT, start_new_session=True)
    print(f"watchdog local iniciado (pid {proc.pid}); log: {log_path}")
    print("Acompanhe com: python scripts/colab_monitor.py --watch")
    return proc


def _kaggle_env():
    """Env de autenticacao da CLI do Kaggle (bearer KGAT ou kaggle.json)."""
    env = dict(os.environ)
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.is_file() and not env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_API_TOKEN"] = token_file.read_text().strip()
    if env.get("KAGGLE_API_TOKEN"):
        # No fluxo bearer o KAGGLE_USERNAME so satisfaz o gate legado da CLI;
        # o namespace real dos slugs vem de _kaggle_user().
        env.setdefault("KAGGLE_USERNAME", "gate")
        env.setdefault("KAGGLE_KEY", "unused-with-api-token")
    return env


def _kaggle_run(*args, check=True):
    exe = shutil.which("kaggle")
    if not exe:
        raise SystemExit("CLI kaggle ausente no ambiente: pip install kaggle")
    print("+", "kaggle", " ".join(str(a) for a in args), file=sys.stderr)
    result = subprocess.run([exe, *(str(a) for a in args)], capture_output=True,
                            text=True, env=_kaggle_env())
    if check and result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()[:400]
        raise SystemExit(f"kaggle {' '.join(str(a) for a in args)} falhou: {message}")
    return result


_KAGGLE_USER_CACHE = None


def _kaggle_user():
    """Usuario real da conta (namespace dos slugs), via kernels list --mine."""
    global _KAGGLE_USER_CACHE
    if _KAGGLE_USER_CACHE:
        return _KAGGLE_USER_CACHE
    result = _kaggle_run("kernels", "list", "--mine", check=False)
    for line in (result.stdout or "").splitlines():
        if "/" in line and not line.startswith("-"):
            _KAGGLE_USER_CACHE = line.split("/", 1)[0].strip()
            return _KAGGLE_USER_CACHE
    raise SystemExit(
        "Usuario Kaggle nao detectado (kaggle kernels list --mine vazio). "
        "Verifique o token em ~/.kaggle/access_token ou exporte KAGGLE_USERNAME."
    )


def cmd_kaggle(args):
    """Versiona o Dataset com o pacote e empurra 1 kernel GPU por worker."""
    plan = _load_plan(args.plan)
    workers = [w for w in plan["workers"] if w.get("backend") == "kaggle"]
    if not workers:
        raise SystemExit("Plano sem workers Kaggle; regenere com --kaggle-gpus >= 1")

    zips = sorted((PROJECT_ROOT / "dist").glob("colab_round2_package_*.zip"),
                  key=lambda p: p.stat().st_mtime)
    if args.repackage or not zips:
        cmd_package(args)
        zips = sorted((PROJECT_ROOT / "dist").glob("colab_round2_package_*.zip"),
                      key=lambda p: p.stat().st_mtime)
    zip_path = zips[-1]

    user = _kaggle_user()
    dataset_slug = plan["kaggle"]["dataset_slug"]
    full_id = f"{user}/{dataset_slug}"
    if not args.dry_run:
        folder = PROJECT_ROOT / "dist" / "kaggle" / "dataset"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        shutil.copy2(zip_path, folder / zip_path.name)
        with open(folder / "dataset-metadata.json", "w", encoding="utf-8") as handle:
            json.dump({"title": "wind speed forecasting (seq2seq lidar)",
                       "id": full_id,
                       "licenses": [{"name": "CC0-1.0"}]}, handle, indent=2)
        exists = _kaggle_run("datasets", "status", full_id, check=False).returncode == 0
        verb = "versionando" if exists else "criando"
        print(f"dataset {full_id}: {verb} com {zip_path.name}")
        if exists:
            _kaggle_run("datasets", "version", "-p", str(folder), "-m",
                        f"{plan['study_name']} {time.strftime('%F %T')}", "--dir-mode", "zip")
        else:
            _kaggle_run("datasets", "create", "-p", str(folder), "--dir-mode", "zip")

    from scripts.kaggle_runner import KERNEL_TEMPLATE

    for worker in workers:
        script = KERNEL_TEMPLATE.format(
            command=worker["command"],
            output_glob=plan["kaggle"]["output_glob"],
            progress_subdir=plan["storage_subdir"],
        )
        kernel_dir = PROJECT_ROOT / "dist" / "kaggle" / f"kernel-{worker['worker']}"
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        kernel_dir.mkdir(parents=True)
        script_path = kernel_dir / f"{worker['slug']}.py"
        script_path.write_text(script, encoding="utf-8")
        metadata = {
            "id": f"{user}/{worker['slug']}",
            "title": worker["slug"].replace("-", " "),
            "code_file": script_path.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [full_id],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        with open(kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        if args.dry_run:
            print(f"--- dry-run: {script_path} ---")
            print(script)
            print(json.dumps(metadata, indent=2))
            continue
        _kaggle_run("kernels", "push", "-p", str(kernel_dir))
        print(f"kernel {user}/{worker['slug']} enviado ({', '.join(worker['wrappers'])})")
    if not args.dry_run:
        print("Acompanhe com: python scripts/colab_monitor.py --watch 60")
        print("Output (ao terminar): python scripts/colab_round2.py collect --kaggle")


def cmd_collect(args):
    import sqlite3

    plan = _load_plan(args.plan)
    mirror = None
    if args.drive_dir:
        mirror = Path(args.drive_dir)
    elif os.environ.get("COLAB_DRIVE_DIR"):
        mirror = Path(os.environ["COLAB_DRIVE_DIR"])

    if args.kaggle:
        user = _kaggle_user()
        dest_root = PROJECT_ROOT / args.kaggle_dest
        for worker in (w for w in plan["workers"] if w.get("backend") == "kaggle"):
            slug_dir = dest_root / worker["slug"]
            slug_dir.mkdir(parents=True, exist_ok=True)
            result = _kaggle_run("kernels", "output", f"{user}/{worker['slug']}",
                                 "-p", str(slug_dir), check=False)
            if result.returncode != 0:
                print(f"[aviso] output de {worker['slug']}: "
                      f"{(result.stderr or result.stdout or '').strip()[:200]}")
                continue
            for zip_path in slug_dir.glob(plan["storage_subdir"] + "*.zip"):
                print("extraindo", zip_path.name, "->", PROJECT_ROOT)
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(PROJECT_ROOT)

    if mirror is not None and not mirror.is_dir():
        raise SystemExit(
            f"--drive-dir {mirror} nao existe (espelho local de "
            f"MyDrive/{plan['drive_root']}) — baixe a pasta do drive.google.com "
            "ou use rclone/google-drive-desktop."
        )
    if mirror is None and not args.kaggle:
        raise SystemExit(
            "Informe --drive-dir (espelho do Drive) e/ou --kaggle (baixa outputs "
            "dos kernels Kaggle)."
        )
    dest_root = PROJECT_ROOT / "pipeline" / "tmp" / plan["storage_subdir"]
    storage = mirror / "optuna" if mirror else None
    summary = []
    if storage is not None and storage.is_dir():
        for src_dir in sorted(p for p in storage.glob("*") if p.is_dir()):
            db = src_dir / "optuna.db"
            if not db.is_file():
                continue
            dst = dest_root / src_dir.name / "optuna.db"
            dst.parent.mkdir(parents=True, exist_ok=True)
            src_conn = sqlite3.connect(db)
            dst_conn = sqlite3.connect(dst)
            with dst_conn:
                src_conn.backup(dst_conn)
            src_conn.close()
            dst_conn.close()
            for f in src_dir.glob("*.json"):
                shutil.copy2(f, dst.parent / f.name)
            summary.append(dst.parent)
            print("coletado:", dst.parent.relative_to(PROJECT_ROOT))

    # Regenera trials.csv / best_trial.json a partir do sqlite coletado para os
    # estudos que ainda nao escreveram localmente.
    try:
        import optuna
        import pandas as pd
    except ImportError:
        print("[aviso] optuna/pandas indisponiveis; pulei a regeracao de best_trial.json")
        optuna = None

    key_by_dir = {v: k for k, v in WRAPPER_DIRS.items()}
    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for wrapper_dir in dest_root.iterdir():
            if not wrapper_dir.is_dir():
                continue
            db = wrapper_dir / "optuna.db"
            if not db.is_file():
                continue
            try:
                study = optuna.load_study(study_name=plan["study_name"],
                                          storage=f"sqlite:///{db}")
            except Exception as exc:
                print(f"[aviso] {wrapper_dir.name}: {exc}")
                continue
            pd.DataFrame(study.trials_dataframe()).to_csv(wrapper_dir / "trials.csv", index=False)
            completed = [t for t in study.trials if t.state.name == "COMPLETE"]
            if not completed:
                continue
            best = min(completed, key=lambda t: t.value)
            payload = {
                "wrapper": key_by_dir.get(wrapper_dir.name, wrapper_dir.name),
                "objective": "val_block_mse(k=4)",
                "best_value": best.value,
                "best_params": best.params,
                "n_trials": len(study.trials),
                "source": "colab_round2 collect",
            }
            with open(wrapper_dir / "best_trial.json", "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(f"  {wrapper_dir.name}: {len(study.trials)} trials | "
                  f"best={best.value:.6f}")

    if args.promote:
        for wrapper_dir in dest_root.iterdir():
            if not wrapper_dir.is_dir():
                continue
            target = PROJECT_ROOT / "pipeline" / "tmp" / "optuna" / wrapper_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for name in ("best_trial.json", "trials.csv", "optuna.db"):
                src = wrapper_dir / name
                if src.is_file():
                    shutil.copy2(src, target / name)
        print("\\n--promote: resultados copiados para pipeline/tmp/optuna/ "
              "(estagios seguintes os encontram via find_optuna_result).")
    print("Collect concluido.")


def cmd_all(args):
    plan = cmd_plan(args)
    cmd_package(args)
    cmd_notebooks(argparse.Namespace(plan=None))
    kaggle_workers = [w for w in plan["workers"] if w.get("backend") == "kaggle"]
    if plan.get("local"):
        print("\nWrappers leves locais — inicie com:")
        print("  python scripts/colab_round2.py local --start")
    if kaggle_workers:
        print("\nWorkers Kaggle — empurre os kernels com:")
        print("  python scripts/colab_round2.py kaggle        (ou --dry-run p/ inspecionar)")
    print("\nProximos passos manuais:")
    print(f"  1. Suba dist/colab_round2_package_*.zip para MyDrive/{plan['drive_root']}/package/")
    print("  2. Abra cada notebook em notebooks/colab_round2/ no Colab com Runtime GPU")
    print("  3. Monitore tudo com: python scripts/colab_monitor.py --watch 60")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="distribui wrappers entre GPUs Colab/Kaggle e local")
    p.add_argument("--config", default=None,
                   help="config do estudo (default: pipeline/pipeline.optuna_round2.json)")
    p.add_argument("--colab-gpus", type=int, default=1,
                   help="numero de notebooks Colab (1 GPU cada) para os pesados")
    p.add_argument("--kaggle-gpus", type=int, default=1,
                   help="numero de kernels Kaggle (1 GPU cada) para os pesados")
    p.add_argument("--max-ram-gb", dest="max_ram_gb", type=float, default=None,
                   help="override do MAX_USED_GB local (default: auto-calibrado)")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("package", help="gera dist/colab_round2_package_<data>.zip")
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_package)

    p = sub.add_parser("notebooks", help="gera notebooks/colab_round2/<worker>.ipynb")
    p.add_argument("--plan", default=None)
    p.set_defaults(func=cmd_notebooks)

    p = sub.add_parser("local", help="comando (ou --start) do watchdog local dos leves")
    p.add_argument("--plan", default=None)
    p.add_argument("--start", action="store_true")
    p.add_argument("--max-ram-gb", type=float, default=None,
                   help="override do MAX_USED_GB (default: 85% da RAM pelo watchdog)")
    p.set_defaults(func=cmd_local)

    p = sub.add_parser("kaggle", help="dataset + push dos kernels Kaggle da rodada")
    p.add_argument("--config", default=None)
    p.add_argument("--plan", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="mostra script/metadata dos kernels sem empurrar")
    p.add_argument("--repackage", action="store_true",
                   help="regenera o pacote antes de versionar o dataset")
    p.set_defaults(func=cmd_kaggle)

    p = sub.add_parser("collect", help="restaura progresso do Drive e/ou Kaggle")
    p.add_argument("--drive-dir", default=None,
                   help="pasta local espelho de MyDrive/%s" % DRIVE_ROOT)
    p.add_argument("--kaggle", action="store_true",
                   help="baixa outputs dos kernels Kaggle antes de processar")
    p.add_argument("--kaggle-dest", default="dist/kaggle_output")
    p.add_argument("--plan", default=None)
    p.add_argument("--promote", action="store_true",
                   help="copia best_trial/trials/optuna.db para pipeline/tmp/optuna/")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("all", help="plan + package + notebooks")
    p.add_argument("--config", default=None)
    p.add_argument("--colab-gpus", type=int, default=1)
    p.add_argument("--kaggle-gpus", type=int, default=1)
    p.add_argument("--max-ram-gb", dest="max_ram_gb", type=float, default=None)
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
