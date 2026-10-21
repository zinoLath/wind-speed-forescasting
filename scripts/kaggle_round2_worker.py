"""Worker watchdog da rodada 2 do Optuna para kernels do Kaggle.

Diferente do notebook Colab, o kernel Kaggle nao tem Drive para heartbeat:
o status vive no proprio kernel (``kaggle kernels status``), e o progresso
(sqlite) volta pelo output do kernel com ``kaggle_runner.py output``.

O que este worker adiciona sobre um ``python pipeline/step_optuna.py`` seco:
  - limite de RAM calibrado pelo ambiente de execucao (RAM_FRAC x MemTotal
    de /proc/meminfo, lido em runtime);
  - kill + restart do treino quando a RAM estoura (o estudo resume do sqlite);
  - trials orfaos (RUNNING de um processo morto) marcados FAIL no restart;
  - deteccao de crash-loop (5 quedas rapidas -> erro com log no output).

Usado dentro do kernel gerado por scripts/colab_round2.py kaggle (via
scripts/kaggle_runner.py push), com o repo ja extraido em /kaggle/working:

    python scripts/kaggle_round2_worker.py --config <cfg> --wrappers tcn
"""

import argparse
import gc
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.colab_round2 import WRAPPER_DIRS  # mapa estatico, sem TF


def meminfo_gb():
    info = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            info[key] = int(value.strip().split()[0])  # kB
    return info["MemTotal"] / 1024 / 1024, (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024


def used_ram_gb():
    return meminfo_gb()[1]


def storage_dir(config, subdir_key="storage_subdir"):
    import json

    with open(PROJECT_ROOT / config, encoding="utf-8") as handle:
        cfg = json.load(handle)
    return (PROJECT_ROOT / "pipeline" / "tmp"
            / cfg.get("optuna", {}).get(subdir_key, "optuna"))


def fail_orphan_trials(storage, study_name):
    for db in storage.glob("*/optuna.db"):
        try:
            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE trials SET state = 'FAIL' "
                "WHERE state = 'RUNNING' AND study_id IN "
                "(SELECT study_id FROM studies WHERE study_name = ?)",
                (study_name,),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            print("fail_orphans", db, "->", exc, flush=True)


def budget_state(wrappers, storage, study_name, budget):
    state = {}
    for wrapper in wrappers:
        db = storage / WRAPPER_DIRS[wrapper] / "optuna.db"
        trials = None
        if db.is_file():
            try:
                conn = sqlite3.connect(db)
                trials = conn.execute(
                    "SELECT COUNT(*) FROM trials t JOIN studies s "
                    "ON t.study_id = s.study_id WHERE s.study_name = ?",
                    (study_name,),
                ).fetchone()[0]
                conn.close()
            except sqlite3.Error:
                trials = None
        state[wrapper] = {"trials": trials, "budget": budget,
                          "done": trials is not None and trials >= budget}
    return state


def snapshot_output_loop(storage, interval=600):
    """Re-zipa o progresso em /kaggle/working a cada intervalo.

    O template do kernel so zipa o output DEPOIS do comando; se o Kaggle
    derrubar a sessao no limite de horas, esse snapshot periodico e o que
    sobra como output (perda maxima = um intervalo).
    """
    work = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    while True:
        time.sleep(interval)
        try:
            work.mkdir(parents=True, exist_ok=True)
            if storage.is_dir():
                shutil.make_archive(str(work / storage.name), "zip", ".", str(storage))
                print(f"[snapshot] {work / (storage.name + '.zip')}", flush=True)
        except Exception as exc:
            print("[snapshot] falhou:", exc, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--wrappers", nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=None,
                        help="n_trials do config quando omitido")
    parser.add_argument("--ram-frac", type=float, default=0.82,
                        help="fracao da MemTotal que derruba o treino (calibra "
                             "pelo ambiente; Kaggle ~13-30GB conforme instancia)")
    args = parser.parse_args()

    import json

    with open(PROJECT_ROOT / args.config, encoding="utf-8") as handle:
        optuna_cfg = json.load(handle).get("optuna", {})
    study_name = optuna_cfg.get("study_name", "round2")
    budget = args.budget or int(optuna_cfg.get("n_trials", 120))
    storage = storage_dir(args.config)
    max_ram_gb = round(meminfo_gb()[0] * args.ram_frac, 1)

    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")
    print(f"worker kaggle | wrappers={args.wrappers} budget={budget} | "
          f"RAM total={meminfo_gb()[0]:.1f}GB limite={max_ram_gb}GB "
          f"(frac={args.ram_frac})", flush=True)
    threading.Thread(target=snapshot_output_loop, args=(storage,),
                     daemon=True).start()

    cmd = [sys.executable, "pipeline/step_optuna.py", "--config", args.config,
           "--wrappers", *args.wrappers]
    fail_orphan_trials(storage, study_name)
    restarts = fast_crashes = 0
    backoff = 5
    while True:
        state = budget_state(args.wrappers, storage, study_name, budget)
        print("[orcamento] " + " ".join(
            f"{w}={s['trials']}/{s['budget']}" for w, s in state.items()), flush=True)
        if all(s["done"] for s in state.values()):
            print("orcamento alcancado.", flush=True)
            break

        print("$ " + " ".join(cmd), flush=True)
        started = time.time()
        proc = subprocess.Popen(cmd, env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2"))
        killed = False
        while proc.poll() is None:
            time.sleep(5)
            if used_ram_gb() > max_ram_gb:
                print(f"RAM {used_ram_gb():.1f}GB > {max_ram_gb}GB -> kill + restart",
                      flush=True)
                proc.terminate()
                time.sleep(3)
                proc.kill()
                killed = True
                break
        rc = proc.wait()
        runtime = time.time() - started

        if rc == 0 and not killed:
            print("step_optuna terminou (rc=0).", flush=True)
            break

        restarts += 1
        fail_orphan_trials(storage, study_name)
        if runtime < 120:
            fast_crashes += 1
        else:
            fast_crashes = 0
            backoff = 5
        if fast_crashes >= 5:
            print(f"ERRO: crash loop (5 quedas rapidas). rc={rc} "
                  "ultimas linhas no log acima.", flush=True)
            sys.exit(3)
        print(f"rc={rc} killed={killed} runtime={runtime:.0f}s -> "
              f"restart #{restarts} em {backoff}s", flush=True)
        gc.collect()
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)

    print("worker finalizado com sucesso.", flush=True)


if __name__ == "__main__":
    main()
