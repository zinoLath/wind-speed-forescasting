"""Notifica o progresso do estudo round4b (TCN, local) a cada N trials.

Lê o sqlite do estudo ao vivo e, sempre que o nº de trials COMPLETE cruza um
novo múltiplo de ``STEP`` (default 10), publica um resumo (ntfy + desktop):
trials feitos, melhor skill, nº do melhor trial, tempo decorrido e ETA.

Idempotente por marco (estado em $XDG_RUNTIME_DIR/exp-notify).

    python scripts/round4b_progress_notify.py            # só notifica em marco novo
    python scripts/round4b_progress_notify.py --force    # notifica agora (teste)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna  # noqa: E402

DB = PROJECT_ROOT / "pipeline" / "tmp" / "optuna_round4b" / "Seq2Seq_TCN" / "optuna.db"
STUDY = "round4b"
STATE_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "exp-notify"
STATE = STATE_DIR / "round4b_progress.json"
STEP = 10
TOTAL = 120

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from wfo_notify import notify  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="notifica agora, sem esperar marco novo")
    args = parser.parse_args()

    if not DB.is_file():
        sys.exit(0)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.load_study(study_name=STUDY, storage=f"sqlite:///{DB}")
    except Exception:
        sys.exit(0)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n = len(done)
    if not n:
        sys.exit(0)
    milestone = n // STEP * STEP
    if not args.force and milestone < STEP:
        sys.exit(0)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    last = 0
    if STATE.is_file():
        try:
            last = json.loads(STATE.read_text()).get("milestone", 0)
        except Exception:
            last = 0
    if not args.force and milestone <= last:
        sys.exit(0)

    best = min(done, key=lambda t: t.value)
    skill = -best.value
    elapsed = max((t.datetime_complete.timestamp() for t in done
                   if t.datetime_complete), default=time.time()) \
        - min((t.datetime_start.timestamp() for t in done if t.datetime_start),
              default=time.time())
    per_trial = elapsed / n
    eta_min = per_trial * (TOTAL - n) / 60
    msg = (f"{n}/{TOTAL} trials • melhor skill {skill:+.4f} "
           f"(trial #{best.number}) • {elapsed/60:.0f} min decorridos • "
           f"ETA ~{eta_min:.0f} min")
    notify("Round4b TCN — progresso", msg, priority="default",
           tags="chart_with_upwards_trend", desktop=True)
    STATE.write_text(json.dumps({"milestone": milestone, "n": n}))
    print(msg)


if __name__ == "__main__":
    main()
