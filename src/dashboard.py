"""Logger de treino por epoca + leitores para o dashboard de terminal.

O callback gravado por ``epoch_logger`` escreve uma linha JSON por epoca em
``<dir>/<run_id>.jsonl`` (default: ``pipeline/tmp/train_dashboard/``). O
script ``scripts/train_dashboard.py`` le esses arquivos ao vivo e desenha as
curvas no terminal — mesmo fluxo do nvtop: treino em um painel, dashboard no
outro, nada compartilhado em memoria.

O modulo NAO importa keras no import: o callback e criado por factory
(mesmo padrao de ``common.optuna_pruning_callback``), mantendo o pacote
``src`` importavel sem TensorFlow. Desligue com a variavel de ambiente
``TRAIN_DASHBOARD=0``.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DIR = Path("pipeline/tmp/train_dashboard")
ENV_DIR = "TRAIN_DASHBOARD_DIR"
ENV_ENABLED = "TRAIN_DASHBOARD"

# Campos de uma linha de epoca no JSONL (status="epoch"); linhas de fim de
# treino (status="done"/"early_stop") repetem a ultima epoca sem metricas.
LINE_STATUS_EPOCH = "epoch"


def dashboard_dir():
    """Diretorio dos runs (env TRAIN_DASHBOARD_DIR > default do projeto)."""
    raw = os.environ.get(ENV_DIR)
    path = Path(raw) if raw else DEFAULT_DIR
    return path if path.is_absolute() else PROJECT_ROOT / path


def dashboard_enabled():
    return os.environ.get(ENV_ENABLED, "1") not in ("0", "false", "False", "no")


def epoch_logger(run_id, total_epochs=None, run_dir=None):
    """Callback Keras que appenda uma linha JSONL por epoca (flush imediato).

    Criado por factory para nao importar keras no modulo.
    """
    from keras.callbacks import Callback

    directory = Path(run_dir) if run_dir else dashboard_dir()

    class _EpochLogger(Callback):
        def __init__(self):
            super().__init__()
            self.run_id = run_id
            self.total_epochs = int(total_epochs) if total_epochs else None
            self._epoch_started = None
            self._path = None
            self._last_logs = {}

        def _write(self, payload):
            if self._path is None:
                directory.mkdir(parents=True, exist_ok=True)
                self._path = directory / f"{self.run_id}.jsonl"
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        def _base(self):
            return {
                "run": self.run_id,
                "wrapper": self.run_id.split("_", 1)[1] if "_" in self.run_id else self.run_id,
                "total_epochs": self.total_epochs,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self._last_logs = logs
            line = self._base()
            line.update(
                status=LINE_STATUS_EPOCH,
                epoch=epoch + 1,
                epoch_sec=round(time.perf_counter() - self._epoch_started, 3)
                if self._epoch_started is not None
                else None,
                lr=_current_lr(self.model),
                **{key: _as_float(logs.get(key)) for key in (
                    "loss", "val_loss", "mae", "val_mae")},
            )
            self._write(line)

        def on_train_end(self, logs=None):
            # So sinaliza o fim do treino; a epoca final fica nas linhas
            # "epoch" acumuladas (o leitor conta/usa a ultima delas).
            line = self._base()
            stopped = getattr(self.model, "stop_training", False)
            line["status"] = "early_stop" if stopped else "done"
            line["final"] = {
                key: _as_float(self._last_logs.get(key))
                for key in ("loss", "val_loss", "mae", "val_mae")
            }
            self._write(line)

    return _EpochLogger()


def new_run_id(wrapper_name):
    """ID unico do run: <timestamp>_<wrapper> (o wrapper duplica no arquivo)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = str(wrapper_name).replace("/", "_")
    return f"{stamp}_{safe}"


# --- Leitores (stdlib puro; usados pelo script do dashboard) -----------------


def list_runs(run_dir=None):
    """Runs mais recentes primeiro: [(path, mtime)], apenas *.jsonl validos."""
    directory = Path(run_dir) if run_dir else dashboard_dir()
    if not directory.is_dir():
        return []
    runs = []
    for path in directory.glob("*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        runs.append((path, mtime))
    runs.sort(key=lambda item: item[1], reverse=True)
    return runs


def pick_run(name=None, run_dir=None, max_age=None):
    """Escolhe o run ativo mais recente (ou cujo nome contenha ``name``).

    ``max_age`` (segundos): se o run mais recente for mais velho que isso,
    devolve (path, False) para o dashboard avisar que o treino esta parado.
    Retorna (path | None, is_fresh).
    """
    runs = list_runs(run_dir)
    if name:
        runs = [(p, m) for p, m in runs if name in p.stem]
    if not runs:
        return None, False
    path, mtime = runs[0]
    fresh = max_age is None or (time.time() - mtime) <= max_age
    return path, fresh


def read_run(path):
    """Linhas de epoca do run: lista de dicts com status=epoch, em ordem."""
    path = Path(path)
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # linha parcial (escrita simultanea)
                rows.append(row)
    except FileNotFoundError:
        return []
    return rows


def epoch_rows(rows):
    return [row for row in rows if row.get("status") == LINE_STATUS_EPOCH]


def last_status(rows):
    """Ultimo evento do run: dict com status done/early_stop, ou None."""
    for row in reversed(rows):
        if row.get("status") != LINE_STATUS_EPOCH:
            return row
    return None


def _as_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _current_lr(model):
    """LR atual do otimizador; None quando indisponivel (ex.: schedule)."""
    try:
        lr = model.optimizer.learning_rate
        return _as_float(lr.numpy() if hasattr(lr, "numpy") else lr)
    except Exception:
        return None
