"""Banco SQLite das execuções de predição (``data/predictions.db``).

Substitui o rastro solto de saídas de avaliação (evaluation.json sem
controle do que foi executado) por um esquema relacional:

- ``run_category``         — tipo geral da predição (ex.: ``lidar 8020``);
- ``runs``                 — uma execução de avaliação (título, descrição,
  ``report_id`` apontando para o relatório do hub, protocolo, origem);
- ``run_model``            — cada modelo da run: nome, classe do wrapper,
  flag de homologação (+ timestamp de quando aconteceu), loss, tamanho;
- ``run_model_hp``         — hiperparâmetros por modelo (nome/valor);
- ``run_model_feature``    — colunas de entrada efetivamente usadas;
- ``run_model_stath``      — estatísticas por horizonte (``horizon = 0`` é
  o agregado total; 1..N são os horizontes), com ``regime``
  (``all``/``rolling``) e ``target`` (``raw``/``denoised``);
- ``run_model_prediction`` — cada valor predito: ``prediction_id`` (janela
  de origem), ``timestamp`` (amostra base da predição), ``horizon``
  (horizonte ocupado pela linha), ``actual`` (valor real da amostra) e
  ``predicted``.

Os relatórios do hub (``reports/src/resultados.py``) leem primeiro daqui
(resolução por ``report_id``, ex.: ``lidar/8020``) e só caem para o
``evaluation.json`` quando a run não existe no banco.

Uso programático:

    from src import predictions_db
    conn = predictions_db.connect()
    run_id = predictions_db.record_run_from_evaluation(conn, results=..., extras=..., ...)

Inspeção pela linha de comando::

    python -m src.predictions_db list
    python -m src.predictions_db show <run_id|report_id>
    python -m src.predictions_db categories
    python -m src.predictions_db homologar <run_id> [--model NOME]
    python -m src.predictions_db prune <run_id>
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "predictions.db"

METRIC_COLUMNS = ("samples", "mae", "mse", "rmse", "nmse", "nrmse", "nmae",
                  "r2", "persistence_mae")
REPORT_METRICS = ("samples", "mae", "mse", "rmse", "nmse", "nrmse", "nmae",
                  "r2")

SCHEMA = """
CREATE TABLE IF NOT EXISTS run_category (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id   INTEGER NOT NULL REFERENCES run_category(id),
    title         TEXT NOT NULL,
    description   TEXT,
    report_id     TEXT,
    models_dir    TEXT,
    dataset_rows  INTEGER,
    train_ratio   REAL,
    val_ratio     REAL,
    denoise_level INTEGER,
    source        TEXT,
    source_file   TEXT,
    source_cmd    TEXT,
    out_file      TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_report ON runs(report_id);

CREATE TABLE IF NOT EXISTS run_model (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    class_name    TEXT NOT NULL,
    wrapper_key   TEXT,
    seed          INTEGER,
    homologado    INTEGER NOT NULL DEFAULT 0,
    homologado_em TEXT,
    loss          TEXT,
    params        INTEGER,
    size_mb       REAL,
    color         TEXT,
    timing_json   TEXT,
    training_json TEXT,
    extra_json    TEXT,
    UNIQUE (run_id, name)
);

CREATE TABLE IF NOT EXISTS run_model_hp (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL REFERENCES run_model(id) ON DELETE CASCADE,
    name     TEXT NOT NULL,
    value    TEXT NOT NULL,
    UNIQUE (model_id, name)
);

CREATE TABLE IF NOT EXISTS run_model_feature (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL REFERENCES run_model(id) ON DELETE CASCADE,
    name     TEXT NOT NULL,
    UNIQUE (model_id, name)
);

-- Estatísticas por horizonte; horizon = 0 é o agregado total.
CREATE TABLE IF NOT EXISTS run_model_stath (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        INTEGER NOT NULL REFERENCES run_model(id) ON DELETE CASCADE,
    horizon         INTEGER NOT NULL,
    regime          TEXT NOT NULL DEFAULT 'all',
    target          TEXT NOT NULL DEFAULT 'raw',
    samples         INTEGER,
    mae             REAL,
    mse             REAL,
    rmse            REAL,
    nmse            REAL,
    nrmse           REAL,
    nmae            REAL,
    r2              REAL,
    persistence_mae REAL,
    UNIQUE (model_id, regime, target, horizon)
);

-- Cada linha predita: prediction_id identifica a janela de origem,
-- timestamp marca a amostra base (origem), horizon o horizonte ocupado.
CREATE TABLE IF NOT EXISTS run_model_prediction (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      INTEGER NOT NULL REFERENCES run_model(id) ON DELETE CASCADE,
    mode          TEXT NOT NULL DEFAULT 'dense',
    prediction_id INTEGER NOT NULL,
    timestamp     TEXT NOT NULL,
    origin        TEXT,
    horizon       INTEGER NOT NULL,
    actual        REAL NOT NULL,
    predicted     REAL NOT NULL,
    UNIQUE (model_id, mode, prediction_id, horizon)
);
"""


def _now() -> str:
    """Timestamp ISO local com offset (mesma convenção do model.json)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ts(value) -> str:
    """Serializa um timestamp pandas/numpy para 'YYYY-MM-DD HH:MM:SS'."""
    return pd.Timestamp(value).isoformat(sep=" ")


def _json(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


# ----------------------------------------------------------------------------
# Conexão e escrita
# ----------------------------------------------------------------------------
# Colunas acrescentadas após a primeira versão do esquema; bancos antigos
# são atualizados in place (ALTER TABLE) pelo _migrate.
_MIGRATIONS = {
    "runs": {"source_file": "TEXT", "source_cmd": "TEXT", "out_file": "TEXT"},
    "run_model": {"seed": "INTEGER"},
}


def _migrate(conn) -> None:
    for table, columns in _MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def connect(path: str | Path | None = None, *, read_only: bool = False):
    """Abre o banco criando o esquema (exceto em leitura).

    ``read_only=True`` não cria o arquivo nem o esquema — falha se não
    existir (uso nos builders, evita efeito colateral de criação).
    """
    path = Path(path) if path else DB_PATH
    if read_only:
        if not path.is_file():
            raise FileNotFoundError(f"Banco ausente: {path}")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def ensure_category(conn, name: str, description: str | None = None) -> int:
    """Retorna o id da categoria, criando-a se necessário."""
    row = conn.execute("SELECT id FROM run_category WHERE name = ?",
                       (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO run_category (name, description) VALUES (?, ?)",
        (name, description))
    return cur.lastrowid


def create_run(conn, *, category: str, title: str, description: str | None = None,
               report_id: str | None = None, models_dir: str | None = None,
               dataset_rows: int | None = None, train_ratio: float | None = None,
               val_ratio: float | None = None, denoise_level: int | None = None,
               source: str | None = None, source_file: str | None = None,
               source_cmd: str | None = None, out_file: str | None = None,
               replace: bool = True) -> int:
    """Cria a run; com ``replace``, descarta a run anterior do mesmo report.

    Runs ligadas a um relatório mantêm uma única run canônica por
    ``report_id`` (o histórico fica nos backups/arquivos); runs sem
    ``report_id`` nunca são substituídas — acumulam como histórico.

    Reprodutibilidade: ``source_file`` (código gerador), ``source_cmd``
    (linha de comando completa) e ``out_file`` (arquivo de saída) — junto
    com models_dir/split, permitem reexecutar a run exata.
    """
    if replace and report_id:
        for old in conn.execute(
                "SELECT id FROM runs WHERE report_id = ?",
                (report_id,)).fetchall():
            conn.execute("DELETE FROM runs WHERE id = ?", (old["id"],))
    category_id = ensure_category(conn, category)
    cur = conn.execute(
        "INSERT INTO runs (category_id, title, description, report_id, "
        "models_dir, dataset_rows, train_ratio, val_ratio, denoise_level, "
        "source, source_file, source_cmd, out_file, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (category_id, title, description, report_id, models_dir,
         dataset_rows, train_ratio, val_ratio, denoise_level,
         source or source_file, source_file, source_cmd, out_file, _now()))
    return cur.lastrowid


def add_run_model(conn, run_id: int, *, name: str, class_name: str,
                  wrapper_key: str | None = None, seed: int | None = None,
                  loss: str | None = None,
                  params: int | None = None, size_mb: float | None = None,
                  color: str | None = None, timing: dict | None = None,
                  training: dict | None = None, extra: dict | None = None,
                  hp: dict | None = None, features=None) -> int:
    """Insere o modelo da run (e hp/features quando fornecidos)."""
    cur = conn.execute(
        "INSERT INTO run_model (run_id, name, class_name, wrapper_key, seed, "
        "loss, params, size_mb, color, timing_json, training_json, "
        "extra_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, name, class_name, wrapper_key, seed, loss, params, size_mb,
         color, _json(timing), _json(training), _json(extra)))
    model_id = cur.lastrowid
    if hp:
        set_hp(conn, model_id, hp)
    if features:
        set_features(conn, model_id, features)
    return model_id


def set_hp(conn, model_id: int, hp: dict) -> None:
    """Grava o dicionário de hiperparâmetros do modelo."""
    conn.executemany(
        "INSERT OR REPLACE INTO run_model_hp (model_id, name, value) "
        "VALUES (?, ?, ?)",
        [(model_id, str(k), _json(v) if isinstance(v, (dict, list))
          else str(v)) for k, v in hp.items()])


def set_features(conn, model_id: int, features) -> None:
    """Grava as colunas de entrada do modelo (ordem preservada)."""
    conn.executemany(
        "INSERT OR IGNORE INTO run_model_feature (model_id, name) "
        "VALUES (?, ?)",
        [(model_id, str(f)) for f in features])


def add_stats(conn, model_id: int, horizon: int, regime: str = "all",
              target: str = "raw", metrics: dict | None = None) -> None:
    """Grava uma linha de run_model_stath (horizon 0 = agregado total)."""
    values = {k: metrics.get(k) for k in METRIC_COLUMNS} if metrics else {}
    if values.get("samples") is not None:
        values["samples"] = int(values["samples"])
    conn.execute(
        "INSERT OR REPLACE INTO run_model_stath (model_id, horizon, regime, "
        "target, samples, mae, mse, rmse, nmse, nrmse, nmae, r2, "
        "persistence_mae) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (model_id, int(horizon), regime, target,
         *[values.get(k) for k in METRIC_COLUMNS]))


def add_predictions(conn, model_id: int, df: pd.DataFrame, *,
                    mode: str = "dense", horizon: int | None = None) -> int:
    """Grava as predições em lote; retorna o nº de linhas inseridas.

    ``df`` precisa das colunas ``origin``, ``timestamp``, ``actual`` (valor
    real) e ``predicted``; ``horizon`` vem da coluna de mesmo nome ou do
    parâmetro (caso rolling, um único horizonte). ``prediction_id`` é o nº
    sequencial da janela de origem (ordenada).
    """
    frame = df.dropna(subset=("actual", "predicted"))
    if frame.empty:
        return 0
    codes = pd.factorize(frame["origin"], sort=True)[0] + 1
    if "horizon" in frame.columns:
        horizons = frame["horizon"].to_numpy(dtype=int)
    else:
        if horizon is None:
            raise ValueError("horizon é obrigatório quando df não tem a coluna")
        horizons = np.full(len(frame), int(horizon), dtype=int)
    rows = [
        (model_id, mode, int(pid), _ts(ts), _ts(org), int(h), float(a),
         float(p))
        for pid, ts, org, h, a, p in zip(
            codes, frame["timestamp"], frame["origin"], horizons,
            frame["actual"], frame["predicted"], strict=True)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO run_model_prediction (model_id, mode, "
        "prediction_id, timestamp, origin, horizon, actual, predicted) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def set_homologado(conn, model_id: int, homologado: bool = True,
                   when: str | None = None) -> None:
    """Marca/desmarca a homologação do modelo (com timestamp de quando)."""
    stamp = (when or _now()) if homologado else None
    conn.execute(
        "UPDATE run_model SET homologado = ?, homologado_em = ? WHERE id = ?",
        (int(bool(homologado)), stamp, model_id))


def record_run_from_evaluation(conn, *, results: dict, extras: dict,
                               category: str, title: str,
                               description: str | None = None,
                               report_id: str | None = None,
                               models_dir: str | None = None,
                               dataset_rows: int | None = None,
                               train_ratio: float | None = None,
                               val_ratio: float | None = None,
                               denoise_level: int | None = None,
                               source: str | None = None,
                               source_file: str | None = None,
                               source_cmd=None, out_file: str | None = None,
                               replace: bool = True,
                               verbose: bool = True) -> int:
    """Grava uma run completa a partir da saída de evaluate_models.

    ``results``/``extras`` seguem o formato de reports/src/relatorio_resultados.py:
    ``results[name]`` com key/short/color/loss/params/size_mb/hp/training/
    timing/features/all*/rolling*/per_horizon* e ``extras[name]`` com
    ``all_df`` (denso) e ``roll_df`` (rolling). ``source_cmd`` aceita a
    lista de argv (virada em linha de comando) ou uma string pronta.
    Retorna o id da run.
    """
    from src import common

    if isinstance(source_cmd, (list, tuple)):
        source_cmd = " ".join(str(part) for part in source_cmd)
    run_id = create_run(conn, category=category, title=title,
                        description=description, report_id=report_id,
                        models_dir=models_dir, dataset_rows=dataset_rows,
                        train_ratio=train_ratio, val_ratio=val_ratio,
                        denoise_level=denoise_level, source=source,
                        source_file=source_file, source_cmd=source_cmd,
                        out_file=out_file, replace=replace)
    total_predictions = 0
    for name, entry in results.items():
        key = entry.get("key")
        module_path, class_name = common.WRAPPERS.get(key, (None, "?"))
        training = entry.get("training") or {}
        model_id = add_run_model(
            conn, run_id, name=name,
            class_name=class_name or entry.get("class_name", "?"),
            wrapper_key=key, seed=training.get("seed"),
            loss=entry.get("loss"),
            params=entry.get("params"), size_mb=entry.get("size_mb"),
            color=entry.get("color"), timing=entry.get("timing"),
            training=training, hp=entry.get("hp"),
            features=entry.get("features"))
        add_stats(conn, model_id, 0, "all", "raw", entry.get("all"))
        add_stats(conn, model_id, 0, "all", "denoised",
                  entry.get("all_denoised"))
        add_stats(conn, model_id, 0, "rolling", "raw", entry.get("rolling"))
        add_stats(conn, model_id, 0, "rolling", "denoised",
                  entry.get("rolling_denoised"))
        for row in entry.get("per_horizon") or ():
            add_stats(conn, model_id, row["horizon"], "all", "raw", row)
        for row in entry.get("per_horizon_den") or ():
            add_stats(conn, model_id, row["horizon"], "all", "denoised", row)

        model_extras = extras.get(name) or {}
        all_df = model_extras.get("all_df")
        if all_df is not None:
            # valor real = ws100 bruto (actual_raw); a coluna 'actual' do
            # all_df é o alvo wavelet e não é persistida
            dense = pd.DataFrame({
                "origin": all_df["origin"],
                "timestamp": all_df["timestamp"],
                "horizon": all_df["horizon"],
                "actual": (all_df["actual_raw"] if "actual_raw" in all_df
                           else all_df["actual"]),
                "predicted": all_df["predicted"],
            })
            total_predictions += add_predictions(conn, model_id, dense,
                                                 mode="dense")
        roll_df = model_extras.get("roll_df")
        if roll_df is not None:
            rolling = pd.DataFrame({
                "origin": roll_df["origin"],
                "timestamp": roll_df["timestamp"],
                "actual": (roll_df["actual_raw"] if "actual_raw" in roll_df
                           else roll_df["actual_denoised"]),
                "predicted": roll_df["predicted"],
            })
            total_predictions += add_predictions(
                conn, model_id, rolling, mode="rolling",
                horizon=int(entry["output_steps"])
                if entry.get("output_steps") else None)
        if verbose:
            print(f"  banco: {name} (modelo #{model_id}) gravado")
    conn.commit()
    if verbose:
        print(f"  banco: run #{run_id} com {len(results)} modelos e "
              f"{total_predictions:,} predições".replace(",", "."))
    return run_id


# ----------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------
def run_by_id(conn, run_id: int):
    return conn.execute(
        "SELECT r.*, c.name AS category FROM runs r "
        "JOIN run_category c ON c.id = r.category_id WHERE r.id = ?",
        (run_id,)).fetchone()


def run_by_report(conn, report_id: str):
    """Run canônica de um relatório (mais recente, desempate pelo maior id)."""
    return conn.execute(
        "SELECT r.*, c.name AS category FROM runs r "
        "JOIN run_category c ON c.id = r.category_id "
        "WHERE r.report_id = ? ORDER BY r.id DESC LIMIT 1",
        (report_id,)).fetchone()


def resolve_run(conn, key):
    """Resolve uma run por id inteiro ou report_id (ex.: 'lidar/8020')."""
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        return run_by_id(conn, int(key))
    return run_by_report(conn, str(key))


def models_of_run(conn, run_id: int) -> list:
    return conn.execute(
        "SELECT * FROM run_model WHERE run_id = ? ORDER BY id",
        (run_id,)).fetchall()


def hp_of_model(conn, model_id: int) -> dict:
    rows = conn.execute(
        "SELECT name, value FROM run_model_hp WHERE model_id = ? ORDER BY id",
        (model_id,)).fetchall()
    out = {}
    for row in rows:
        try:
            out[row["name"]] = json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            out[row["name"]] = row["value"]
    return out


def features_of_model(conn, model_id: int) -> list:
    return [r["name"] for r in conn.execute(
        "SELECT name FROM run_model_feature WHERE model_id = ? ORDER BY id",
        (model_id,)).fetchall()]


def stats_of_model(conn, model_id: int) -> dict:
    """{(regime, target): {horizon: Row}} com todas as linhas de stath."""
    out: dict = {}
    for row in conn.execute(
            "SELECT * FROM run_model_stath WHERE model_id = ? "
            "ORDER BY regime, target, horizon", (model_id,)):
        out.setdefault((row["regime"], row["target"]), {})[row["horizon"]] = row
    return out


def predictions_df(conn, model_id: int, mode: str = "dense") -> pd.DataFrame:
    """Predições do modelo como DataFrame longo (timestamps parseados)."""
    df = pd.read_sql_query(
        "SELECT prediction_id, timestamp, origin, horizon, actual, predicted "
        "FROM run_model_prediction WHERE model_id = ? AND mode = ? "
        "ORDER BY prediction_id, horizon", conn, params=(model_id, mode))
    for col in ("timestamp", "origin"):
        df[col] = pd.to_datetime(df[col])
    return df


def _metrics(row) -> dict:
    return {k: row[k] for k in REPORT_METRICS}


def results_dict(conn, run_id: int) -> dict:
    """Reconstrói o formato de evaluation.json (com hp/training/features).

    As chaves ``all``/``all_denoised``/``rolling``/``rolling_denoised``
    vêm de run_model_stath no horizonte 0; ``per_horizon``/``per_horizon_den``
    dos horizontes 1..N do regime ``all``. ``model_id``/``run_id`` vêm
    junto para navegação reversa.
    """
    data = {}
    for model in models_of_run(conn, run_id):
        mid = model["id"]
        stats = stats_of_model(conn, mid)
        name = model["name"]
        entry = {
            "run_id": run_id,
            "model_id": mid,
            "key": model["wrapper_key"],
            "name": name,
            "short": name.replace("Seq2Seq_", "").replace("_", " "),
            "class_name": model["class_name"],
            "seed": model["seed"],
            "loss": model["loss"],
            "params": model["params"],
            "size_mb": model["size_mb"],
            "color": model["color"],
            "homologado": bool(model["homologado"]),
            "homologado_em": model["homologado_em"],
            "hp": hp_of_model(conn, mid),
            "features": features_of_model(conn, mid),
            "timing": json.loads(model["timing_json"] or "{}"),
            "training": json.loads(model["training_json"] or "{}"),
        }
        for field, (regime, target) in (
                ("all", ("all", "raw")),
                ("all_denoised", ("all", "denoised")),
                ("rolling", ("rolling", "raw")),
                ("rolling_denoised", ("rolling", "denoised"))):
            row = stats.get((regime, target), {}).get(0)
            entry[field] = _metrics(row) if row else {}
        for field, target in (("per_horizon", "raw"),
                              ("per_horizon_den", "denoised")):
            rows = stats.get(("all", target), {})
            entry[field] = [
                {**_metrics(row), "horizon": h,
                 "persistence_mae": row["persistence_mae"]}
                for h, row in sorted(rows.items()) if h > 0]
        data[name] = entry
    return data


def runs_list(conn) -> list:
    return conn.execute(
        "SELECT r.id, r.created_at, r.title, r.description, r.report_id, "
        "r.models_dir, c.name AS category, r.train_ratio, r.val_ratio, "
        "(SELECT COUNT(*) FROM run_model m WHERE m.run_id = r.id) AS models, "
        "(SELECT COUNT(*) FROM run_model_prediction p "
        " JOIN run_model m2 ON m2.id = p.model_id WHERE m2.run_id = r.id) "
        "  AS predictions, "
        "(SELECT COUNT(*) FROM run_model m3 WHERE m3.run_id = r.id "
        "  AND m3.homologado = 1) AS homologados "
        "FROM runs r JOIN run_category c ON c.id = r.category_id "
        "ORDER BY r.id DESC").fetchall()


def categories_list(conn) -> list:
    return conn.execute(
        "SELECT c.id, c.name, c.description, COUNT(r.id) AS runs "
        "FROM run_category c LEFT JOIN runs r ON r.category_id = c.id "
        "GROUP BY c.id ORDER BY c.name").fetchall()


def delete_run(conn, run_id: int) -> None:
    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    conn.commit()


# ----------------------------------------------------------------------------
# CLI de inspeção/controle
# ----------------------------------------------------------------------------
def _fmt_ts(value: str | None) -> str:
    if not value:
        return "—"
    return str(value).replace("T", " ")[:19]


def _cmd_list(args) -> int:
    conn = connect()
    try:
        rows = runs_list(conn)
        if not rows:
            print("Nenhuma run gravada.")
            return 0
        print(f"{'id':>4}  {'criada em':<19}  {'categoria':<16}  "
              f"{'modelos':>7}  {'predições':>10}  {'hom.':>4}  "
              f"{'report':<22}  título")
        for r in rows:
            print(f"{r['id']:>4}  {_fmt_ts(r['created_at'])}  "
                  f"{r['category']:<16.16}  {r['models']:>7}  "
                  f"{r['predictions']:>10,}  {r['homologados']:>4}  "
                  f"{(r['report_id'] or '—'):<22.22}  {r['title']}")
        print(f"\n{len(rows)} runs · {DB_PATH}")
    finally:
        conn.close()
    return 0


def _cmd_show(args) -> int:
    conn = connect()
    try:
        run = resolve_run(conn, args.run)
        if run is None:
            print(f"Run não encontrada: {args.run}", file=sys.stderr)
            return 1
        print(f"run #{run['id']} — {run['title']}")
        print(f"  categoria : {run['category']}")
        print(f"  report    : {run['report_id'] or '—'}")
        print(f"  criada em : {_fmt_ts(run['created_at'])}")
        print(f"  modelos   : {run['models_dir'] or '—'}")
        print(f"  split     : train {run['train_ratio']} / val "
              f"{run['val_ratio']} · {run['dataset_rows']} linhas")
        print(f"  gerador   : {run['source_file'] or run['source'] or '—'}")
        if run["source_cmd"]:
            print(f"  comando   : {run['source_cmd']}")
        if run["out_file"]:
            print(f"  saída     : {run['out_file']}")
        if run["description"]:
            print(f"  descrição : {run['description']}")
        for model in models_of_run(conn, run["id"]):
            stats = stats_of_model(conn, model["id"])
            total = stats.get(("all", "raw"), {}).get(0)
            flag = "✓" if model["homologado"] else " "
            when = (f" em {_fmt_ts(model['homologado_em'])}"
                    if model["homologado_em"] else "")
            line = (f"  [{flag}] {model['name']:<28} "
                    f"{model['class_name']} · seed "
                    f"{model['seed'] if model['seed'] is not None else '—'} · "
                    f"loss {model['loss']} · "
                    f"{model['params']:,} params".replace(",", "."))
            if total is not None:
                line += (f" · MSE {total['mse']:.4f} · R² {total['r2']:.3f}")
            print(line + (f" · HOMOLOGADO{when}" if model["homologado"] else ""))
    finally:
        conn.close()
    return 0


def _cmd_categories(args) -> int:
    conn = connect()
    try:
        for row in categories_list(conn):
            print(f"{row['id']:>4}  {row['name']:<24}  "
                  f"{row['runs']} runs  {row['description'] or ''}")
    finally:
        conn.close()
    return 0


def _cmd_homologar(args) -> int:
    conn = connect()
    try:
        run = resolve_run(conn, args.run)
        if run is None:
            print(f"Run não encontrada: {args.run}", file=sys.stderr)
            return 1
        models = models_of_run(conn, run["id"])
        if args.model:
            wanted = {args.model.lower(), args.model.lower().replace(" ", "_")}
            models = [m for m in models if m["name"].lower() in wanted
                      or (m["wrapper_key"] or "").lower() in wanted]
            if not models:
                print(f"Modelo não encontrado na run: {args.model}",
                      file=sys.stderr)
                return 1
        for model in models:
            set_homologado(conn, model["id"], homologado=not args.remove)
            state = "DESMARCADO" if args.remove else \
                f"HOMOLOGADO em {_now()}"
            print(f"  {model['name']}: {state}")
        conn.commit()
    finally:
        conn.close()
    return 0


def _cmd_prune(args) -> int:
    conn = connect()
    try:
        run = resolve_run(conn, args.run)
        if run is None:
            print(f"Run não encontrada: {args.run}", file=sys.stderr)
            return 1
        delete_run(conn, run["id"])
        print(f"Run #{run['id']} ({run['title']}) removida.")
    finally:
        conn.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.predictions_db",
        description="Controle das runs de predição (data/predictions.db).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="lista as runs gravadas")
    p = sub.add_parser("show", help="detalha uma run (id ou report_id)")
    p.add_argument("run")
    sub.add_parser("categories", help="lista as categorias")
    p = sub.add_parser("homologar",
                       help="marca/desmarca homologação (run inteira ou --model)")
    p.add_argument("run")
    p.add_argument("--model", default=None,
                   help="restringe a um modelo da run (nome ou wrapper_key)")
    p.add_argument("--remove", action="store_true",
                   help="remove a marca de homologação")
    p = sub.add_parser("prune", help="remove uma run do banco")
    p.add_argument("run")

    args = parser.parse_args(argv)
    return {"list": _cmd_list, "show": _cmd_show,
            "categories": _cmd_categories, "homologar": _cmd_homologar,
            "prune": _cmd_prune}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
