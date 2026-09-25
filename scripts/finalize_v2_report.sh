#!/usr/bin/env bash
# Finalizacao do estudo v2 + regeneracao do relatorio HTML.
#
# 1. espera o watchdog local (tcn_bi) terminar
# 2. espera o kernel optuna-nogate-v2-tcn do Kaggle sair de RUNNING
# 3. baixa o savepoint do Kaggle e mantem, por wrapper, o db com mais trials
# 4. extrai o melhor trial COMPLETE de cada estudo para um staging de hp
# 5. treina os 4 foco com os novos hp (protocolo nogate + ctx horizon)
# 6. regenera data/results/report/relatorio_resultados.html (--retrain)
#
# Log: pipeline/tmp/finalize_v2.log | Concluido: pipeline/tmp/finalize_v2.done

set -u
cd "$(dirname "$0")/.."

LOG="${LOG:-pipeline/tmp/finalize_v2.log}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

source .venv/bin/activate
if [ -f scripts/tf_gpu_env.sh ]; then
    # shellcheck disable=SC1091
    source scripts/tf_gpu_env.sh
fi
export MALLOC_TRIM_THRESHOLD_=134217728
export KAGGLE_USERNAME=joaohenriqueflauzino
export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
export KAGGLE_KEY="unused-with-api-token"

log "aguardando watchdog local (tcn_bi)..."
while pgrep -f "optuna_watchdo[g]" >/dev/null 2>&1; do sleep 60; done
log "watchdog local encerrou"

log "aguardando kernel kaggle optuna-nogate-v2-tcn..."
for _ in $(seq 1 90); do
    st="$(.venv/bin/kaggle kernels status joaohenriqueflauzino/optuna-nogate-v2-tcn 2>/dev/null | tail -1)"
    [ -n "$st" ] && log "status: $st"
    case "$st" in *RUNNING*) sleep 600 ;; *) break ;; esac
done

log "baixando output do kaggle e mesclando dbs..."
rm -rf /tmp/opencode/finalize_kout && mkdir -p /tmp/opencode/finalize_kout
.venv/bin/kaggle kernels output joaohenriqueflauzino/optuna-nogate-v2-tcn \
    -p /tmp/opencode/finalize_kout >>"$LOG" 2>&1 || log "falha ao baixar output (seguindo com estado local)"

python - <<'PY' 2>&1 | tee -a "$LOG"
import glob
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

ROOT = Path.cwd()
V2 = ROOT / "pipeline/tmp/optuna_nogate_v2"

# --- 3. merge: por wrapper, fica o db com mais trials ---
zips = list(Path("/tmp/opencode/finalize_kout").glob("optuna_nogate_v2.zip"))
if zips:
    zext = Path("/tmp/opencode/finalize_kout/extracted")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(zext)
    for name in ("Seq2Seq_TCN", "Seq2Seq_TCN_Bidirectional"):
        cand = zext / f"pipeline/tmp/optuna_nogate_v2/{name}/optuna.db"
        local = V2 / name / "optuna.db"
        if not cand.is_file():
            print(f"{name}: sem db no kaggle; mantendo local")
            continue
        def n(db):
            con = sqlite3.connect(db)
            c = con.execute("select count(*) from trials").fetchone()[0]
            con.close()
            return c
        cn, cl = n(cand), n(local) if local.exists() else -1
        if cn > cl:
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cand, local)
            print(f"{name}: kaggle vence ({cn} > {cl} trials) -> db local atualizado")
        else:
            print(f"{name}: local vence ({cl} >= {cn} trials)")
else:
    print("nenhum savepoint zip no output do kaggle; usando dbs locais")

# --- 4. staging com o melhor trial COMPLETE de cada estudo ---
stage = ROOT / "pipeline/tmp/hp_v2_stage/optuna"
stage.mkdir(parents=True, exist_ok=True)

def best_from_db(db):
    # Usa a API do optuna: no sqlite, categoricos ficam como indices
    # numericos (ex.: loss=2.0); trial.params devolve os tipos corretos.
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{db}")
    summaries = optuna.study.get_all_study_summaries(storage)
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        return None
    best = min(complete, key=lambda t: t.value if t.value is not None else float("inf"))
    return best.value, dict(best.params), len(study.trials)

sources = {
    "Seq2Seq_LSTM": ROOT / "pipeline/tmp/optuna_nogate_v2_kaggle/Seq2Seq_LSTM/best_trial.json",
    "Seq2Seq_LSTM_Bidirectional": ROOT / "pipeline/tmp/optuna_nogate_v2_kaggle/Seq2Seq_LSTM_Bidirectional/best_trial.json",
    "Seq2Seq_TCN": V2 / "Seq2Seq_TCN/optuna.db",
    "Seq2Seq_TCN_Bidirectional": V2 / "Seq2Seq_TCN_Bidirectional/optuna.db",
}
for name, src in sources.items():
    dst = stage / name
    dst.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".json":
        data = json.load(open(src))
        best = {
            "wrapper": data.get("wrapper"),
            "objective": "val_block_mse(k=4)",
            "best_value": data["best_value"],
            "best_params": data["best_params"],
            "n_trials": data.get("n_trials"),
            "seed": 42,
            "source": str(src),
        }
    else:
        got = best_from_db(src)
        if got is None:
            print(f"{name}: NENHUM trial COMPLETE -> sem hp")
            continue
        value, params, total = got
        best = {
            "objective": "val_block_mse(k=4)",
            "best_value": value,
            "best_params": params,
            "n_trials": total,
            "seed": 42,
            "source": str(src),
        }
    json.dump(best, open(dst / "best_trial.json", "w"), indent=2)
    print(f"{name}: best_value={best['best_value']:.4f} n_trials={best['n_trials']}")

# --- config de treino comparativo (protocolo v2: nogate + ctx horizon) ---
base = json.load(open(ROOT / "pipeline/pipeline.train_tcn_compare.json"))
base["common"]["persistence_gate"] = False
base["common"]["context_mode"] = "horizon"
base["paths"]["tmp_dir"] = "pipeline/tmp/hp_v2_stage"
json.dump(base, open("/tmp/opencode/report_compare.json", "w"), indent=1)
print("config de compare gerada em /tmp/opencode/report_compare.json")
PY

log "treinando os 4 wrappers com os hp do v2..."
python tests/train_compare_tcn.py --wrappers lstm lstm_bi tcn tcn_bi \
    --config /tmp/opencode/report_compare.json >>"$LOG" 2>&1 \
    || log "train_compare terminou com erro"

log "regenerando o relatorio HTML..."
python tests/relatorio_resultados.py --retrain >>"$LOG" 2>&1 \
    || log "relatorio terminou com erro"

log "CONCLUIDO"
touch pipeline/tmp/finalize_v2.done
