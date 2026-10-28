"""Seleção 1-SE dos hiperparâmetros de uma rodada Optuna (round 7+).

Em vez de promover o ``best_trial`` cru (ponto de sorte em objetivo ruidoso
— ver findings/knowledge/optuna-hpo.md), aplica a regra 1-SE (Breiman et
al. 1984):

  1. melhor trial por valor (o objetivo da round 7 já embute o termo
     ``-lambda*std(blocos)``);
  2. SE estimado dos blocos do melhor trial: std(skills)/sqrt(n_blocos);
  3. conjunto candidato = trials a <= 1 SE do melhor;
  4. dentro do conjunto, escolhe o mais CONSERVADOR: menor learning_rate,
     maior soma de dropouts, menor units/filters.

Escreve ``best_trial.json`` (escolhido) e ``best_trial_raw.json`` (argmin
cru) por wrapper. Sem GPU; lê o sqlite da rodada.

    python scripts/experiments/optuna_select_1se.py --storage pipeline/tmp/optuna_round7 \
        --study round7 [--dry-run]
"""

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

WRAPPER_DIRS = {
    "lstm": "Seq2Seq_LSTM",
    "lstm_bi": "Seq2Seq_LSTM_Bidirectional",
    "tcn": "Seq2Seq_TCN",
    "tcn_bi": "Seq2Seq_TCN_Bidirectional",
}
DEFAULT_SE = 0.02  # se skill_blocks faltar no estudo (rounds antigas)


def conservatism(params):
    """Menor = mais conservador: lr asc, soma de dropouts desc, tamanho asc."""
    lr = float(params.get("learning_rate", 1.0))
    dropout_sum = sum(float(v) for k, v in params.items()
                      if "dropout" in k)
    size = min(float(params.get(k, 0))
               for k in ("lstm_units", "filters", "d_model")
               if k in params) if any(
                   k in params for k in ("lstm_units", "filters")) else 0.0
    return (lr, -dropout_sum, size)


def select_study(db_path, study_name, wrapper_key, tiebreak="conservative"):
    study = optuna.load_study(study_name=study_name,
                              storage=f"sqlite:///{db_path}")
    done = [t for t in study.trials if t.state.name == "COMPLETE"
            and t.value is not None]
    if not done:
        return None
    best = min(done, key=lambda t: t.value)
    # Round 8 (block_mse): val_mse_blocks — os blocos do OBJETIVO. O
    # skill_blocks da round 8 é só diagnóstico (e também existe).
    blocks = (best.user_attrs.get("val_mse_blocks")
              or best.user_attrs.get("skill_blocks") or [])
    if len(blocks) >= 2:
        mean = sum(blocks) / len(blocks)
        var = sum((b - mean) ** 2 for b in blocks) / (len(blocks) - 1)
        se = math.sqrt(var) / math.sqrt(len(blocks))
    else:
        se = DEFAULT_SE
    threshold = best.value + se
    candidates = [t for t in done if t.value <= threshold]
    if tiebreak == "argmin":
        chosen = min(candidates, key=lambda t: t.value)
    else:
        chosen = min(candidates, key=lambda t: conservatism(t.params))
    return {"best": best, "se": se, "candidates": candidates,
            "chosen": chosen, "tiebreak": tiebreak, "blocks": blocks}


def payload(wrapper_key, trial, extra):
    return {
        "wrapper": wrapper_key,
        "objective": extra.get("objective", "skill_vs_arima stability"),
        "stability_lambda": extra.get("stability_lambda"),
        "selection": extra["selection"],
        "n_candidates_1se": extra["n_candidates_1se"],
        "se_blocks": round(extra["se"], 5),
        "best_value": trial.value,
        "skill": trial.user_attrs.get("skill"),
        "skill_stability": trial.user_attrs.get("skill_stability"),
        "skill_blocks": trial.user_attrs.get("skill_blocks"),
        "best_params": trial.params,
        "train_epochs": trial.user_attrs.get("train_epochs"),
        "n_trials": extra["n_trials"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--storage", default="pipeline/tmp/optuna_round7")
    parser.add_argument("--study", default="round7")
    parser.add_argument("--wrappers", nargs="*", default=None,
                        help="Subset (default: todos os que tiverem optuna.db)")
    parser.add_argument("--dry-run", action="store_true",
                        help="so imprime; nao escreve best_trial.json")
    parser.add_argument("--tiebreak", choices=("conservative", "argmin"),
                        default="conservative",
                        help="desempate dentro da banda 1-SE (round 8 usa "
                             "argmin: o desempate conservador escolheu um "
                             "canto underfit na round 7)")
    args = parser.parse_args()

    storage = PROJECT_ROOT / args.storage
    keys = args.wrappers or [k for k, d in WRAPPER_DIRS.items()
                             if (storage / d / "optuna.db").is_file()]
    if not keys:
        raise SystemExit(f"Nenhum optuna.db em {storage}")

    for key in keys:
        db = storage / WRAPPER_DIRS[key] / "optuna.db"
        if not db.is_file():
            print(f"[{key}] sem {db}; pulando")
            continue
        result = select_study(db, args.study, key, tiebreak=args.tiebreak)
        if result is None:
            print(f"[{key}] sem trials COMPLETE; pulando")
            continue
        best, chosen = result["best"], result["chosen"]
        study = optuna.load_study(study_name=args.study,
                                  storage=f"sqlite:///{db}")
        lam = None
        for t in study.trials:
            if "stability_lambda" in t.user_attrs:
                lam = t.user_attrs["stability_lambda"]
                break
        print(f"\n[{key}] trials={len(study.trials)} "
              f"COMPLETE={sum(1 for t in study.trials if t.state.name == 'COMPLETE')}")
        print(f"  argmin:  trial {best.number} value={best.value:+.4f} "
              f"blocks={[round(b, 3) for b in result['blocks']]}")
        print(f"  SE(blocos do argmin)={result['se']:.4f} -> "
              f"conjunto 1-SE: {len(result['candidates'])} trials")
        chosen_params = ", ".join(
            f"{k}={round(v, 5) if isinstance(v, float) else v}"
            for k, v in sorted(chosen.params.items()))
        print(f"  escolhido: trial {chosen.number} "
              f"value={chosen.value:+.4f} params=[{chosen_params}]")
        if chosen.number == best.number:
            print("  (argmin ja e o mais conservador do conjunto)")
        if args.dry_run:
            continue
        out_dir = db.parent
        common_extra = {
            "objective": ("skill_vs_arima stability "
                          f"(lambda={lam})" if lam is not None
                          else "skill_vs_arima"),
            "stability_lambda": lam,
            "n_trials": len(study.trials),
        }
        (out_dir / "best_trial_raw.json").write_text(
            json.dumps(payload(key, best, {
                **common_extra, "selection": "argmin",
                "n_candidates_1se": 1, "se": result["se"]}), indent=2),
            encoding="utf-8")
        (out_dir / "best_trial.json").write_text(
            json.dumps(payload(key, chosen, {
                **common_extra, "selection": "1se_conservative",
                "n_candidates_1se": len(result["candidates"]),
                "se": result["se"]}), indent=2),
            encoding="utf-8")
        print(f"  escrito: {out_dir / 'best_trial.json'} "
              f"(raw em best_trial_raw.json)")


if __name__ == "__main__":
    main()
