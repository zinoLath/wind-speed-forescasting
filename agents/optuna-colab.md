# agents/optuna-colab.md — rodada distribuída do Optuna

> Base canônica: `findings/knowledge/optuna-hpo.md` +
> `findings/playbooks/optuna-distribuida.md` (skill `canonical-findings`).

Resumo operacional — o runbook completo (com playbook de erros) está na
skill **optuna-colab** (`.opencode/skills/optuna-colab/SKILL.md`).

`pipeline/pipeline.optuna_round2.json` define os estudos `round2` (wrappers
`lstm, lstm_bi, tcn, tcn_bi`; um sqlite por wrapper em
`pipeline/tmp/optuna_round2/`; todo worker resume via `load_if_exists=True`).

`scripts/colab_round2.py` planeja a distribuição (pesados: tcn/tcn_bi →
notebooks Colab e/ou kernels Kaggle; leves: lstm/lstm_bi → watchdog local
`scripts/optuna_watchdog.sh`), balanceia pelo custo mediano dos trials,
empacota o zip do repo (Drive `MyDrive/wind-speed-colab/round2/package/` +
dataset Kaggle) e gera os notebooks em `notebooks/colab_round2/` —
**nunca editar esses notebooks à mão; regenerar**. Limites de RAM calibrados
por ambiente em runtime (local 85% do MemTotal, Colab/Kaggle 82%).

Workers Colab publicam heartbeats + snapshots sqlite no Drive; workers
Kaggle (`scripts/kaggle_round2_worker.py`, enviados via `kaggle_runner.py`
com `--progress-subdir optuna_round2`) fazem snapshot do progresso no
output do kernel a cada 10 min e são monitorados via `kaggle kernels
status`. `scripts/colab_monitor.py` consolida tudo (exit 3 = erro/stale).

```bash
python scripts/colab_round2.py all --colab-gpus 1 --kaggle-gpus 1
python scripts/colab_round2.py local --start
python scripts/colab_round2.py kaggle
python scripts/colab_monitor.py --watch 60
python scripts/colab_round2.py collect --kaggle --drive-dir <mirror> --promote
```
