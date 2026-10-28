---
title: Rodada distribuída do Optuna (Colab/Kaggle/local)
area: playbook
updated: 2026-10-19
---

# Playbook: rodada distribuída do Optuna

Fonte: skill **optuna-colab** (runbook autoritativo com playbook de erros).

```bash
# planejar/empacotar/gerar notebooks (pesados -> Colab/Kaggle; leves -> local)
python scripts/dist/colab_round2.py all --colab-gpus 1 --kaggle-gpus 1

# watchdog local (wrappers leves)
python scripts/dist/colab_round2.py local --start

# submeter kernels Kaggle
python scripts/dist/colab_round2.py kaggle

# monitorar (exit 3 = erro/stale)
python scripts/dist/colab_monitor.py --watch 60

# coletar e promover resultados
python scripts/dist/colab_round2.py collect --kaggle --drive-dir <mirror> --promote
```

Pontos-chave:

- Estudo definido em `pipeline/pipeline.optuna_round*.json` (round2:
  lstm, lstm_bi, tcn, tcn_bi; sqlite por wrapper em
  `pipeline/tmp/optuna_round2/`); workers resumem via
  `load_if_exists=True`.
- Notebooks em `notebooks/colab_round2/` são **gerados** — nunca editar à
  mão; regenerar.
- Colab publica heartbeat + snapshot sqlite no Drive
  (`MyDrive/wind-speed-colab/round2/`); Kaggle snapshot no output do
  kernel a cada 10 min (`kaggle_runner.py --progress-subdir
  optuna_round2`).
- Watchdog anti-OOM com RAM calibrada por ambiente (local 85% do
  MemTotal; Colab/Kaggle 82%).
- Para recomeçar um estudo: apagar o `optuna.db` correspondente.
