---
name: optuna-colab
description: Prepara, executa, monitora e conserte a rodada distribuída do Optuna (wrappers pesados em notebooks do Google Colab e kernels do Kaggle, leves na GPU local, watchdogs anti-OOM com RAM calibrada por ambiente). Use quando o usuário mencionar "rodada optuna", "optuna colab", "optuna kaggle", "distribuir optuna", "round2", notebooks/kernels travados/parados/OOM em `notebooks/colab_round2/`, ou comandos `scripts/colab_round2.py` / `scripts/colab_monitor.py` / `scripts/kaggle_round2_worker.py`.
---

# Optuna Round 2 — distribuído (Colab + Kaggle + GPU local)

Infra: `scripts/colab_round2.py` (plan/package/notebooks/local/kaggle/collect),
`scripts/colab_monitor.py` (status/watch), `scripts/kaggle_round2_worker.py`
(watchdog interno dos kernels), `notebooks/colab_round2/*.ipynb` (gerados, NÃO
editar à mão — regenerar), config `pipeline/pipeline.optuna_round2.json`
(wrappers `lstm, lstm_bi, tcn, tcn_bi`; estudo `round2` em
`pipeline/tmp/optuna_round2/`).

Distribuição padrão (`plan --colab-gpus 1 --kaggle-gpus 1`):
local (GTX 1660) = `lstm, lstm_bi`; Colab = `tcn_bi`; Kaggle = `tcn`
(balanceado por LPT com custo mediano real dos trials históricos; ajuste as
contagens de GPU conforme a disponibilidade).

**RAM calibrada por ambiente** (nunca hardcode): local → `optuna_watchdog.sh`
usa 85% de `/proc/meminfo` (override `MAX_USED_GB`/`RAM_FRAC`); Colab → célula
runner usa 82% de `/proc/meminfo` do runtime; Kaggle → worker usa 82%
(`--ram-frac`), lido em runtime conforme a instância (~13–30 GB).

Cada wrapper tem seu próprio sqlite; tudo resume de onde parou. O notebook
Colab grava heartbeat em `MyDrive/wind-speed-colab/round2/heartbeats/`; o
kernel Kaggle não tem heartbeat — seu status vem de `kaggle kernels status` e
o progresso volta pelo output (snapshot a cada 10 min + zip final).

Sempre ative o ambiente antes de qualquer comando local que toque TF/optuna:

```bash
source .venv/bin/activate && source scripts/tf_gpu_env.sh
```

## 1. Preparar (rodada nova)

```bash
python scripts/colab_round2.py all --colab-gpus 1 --kaggle-gpus 1  # plan+package+notebooks
python scripts/colab_round2.py local --start   # watchdog local (lstm, lstm_bi)
python scripts/colab_round2.py kaggle          # dataset + push dos kernels (--dry-run p/ inspecionar)
```

Passos manuais do usuário (impressos pelo `all`):
1. subir `dist/colab_round2_package_*.zip` para `MyDrive/wind-speed-colab/round2/package/`;
2. abrir cada notebook de `notebooks/colab_round2/` no Colab com Runtime → GPU → Run all;
3. para observação remota dos notebooks, espelho do Drive: `--drive-dir <pasta>`
   (desktop, `rclone --rclone-remote gdrive`, ou download manual).

Observação: `python scripts/colab_round2.py kaggle` empurra 1 kernel por worker
Kaggle do plano (GPU + internet, restaura progresso do dataset, roda o worker
watchdog e zipa o output). Auth: `~/.kaggle/access_token` (bearer) — o usuário
real do namespace é auto-detectado via `kaggle kernels list --mine`.

Iniciar a observação: `python scripts/colab_monitor.py --watch 60`.

## 2. Observar

```bash
python scripts/colab_monitor.py --json      # snapshot máquina (exit 3 = problema)
python scripts/colab_monitor.py --watch 60  # painel humano
```

Estados por wrapper: `done` (atingiu n_trials), `running`, `STALE`
(heartbeat Colab > 12 min = sessão caiu), `ERROR` (crash-loop no worker /
kernel Kaggle em erro), `MISSING` (worker nunca começou), `complete (colete!)`
(kernel Kaggle terminou; traga o output com `collect --kaggle`), `UNKNOWN`
(status Kaggle indisponível — CLI/auth), `RUNNING-ERR` (log local com erros).
`all_done: true` → etapa 4.

Loop de observação contínuo: repita `--json` a cada ~5 min; aja imediatamente
em qualquer problema (etapa 3).

## 3. Playbook de correção imediata

| Sintoma | Causa provável | Ação imediata |
| --- | --- | --- |
| heartbeat `STALE` | sessão Colab caiu/preemptada | Avisar o usuário para reabrir o notebook e Run all (restaura do Drive sozinho). Nada a corrigir no código. |
| `phase=error` com "crash loop" no heartbeat Colab | 5 crashes rápidos consecutivos | Ler `hb_last_error`/`hb_errors` no JSON do monitor (ou `last_log_lines` do heartbeat); corrigir a causa raiz abaixo; regerar notebooks (`colab_round2.py notebooks`) e pedir re-execução |
| kernel Kaggle `ERROR` | crash-loop do worker (exit 3) ou erro do template | Baixar output (`collect --kaggle`) e ler o log do kernel no site; corrigir, `kaggle --repackage` e re-empurrar |
| kernel Kaggle `UNKNOWN` 403 | kernel ainda não empurrado OU token inválido | Se já empurrou, testar `kaggle kernels list --mine`; verificar `~/.kaggle/access_token` |
| `ModuleNotFoundError`/`ImportError` no log | dependência faltando/pin errado | corrigir os pins em `CODE_PIP`/`CODE_RUNNER` (Colab) ou requirements (Kaggle usa o pip do kernel com `tensorflow[and-cuda]` do zip); regerar/re-empacotar |
| `FileNotFoundError dataset.csv` | pacote gerado sem dados | `colab_round2.py package` de novo, re-upload (Drive) e `kaggle --repackage` |
| "Nenhum colab_round2_package_*.zip" | zip não subiu / pasta errada | confirmar caminho `MyDrive/wind-speed-colab/round2/package/` com o usuário |
| kernel Kaggle não acha o repo (`pacote do projeto nao encontrado`) | dataset sem o zip novo | `colab_round2.py kaggle --repackage` (re-versiona o dataset) e re-empurrar o kernel |
| muitos `TrialPruned("OOM")` / `user_attr oom` | espaço de busca acima da VRAM do runtime | normal em parte (<30%); se massivo em tcn_bi, sugerir runtime maior (L4/A100) OU reduzir `TCN_RANGES` no código — nunca mudar só um worker (viés no estudo) |
| watchdog local reinicia muito (RAM) | `MAX_USED_GB` alto demais p/ a máquina | reduzir: `MAX_USED_GB=<0.8*RAM> bash scripts/optuna_watchdog.sh ...`; conferir outras cargas |
| `sqlite3.OperationalError: database is locked` | dois escritores no mesmo estudo | verificar plano: cada wrapper em exatamente 1 worker (`plan.json`); matar o duplicado |
| wrapper parado no mesmo nº de trials com status running | kernel/notebook travado sem crash | heartbeat congelado + RAM estável = runtime morto; pedir reabrir (Colab) ou re-empurrar kernel (Kaggle) |
| heartbeat com `sync_drive: ...` erro | mount do Drive caiu em sessão longa | se persistir por >3 syncs, pedir reabrir notebook; progresso até o último sync é preservado |

Regras: nunca editar `notebooks/colab_round2/*.ipynb` nem kernels em
`dist/kaggle/` à mão (fonte é `scripts/colab_round2.py` / `kaggle_runner.py`);
nunca matar processos do usuário sem perguntar; testar `python -m py_compile` e
`pyflakes` após editar scripts; commits só sob pedido.

## 4. Encerrar (todos `done`)

```bash
python scripts/colab_round2.py collect --kaggle --drive-dir <mirror> --promote
python scripts/colab_monitor.py --json   # confirmar all_done
```

`collect` traz: outputs dos kernels Kaggle (zips com `pipeline/tmp/...`,
extraídos no lugar certo) + sqlite do espelho do Drive; regenera
`trials.csv`/`best_trial.json`. `--promote` copia para `pipeline/tmp/optuna/`
(onde `find_optuna_result` procura; sobrescreve resultados da rodada 1 — só
quando o usuário quiser promover). Resumir trials/best por wrapper; próximos
estágios (`pipeline.py --stage train evaluate`) usam os resultados promovidos.
