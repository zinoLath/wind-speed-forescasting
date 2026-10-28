---
title: Reorganização para revisão humana + persistência de configuração
area: infra
updated: 2026-10-28
---

# Reorganização para revisão humana + persistência de configuração

Auditoria e reorganização de 28/10/2026, motivadas por revisão humana do
código: um clone fresco precisa conseguir **entender o repositório** e
**reproduzir a última rodada** (reval). Nenhuma mudança de comportamento
— apenas caminhos, bootstraps e referências.

## Problemas da auditoria (e resolução)

1. **HPs canônicos só existiam em `pipeline/tmp/` (gitignored)** —
   `scripts/lidar_reval_*` e `lidar_8020_sbsr_*` liam
   `trained_8020_itildeq/<wrapper>/model.json`: clone fresco não rodava a
   rodada. **Resolvido**: snapshot verbatim em
   `pipeline/configs/canonical_hparams_8020_itildeq.json` + loader
   `src/common.py:canonical_round_meta(wrapper_key)` (snapshot primeiro,
   fallback ao artefato local; mesmo valores). Guia de reprodução:
   `docs/revisao-do-codigo.md`.
2. **`tests/` misturava smoke tests com ~35 estudos/ablações** e o
   gerador de resultados morava lá. **Resolvido**: `studies/` novo
   (ex-`tests/`), `tests/` só com `test_*.py`;
   `tests/relatorio_resultados.py` → `reports/src/relatorio_resultados.py`
   (importadores por path/sys.path atualizados: `sbsr_tables`,
   `prediction_diagnostics`, `arima_val_cache` → `studies/`, e os 3
   estudos que faziam `import relatorio_resultados`).
3. **`scripts/` plano com ~85 arquivos** (rodada canônica = históricos =
   automação distribuída). **Resolvido**: raiz = rodada canônica do
   artigo + ferramentas do paper + utilitários vivos;
   `scripts/experiments/` = rodadas concluídas/ablações (incl. família
   `wfo_*`, `itildeq_*`, treinadores `lidar_8020_*`/`lidar_7525`);
   `scripts/dist/` = Colab/Kaggle/TPU/watchers/notificações/cron.
   Bootstraps em subpasta passaram de `parents[1]` para `parents[2]`
   (padrão em todos os `.py` movidos); `kaggle_runner`/`kaggle_package`/
   `optuna_watchdog`/`finalize_v2_report` ganharam `/../..` no ROOT;
   imports cruzados viraram `scripts.dist.*` (e `scripts/__init__.py`
   criado). `sbsr_seeds_kaggle` empacota também `studies`, `reports/src`
   e `pipeline/configs/*.json`.
4. **Config de experimento invadindo o default da pipeline.**
   **Resolvido** com regra + camada de persistência (abaixo).
5. **Inconsistências de gitignore**: `OLD/`, `images/`, `ptxas` listados
   no `.gitignore` mas rastreados (`ptxas` = symlink pendente em clone
   fresco). **Resolvido**: des-rastreados em 28/10 (arquivos ficam no
   disco local).

## Regra: rodada canônica é sempre o default da pipeline

`pipeline/pipeline.json` reflete a rodada canônica (desde 28/10:
protocolo reval 72,5/12,5/15 + ES/RLRP, 60 épocas, MSE pura, seed 42,
denoise `["ws100"]` nível 2 — antes: 75/20/5, patience 8, épocas 100, e
`pipeline.json` antigo com denoise 3 canais nível 1) e **nunca é editado
por experimento**. Camada de persistência em `pipeline/config.py`:

- `python pipeline/config.py show` — config resolvida;
- `set SECTION.KEY=VALUE` / `unset` — grava/remove em
  `pipeline/pipeline.local.json` (gitignored, aplicado só sem `--config`);
- `diff` — o que o local sobrepõe ao canônico;
- `profile-save NAME` / `profile-use NAME` / `profiles` — perfis em
  `pipeline/profiles/*.json` (gitignored);
- `reset` — apaga o local, volta ao canônico;
- `--config` explícito ignora o local por design (rodadas nomeadas
  determinísticas). `_inherit_common` é preservado — o merge local
  acontece antes da herança.

Validação: ciclo set→diff→profile-save→reset→profile-use→reset testado
(28/10) com asserts em `load_config`.

## Estado pós-reorganização (referência de caminhos)

- Raiz `scripts/`: `lidar_reval_{sbsr,h160,allheights,cisltd}_train.py`,
  `lidar_8020_sbsr_train.py`, `lidar_8020_sbsr_seeds_train.py`,
  `sbsr_seeds_kaggle.py`, `kaggle_seeds_dispatch.py`,
  `arima_val_cache.py` (citado por `pipeline/step_optuna.py`),
  `articles_sync.py`, `paper_sync.py`, `sbsr_tables.py`,
  `fig_testes_hipotese_sbsr.py`, `prediction_diagnostics.py`,
  `train_dashboard.py`, `load_experiment_model.py`, `tf_gpu_env.*`,
  `hooks/`.
- ~500 referências atualizadas em 163 arquivos (código, argv
  `regenerate`/`retrain` do hub, docs vivos, playbooks, skills, README,
  AGENTS). `reports/*/report.md`/`report.html` históricos foram
  atualizados mecanicamente onde citavam caminhos movidos; relatórios
  gerados (`_site`) não são editados.
- Automação desta máquina: unit systemd `itildeq-report.service`
  atualizada para `scripts/dist/cron_itildeq_coeffs_report.sh` +
  `daemon-reload` (timer ativo, rodada já auto-desarmada).
- `sbsr_seeds_kaggle.py`/`colab_round2.py` geram pacotes/kernels com a
  árvore nova (padrões `src|scripts|tests|studies|reports/src`).

## Gotchas novos

- Script em subpasta de `scripts/` precisa `parents[2]` no bootstrap
  (raiz de `scripts/` usa `parents[1]`).
- Shell de `scripts/dist/` com `ROOT="/home/zino/..."` absoluto: são
  automação local, não portáveis sem ajuste.
- `pipeline/pipeline.json` (canônico) e `pipeline.default.json` devem
  ficar alinhados — o arquivo de usuário sobrescreve o default, então
  mudar só um deles cria divergência silenciosa.

## Fontes

- Movimentações: `git status` de 28/10/2026 (renames `studies/`,
  `scripts/dist/`, `scripts/experiments/`).
- Persistência: `pipeline/config.py` (`LOCAL_CONFIG_PATH`,
  `PROFILES_DIR`, `load_config`).
- Snapshot canônico: `pipeline/configs/canonical_hparams_8020_itildeq.json`.
- Guia de revisão: `docs/revisao-do-codigo.md`.
