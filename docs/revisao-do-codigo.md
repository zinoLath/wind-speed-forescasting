# Guia de revisão do código

> Para quem chegou num clone fresco e precisa (1) entender como o
> repositório funciona e (2) reproduzir a última rodada do artigo
> (protocolo **reval 72,5/12,5/15**). Atualizado em 28/10/2026, quando o
> repositório foi reorganizado para revisão humana (registro canônico:
> `findings/knowledge/reorganizacao-revisao.md`).

## 1. O projeto em um parágrafo

Previsão multi-horizonte de velocidade do vento (6 h = 36 passos de
10 min a partir de 12 h = 72 passos) com arquiteturas **seq2seq direct**
(LSTM, LSTM-Bi, TCN, TCN-Bi, Transformer Pre-LN — registry
`src/common.py:WRAPPERS`) sobre dados LiDAR do Complexo Eólico Delta do
Maranhão. Alvo `ws100` treinado sobre `ws100_wavelet` (denoising sym18
nível 2); métricas contra o `ws100` **bruto**. Keras/TF 2.21, Optuna
para HPO, baselines ARIMA/RF/persistência e bateria estatística
(DM/KSPA/MCS/bootstrap em blocos).

## 2. Mapa de leitura (ordem sugerida)

| Camada | Onde | O que ler primeiro |
|---|---|---|
| Contrato do domínio | `src/common.py` | `WRAPPERS`, `load_dataset`, `split_dataset`, `predict_all_horizons`, `compute_metrics`, `default_callbacks`, `canonical_round_meta` |
| Arquiteturas | `src/models/` | um wrapper simples (`s2s_lstm_wrapper.py`) e depois `s2s_tcn_wrapper.py` (TCN hp compartilhado) |
| Utilidades | `src/utils.py` | `wavelet_denoising`, `diebold_mariano*`, bootstrap em blocos |
| Orquestração | `pipeline/pipeline.py` + `pipeline/step_*.py` | `step_train.py` (fluxo treino completo) e `pipeline/config.py` (3 camadas de config + persistência local) |
| Rodada canônica | `scripts/lidar_reval_sbsr_train.py` (e `lidar_reval_h160_train.py`) | treinador standalone por wrapper, HPs do snapshot `pipeline/configs/` |
| Avaliação/relatório | `reports/src/relatorio_resultados.py` | gera `data/results/report_*`: métricas dense/rolling, baselines, testes de hipótese, banco `data/predictions.db` |
| Hub | `reports/src/builder.py`, `server.py`, `components.py` | site estático + servidor com botões regenerar/retreinar |
| Config canônica | `pipeline/pipeline.json` + `pipeline/configs/` | default da pipeline + HPs snapshotados |
| Conhecimento | `findings/INDEX.md` | fatos, números e decisões consolidados |

`tests/` contém **apenas smoke tests** (`test_*.py`, sem pytest).
Estudos/ablações/A-Bs vivem em `studies/`; rodadas concluídas em
`scripts/experiments/`; automação distribuída (Colab/Kaggle/watchers)
em `scripts/dist/`.

## 3. Fluxo ponta a ponta

```
data/dataset.csv ──► src/common.load_dataset ──► wrapper.prepare (janelas 72→36, MinMax no treino)
                        │
                        ▼
        treino (scripts/lidar_reval_*.py ou pipeline step_train)
                        │  salva model.keras + model.json (pipeline/tmp/trained_*)
                        ▼
        reports/src/relatorio_resultados.py
            ├─ métricas dense (todas as origens × 36 passos) e rolling
            ├─ baselines ARIMA/RF/persistência na mesma grade (trim -72 origens)
            ├─ testes de hipótese (DM/Friedman/Wilcoxon/MCS) → JSON
            └─ grava tudo em data/predictions.db (run por report_id)
                        ▼
        data/results/report_* ──► reports/<categoria>/<nome>/report.py ──► hub (reports/_site)
                        ▼
        scripts/sbsr_tables.py + sbsr-integrate ──► artigo SBSR (articles/sbr/)
```

## 4. Reproduzir a rodada reval a partir do clone

Pré-requisitos:

- `./setup_environment.sh` (Python 3.10 + TF 2.21 CUDA) e
  `source scripts/tf_gpu_env.sh` em máquina NVIDIA.
- **`data/dataset.csv`** (gitignored — série LiDAR 10 min com as colunas
  de perfil `ws*/v*/dir*` das alturas 40–140, `humid`, `temp` e `ws100`)
  e, para baselines/walkforward, `data/wind_data.csv`. Sem eles, os
  passos 2–3 ainda funcionam com `--from-db` a partir de
  `data/predictions.db` (também gitignored).

Passos (≈17 min de GPU no treino + ≈5 min na avaliação):

```bash
# 1) treino do lineup SBSR (HPs do snapshot versionado; seed 42, MSE pura)
python scripts/lidar_reval_sbsr_train.py
#    → pipeline/tmp/trained_reval_sbsr/{lstm,lstm_bi,tcn,tcn_bi}/

# 2) avaliação completa + relatório (grade oficial corta as 72 primeiras origens)
python reports/src/relatorio_resultados.py \
    --models-dir pipeline/tmp/trained_reval_sbsr \
    --out-dir data/results/report_reval_sbsr \
    --trim-boundary-origins 72 \
    --run-title "LiDAR 72,5/12,5/15 — revalidação SBSR"

# 3) hub com o relatório novo (reports/lidar/reval_sbsr)
python reports/src/builder.py build
python reports/src/server.py        # http://127.0.0.1:8765/
```

Variante do perfil 40–160 m (rodada atual do paper): trocar o passo 1 por
`python scripts/lidar_reval_h160_train.py` e usar
`trained_reval_h160`/`report_reval_h160` no passo 2 (relatório
`reports/lidar/reval-h160`).

Protocolo embutido nesses treinadores: split 72,5/12,5/15 **com
validação real**, EarlyStopping(val_loss, patience 12,
restore_best) + ReduceLROnPlateau, 60 épocas, seed 42, MSE pura,
alvo `ws100_wavelet`. O mesmo protocolo é o **default do estágio
`train` da pipeline** (`pipeline/pipeline.json`).

## 5. Configuração e persistência

- `pipeline/pipeline.default.json` → `pipeline.json` (canônico
  versionado) → `pipeline.local.json` (overlay gitignored, via
  `python pipeline/config.py set ...`; perfis em `pipeline/profiles/`).
  `--config` explícito ignora o overlay (determinismo de rodada).
- HPs canônicos: `pipeline/configs/canonical_hparams_8020_itildeq.json`
  (snapshot verbatim; `src/common.py:canonical_round_meta` lê o snapshot
  e faz fallback aos artefatos de `pipeline/tmp/trained_8020_itildeq/`).

## 6. Lacunas conhecidas (esperadas num clone)

- `data/` completo (dataset, `predictions.db`, imputações) é
  gitignored — os códigos assumem `data/dataset.csv` presente.
- `scripts/dist/` tem automação **desta máquina**: `ROOT` absoluto
  (`/home/zino/...`), timers systemd, chaves ntfy — úteis como
  referência, não portáveis sem ajuste.
- `notebooks/colab_round2/` e `reports/_site/` são **gerados**
  (regeneráveis via `colab_round2.py notebooks` e `builder.py build`).
- `OLD/`, `results/` (raiz) e `images/` são legado do TCC — mantidos
  fora do git (OLD/images) ou congelados (results/).
- `reports/*/report.md` são análises históricas: citam caminhos da época
  em que foram escritos; os vigentes estão neste guia e no AGENTS.md.

## 7. O que mudou na reorganização de 28/10/2026

Movimentações (nenhuma mudança de comportamento; só caminhos e
`sys.path` bootstrap):

- `tests/*.py` (análises/ablações) → **`studies/`**; `tests/` ficou só
  com os smoke tests `test_*.py`.
- `tests/relatorio_resultados.py` → **`reports/src/`** (o gerador de
  resultados mora com o framework de relatórios).
- `scripts/` dividido: raiz = rodada canônica do artigo + ferramentas do
  paper; `scripts/experiments/` = rodadas concluídas/ablações;
  `scripts/dist/` = Colab/Kaggle/watchers/notificações.
- HPs canônicos snapshotados em `pipeline/configs/` (antes só existiam
  em `pipeline/tmp/`, gitignored — bloqueio de clone fresco).
- `OLD/`, `images/` e `ptxas` des-rastreados (já eram gitignored).
- Referências atualizadas em código, docs, playbooks, skills e argv
  `regenerate`/`retrain` do hub (que roda com `cwd` na raiz).
