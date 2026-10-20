# Previsão de Velocidade do Vento com Seq2Seq e Mecanismo de Atenção

Modelo de aprendizado profundo para previsão de velocidade do vento com 6
horas de antecedência (36 passos de 10 min) a partir de 12 horas de histórico
(72 passos), usando arquiteturas Sequence-to-Sequence com atenção — LSTM,
LSTM bidirecional, TCN, TCN bidirecional e Transformer — treinadas com
dados LiDAR do Complexo Eólico Delta do Maranhão (campanha EOSOLAR).

## 📊 Estrutura do Repositório

```
wind-speed-forescasting/
├── pipeline/                  # Pipeline em estágios (entrada principal)
│   ├── pipeline.py            # Orquestrador (roda todos os estágios)
│   ├── pipeline.json          # Configuração explícita do usuário
│   └── pipeline.default.json  # Defaults de todos os parâmetros
├── src/                       # Código-fonte compartilhado
│   ├── models/                # Wrappers Seq2Seq (registry em common.py)
│   ├── impute/                # Imputação: RF, LightGBM, KNN, Kalman
│   ├── common.py              # Fonte única: GPU, dados, WRAPPERS, métricas
│   ├── periods.py             # Detecção/exportação de períodos completos
│   └── utils.py               # Denoising wavelet (sym18, nível 2)
├── scripts/                   # Automação: treinos LiDAR, Optuna distribuída,
│   ├── lidar_7525_train.py    #   sync Overleaf, dashboard de treino
│   ├── colab_round2.py        #   orquestrador da rodada Optuna distribuída
│   ├── colab_monitor.py       #   watchdog/monitor da máquina local
│   └── kaggle_runner.py       #   submissão/monitor de kernels Kaggle
├── tests/                     # Scripts de benchmark/regressão (standalone,
│   │                          #   SEM pytest — roda com python direto)
│   └── relatorio_resultados.py # Gera o relatório de resultados (figuras+JSON)
├── reports/                   # HUB DE RELATÓRIOS (ver seção própria)
├── articles/                  # Artigos: sbr/ e steam/ (clones Overleaf,
│   │                          #   gitignored) + knowledge/ (catálogo bibtex)
├── findings/                  # Base canônica de conhecimento (INDEX.md +
│   │                          #   knowledge/ + playbooks/ + scripts)
├── agents/                    # Contexto por área (para agentes de IA;
│   │                          #   índice no AGENTS.md)
├── notebooks/                 # Notebooks Colab/Kaggle da rodada Optuna
├── docs/                      # Documentação de estudos e protocolos
├── data/                      # Dados e outputs (gitignored)
├── paper/                     # Figuras/manifest do paper (sync Overleaf)
└── OLD/                       # Código aposentado (gitignored)
```

## 🔧 Como Usar

### Configuração do Ambiente

```bash
./setup_environment.sh           # cria .venv (Python 3.10) e instala deps
source .venv/bin/activate
source ./scripts/tf_gpu_env.sh   # OBRIGATÓRIO no Arch Linux com NVIDIA GPU
```

TensorFlow 2.21 com CUDA via pip. Não há linter/CI configurado; para lint
durante o desenvolvimento, `pyflakes` e `radon` são úteis.

### Pipeline (entrada principal)

```bash
python pipeline/pipeline.py                          # todos os estágios
python pipeline/pipeline.py --stage train evaluate   # só alguns estágios
python pipeline/step_train.py                        # um estágio isolado
```

Estágios: `optuna` (hiperparâmetros) → `train` (salva modelos +
`model.json`) → `evaluate` (métricas e gráficos) → `impute`
(RF/LightGBM/KNN/Kalman) → `walkforward` (validação com re-treino por
janela). Toda a configuração vive em `pipeline/pipeline.json` (mesclada
sobre `pipeline.default.json`; estágios aceitam `--config`). Artefatos em
`pipeline/tmp/`. Detalhes: [`pipeline/README.md`](pipeline/README.md).

### Testes e benchmarks (scripts standalone)

```bash
python tests/test_wrappers_build.py               # build+predict de todos wrappers
python tests/test_create_sequences_equivalence.py # vetorizado vs loop oráculo
python tests/ab_raw_vs_wavelet.py --epochs 15     # A/B alvo cru vs wavelet
python tests/benchmark_transformer.py --wrapper lstm_bi --epochs 40
```

Dashboard de treino no terminal (rode em outro pane enquanto qualquer
`fit()` treina — grava métricas por época em `pipeline/tmp/train_dashboard/`):

```bash
python scripts/train_dashboard.py               # segue o run ativo
python scripts/train_dashboard.py --list        # lista runs (ativo/parado)
```

### Treinos LiDAR (protocolo de reamostragem 80/20)

Protocolo sem conjunto de validação (60 épocas, seed 42), usado no
artigo SBSR:

```bash
python scripts/lidar_7525_train.py --train-ratio 0.80   # 80/20 → trained_8020
python scripts/lidar_7525_train.py --train-ratio 0.75   # 75/25 → trained_7525
python scripts/lidar_8020_mseloss_train.py              # ablação de loss
# Regenera relatório de resultados (~5 min de GPU):
python tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020 \
    --out-dir data/results/report_8020 --no-baselines
```

## 📈 Hub de Relatórios (`reports/`)

Todo resultado vira um relatório padronizado em
`reports/<categoria>/<nome>/` (categorias: `treinamento`, `lidar`,
`resultados`, `imputacao`, `protocolo`). Cada relatório é um
`report.py` (gerador) ou `report.md`/`report.html` estático.

```bash
python reports/src/builder.py build    # gera o site inteiro (hub + páginas)
python reports/src/server.py           # http://127.0.0.1:8765/ + API
```

O servidor (`http://127.0.0.1:8765/`) adiciona os botões interativos:
**regenerar** (roda o `regenerate` do relatório), **retreinar** (roda o
`retrain` em background com log), **★ favorito** e **✏ renomear**
(persistidos em `reports/state.json`, que é gerado — nunca editar à
mão). Todo inserível tem botão **LaTeX** com popup nos formatos
`steam`/`sbsr`/genérico (label `fig:`/`tab:` e legenda editáveis).

## 🌐 Rodada Optuna distribuída (local + Colab + Kaggle)

A busca de hiperparâmetros pode ser distribuída entre a GPU local,
notebooks do Google Colab e kernels do Kaggle — todos escrevendo no
mesmo storage sqlite (retomável):

```bash
python scripts/colab_round2.py all --colab-gpus 1 --kaggle-gpus 1
#   ↑ planeja, empacota dados, gera os notebooks/kernels
python scripts/colab_round2.py local --start     # watchdog local (anti-OOM)
python scripts/colab_round2.py kaggle            # dataset + push dos kernels
python scripts/colab_round2.py collect --kaggle --promote   # coleta resultados
python scripts/colab_monitor.py --watch 60       # painel da máquina local
```

Os watchdogs têm RAM calibrada por ambiente e reiniciam trials que
estouram memória. Notebooks gerados ficam em `notebooks/colab_round2/`.

## 📚 Artigos e sincronização Overleaf

Os artigos vivem em `articles/` (gitignored; cada clone Overleaf é um
git próprio com remote apontando para o git bridge):

```bash
python scripts/articles_sync.py list       # lista figuras sincronizadas
python scripts/articles_sync.py sync       # copia figuras dos relatórios
python scripts/articles_sync.py push-all   # commit+push para o Overleaf
```

Push exige `OVERLEAF_USER` e `OVERLEAF_TOKEN` (Git integration token) no
ambiente — as credenciais são injetadas na URL apenas na hora do push e
nunca ficam gravadas no repo. O `paper/` segue o mesmo esquema via
`scripts/paper_sync.py` (com hook post-commit como ponto de push).

## 📈 Convenções de Métricas

As métricas principais comparam as previsões contra o **ws100 cru** (sem
suavização wavelet), e o forecast rolante usa apenas janelas de histórico
conhecido — sem valores futuros no encoder. A função de perda padrão é
MSE ponderada por horizonte (rampa ~0,7→1,3). Números históricos do TCC
(MAE ≈ 0.16) vêm de um protocolo com vazamento e dataset antigo — **não
usar**; os números honestos estão em `reports/` e `docs/`.

## 📝 Detalhes Técnicos

- **Modelos**: Seq2Seq com atenção (registry `src/common.py:WRAPPERS` —
  LSTM, LSTM-Bi, TCN, TCN-Bi, Transformer Pre-LN); encoder 72 passos,
  decoder direto de 36 passos na validação/inferência
- **Dados**: `data/dataset.csv` (série de treino/avaliação, 10 min,
  16/09–08/11/2021) e `data/wind_data.csv` (série longa para imputação e
  walk-forward)
- **Alvo**: `ws100_wavelet` (denoising sym18 nível 2, limiar soft de
  Donoho-Johnstone) — validado por A/B (`docs/ab_target_denoise.md`)
- **Otimização**: Optuna (TPE multivariado + MedianPruner, objetivo =
  RMSE com decoder de inferência, storage sqlite retomável)
- **Validação**: walk-forward com re-treino por janela (60/30/30 dias por
  padrão; `step_days` menor que a janela de teste permite sobreposição)
- **Testes estatísticos**: Diebold–Mariano (HAC + correção de Harvey),
  KSPA com bootstrap e Model Confidence Set acompanham os resultados
  (`tests/metodos_complementares.py`, `mcs.json`)

## 📚 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---

*Projeto desenvolvido como Trabalho de Conclusão de Curso, utilizando
dados reais do Complexo Eólico Delta do Maranhão (campanha EOSOLAR) para
validação e teste.*
