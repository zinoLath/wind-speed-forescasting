# Previsão de Velocidade do Vento com Seq2Seq e Mecanismo de Atenção

Modelo de aprendizado profundo para previsão de velocidade do vento com 6
horas de antecedência (36 passos de 10 min) a partir de 12 horas de histórico
(72 passos), usando arquiteturas Sequence-to-Sequence com atenção — LSTM,
LSTM bidirecional, TCN, TCN bidirecional, TCN+LSTM e Transformer — treinadas
com dados do Parque Eólico Delta do Maranhão.

## 📊 Estrutura do Repositório

```
Wind-Speed-Forecasting/
├── pipeline/                  # Pipeline em estágios (entrada principal)
│   ├── pipeline.py            # Orquestrador (roda todos os estágios)
│   ├── pipeline.json          # Configuração explícita do usuário
│   ├── pipeline.default.json  # Defaults de todos os parâmetros
│   └── README.md              # Documentação completa da pipeline
├── src/
│   ├── models/                # Wrappers Seq2Seq (LSTM, TCN, Transformer)
│   ├── impute/                # Imputação: RF, LightGBM, KNN, Kalman
│   ├── common.py              # Fonte única: GPU, dados, registry, métricas
│   ├── periods.py             # Detecção/exportação de períodos completos
│   └── utils.py               # Denoising wavelet
├── tests/                     # Scripts de benchmark e regressão (standalone)
├── notebooks/                 # Notebooks do TCC e diagramas de modelo
├── docs/                      # Relatórios de experimentos e estudos
├── data/                      # Dados (gitignored); outputs em data/results/
├── models/                    # Modelos salvos (gitignored)
└── OLD/                       # Código aposentado (mantido por segurança)
```

## 🔧 Como Usar

### Configuração do Ambiente

```bash
./setup_environment.sh
source .venv/bin/activate
```

TensorFlow com GPU (Arch Linux + NVIDIA):

```bash
source ./scripts/tf_gpu_env.sh
```

### Pipeline (entrada principal)

```bash
python pipeline/pipeline.py                          # todos os estágios
python pipeline/pipeline.py --stage train evaluate   # só alguns estágios
python pipeline/step_impute.py                       # um estágio isolado
```

Estágios: `optuna` (hiperparâmetros) → `train` (salva
`models/best_model/model.keras` + `model.json`) → `evaluate` (métricas e
gráficos) → `impute` (RF/LightGBM/KNN/Kalman) → `walkforward` (validação
2 meses treino / 1 mês predição). Toda a configuração vive em
`pipeline/pipeline.json`; os artefatos de cada estágio ficam em
`pipeline/tmp/`. Detalhes: [`pipeline/README.md`](pipeline/README.md).

### Notebook do TCC

```bash
./run_notebook.sh
```

## 📈 Convenções de Métricas

As métricas principais comparam as previsões contra o **ws100 cru** (sem
suavização wavelet), e o forecast rolante usa apenas janelas de histórico
conhecido — sem valores futuros no encoder. A função de perda padrão é
**MSE** (penaliza erros grandes); os números históricos do TCC (MAE ≈ 0.16)
foram produzidos com um protocolo diferente e um dataset antigo — ver
`docs/investigacao_baselines.md`. Experimentos atuais e o protocolo honesto
estão documentados em `docs/`.

## 📚 Detalhes Técnicos

- **Modelos**: Seq2Seq com atenção; encoder 72 passos, decoder 36 passos
- **Dados**: `data/dataset.csv` (série de treino/avaliação, 10 min) e
  `data/wind_data.csv` (série longa para imputação e walk-forward)
- **Otimização**: Optuna (TPE + pruning, objetivo = val RMSE com decoder de
  inferência, storage sqlite retomável)
- **Alvo**: `ws100_wavelet` (denoising sym18 nível 2) — escolha validada por
  A/B contra o alvo cru (`docs/ab_target_denoise.md`)
- **Validação**: walk-forward com re-treino por janela (60/30/30 dias por
  padrão; `step_days` menor que a janela de teste permite sobreposição)

## 📝 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---

*Projeto desenvolvido como Trabalho de Conclusão de Curso, utilizando dados
reais do Parque Eólico Delta do Maranhão para validação e teste.*
