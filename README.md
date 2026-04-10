# Previsão de Velocidade do Vento com Seq2Seq e Mecanismo de Atenção

Este repositório apresenta um modelo de aprendizado profundo para previsão de velocidade do vento utilizando uma arquitetura Sequence-to-Sequence (Seq2Seq) com mecanismo de atenção. O projeto, desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC), demonstrou excelente desempenho na previsão de velocidade do vento com 6 horas de antecedência no Parque Eólico Delta do Maranhão.

## 🌪️ Sobre o Projeto

O modelo implementado alcançou resultados significativos na previsão de velocidade do vento a 100 metros de altura, com um MAE de 0.1589 m/s. A arquitetura Seq2Seq com mecanismo de atenção demonstrou alta capacidade de capturar padrões complexos e prever tendências, permitindo a detecção antecipada de variações críticas na velocidade do vento.

### Características e Resultados

- **Arquitetura**: LSTM Bidirecional (512 unidades) + LSTM com Atenção (512 unidades)
- **Horizonte de Previsão**: 6 horas (36 timesteps)
- **Contexto Histórico**: 12 horas (72 timesteps)
- **Métricas Alcançadas**:
  - MAE: 0.1589 m/s
  - MSE: 0.0447
  - RMSE: 0.2115 m/s

## 📊 Estrutura do Repositório

```
Wind-Speed-Forecasting/
├── data/                     # Arquivos de dados (não incluídos no repositório)
│   ├── dataset.csv           # Dados brutos
│   ├── dataset1.csv          # Dados pré-processados para treino
│   └── dataset2.csv          # Dados para simulação
├── notebooks/                # Jupyter notebooks
│   └── wind_speed_forecasting.ipynb  # Notebook completo com o modelo
├── models/                   # Modelos treinados
│   └── best_model.h5.keras   # Modelo salvo (não incluído no repositório)
├── images/                   # Gráficos e visualizações gerados
├── src/                      # Código fonte
│   └── simulation.py         # Script para simulação de previsões em tempo real
└── README.md                 # Este arquivo
```

## 🔧 Como Usar

### Configuração do Ambiente

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/Wind-Speed-Forecasting.git
cd Wind-Speed-Forecasting
```

2. Configure o ambiente virtual:
```bash
./setup_environment.sh
```

3. Ative o ambiente virtual:
```bash
source .venv/bin/activate
```

### TensorFlow com GPU (Arch Linux)

Se você estiver no Arch Linux com NVIDIA, execute também:

```bash
source ./scripts/tf_gpu_env.sh
```

Se seu shell for fish:

```fish
source ./scripts/tf_gpu_env.fish
```

Esse passo ajusta o `LD_LIBRARY_PATH` para garantir que o TensorFlow encontre a biblioteca `libcusolver.so.11` no ambiente virtual.

### Executando o Projeto

1. Para treinar o modelo:
```bash
./run_notebook.sh
```
O modelo treinado será salvo automaticamente em `models/best_model.h5.keras`

### Executando a Simulação

Para visualizar as previsões em tempo real:

```bash
./run_simulation.sh
```

A interface gráfica permite:
- Visualizar previsões em tempo real
- Comparar valores previstos com reais
- Analisar métricas de desempenho dinamicamente
- Navegar temporalmente através dos dados

## 📈 Resultados Detalhados

O modelo demonstrou excelente desempenho no conjunto de teste:

### Métricas de Precisão
- **MAE**: 0.1589 m/s (erro médio absoluto)
- **MSE**: 0.0447 (erro quadrático médio)
- **RMSE**: 0.2115 m/s (raiz do erro quadrático médio)

### Capacidades Demonstradas
- Previsão precisa com 6 horas de antecedência
- Detecção eficaz de padrões e tendências
- Robustez na identificação de anomalias
- Desempenho consistente em dados brutos e processados

### Visualizações Disponíveis
- Comparação temporal entre valores reais e previstos
- Análise de performance com dados filtrados por wavelet
- Monitoramento em tempo real das métricas
- Gráficos de evolução do treinamento

## 📚 Detalhes Técnicos

### Arquitetura do Modelo
- **Encoder**: LSTM Bidirecional com 512 unidades totais
- **Decoder**: LSTM com 512 unidades e mecanismo de atenção
- **Otimizador**: Adam com taxa de aprendizado adaptativa
- **Função de Perda**: MSE (Mean Squared Error)
- **Callbacks**: Early Stopping, ReduceLROnPlateau, ModelCheckpoint

### Processamento de Dados
- Normalização via MinMaxScaler
- Pré-processamento com transformada Wavelet
- Divisão dos dados: 77% treino, 18% validação, 5% teste

## 📝 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---

*Projeto desenvolvido como Trabalho de Conclusão de Curso, utilizando dados reais do Parque Eólico Delta do Maranhão para validação e teste do modelo.*
