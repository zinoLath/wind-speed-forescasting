#!/bin/bash

# Script para executar o notebook de previsão de velocidade do vento

# Ativar o ambiente virtual
source ./.venv/bin/activate

# Configurar bibliotecas de runtime para TensorFlow com GPU
source ./scripts/tf_gpu_env.sh

# Otimizar alocação de memória GPU (reduz fragmentação)
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Criar diretórios necessários caso não existam
mkdir -p models
mkdir -p images

# Iniciar o notebook 
jupyter notebook notebooks/wind_speed_forecasting.ipynb

# Desativar o ambiente virtual ao terminar
deactivate
