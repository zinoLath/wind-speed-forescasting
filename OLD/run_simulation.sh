#!/bin/bash

# Script para executar a simulação de previsão de velocidade do vento

# Ativar o ambiente virtual
source ./.venv/bin/activate

# Configurar bibliotecas de runtime para TensorFlow com GPU
source ./scripts/tf_gpu_env.sh

# Executar o script de simulação
echo "Iniciando simulação de previsão de velocidade do vento..."
cd src
python simulation.py

# Desativar o ambiente virtual ao terminar
deactivate
