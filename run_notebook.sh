#!/bin/bash

# Script para executar o notebook de previsão de velocidade do vento

# Ativar o ambiente virtual
source ./.venv/bin/activate

# Criar diretórios necessários caso não existam
mkdir -p models
mkdir -p images

# Iniciar o notebook 
jupyter notebook notebooks/wind_speed_forecasting.ipynb

# Desativar o ambiente virtual ao terminar
deactivate
