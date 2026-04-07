#!/bin/bash
# filepath: /home/lucas/Projetos/Wind-Speed-Forecasting/setup_environment.sh

echo "Criando ambiente virtual Python..."

# Criando o ambiente virtual
python3.10 -m venv .venv

# Verificando se o ambiente foi criado corretamente
if [ ! -d ".venv" ] || [ ! -f ".venv/bin/activate" ]; then
    echo "Erro ao criar o ambiente virtual!"
    exit 1
fi

# Ativando o ambiente virtual
source .venv/bin/activate

echo "Instalando dependências..."

# Criando requirements.txt se não existir
if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt << EOF
pandas==2.0.0
numpy==1.24.3
matplotlib==3.7.1
scikit-learn==1.2.2
tensorflow==2.15.0
jupyter==1.0.0
ipykernel==6.22.0
EOF
    echo "Arquivo requirements.txt criado."
fi

# Instalando as dependências
pip install --upgrade pip
pip install -r requirements.txt

# Criando diretórios necessários
mkdir -p models
mkdir -p images
mkdir -p data

echo "Ambiente virtual criado e configurado com sucesso!"
echo "Para ativar o ambiente, execute: source .venv/bin/activate"
echo "Para desativar o ambiente, execute: deactivate"

# Dicas para resolver problemas comuns
echo ""
echo "Dicas de solução de problemas:"
echo "- Se o comando 'source venv/bin/activate' não funcionar, verifique se o ambiente foi criado na pasta correta"
echo "- Para verificar se o ambiente está ativo, execute: which python"
echo "- O ambiente deve estar na pasta: $(pwd)/.venv"