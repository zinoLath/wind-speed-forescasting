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

# Instalando as dependências
pip install --upgrade pip
pip install -r requirements.txt

# TensorFlow com suporte a GPU (CUDA userspace via pip)
pip install "tensorflow[and-cuda]==2.21.0"

# Criando diretórios necessários
mkdir -p models
mkdir -p images
mkdir -p data
mkdir -p scripts

# Script de configuracao de runtime para GPU
cat > scripts/tf_gpu_env.sh << 'EOF'
#!/bin/bash
# Configure runtime libraries for TensorFlow GPU on Arch Linux.
# This is required because TensorFlow may fail to locate cuSOLVER in some venv layouts.

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[tf_gpu_env] Aviso: ambiente virtual nao esta ativo."
    return 0 2>/dev/null || exit 0
fi

CUDA_VENV_BASE="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia"
CUSOLVER_DIR="$CUDA_VENV_BASE/cusolver/lib"

if [ -d "$CUSOLVER_DIR" ]; then
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="$CUSOLVER_DIR:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$CUSOLVER_DIR"
    fi
fi

export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-1}
export TF_GPU_ALLOCATOR=cuda_malloc_async
EOF
chmod +x scripts/tf_gpu_env.sh

cat > scripts/tf_gpu_env.fish << 'EOF'
# Configure runtime libraries for TensorFlow GPU on Arch Linux (fish shell).

if test -z "$VIRTUAL_ENV"
    echo "[tf_gpu_env] Aviso: ambiente virtual nao esta ativo."
    exit 0
end

set CUSOLVER_DIR "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusolver/lib"

if test -d "$CUSOLVER_DIR"
    if test -n "$LD_LIBRARY_PATH"
        set -gx LD_LIBRARY_PATH "$CUSOLVER_DIR:$LD_LIBRARY_PATH"
    else
        set -gx LD_LIBRARY_PATH "$CUSOLVER_DIR"
    end
end

if test -z "$TF_CPP_MIN_LOG_LEVEL"
    set -gx TF_CPP_MIN_LOG_LEVEL 1
end

set -gx TF_GPU_ALLOCATOR cuda_malloc_async
EOF

echo "Ambiente virtual criado e configurado com sucesso!"
echo "Para ativar o ambiente, execute: source .venv/bin/activate"
echo "Para desativar o ambiente, execute: deactivate"
echo "Para habilitar GPU no TensorFlow, execute apos ativar o venv: source ./scripts/tf_gpu_env.sh"
echo "Se estiver no fish: source ./scripts/tf_gpu_env.fish"

# Dicas para resolver problemas comuns
echo ""
echo "Dicas de solução de problemas:"
echo "- Se o comando 'source venv/bin/activate' não funcionar, verifique se o ambiente foi criado na pasta correta"
echo "- Para verificar se o ambiente está ativo, execute: which python"
echo "- O ambiente deve estar na pasta: $(pwd)/.venv"