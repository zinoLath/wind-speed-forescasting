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
