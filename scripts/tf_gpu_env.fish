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
