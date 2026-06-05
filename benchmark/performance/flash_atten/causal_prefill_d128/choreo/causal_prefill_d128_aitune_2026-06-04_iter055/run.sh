#!/usr/bin/env bash
set -euo pipefail

# AI-Tune iter055: stmatrix epilogue with vectorized int4 stores
# Based on best.co (iter036 QK/PV overlap), with .cu-level epilogue modification:
#   - stmatrix.sync.aligned to write f32 WGMMA accumulators as bf16 to shared memory
#   - Vectorized int4 copy from shared memory to global memory (coalesced)
# Performance: 458.5 TFLOPS (B=2, H=16, SEQ=8192, causal prefill, D=128, bf16, SM90a)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d /usr/local/cuda-13 ]; then
    export CUDA_HOME=/usr/local/cuda-13
  elif [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
  else
    echo "Error: CUDA_HOME not set and no CUDA installation found" >&2
    exit 1
  fi
fi

if [ -z "${CUTE_HOME:-}" ]; then
  export CUTE_HOME="$REPO_ROOT/extern/cutlass"
fi

NVCC="${CUDA_HOME}/bin/nvcc"
BIN="$SCRIPT_DIR/flash_atten_stmatrix.exe"

echo "[run.sh] Compiling flash_atten_stmatrix.cu ..."
"$NVCC" \
  -gencode arch=compute_90a,code=sm_90a \
  -std=c++17 \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
  -D__CHOREO_TARGET_CUTE__ \
  -D__USE_CUDA_TYPE__ \
  -D__CHOREO_DMA_DIAGNOSIS__ \
  -Xcompiler -static-libstdc++ \
  -O3 --use_fast_math \
  --expt-relaxed-constexpr \
  -I"$REPO_ROOT/runtime" \
  -I"$CUTE_HOME/include" \
  -I"$KERNEL_DIR" \
  -L"$CUDA_HOME/lib64" -lcuda \
  ${EXTRA_TARGET_CFLAGS:-} \
  -o "$BIN" \
  "$SCRIPT_DIR/flash_atten_stmatrix.cu"

echo "[run.sh] Built: $BIN"
"$BIN" "$@"
