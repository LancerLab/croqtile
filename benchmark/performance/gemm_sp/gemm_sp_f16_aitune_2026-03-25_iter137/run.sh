#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

BIN="$SCRIPT_DIR/gemm_sp_f16_iter137"

nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
  -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
  -Xcompiler -static-libstdc++ -O3 --use_fast_math -ftz=true \
  -I"$REPO_ROOT/runtime" -I"$REPO_ROOT/extern/cutlass/include" \
  -L/usr/local/cuda/lib64 -lcuda \
  -o "$BIN" \
  "$SCRIPT_DIR/gemm_sp_f16_iter137_ftz.cu"

echo "Built: $BIN"
"$BIN" "$@"
