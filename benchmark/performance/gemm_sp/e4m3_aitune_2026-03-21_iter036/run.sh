#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

BIN="$SCRIPT_DIR/gemm_sp_e4m3_iter036"

nvcc -arch sm_90a -std=c++17 \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
  -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
  -Xcompiler -static-libstdc++ -O2 --use_fast_math \
  -I"$REPO_ROOT/runtime" -I"$REPO_ROOT/extern/cutlass/include" -I"$REPO_ROOT" \
  -L/usr/local/cuda/lib64 -lcuda \
  -o "$BIN" \
  "$SCRIPT_DIR/gemm_sp_e4m3_aitune_2026-03-21_iter036.cu"

echo "Built: $BIN"
"$BIN" "$@"
