#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

BIN="$SCRIPT_DIR/moe_gemm_fp8_bf16_iter012_co"
CHOREO="$REPO_ROOT/build/choreo"

EXTRA_TARGET_CFLAGS="-Xptxas --allow-expensive-optimizations=true" \
  "$CHOREO" -t cute -arch=sm_90a \
  --disable-runtime-check \
  -o "$BIN" \
  "$SCRIPT_DIR/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co"

echo "Built: $BIN"
"$BIN" "$@"
