#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

BIN="$SCRIPT_DIR/gemm_sp_f16_iter137"
CHOREO="$REPO_ROOT/build/choreo"

"$CHOREO" -t cute -arch=sm_90a --use-warpspec --use-prepack --stmatrix \
  -o "$BIN" \
  "$SCRIPT_DIR/gemm_sp_f16_aitune_2026-03-25_iter137.co"

echo "Built: $BIN"
"$BIN" "$@"
