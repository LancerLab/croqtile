#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

BIN="$SCRIPT_DIR/gemm_sp_e4m3_iter068"
CHOREO="$REPO_ROOT/build/choreo"

"$CHOREO" -t cute -arch=sm_90a --tma-cluster-aware --use-prepack \
  --hoist-wgmma-arrive --skip-epilogue-group-sync --stmatrix \
  -o "$BIN" \
  "$SCRIPT_DIR/gemm_sp_e4m3_aitune_2026-03-21_iter068.co"

echo "Built: $BIN"
"$BIN" "$@"
