#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

BIN="$SCRIPT_DIR/blockscale_gemm_e4m3_iter051_co"
CHOREO="$REPO_ROOT/build/choreo"

"$CHOREO" -t cute -arch=sm_90a \
  --disable-runtime-check \
  --event-arrive-tx \
  --hoist-wgmma-arrive \
  --hoist-offset \
  -o "$BIN" \
  "$SCRIPT_DIR/blockscale_gemm_e4m3_aitune_2026-03-22_iter051.co"

echo "Built: $BIN"
"$BIN" "$@"
