#!/usr/bin/env bash
# Run Choreo kernel + external baselines with matched timing env.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHOREO_DIR="$SCRIPT_DIR/../choreo"
KERNEL="${KERNEL:-flash_mha_fwd_aitune_2026-06-01_iter038.co}"

export CHOREO_TIMING_WARMUP="${CHOREO_TIMING_WARMUP:-10}"
export CHOREO_TIMING_REPEAT="${CHOREO_TIMING_REPEAT:-50}"

gpu_args=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_args=(--gpu "$CUDA_VISIBLE_DEVICES")
fi

echo "=== Choreo ($(basename "$KERNEL")) ==="
bash "$CHOREO_DIR/bench.sh" --kernel "$KERNEL" --no-verify "${gpu_args[@]}"

echo ""
echo "=== External baselines (TileLang + FlashAttention) ==="
"$SCRIPT_DIR/bench_baselines.sh" "${gpu_args[@]}"
