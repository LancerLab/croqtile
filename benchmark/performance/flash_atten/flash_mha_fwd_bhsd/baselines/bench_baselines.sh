#!/usr/bin/env bash
# Run TileLang and FlashAttention baselines using ml-hopper Python env.
# Timing and problem sizes match choreo mha_helper.hpp / bench.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)
ML_HOPPER_ENV="${ML_HOPPER_ENV:-/home/fem/.env/ml-hopper}"

usage() {
  cat <<'EOF'
Usage: ./bench_baselines.sh [options]

Runs external flash-attention baselines with the same BHSD decoder configs
and cuda-event timing as choreo bench.sh (warmup=10, repeat=50).

Requires: ml-hopper venv with tilelang and flash-attn installed.

Options:
  --backend NAME    all | tilelang | flash_attn  (default: all)
  --verify          Run correctness checks
  --tsv PATH        Append results TSV (default: baselines/results.tsv)
  --gpu ID          CUDA_VISIBLE_DEVICES
  --help

Environment:
  CHOREO_TIMING_WARMUP / CHOREO_TIMING_REPEAT  (same as choreo bench)
  ML_HOPPER_ENV       Path to venv (default: /home/fem/.env/ml-hopper)

Examples:
  ./bench_baselines.sh
  ./bench_baselines.sh --backend flash_attn --verify
  ./bench_baselines.sh --backend tilelang --gpu 0
EOF
}

backend="all"
verify=0
tsv_arg=()
gpu_override=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="$2"
      shift 2
      ;;
    --verify)
      verify=1
      shift
      ;;
    --tsv)
      tsv_arg=(--tsv "$2")
      shift 2
      ;;
    --gpu)
      gpu_override="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$ML_HOPPER_ENV/bin/activate" ]]; then
  echo "ml-hopper env not found: $ML_HOPPER_ENV" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$ML_HOPPER_ENV/bin/activate"

python -c "import torch, flash_attn, tilelang" 2>/dev/null || {
  echo "Missing packages in ml-hopper. Need: torch, flash_attn, tilelang" >&2
  exit 1
}

cmd=(python "$SCRIPT_DIR/bench_baselines.py" --backend "$backend")
if [[ $verify -eq 1 ]]; then
  cmd+=(--verify)
fi
if [[ ${#tsv_arg[@]} -gt 0 ]]; then
  cmd+=("${tsv_arg[@]}")
fi

echo "[baselines] Python: $(which python)"
echo "[baselines] Backend: $backend"
if [[ -n "$gpu_override" ]]; then
  echo "[baselines] CUDA_VISIBLE_DEVICES=$gpu_override"
  env CUDA_VISIBLE_DEVICES="$gpu_override" "${cmd[@]}"
else
  "${cmd[@]}"
fi
