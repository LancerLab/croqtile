#!/usr/bin/env bash
# Run FA3 baselines for all six flash_atten variants.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck source=_bench_common.sh
source "$SCRIPT_DIR/_bench_common.sh"

flash_atten_activate_ml_hopper
parse_flash_atten_gpu "$@"
set -- "${FLASH_ATTEN_REMAINING_ARGS[@]}"

echo "[run_all_fa3] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2

for variant in \
  decoder_causal_d64 \
  prefill_d128 \
  gqa_decoder_d64 \
  kv_cache_decode_d64 \
  causal_prefill_d128 \
  fp8_d128; do
  echo "========== $variant =========="
  if bash "$ROOT/$variant/baselines/bench.sh" --gpu "${CUDA_VISIBLE_DEVICES}"; then
    :
  else
    echo "[warn] $variant failed (see message above)" >&2
  fi
  echo
done
