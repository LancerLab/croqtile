#!/usr/bin/env bash
# Compare driver. Set FLASH_ATTEN_VARIANT_DIR to variant root before exec.
set -euo pipefail
VARIANT_DIR="${FLASH_ATTEN_VARIANT_DIR:?set FLASH_ATTEN_VARIANT_DIR}"
FLASH_ATTEN_ROOT=$(cd "$VARIANT_DIR/.." && pwd)
# shellcheck source=_bench_common.sh
source "$FLASH_ATTEN_ROOT/scripts/_bench_common.sh"

flash_atten_activate_ml_hopper
parse_flash_atten_gpu "$@"
set -- "${FLASH_ATTEN_REMAINING_ARGS[@]}"

echo "[compare_all] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
exec python3 "$VARIANT_DIR/compare_all.py" --gpu "${CUDA_VISIBLE_DEVICES}" "$@"
