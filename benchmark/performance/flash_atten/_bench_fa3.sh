#!/usr/bin/env bash
# FA3 baseline driver. Set FLASH_ATTEN_BASELINES_DIR to variant baselines/ before exec.
set -euo pipefail
BASELINES_DIR="${FLASH_ATTEN_BASELINES_DIR:?set FLASH_ATTEN_BASELINES_DIR}"
FLASH_ATTEN_ROOT=$(cd "$BASELINES_DIR/../.." && pwd)
# shellcheck source=_bench_common.sh
source "$FLASH_ATTEN_ROOT/_bench_common.sh"

flash_atten_activate_ml_hopper
parse_flash_atten_gpu "$@"
set -- "${FLASH_ATTEN_REMAINING_ARGS[@]}"

echo "[bench] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
exec python3 "$BASELINES_DIR/bench_fa3.py" "$@"
