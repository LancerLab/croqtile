#!/usr/bin/env bash
# Shared helpers for flash_atten bench/compare shell scripts.
# Source from other scripts; do not execute directly.

flash_atten_activate_ml_hopper() {
  local venv="${ML_HOPPER_ENV:-/home/fem/.env/ml-hopper}"
  if [[ -f "$venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$venv/bin/activate"
  fi
}

# Parse --gpu from "$@". Sets CUDA_VISIBLE_DEVICES (default: 1).
# Remaining flags are stored in FLASH_ATTEN_REMAINING_ARGS (array).
parse_flash_atten_gpu() {
  local default_gpu="${FLASH_ATTEN_DEFAULT_GPU:-1}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_gpu}"
  FLASH_ATTEN_REMAINING_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu)
        if [[ $# -lt 2 ]]; then
          echo "error: --gpu requires an argument" >&2
          return 1
        fi
        export CUDA_VISIBLE_DEVICES="$2"
        shift 2
        ;;
      *)
        FLASH_ATTEN_REMAINING_ARGS+=("$1")
        shift
        ;;
    esac
  done
}
