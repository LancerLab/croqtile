#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
CHOREO_BIN="$REPO_ROOT/choreo"

default_kernel="$SCRIPT_DIR/v2_manual_s2_1p1c_tma.co"

kernel_path="$default_kernel"
output_script=""
enable_verify=1
enable_timing=1
enable_profile=0
profile_iter=""
gpu_override=""
force_compile=0

trim_field() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

pick_free_gpu() {
  local max_used="${CHOREO_BATCH_GPU_MAX_USED_MB:-2048}"
  local max_util="${CHOREO_BATCH_GPU_MAX_UTIL:-20}"
  local lines=""
  lines="$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$lines" ]]; then
    return 1
  fi

  local best_gpu=""
  local best_free=-1
  while IFS=',' read -r idx_raw used_raw free_raw util_raw; do
    local idx used free util
    idx="$(trim_field "$idx_raw")"
    used="$(trim_field "$used_raw")"
    free="$(trim_field "$free_raw")"
    util="$(trim_field "$util_raw")"
    [[ "$used" =~ ^[0-9]+$ ]] || used=999999
    [[ "$free" =~ ^[0-9]+$ ]] || free=0
    [[ "$util" =~ ^[0-9]+$ ]] || util=100

    if (( used <= max_used && util <= max_util && free > best_free )); then
      best_gpu="$idx"
      best_free=$free
    fi
  done <<< "$lines"

  [[ -n "$best_gpu" ]] || return 1
  printf '%s\n' "$best_gpu"
}

pick_fallback_gpu() {
  local lines=""
  lines="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$lines" ]]; then
    return 1
  fi

  local best_gpu=""
  local best_free=-1
  while IFS=',' read -r idx_raw free_raw; do
    local idx free
    idx="$(trim_field "$idx_raw")"
    free="$(trim_field "$free_raw")"
    [[ "$free" =~ ^[0-9]+$ ]] || free=0
    if (( free > best_free )); then
      best_gpu="$idx"
      best_free=$free
    fi
  done <<< "$lines"

  [[ -n "$best_gpu" ]] || return 1
  printf '%s\n' "$best_gpu"
}

detect_arch() {
  local gpu_index="$1"
  local compute_cap=""

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  compute_cap="$(nvidia-smi --id="$gpu_index" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  case "$compute_cap" in
    9.0)
      printf '%s\n' "sm_90a"
      return 0
      ;;
    8.9)
      printf '%s\n' "sm_89"
      return 0
      ;;
    8.6)
      printf '%s\n' "sm_86"
      return 0
      ;;
    8.0)
      printf '%s\n' "sm_80"
      return 0
      ;;
  esac
  return 1
}

usage() {
  cat <<'EOF'
Usage: ./bench.sh [options]

Options:
  --kernel PATH           Kernel .co file to build
  --output PATH           Generated compile script path
  --gpu ID                Bind execution to a specific GPU
  --no-verify             Disable verification
  --no-timing             Disable timing output
  --profile [ITER]        Profile with ncu (--set full)
  --force-compile         Force recompilation even if the output is up-to-date
  --help                  Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel)
      kernel_path="$2"
      shift 2
      ;;
    --output)
      output_script="$2"
      shift 2
      ;;
    --gpu)
      gpu_override="$2"
      shift 2
      ;;
    --no-verify)
      enable_verify=0
      shift
      ;;
    --no-timing)
      enable_timing=0
      shift
      ;;
    --profile)
      enable_profile=1
      if [[ $# -ge 2 && "$2" != --* ]]; then
        profile_iter="$2"
        shift 2
      else
        shift
      fi
      ;;
    --force-compile)
      force_compile=1
      shift
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

if [[ "$kernel_path" != /* ]]; then
  kernel_path="$SCRIPT_DIR/$kernel_path"
fi

if [[ ! -f "$kernel_path" ]]; then
  echo "Kernel file not found: $kernel_path" >&2
  exit 1
fi

if [[ -z "$output_script" ]]; then
  mkdir -p "$SCRIPT_DIR/build"
  kernel_base=$(basename "$kernel_path" .co)
  output_script="$SCRIPT_DIR/build/${kernel_base}.cute.result"
fi

FLASH_ATTEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VARIANT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
python3 "$FLASH_ATTEN_ROOT/scripts/gen_bench_configs.py" "$VARIANT_DIR"

selected_gpu=""
if [[ -n "$gpu_override" ]]; then
  selected_gpu="$gpu_override"
  echo "[bench] Using GPU override CUDA_VISIBLE_DEVICES=$selected_gpu"
elif command -v nvidia-smi >/dev/null 2>&1; then
  if selected_gpu="$(pick_free_gpu)"; then
    echo "[bench] Auto-selected free GPU $selected_gpu"
  else
    echo "[bench] Warning: no free GPU found; falling back to GPU with most free memory" >&2
    if selected_gpu="$(pick_fallback_gpu)"; then
      echo "[bench] Fallback GPU $selected_gpu"
    else
      echo "[bench] Warning: failed to query GPU status; running without explicit binding" >&2
    fi
  fi
else
  echo "[bench] Warning: nvidia-smi not found; running without explicit binding" >&2
fi

arch_gpu="$selected_gpu"
if [[ -z "$arch_gpu" ]]; then
  arch_gpu="0"
fi

if ! detected_arch="$(detect_arch "$arch_gpu")"; then
  echo "[bench] Warning: failed to detect GPU arch; defaulting to sm_90a" >&2
  detected_arch="sm_90a"
fi

combined_target_cflags="-DMHA_ENABLE_VERIFY=$enable_verify -DMHA_ENABLE_TIMING=$enable_timing ${EXTRA_NVCC_FLAGS:-}"
extra_choreo_flags="${EXTRA_CHOREO_FLAGS:-}"
choreo_cmd=("$CHOREO_BIN" -gs -t cute -arch="$detected_arch" --use-fast-math)
if [[ -n "$extra_choreo_flags" ]]; then
  read -ra __cf <<< "$extra_choreo_flags"
  choreo_cmd+=("${__cf[@]}")
fi
if [[ $enable_timing -eq 1 ]]; then
  choreo_cmd+=("--disable-runtime-check")
fi
choreo_cmd+=("$kernel_path" -o "$output_script")

echo "[bench] Detected arch $detected_arch"

bench_configs_inc="$SCRIPT_DIR/build/bench_configs.inc"
need_compile=1
if [[ $force_compile -eq 1 ]]; then
  echo "[bench] Forced recompilation requested"
elif [[ -f "$output_script" && "$output_script" -nt "$kernel_path" \
        && ( ! -f "$bench_configs_inc" || "$output_script" -nt "$bench_configs_inc" ) ]]; then
  need_compile=0
  echo "[bench] Skipping compilation (output is up-to-date with kernel source)"
fi

if [[ $need_compile -eq 1 ]]; then
  echo "[bench] Generating $output_script"
  "${choreo_cmd[@]}"

  if grep -q '__launch_bounds__(256, 1)' "$output_script" 2>/dev/null; then
    sed -i 's/__launch_bounds__(256, 1)/__launch_bounds__(256, 2)/' "$output_script"
    echo "[bench] Patched launch_bounds(256,1) -> (256,2)"
  fi
fi

echo "[bench] Running --execute for $(basename "$kernel_path")"
if [[ -n "$selected_gpu" ]]; then
  env CUDA_VISIBLE_DEVICES="$selected_gpu" EXTRA_TARGET_CFLAGS="$combined_target_cflags" \
    bash "$output_script" --execute
else
  EXTRA_TARGET_CFLAGS="$combined_target_cflags" bash "$output_script" --execute
fi

if [[ $enable_profile -eq 1 ]]; then
  NCU_BIN=""
  for candidate in /usr/local/cuda/bin/ncu /usr/local/bin/ncu; do
    if [[ -x "$candidate" ]]; then
      NCU_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$NCU_BIN" ]]; then
    NCU_BIN="$(command -v ncu 2>/dev/null || true)"
  fi
  if [[ -z "$NCU_BIN" ]]; then
    echo "[bench] ERROR: ncu not found." >&2
    exit 1
  fi

  iter_tag="${profile_iter:-0}"
  ncu_output="${output_script}_ncu_iter${iter_tag}"
  echo "[bench] Profiling with ncu (iter=${iter_tag}) -> ${ncu_output}.ncu-rep"

  extra_ncu_flags=()
  if [[ -n "${EXTRA_NCU_FLAGS:-}" ]]; then
    read -ra extra_ncu_flags <<< "$EXTRA_NCU_FLAGS"
  fi
  ncu_cmd=("$NCU_BIN" --set full --target-processes all "${extra_ncu_flags[@]}" -o "$ncu_output"
           bash "$output_script" --execute)

  if [[ -n "$selected_gpu" ]]; then
    env CUDA_VISIBLE_DEVICES="$selected_gpu" EXTRA_TARGET_CFLAGS="$combined_target_cflags" \
      "${ncu_cmd[@]}"
  else
    EXTRA_TARGET_CFLAGS="$combined_target_cflags" "${ncu_cmd[@]}"
  fi

  echo "[bench] ncu profile saved: ${ncu_output}.ncu-rep"
fi
