#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/batch_compile_and_profile_kernels.sh [options] [dir ...]

Options:
  --gpu <id>             Bind execution to a specific GPU. By default auto-pick an idle GPU.
  --timeout-sec <n>      Per-kernel runtime timeout in seconds. Default: 300
  --kill-sec <n>         Grace period before force-kill after timeout. Default: 10
  --output-csv <path>    Override CSV path. Default: performance/<commit_time>_<commit_id>.csv
  --limit <n>            Only run the first n kernels after filtering.
  --pattern <regex>      Only keep kernels whose path matches the regex.
  -h, --help             Show this help.

Default directories:
  benchmark/performance/blockscale_gemm
  benchmark/performance/blockscale_gemm_v2
  benchmark/performance/gemm_sp
  benchmark/performance/matmul
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
  local dir="${SCRIPT_DIR}"
  while [[ "${dir}" != "/" ]]; do
    if [[ -x "${dir}/build/choreo" ]]; then
      printf '%s\n' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  return 1
}

if ! REPO_ROOT="$(find_repo_root)"; then
  echo "error: failed to locate repository root containing build/choreo" >&2
  exit 1
fi

CHOREO_BIN="${REPO_ROOT}/build/choreo"
if [[ ! -x "${CHOREO_BIN}" ]]; then
  echo "error: choreo binary not found or not executable: ${CHOREO_BIN}" >&2
  exit 1
fi

DEFAULT_DIRS=(
  "benchmark/performance/blockscale_gemm"
  "benchmark/performance/blockscale_gemm_v2"
  "benchmark/performance/gemm_sp"
  "benchmark/performance/matmul"
)

gpu_override=""
timeout_sec=300
kill_sec=10
output_csv=""
limit=""
pattern=""
input_dirs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      gpu_override="${2:-}"
      shift 2
      ;;
    --timeout-sec)
      timeout_sec="${2:-}"
      shift 2
      ;;
    --kill-sec)
      kill_sec="${2:-}"
      shift 2
      ;;
    --output-csv)
      output_csv="${2:-}"
      shift 2
      ;;
    --limit)
      limit="${2:-}"
      shift 2
      ;;
    --pattern)
      pattern="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "error: unknown option '$1'" >&2
      usage
      exit 1
      ;;
    *)
      input_dirs+=("$1")
      shift
      ;;
  esac
done

if [[ ${#input_dirs[@]} -eq 0 ]]; then
  input_dirs=("${DEFAULT_DIRS[@]}")
fi

for value_name in timeout_sec kill_sec; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "error: ${value_name} must be a positive integer" >&2
    exit 1
  fi
done

if [[ -n "${limit}" ]] && { [[ ! "${limit}" =~ ^[0-9]+$ ]] || (( limit <= 0 )); }; then
  echo "error: --limit must be a positive integer" >&2
  exit 1
fi

commit_stamp="$(git -C "${REPO_ROOT}" show -s --format=%cd --date=format-local:%Y%m%d_%H%M%S HEAD)"
commit_short="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
run_tag="${commit_stamp}_${commit_short}"

if [[ -z "${output_csv}" ]]; then
  output_csv="${REPO_ROOT}/performance/${run_tag}.csv"
elif [[ "${output_csv}" != /* ]]; then
  output_csv="${REPO_ROOT}/${output_csv}"
fi

run_root="${REPO_ROOT}/build/kernel-batch/${run_tag}"
bin_root="${run_root}/bin"
log_root="${run_root}/logs"
mkdir -p "$(dirname "${output_csv}")" "${bin_root}" "${log_root}"

trim_field() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "${x}"
}

pick_gpu_by_free_mem() {
  local current="${1:-}"
  local max_used="${CHOREO_BATCH_GPU_MAX_USED_MB:-2048}"
  local max_util="${CHOREO_BATCH_GPU_MAX_UTIL:-20}"
  local lines=""
  lines="$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${lines}" ]]; then
    return 1
  fi

  local best_idle=""
  local best_idle_free=-1
  local best_any=""
  local best_any_free=-1
  while IFS=',' read -r idx_raw used_raw free_raw util_raw; do
    local idx used free util
    idx="$(trim_field "${idx_raw}")"
    used="$(trim_field "${used_raw}")"
    free="$(trim_field "${free_raw}")"
    util="$(trim_field "${util_raw}")"
    [[ -n "${current}" && "${idx}" == "${current}" ]] && continue
    [[ "${used}" =~ ^[0-9]+$ ]] || used=999999
    [[ "${free}" =~ ^[0-9]+$ ]] || free=0
    [[ "${util}" =~ ^[0-9]+$ ]] || util=100

    if (( free > best_any_free )); then
      best_any="${idx}"
      best_any_free=${free}
    fi
    if (( used <= max_used && util <= max_util )) && (( free > best_idle_free )); then
      best_idle="${idx}"
      best_idle_free=${free}
    fi
  done <<< "${lines}"

  if [[ -n "${best_idle}" ]]; then
    printf '%s\n' "${best_idle}"
    return 0
  fi
  if [[ -n "${best_any}" ]]; then
    printf '%s\n' "${best_any}"
    return 0
  fi
  return 1
}

selected_gpu="${gpu_override}"
if [[ -z "${selected_gpu}" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi not found; use --gpu to set device explicitly" >&2
    exit 1
  fi
  if ! selected_gpu="$(pick_gpu_by_free_mem)"; then
    echo "error: failed to auto-select a GPU" >&2
    exit 1
  fi
fi

infer_arch() {
  local input="$1"
  if [[ "${input}" == *"sm90"* ]]; then
    printf '%s\n' "sm_90a"
    return 0
  fi
  if [[ "${input}" == *"sm86"* ]]; then
    printf '%s\n' "sm_86"
    return 0
  fi
  printf '%s\n' ""
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

append_flag_if_needed() {
  local input_path="$1"
  local flag="$2"
  local needle="$3"
  local -n cmd_ref="$4"
  local arg
  if [[ "${input_path}" != *"${needle}"* ]]; then
    return 0
  fi
  for arg in "${cmd_ref[@]}"; do
    if [[ "${arg}" == "${flag}" ]]; then
      return 0
    fi
  done
  cmd_ref+=("${flag}")
}

run_with_timeout() {
  local out_log="$1"
  shift
  local -a cmd=( "$@" )

  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout -k "${kill_sec}" "${timeout_sec}" "${cmd[@]}" >"${out_log}" 2>&1
    local ec=$?
    set -e
    return "${ec}"
  fi

  "${cmd[@]}" >"${out_log}" 2>&1 &
  local pid=$!
  local start_ts
  start_ts="$(date +%s)"
  local ec=0

  while kill -0 "${pid}" 2>/dev/null; do
    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout_sec )); then
      kill "${pid}" 2>/dev/null || true
      sleep "${kill_sec}"
      kill -9 "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      set -e
      return 124
    fi
    sleep 1
  done

  wait "${pid}" || ec=$?
  set -e
  return "${ec}"
}

extract_metric() {
  local pattern="$1"
  local log_file="$2"
  local value
  value="$(
    sed -n -E "s/.*${pattern}[[:space:]]*([0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?).*/\\1/p" "${log_file}" 2>/dev/null | tail -1 || true
  )"
  printf '%s' "${value}"
}

declare -a kernels=()
while IFS= read -r file; do
  kernels+=("${file}")
done < <(
  find "${input_dirs[@]}" -maxdepth 1 -type f -name '*.co' 2>/dev/null | sort
)

if [[ -n "${pattern}" ]]; then
  declare -a filtered=()
  for file in "${kernels[@]}"; do
    if [[ "${file}" =~ ${pattern} ]]; then
      filtered+=("${file}")
    fi
  done
  kernels=("${filtered[@]}")
fi

if [[ -n "${limit}" ]] && (( ${#kernels[@]} > limit )); then
  kernels=("${kernels[@]:0:limit}")
fi

if [[ ${#kernels[@]} -eq 0 ]]; then
  echo "error: no kernel files matched" >&2
  exit 1
fi

{
  printf 'commit_time,commit_id,gpu_index,folder,kernel,arch,compile_status,run_status,verify_passed,timing_avg_ms,tflops,status,compile_log,run_log,notes\n'
} >"${output_csv}"

echo "Selected GPU: ${selected_gpu}"
echo "Kernel count: ${#kernels[@]}"
echo "Result CSV: ${output_csv}"

for kernel_path in "${kernels[@]}"; do
  rel_path="${kernel_path#${REPO_ROOT}/}"
  folder_name="$(basename "$(dirname "${kernel_path}")")"
  kernel_name="$(basename "${kernel_path}")"
  kernel_stem="${rel_path%.co}"
  safe_stem="${kernel_stem//\//__}"
  output_bin="${bin_root}/${safe_stem}"
  compile_log="${log_root}/${safe_stem}.compile.log"
  run_log="${log_root}/${safe_stem}.run.log"
  arch="$(infer_arch "${kernel_path}")"
  compile_status="ok"
  run_status="not_run"
  verify_passed="0"
  timing_avg_ms=""
  tflops=""
  status="ok"
  notes=""

  mkdir -p "$(dirname "${output_bin}")" "$(dirname "${compile_log}")" "$(dirname "${run_log}")"

  compile_cmd=("${CHOREO_BIN}" "${kernel_path}" -o "${output_bin}")
  if [[ -n "${arch}" ]]; then
    compile_cmd+=(-t cute "-arch=${arch}")
  fi
  append_flag_if_needed "${kernel_path}" "--use-warpspec" "warpspec" compile_cmd
  append_flag_if_needed "${kernel_path}" "--use-prepack" "prepack" compile_cmd

  echo "==> [compile] ${rel_path}"
  printf '    '
  printf '%q ' "${compile_cmd[@]}"
  printf '\n'

  set +e
  "${compile_cmd[@]}" >"${compile_log}" 2>&1
  compile_ec=$?
  set -e

  if [[ ${compile_ec} -ne 0 ]]; then
    compile_status="fail(${compile_ec})"
    run_status="skipped"
    status="compile_failed"
    notes="compile failed"
  elif [[ ! -e "${output_bin}" ]]; then
    compile_status="fail(no_output)"
    run_status="skipped"
    status="compile_failed"
    notes="compile output missing"
  elif [[ ! -x "${output_bin}" ]]; then
    chmod +x "${output_bin}" 2>/dev/null || true
    if [[ ! -x "${output_bin}" ]]; then
      compile_status="fail(no_executable)"
      run_status="skipped"
      status="compile_failed"
      notes="compile output not executable"
    fi
  fi

  if [[ "${status}" == "ok" ]]; then
    echo "==> [run] ${rel_path} on GPU ${selected_gpu}"
    if run_with_timeout "${run_log}" env CUDA_VISIBLE_DEVICES="${selected_gpu}" "${output_bin}"; then
      run_ec=0
    else
      run_ec=$?
    fi

    if [[ ${run_ec} -eq 0 ]]; then
      run_status="ok"
    elif [[ ${run_ec} -eq 124 || ${run_ec} -eq 137 ]]; then
      run_status="timeout(${run_ec})"
      status="timeout"
      notes="possible deadlock or long-running kernel"
    else
      run_status="fail(${run_ec})"
      status="run_failed"
      notes="runtime failed"
    fi

    if grep -q "Test Passed" "${run_log}" 2>/dev/null; then
      verify_passed="1"
    fi
    timing_avg_ms="$(extract_metric 'Timing avg ms:' "${run_log}")"
    tflops="$(extract_metric 'TFLOPS:' "${run_log}")"

    if [[ "${run_status}" == "ok" && "${verify_passed}" != "1" ]]; then
      status="verify_failed"
      notes="missing Test Passed marker"
    fi
  fi

  {
    csv_escape "${commit_stamp}"
    printf ','
    csv_escape "${commit_short}"
    printf ','
    csv_escape "${selected_gpu}"
    printf ','
    csv_escape "${folder_name}"
    printf ','
    csv_escape "${kernel_name}"
    printf ','
    csv_escape "${arch}"
    printf ','
    csv_escape "${compile_status}"
    printf ','
    csv_escape "${run_status}"
    printf ','
    csv_escape "${verify_passed}"
    printf ','
    csv_escape "${timing_avg_ms}"
    printf ','
    csv_escape "${tflops}"
    printf ','
    csv_escape "${status}"
    printf ','
    csv_escape "${compile_log}"
    printf ','
    csv_escape "${run_log}"
    printf ','
    csv_escape "${notes}"
    printf '\n'
  } >>"${output_csv}"
done

echo "Batch run finished. CSV written to ${output_csv}"
