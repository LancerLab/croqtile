#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  verify_choreo_kernel.sh <input.co> [output] [options] [-- extra compile args]

Options:
  --timeout-sec <n>      Runtime timeout in seconds (default: 120)
  --bench-timeout-sec <n>  Bench runtime timeout in seconds (default: 180)
  --run-arg <arg>        One runtime argument to append (repeatable)
  --no-disable-timing    Do not add '--disable-timing' to runtime args
  --no-bench             Skip post-verify bench run
  --no-auto-target       Disable auto '-t cute -arch=sm_90a' for inputs containing 'sm90'
  --no-debug-on-fail     Disable extra diagnostics when compile/run fails
  -h, --help             Show help

Environment:
  CHOREO_VERIFY_GPU          Bind run to a specific GPU id/list via CUDA_VISIBLE_DEVICES
  CHOREO_VERIFY_AUTO_GPU     Auto-select an idle GPU when CHOREO_VERIFY_GPU is unset (default: 1)
  CHOREO_VERIFY_GPU_MAX_USED_MB  Idle GPU threshold for used memory (default: 2048)
  CHOREO_VERIFY_GPU_MAX_UTIL     Idle GPU threshold for utilization percent (default: 20)
  CHOREO_VERIFY_OOM_GPU      Retry once on OOM using this GPU id/list (fallback)
  CHOREO_VERIFY_TIMEOUT_KILL_SEC  Grace period before force-kill after timeout (default: 10)
  CHOREO_VERIFY_USE_SMALL    Compile with smaller MATMUL_DEFAULT_M/N/K for faster verify (default: 1)
  CHOREO_VERIFY_SMALL_MNK    Small verify shape as M,N,K (default: 256,256,256)
  CHOREO_VERIFY_RUN_BENCH    Run post-verify bench (default: 1)
  CHOREO_VERIFY_BENCH_MNK    Bench shape as M,N,K (default: 2048,2048,2048)
  CHOREO_VERIFY_BENCH_TIMEOUT_SEC  Bench timeout seconds override
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

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

input=""
output=""
timeout_sec=120
bench_timeout_sec=180
disable_timing=1
run_bench=1
auto_target=1
debug_on_fail=1
run_args=()
compile_extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --timeout-sec)
      timeout_sec="${2:-}"
      shift 2
      ;;
    --bench-timeout-sec)
      bench_timeout_sec="${2:-}"
      shift 2
      ;;
    --run-arg)
      run_args+=("${2:-}")
      shift 2
      ;;
    --no-disable-timing)
      disable_timing=0
      shift
      ;;
    --no-bench)
      run_bench=0
      shift
      ;;
    --no-auto-target)
      auto_target=0
      shift
      ;;
    --no-debug-on-fail)
      debug_on_fail=0
      shift
      ;;
    --)
      shift
      compile_extra=("$@")
      break
      ;;
    -*)
      echo "error: unknown option '$1'" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -z "${input}" ]]; then
        input="$1"
      elif [[ -z "${output}" ]]; then
        output="$1"
      else
        echo "error: unexpected argument '$1'" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -n "${CHOREO_VERIFY_RUN_BENCH:-}" ]]; then
  run_bench="${CHOREO_VERIFY_RUN_BENCH}"
fi
if [[ -n "${CHOREO_VERIFY_BENCH_TIMEOUT_SEC:-}" ]]; then
  bench_timeout_sec="${CHOREO_VERIFY_BENCH_TIMEOUT_SEC}"
fi

if [[ -z "${input}" ]]; then
  echo "error: missing input .co file" >&2
  usage
  exit 1
fi

if [[ "${input##*.}" != "co" ]]; then
  echo "error: input file must end with .co: ${input}" >&2
  exit 1
fi

input_path="${input}"
if [[ ! -f "${input_path}" ]]; then
  candidate="${REPO_ROOT}/${input}"
  if [[ -f "${candidate}" ]]; then
    input_path="${candidate}"
  fi
fi

if [[ ! -f "${input_path}" ]]; then
  echo "error: input .co file not found: ${input}" >&2
  exit 1
fi

if [[ -z "${output}" ]]; then
  output="${REPO_ROOT}/build/$(basename "${input%.co}")_verify"
fi

if [[ "${output}" != /* ]]; then
  output="${PWD}/${output}"
fi

mkdir -p "$(dirname "${output}")"

log_dir="${REPO_ROOT}/build/skill-logs/verify-choreo-kernel"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"
compile_log="${log_dir}/compile_${stamp}.log"
run_log="${log_dir}/run_${stamp}.log"
compile_bench_log="${log_dir}/compile_bench_${stamp}.log"
run_bench_log="${log_dir}/run_bench_${stamp}.log"
infer_log="${log_dir}/infer_${stamp}.log"
env_log="${log_dir}/env_${stamp}.log"

prepare_input_with_mnk() {
  local src="$1"
  local tag="$2"
  local mnk="$3"
  local out="${src}"
  local m=""
  local n=""
  local k=""
  IFS=',' read -r m n k <<< "${mnk}"
  if [[ ! "${m}" =~ ^[0-9]+$ || ! "${n}" =~ ^[0-9]+$ || ! "${k}" =~ ^[0-9]+$ ]]; then
    echo "warning: invalid ${tag} shape '${mnk}', expected M,N,K integers; skip override" >&2
    printf '%s\n' "${out}"
    return 0
  fi
  if grep -q 'MATMUL_DEFAULT_[MNK]' "${src}"; then
    out="${log_dir}/$(basename "${src%.co}")_${tag}_${stamp}.co"
    cp "${src}" "${out}"
    sed -E -i "s@^#define MATMUL_DEFAULT_M[[:space:]]+[0-9]+@#define MATMUL_DEFAULT_M ${m}@" "${out}" || true
    sed -E -i "s@^#define MATMUL_DEFAULT_N[[:space:]]+[0-9]+@#define MATMUL_DEFAULT_N ${n}@" "${out}" || true
    sed -E -i "s@^#define MATMUL_DEFAULT_K[[:space:]]+[0-9]+@#define MATMUL_DEFAULT_K ${k}@" "${out}" || true
    echo "${tag} enabled: M=${m},N=${n},K=${k}" >&2
  fi
  printf '%s\n' "${out}"
}

append_flag_if_needed() {
  local needle="$1"
  local flag="$2"
  local present=0
  if [[ "${input_path}" != *"${needle}"* ]]; then
    return 0
  fi
  for arg in "${compile_cmd[@]}"; do
    if [[ "${arg}" == "${flag}" ]]; then
      present=1
      break
    fi
  done
  if [[ ${present} -eq 0 ]]; then
    compile_cmd+=("${flag}")
    echo "Detected ${needle} kernel; append compile flag ${flag}"
  fi
}

compile_input_path="${input_path}"
use_small_verify="${CHOREO_VERIFY_USE_SMALL:-1}"
small_mnk="${CHOREO_VERIFY_SMALL_MNK:-256,256,256}"
if [[ "${use_small_verify}" == "1" ]]; then
  compile_input_path="$(prepare_input_with_mnk "${input_path}" "small" "${small_mnk}")"
fi
compile_cmd=("${CHOREO_BIN}" "${compile_input_path}" -o "${output}")
if [[ ${#compile_extra[@]} -gt 0 ]]; then
  compile_cmd+=("${compile_extra[@]}")
elif [[ "${auto_target}" -eq 1 && "${input_path}" == *"sm90"* ]]; then
  compile_cmd+=(-t cute -arch=sm_90a)
fi
append_flag_if_needed "prepack" "--use-prepack"

echo "== Compile =="
printf 'Running:'
printf ' %q' "${compile_cmd[@]}"
printf '\n'

set +e
"${compile_cmd[@]}" >"${compile_log}" 2>&1
compile_ec=$?
set -e
cat "${compile_log}"

if [[ ${compile_ec} -ne 0 ]]; then
  echo "Compile failed (exit ${compile_ec}). Log: ${compile_log}" >&2
  if [[ "${debug_on_fail}" -eq 1 ]]; then
    echo "== Debug (compile) =="
    if grep -Eq "unsupported architecture|Compile Target '.*' is invalid|group-4 level is not supported" "${compile_log}"; then
      echo "Hint: check target/arch pair. For SM90 kernels, try '-t cute -arch=sm_90a'."
      "${CHOREO_BIN}" --help-target || true
    fi
    set +e
    "${CHOREO_BIN}" -i "${input_path}" >"${infer_log}" 2>&1
    infer_ec=$?
    set -e
    echo "Infer diagnostics exit: ${infer_ec}. Log: ${infer_log}"
    sed -n '1,120p' "${infer_log}" || true
  fi
  exit ${compile_ec}
fi

if [[ ! -x "${output}" ]]; then
  chmod +x "${output}" || true
fi

if [[ ! -x "${output}" ]]; then
  echo "error: compile output is not executable: ${output}" >&2
  exit 1
fi

runtime_env=(
  "CHOREO_DISABLE_TIMING=1"
  "CHOREO_TIMING_WARMUP=0"
  "CHOREO_TIMING_REPEAT=1"
)

trim_field() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "${x}"
}

pick_gpu_by_free_mem() {
  local current="${1:-}"
  local max_used="${CHOREO_VERIFY_GPU_MAX_USED_MB:-2048}"
  local max_util="${CHOREO_VERIFY_GPU_MAX_UTIL:-20}"
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

configured_gpu="${CHOREO_VERIFY_GPU:-}"
auto_gpu="${CHOREO_VERIFY_AUTO_GPU:-1}"
oom_fallback_gpu="${CHOREO_VERIFY_OOM_GPU:-}"
selected_gpu="${configured_gpu}"
if [[ -z "${selected_gpu}" && "${auto_gpu}" == "1" ]]; then
  if selected_gpu="$(pick_gpu_by_free_mem)"; then
    echo "Auto-selected GPU: ${selected_gpu}"
  else
    echo "warning: auto GPU selection failed; run without CUDA_VISIBLE_DEVICES pinning" >&2
    selected_gpu=""
  fi
fi
if [[ -n "${selected_gpu}" ]]; then
  runtime_env+=("CUDA_VISIBLE_DEVICES=${selected_gpu}")
fi
if [[ -z "${oom_fallback_gpu}" && -n "${selected_gpu}" ]]; then
  oom_fallback_gpu="$(pick_gpu_by_free_mem "${selected_gpu}" || true)"
fi

run_cmd=("${output}")
if [[ "${disable_timing}" -eq 1 ]]; then
  run_cmd+=(--disable-timing)
fi
if [[ ${#run_args[@]} -gt 0 ]]; then
  run_cmd+=("${run_args[@]}")
fi

echo "== Run & Verify =="
printf 'Running:'
printf ' %q' "${run_cmd[@]}"
printf '\n'

run_once() {
  local out_log="$1"
  local timeout_kill_sec="${CHOREO_VERIFY_TIMEOUT_KILL_SEC:-10}"
  set +e
  if command -v timeout >/dev/null 2>&1; then
    env "${runtime_env[@]}" timeout -k "${timeout_kill_sec}" "${timeout_sec}" "${run_cmd[@]}" >"${out_log}" 2>&1
    run_ec=$?
  else
    env "${runtime_env[@]}" "${run_cmd[@]}" >"${out_log}" 2>&1
    run_ec=$?
  fi
  set -e
}

run_once "${run_log}"
cat "${run_log}"

is_oom=0
if grep -Eiq "out of memory|cudaErrorMemoryAllocation|memory allocation" "${run_log}"; then
  is_oom=1
fi

if [[ ${run_ec} -ne 0 && ${is_oom} -eq 1 && -n "${oom_fallback_gpu}" ]]; then
  current_visible=""
  for kv in "${runtime_env[@]}"; do
    if [[ "${kv}" == CUDA_VISIBLE_DEVICES=* ]]; then
      current_visible="${kv#CUDA_VISIBLE_DEVICES=}"
      break
    fi
  done

  if [[ "${current_visible}" != "${oom_fallback_gpu}" ]]; then
    echo "Detected OOM. Retrying with CUDA_VISIBLE_DEVICES=${oom_fallback_gpu} ..."
    updated_env=()
    replaced=0
    for kv in "${runtime_env[@]}"; do
      if [[ "${kv}" == CUDA_VISIBLE_DEVICES=* ]]; then
        updated_env+=("CUDA_VISIBLE_DEVICES=${oom_fallback_gpu}")
        replaced=1
      else
        updated_env+=("${kv}")
      fi
    done
    if [[ ${replaced} -eq 0 ]]; then
      updated_env+=("CUDA_VISIBLE_DEVICES=${oom_fallback_gpu}")
    fi
    runtime_env=("${updated_env[@]}")
    run_once "${run_log}"
    cat "${run_log}"
  fi
fi

verify_passed=0
if [[ ${run_ec} -eq 0 ]] && grep -q "Test Passed" "${run_log}"; then
  verify_passed=1
fi
if [[ ${verify_passed} -ne 1 ]]; then
  echo "Verification failed (exit ${run_ec}). Run log: ${run_log}" >&2
  if [[ "${debug_on_fail}" -eq 1 ]]; then
    echo "== Debug (runtime) =="
    if [[ ${run_ec} -eq 124 || ${run_ec} -eq 137 ]]; then
      echo "Detected runtime timeout (possible deadlock)."
      echo "Hint: reduce problem size or increase --timeout-sec / CHOREO_VERIFY_TIMEOUT_KILL_SEC."
    fi
    if grep -Eq "CUDA failure|cudaError|no CUDA-capable device|operation not supported" "${run_log}"; then
      echo "Detected CUDA runtime failure. Collecting environment info..."
      {
        echo "Date: $(date)"
        echo "Host: $(hostname)"
        echo "nvidia-smi:"
        nvidia-smi || true
      } >"${env_log}" 2>&1
      sed -n '1,120p' "${env_log}" || true
      echo "Env log: ${env_log}"
    fi

    if grep -q "values are not equal" "${run_log}"; then
      echo "Detected correctness mismatch. Suggested next steps:"
      echo "1) Re-run with --run-arg --skip-verify to isolate runtime stability."
      echo "2) Compare generated source using: ${CHOREO_BIN} -es ${input_path} -o -"
      echo "3) Use compile diagnostics: ${CHOREO_BIN} -i ${input_path}"
    fi
  fi
  exit "${run_ec}"
fi

echo "Verification succeeded. Binary: ${output}"
echo "Verify logs: ${compile_log}, ${run_log}"

if [[ "${run_bench}" != "1" ]]; then
  exit 0
fi

bench_input_path="${input_path}"
bench_mnk="${CHOREO_VERIFY_BENCH_MNK:-2048,2048,2048}"
bench_input_path="$(prepare_input_with_mnk "${input_path}" "bench" "${bench_mnk}")"

bench_output="${output}_bench"
compile_cmd=("${CHOREO_BIN}" "${bench_input_path}" -o "${bench_output}")
if [[ ${#compile_extra[@]} -gt 0 ]]; then
  compile_cmd+=("${compile_extra[@]}")
elif [[ "${auto_target}" -eq 1 && "${input_path}" == *"sm90"* ]]; then
  compile_cmd+=(-t cute -arch=sm_90a)
fi
append_flag_if_needed "prepack" "--use-prepack"

echo "== Compile (bench) =="
printf 'Running:'
printf ' %q' "${compile_cmd[@]}"
printf '\n'
set +e
"${compile_cmd[@]}" >"${compile_bench_log}" 2>&1
compile_ec=$?
set -e
cat "${compile_bench_log}"
if [[ ${compile_ec} -ne 0 ]]; then
  echo "Bench compile failed (exit ${compile_ec}). Log: ${compile_bench_log}" >&2
  exit ${compile_ec}
fi
if [[ ! -x "${bench_output}" ]]; then
  chmod +x "${bench_output}" || true
fi
if [[ ! -x "${bench_output}" ]]; then
  echo "error: bench compile output is not executable: ${bench_output}" >&2
  exit 1
fi

bench_runtime_env=()
for kv in "${runtime_env[@]}"; do
  case "${kv}" in
    CHOREO_DISABLE_TIMING=*|CHOREO_TIMING_WARMUP=*|CHOREO_TIMING_REPEAT=*) ;;
    *) bench_runtime_env+=("${kv}") ;;
  esac
done
bench_runtime_env+=("CHOREO_DISABLE_TIMING=0")

runtime_env=("${bench_runtime_env[@]}")
run_cmd=("${bench_output}" "--skip-verify")
timeout_sec="${bench_timeout_sec}"

echo "== Run (bench) =="
printf 'Running:'
printf ' %q' "${run_cmd[@]}"
printf '\n'
run_once "${run_bench_log}"
cat "${run_bench_log}"
if [[ ${run_ec} -ne 0 ]]; then
  echo "Bench run failed (exit ${run_ec}). Log: ${run_bench_log}" >&2
  if [[ ${run_ec} -eq 124 || ${run_ec} -eq 137 ]]; then
    echo "Bench timeout hit. Increase --bench-timeout-sec or reduce CHOREO_VERIFY_BENCH_MNK."
  fi
  exit "${run_ec}"
fi

echo "Bench succeeded. Bench logs: ${compile_bench_log}, ${run_bench_log}"
exit 0
