#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  compile_choreo_kernel.sh <input.co> [output] [-- extra choreo args]

Examples:
  compile_choreo_kernel.sh benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co
  compile_choreo_kernel.sh benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co my_kernel -- -g -t cute -arch=sm_90a
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

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

input="$1"
shift

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

output=""
if [[ $# -gt 0 && "${1}" != "--" ]]; then
  output="$1"
  shift
else
  output="$(basename "${input%.co}")"
fi

extra_args=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" != "--" ]]; then
    echo "error: use '--' before extra choreo args" >&2
    exit 1
  fi
  shift
  extra_args=("$@")
fi

cmd=("${CHOREO_BIN}" -c "${input_path}" -o "${output}")
if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
elif [[ "${input_path}" == *"sm90"* ]]; then
  # Match common repository naming and avoid defaulting to gcu300 for SM90 kernels.
  cmd+=(-t cute -arch=sm_90a)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

if [[ -e "${output}" ]]; then
  echo "Build succeeded: ${output}"
else
  echo "warning: build command exited successfully but output not found: ${output}" >&2
  exit 1
fi
