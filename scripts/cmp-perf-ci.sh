#!/bin/bash
# cmp-perf-ci.sh -- compile and execute benchmarks for choreo+topscc vs coir+topscc
#
# Subcommands:
#   compile   Compile all .co files with both choreo and cocc (emit-source only)
#   execute   Compile and execute all .co files on GCU hardware
#
# Options:
#   --filter <pattern>   Only process files whose relative path matches pattern
#   --jobs <N>           Run N compilations in parallel (default: 1)
#
# Usage:
#   bash scripts/cmp-perf-ci.sh compile
#   bash scripts/cmp-perf-ci.sh compile --filter matmul
#   bash scripts/cmp-perf-ci.sh execute
#   bash scripts/cmp-perf-ci.sh execute --filter softmax --jobs 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$PROJECT_ROOT/benchmark/cmp-perf"
BUILD_DIR="$PROJECT_ROOT/build"

# Tool paths
CHOREO="${CHOREO:-$BUILD_DIR/choreo}"
COCC="${COCC:-$BUILD_DIR/tools/coir/cocc}"

# Options
FILTER=""
JOBS=1
SUBCMD=""

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[cmp-perf] $*"; }

find_co_files() {
  local dir="$1"
  find "$dir" -name '*.co' -type f | sort
}

apply_filter() {
  local pat="$1"
  if [[ -z "$pat" ]]; then
    cat
  else
    grep -F "$pat" || true
  fi
}

tool_ok() {
  local tool="$1"
  if [[ ! -x "$tool" ]]; then
    die "tool not found or not executable: $tool"
  fi
}

# ---------------------------------------------------------------------------
# compile subcommand
# ---------------------------------------------------------------------------

cmd_compile() {
  info "Starting compile check (emit-source only)..."
  info "  choreo: $CHOREO"
  info "  cocc:   $COCC"
  tool_ok "$CHOREO"
  tool_ok "$COCC"

  local files
  files=$(find_co_files "$BENCH_DIR" | apply_filter "$FILTER")
  if [[ -z "$files" ]]; then
    die "no .co files found under $BENCH_DIR (filter='$FILTER')"
  fi

  local total=0 choreo_fail=0 coir_fail=0

  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    local rel="${f#$BENCH_DIR/}"
    total=$((total + 1))

    # choreo compile
    if ! "$CHOREO" -t topscc -es "$f" >/dev/null 2>&1; then
      echo "  FAIL [choreo] $rel" >&2
      choreo_fail=$((choreo_fail + 1))
    fi

    # coir compile
    if ! "$COCC" -t topscc -es "$f" >/dev/null 2>&1; then
      echo "  FAIL [coir]   $rel" >&2
      coir_fail=$((coir_fail + 1))
    fi
  done <<< "$files"

  echo ""
  info "Compile summary:"
  echo "  total:          $total"
  echo "  choreo passed:  $((total - choreo_fail))"
  echo "  choreo FAILED:  $choreo_fail"
  echo "  coir passed:    $((total - coir_fail))"
  echo "  coir FAILED:    $coir_fail"

  if (( choreo_fail > 0 || coir_fail > 0 )); then
    echo ""
    die "compile check FAILED"
  fi

  info "compile check PASSED"
}

# ---------------------------------------------------------------------------
# execute subcommand
# ---------------------------------------------------------------------------

cmd_execute() {
  info "Starting execution check..."
  tool_ok "$CHOREO"
  tool_ok "$COCC"

  local files
  files=$(find_co_files "$BENCH_DIR" | apply_filter "$FILTER")
  if [[ -z "$files" ]]; then
    die "no .co files found under $BENCH_DIR (filter='$FILTER')"
  fi

  local tmpdir
  tmpdir=$(mktemp -d -t cmp-perf-exec.XXXXXX)
  trap "rm -rf $tmpdir" EXIT

  local total=0 choreo_fail=0 coir_fail=0

  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    local rel="${f#$BENCH_DIR/}"
    local base
    base=$(basename "$f" .co)
    total=$((total + 1))

    # choreo: generate script, then execute
    local choreo_script="$tmpdir/${base}.choreo.sh"
    if "$CHOREO" -t topscc -gs "$f" -o "$choreo_script" >/dev/null 2>&1; then
      if ! bash "$choreo_script" --execute >/dev/null 2>&1; then
        echo "  FAIL [choreo-exec] $rel" >&2
        choreo_fail=$((choreo_fail + 1))
      fi
    else
      echo "  FAIL [choreo-compile] $rel" >&2
      choreo_fail=$((choreo_fail + 1))
    fi

    # coir: pipe script to bash
    if ! "$COCC" -t topscc -gs "$f" 2>/dev/null | bash -s -- --execute >/dev/null 2>&1; then
      echo "  FAIL [coir]   $rel" >&2
      coir_fail=$((coir_fail + 1))
    fi

    rm -f "$choreo_script"
  done <<< "$files"

  echo ""
  info "Execute summary:"
  echo "  total:          $total"
  echo "  choreo passed:  $((total - choreo_fail))"
  echo "  choreo FAILED:  $choreo_fail"
  echo "  coir passed:    $((total - coir_fail))"
  echo "  coir FAILED:    $coir_fail"

  if (( choreo_fail > 0 || coir_fail > 0 )); then
    echo ""
    die "execute check FAILED"
  fi

  info "execute check PASSED"
}

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 <subcommand> [options]

Subcommands:
  compile   Compile all .co files with choreo and cocc (emit-source only)
  execute   Compile and execute all .co files on GCU hardware

Options:
  --filter <pattern>  Only process files whose path contains pattern
  --jobs <N>          Number of parallel jobs (default: 1)
EOF
  exit 1
}

SUBCMD="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --filter)
      FILTER="${2:-}"; shift 2 ;;
    --jobs)
      JOBS="${2:-1}"; shift 2 ;;
    -h|--help)
      usage ;;
    *)
      die "unknown option: $1" ;;
  esac
done

case "$SUBCMD" in
  compile)  cmd_compile ;;
  execute)  cmd_execute ;;
  "")       usage ;;
  *)        die "unknown subcommand: $SUBCMD" ;;
esac
