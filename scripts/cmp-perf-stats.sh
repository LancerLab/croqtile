#!/bin/bash
# cmp-perf-stats.sh -- collect and compare --stats output for
# choreo+topscc vs coir+topscc across all cmp-perf benchmarks.
#
# Subcommands:
#   stats-collect   Run --stats on every .co file for both paths, save JSON
#   stats-compare   Compare choreo vs coir stats; detect regressions
#
# Options:
#   --filter <pattern>   Only process files whose path matches pattern
#   --output <file>      Write output to file (default: stdout)
#   --baseline <file>    Compare against a previous stats JSON for CI gate
#
# Usage:
#   bash scripts/cmp-perf-stats.sh stats-collect --output stats.json
#   bash scripts/cmp-perf-stats.sh stats-compare --baseline prev.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$PROJECT_ROOT/benchmark/cmp-perf"
BUILD_DIR="$PROJECT_ROOT/build"
AWK_PARSER="$SCRIPT_DIR/cmp-perf-parse-stats.awk"

CHOREO="${CHOREO:-$BUILD_DIR/choreo}"
COCC="${COCC:-$BUILD_DIR/tools/coir/cocc}"

FILTER=""
OUTPUT=""
BASELINE=""
SUBCMD=""

# field names in order (must match awk parser output)
FIELDS=(
  total static_true static_false runtime_total
  runtime_entry runtime_low runtime_medium runtime_high
  runtime_enabled runtime_disabled
  unclassified_total shape_compat_total elem_access_total
  loop_bound_total hw_constraint_total
  unclassified_runtime shape_compat_runtime elem_access_runtime
  loop_bound_runtime hw_constraint_runtime
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[cmp-perf-stats] $*" >&2; }

find_co_files() {
  find "$BENCH_DIR" -name '*.co' -type f | sort
}

apply_filter() {
  local pat="$1"
  if [[ -z "$pat" ]]; then cat; else grep -F "$pat" || true; fi
}

# Parse a single stats run (stdin) into a line of tab-separated values
# Output: tab-separated list of values in FIELDS order
parse_stats() {
  awk -f "$AWK_PARSER" | {
    local result=""
    for field in "${FIELDS[@]}"; do
      local val
      val=$(grep "^${field}=" 2>/dev/null | head -1 | cut -d= -f2)
      val="${val:-0}"
      result="${result}${val}\t"
    done
    printf "%s" "${result%\\t}"  # trim trailing tab
  }
}

# Read a tab-separated stats line and print as JSON object
# Usage: print_json_obj "prefix" < values.tsv
print_json_obj() {
  local prefix="$1"
  local vals
  IFS=$'\t' read -r -a vals
  local i=0
  printf '  "%s":{"total":%d,"static_true":%d,"static_false":%d,' \
    "$prefix" "${vals[0]}" "${vals[1]}" "${vals[2]}"
  printf '"runtime_total":%d,"runtime_entry":%d,"runtime_low":%d,' \
    "${vals[3]}" "${vals[4]}" "${vals[5]}"
  printf '"runtime_medium":%d,"runtime_high":%d,' \
    "${vals[6]}" "${vals[7]}"
  printf '"runtime_enabled":%d,"runtime_disabled":%d,' \
    "${vals[8]}" "${vals[9]}"
  printf '"unclassified_total":%d,"shape_compat_total":%d,' \
    "${vals[10]}" "${vals[11]}"
  printf '"elem_access_total":%d,"loop_bound_total":%d,' \
    "${vals[12]}" "${vals[13]}"
  printf '"hw_constraint_total":%d,' \
    "${vals[14]}"
  printf '"unclassified_runtime":%d,"shape_compat_runtime":%d,' \
    "${vals[15]}" "${vals[16]}"
  printf '"elem_access_runtime":%d,"loop_bound_runtime":%d,' \
    "${vals[17]}" "${vals[18]}"
  printf '"hw_constraint_runtime":%d' \
    "${vals[19]}"
  printf '}'
}

# ---------------------------------------------------------------------------
# stats-collect
# ---------------------------------------------------------------------------

cmd_collect() {
  info "Collecting --stats for all benchmarks..."

  local files
  files=$(find_co_files "$BENCH_DIR" | apply_filter "$FILTER")
  if [[ -z "$files" ]]; then
    die "no .co files found (filter='$FILTER')"
  fi

  local tmpdir
  tmpdir=$(mktemp -d -t cmp-perf-stats.XXXXXX)
  trap "rm -rf $tmpdir" EXIT

  local data_file="$tmpdir/data.jsonl"
  local total=0 choreo_miss=0 coir_miss=0

  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    local rel="${f#$BENCH_DIR/}"
    total=$((total + 1))

    # choreo stats
    local choreo_line=""
    if "$CHOREO" -t topscc -es --stats "$f" 2>&1 | parse_stats > "$tmpdir/choreo.tsv" 2>/dev/null; then
      choreo_line=$(cat "$tmpdir/choreo.tsv")
    fi
    if [[ -z "$choreo_line" ]]; then
      choreo_miss=$((choreo_miss + 1))
    fi

    # coir stats
    local coir_line=""
    if "$COCC" -t topscc -es --stats "$f" 2>&1 | parse_stats > "$tmpdir/coir.tsv" 2>/dev/null; then
      coir_line=$(cat "$tmpdir/coir.tsv")
    fi
    if [[ -z "$coir_line" ]]; then
      coir_miss=$((coir_miss + 1))
    fi

    # Write JSON
    printf '{"file":"%s",\n' "$rel"
    echo -e "$choreo_line" | print_json_obj "choreo"
    printf ',\n'
    echo -e "$coir_line" | print_json_obj "coir"
    printf '}\n'
  done <<< "$files" > "$data_file"

  info "Stats collection done: $total files ($choreo_miss choreo misses, $coir_miss coir misses)"
  cat "$data_file"
}

# ---------------------------------------------------------------------------
# stats-compare
# ---------------------------------------------------------------------------

cmd_compare() {
  info "Comparing choreo vs coir stats..."

  # Read JSON lines from stdin or a file
  local data_file="${1:-/dev/stdin}"
  if [[ ! -f "$data_file" && "$data_file" != "/dev/stdin" ]]; then
    die "data file not found: $data_file"
  fi

  local total=0
  local static_fail=0
  local runtime_warn=0
  local regressions=()

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue

    # Extract fields using simple pattern matching
    local file
    file=$(echo "$line" | sed -n 's/.*"file":"\([^"]*\)".*/\1/p')
    [[ -z "$file" ]] && continue

    local c_static_true co_static_true
    c_static_true=$(echo "$line" | sed -n 's/.*"choreo":{.*"static_true":\([0-9]*\).*/\1/p')
    co_static_true=$(echo "$line" | sed -n 's/.*"coir":{.*"static_true":\([0-9]*\).*/\1/p')

    local c_runtime co_runtime
    c_runtime=$(echo "$line" | sed -n 's/.*"choreo":{.*"runtime_total":\([0-9]*\).*/\1/p')
    co_runtime=$(echo "$line" | sed -n 's/.*"coir":{.*"runtime_total":\([0-9]*\).*/\1/p')

    total=$((total + 1))

    local d_static=$((co_static_true - c_static_true))
    local d_runtime=$((co_runtime - c_runtime))

    # Build result line
    local result="OK"
    if (( d_static < 0 )); then
      result="FAIL:static_true"
      static_fail=$((static_fail + 1))
      regressions+=("$result $file (choreo=$c_static_true coir=$co_static_true d=$d_static)")
    fi
    if (( d_runtime > 0 )); then
      if [[ "$result" == "OK" ]]; then result="WARN:runtime"; fi
      runtime_warn=$((runtime_warn + 1))
      regressions+=("$result $file (choreo=$c_runtime coir=$co_runtime d=$d_runtime) [runtime]")
    fi

    printf "%-60s  st=%3d->%-3d(%+4d)  rt=%3d->%-3d(%+4d)  %s\n" \
      "$file" "$c_static_true" "$co_static_true" "$d_static" \
      "$c_runtime" "$co_runtime" "$d_runtime" "$result"
  done < "$data_file"

  # Summary
  echo ""
  local sep="============================================================"
  echo "$sep"
  echo "  COMPARE SUMMARY"
  echo "$sep"
  echo "  total files:              $total"
  echo "  static_true regressions:  $static_fail  (FAIL)"
  echo "  runtime increases:        $runtime_warn  (WARN)"
  echo "$sep"

  if (( ${#regressions[@]} > 0 )); then
    echo ""
    echo "  Regressions:"
    for r in "${regressions[@]}"; do
      echo "    $r"
    done
    echo ""
  fi

  if (( static_fail > 0 )); then
    die "stats comparison FAILED: $static_fail static_true regressions"
  elif (( runtime_warn > 0 )); then
    info "stats comparison: $runtime_warn runtime increases (WARN only, not failing)"
  else
    info "stats comparison: CLEAN (no regressions)"
  fi
}

# ---------------------------------------------------------------------------
# baseline comparison
# ---------------------------------------------------------------------------

cmd_baseline() {
  local baseline_file="$1"

  if [[ ! -f "$baseline_file" ]]; then
    info "no baseline file found at $baseline_file, skipping comparison"
    exit 0
  fi

  info "Comparing current stats against baseline: $baseline_file"
  # For baseline comparison, we compare per-file stats from the collected
  # data (stdin) against the baseline file.
  # Both are JSONL with one object per line.

  local tmpdir
  tmpdir=$(mktemp -d -t cmp-perf-baseline.XXXXXX)
  trap "rm -rf $tmpdir" EXIT

  # Extract per-file static_true and runtime counts from current data
  cat > "$tmpdir/current.txt"
  local baseline_text
  baseline_text=$(cat "$baseline_file")

  local total=0 static_fail=0 runtime_warn=0

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local file
    file=$(echo "$line" | sed -n 's/.*"file":"\([^"]*\)".*/\1/p')
    [[ -z "$file" ]] && continue

    # Find this file in baseline
    local bline
    bline=$(echo "$baseline_text" | grep -F "\"file\":\"$file\"" | head -1)
    if [[ -z "$bline" ]]; then
      echo "  NEW: $file (no baseline)"
      continue
    fi

    local cur_st cur_rt bl_st bl_rt
    cur_st=$(echo "$line" | sed -n 's/.*"coir":{.*"static_true":\([0-9]*\).*/\1/p')
    cur_rt=$(echo "$line" | sed -n 's/.*"coir":{.*"runtime_total":\([0-9]*\).*/\1/p')
    bl_st=$(echo "$bline" | sed -n 's/.*"coir":{.*"static_true":\([0-9]*\).*/\1/p')
    bl_rt=$(echo "$bline" | sed -n 's/.*"coir":{.*"runtime_total":\([0-9]*\).*/\1/p')

    total=$((total + 1))

    local d_st=$((cur_st - bl_st))
    local d_rt=$((cur_rt - bl_rt))

    if (( d_st < 0 )); then
      echo "  FAIL: $file static_true ${bl_st}->${cur_st} (${d_st})"
      static_fail=$((static_fail + 1))
    fi
    if (( d_rt > 0 )); then
      echo "  WARN: $file runtime_total ${bl_rt}->${cur_rt} (+${d_rt})"
      runtime_warn=$((runtime_warn + 1))
    fi
  done < "$tmpdir/current.txt"

  echo ""
  echo "Baseline comparison: $total files, $static_fail FAILs, $runtime_warn WARNs"

  if (( static_fail > 0 )); then
    die "baseline regression: $static_fail FAILs"
  fi
}

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 <subcommand> [options]

Subcommands:
  stats-collect   Run --stats on every .co file for both paths
  stats-compare   Compare choreo vs coir stats
  stats-baseline  Compare against a previous stats JSONL file

Options:
  --filter <pattern>  Only process files whose path contains pattern
  --output <file>     Write output to file (stats-collect)
  --baseline <file>   Baseline JSONL file (stats-baseline, or
                      read from stdin for stats-compare)
EOF
  exit 1
}

SUBCMD="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --filter)   FILTER="${2:-}"; shift 2 ;;
    --output)   OUTPUT="${2:-}"; shift 2 ;;
    --baseline) BASELINE="${2:-}"; shift 2 ;;
    -h|--help)  usage ;;
    *)          die "unknown option: $1" ;;
  esac
done

case "$SUBCMD" in
  stats-collect)
    if [[ -n "$OUTPUT" ]]; then
      cmd_collect > "$OUTPUT"
      info "stats written to $OUTPUT"
    else
      cmd_collect
    fi
    ;;
  stats-compare)
    if [[ -n "$BASELINE" ]]; then
      cmd_baseline "$BASELINE"
    else
      cmd_compare /dev/stdin
    fi
    ;;
  stats-baseline)
    [[ -n "$BASELINE" ]] || die "--baseline <file> required"
    cmd_baseline "$BASELINE"
    ;;
  "") usage ;;
  *)  die "unknown subcommand: $SUBCMD" ;;
esac
