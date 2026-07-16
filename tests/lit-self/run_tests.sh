#!/usr/bin/env bash
# Self-tests for lit.sh filtering flags.
# Run with: bash tests/lit-self/run_tests.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIT_SH="${REPO_ROOT}/tests/lit.sh"
TEST_DIR="${SCRIPT_DIR}"

PASS=0; FAIL=0

_pass() { PASS=$((PASS + 1)); }
_fail() { FAIL=$((FAIL + 1)); }

check() {
  local desc="$1"; shift
  local expected="$1"; shift
  local actual
  actual="$("$@" 2>/dev/null || true)"
  if [[ "$actual" == "$expected" ]]; then
    echo "PASS: $desc"
    _pass
  else
    echo "FAIL: $desc"
    echo "  expected: '$expected'"
    echo "  got:      '$actual'"
    _fail
  fi
}

check_grep() {
  local desc="$1"; shift
  local pattern="$1"; shift
  local expected_count="$1"; shift
  local actual_count
  actual_count=$("$@" 2>&1 | grep -c "$pattern" || true)
  if [[ "$actual_count" == "$expected_count" ]]; then
    echo "PASS: $desc"
    _pass
  else
    echo "FAIL: $desc"
    echo "  expected $expected_count lines matching '$pattern'"
    echo "  got      $actual_count"
    _fail
  fi
}

echo "=== lit.sh self-tests ==="
echo ""

# -- Test 1: --sim=off (default) -----------------------------------------
echo "--- --sim=off ---"

check_grep "sim=off: sim_req.co skipped" \
  'SKIP(sim=off)' 1 \
  bash "$LIT_SH" --sim=off --dry-run "$TEST_DIR/sim_req.co"

check_grep "sim=off: no_req.co passes (no sim skip)" \
  'SKIP(sim=off)' 0 \
  bash "$LIT_SH" --sim=off --dry-run "$TEST_DIR/no_req.co"

check_grep "sim=off: no_req.co gets PASS" \
  'PASS:' 1 \
  bash "$LIT_SH" --sim=off --dry-run "$TEST_DIR/no_req.co"

check_grep "sim=off: nonsim_req.co passes (no sim skip)" \
  'SKIP(sim=off)' 0 \
  bash "$LIT_SH" --sim=off --dry-run "$TEST_DIR/nonsim_req.co"

# -- Test 2: --sim=only --------------------------------------------------
echo "--- --sim=only ---"

check_grep "sim=only: sim_req.co passes" \
  'PASS:' 1 \
  bash "$LIT_SH" --sim=only --dry-run "$TEST_DIR/sim_req.co"

check_grep "sim=only: no_req.co skipped" \
  'SKIP(sim=only)' 1 \
  bash "$LIT_SH" --sim=only --dry-run "$TEST_DIR/no_req.co"

check_grep "sim=only: nonsim_req.co skipped" \
  'SKIP(sim=only)' 1 \
  bash "$LIT_SH" --sim=only --dry-run "$TEST_DIR/nonsim_req.co"

# -- Test 3: --sim=on ----------------------------------------------------
echo "--- --sim=on ---"

check_grep "sim=on: sim_req.co passes" \
  'PASS:' 1 \
  bash "$LIT_SH" --sim=on --dry-run "$TEST_DIR/sim_req.co"

check_grep "sim=on: no_req.co passes" \
  'PASS:' 1 \
  bash "$LIT_SH" --sim=on --dry-run "$TEST_DIR/no_req.co"

check_grep "sim=on: nonsim_req.co passes" \
  'PASS:' 1 \
  bash "$LIT_SH" --sim=on --dry-run "$TEST_DIR/nonsim_req.co"

# -- Test 4: --run-only --------------------------------------------------
echo "--- --run-only ---"

check_grep "run-only=choreo: skips cocc line" \
  'SKIP(run-only=choreo)' 1 \
  bash "$LIT_SH" --sim=on --dry-run --run-only=choreo "$TEST_DIR/multi_bin.co"

check_grep "run-only=choreo: keeps choreo line" \
  "PASS:.*multi_bin" 1 \
  bash "$LIT_SH" --sim=on --dry-run --run-only=choreo "$TEST_DIR/multi_bin.co"

check_grep "run-only=cocc: skips choreo line" \
  'SKIP(run-only=cocc)' 1 \
  bash "$LIT_SH" --sim=on --dry-run --run-only=cocc "$TEST_DIR/multi_bin.co"

check_grep "run-only=cocc: keeps cocc line" \
  "PASS:.*multi_bin" 1 \
  bash "$LIT_SH" --sim=on --dry-run --run-only=cocc "$TEST_DIR/multi_bin.co"

# Test %choreo placeholder matching
check_grep "run-only=choreo: matches %choreo placeholder" \
  'SKIP(run-only=choreo)' 1 \
  bash "$LIT_SH" --sim=on --dry-run --run-only=choreo "$TEST_DIR/placeholder.co"

check_grep "run-only=choreo: keeps %choreo line" \
  "PASS:.*placeholder" 1 \
  bash "$LIT_SH" --sim=on --dry-run --run-only=choreo "$TEST_DIR/placeholder.co"

# -- Test 5: --target-only -----------------------------------------------
echo "--- --target-only ---"

check_grep "target-only=app1: skips app2 line" \
  'SKIP(target-only=app1)' 1 \
  bash "$LIT_SH" --sim=on --dry-run --target-only=app1 "$TEST_DIR/target.co"

check_grep "target-only=app1: keeps app1 line" \
  "PASS:.*target.co" 1 \
  bash "$LIT_SH" --sim=on --dry-run --target-only=app1 "$TEST_DIR/target.co"

check_grep "target-only=app2: skips app1 line" \
  'SKIP(target-only=app2)' 1 \
  bash "$LIT_SH" --sim=on --dry-run --target-only=app2 "$TEST_DIR/target.co"

check_grep "target-only=app2: keeps app2 line" \
  "PASS:.*target.co" 1 \
  bash "$LIT_SH" --sim=on --dry-run --target-only=app2 "$TEST_DIR/target.co"

# Comma-separated values
check_grep "run-only=choreo,cocc: keeps both lines" \
  'SKIP(run-only)' 0 \
  bash "$LIT_SH" --sim=on --dry-run --run-only=choreo,cocc "$TEST_DIR/multi_bin.co"

check_grep "target-only=app1,app2: keeps both lines" \
  'SKIP(target-only)' 0 \
  bash "$LIT_SH" --sim=on --dry-run --target-only=app1,app2 "$TEST_DIR/target.co"

check_grep "run-only=choreo,cocc: 2 PASS lines" \
  "PASS:.*multi_bin" 2 \
  bash "$LIT_SH" --sim=on --dry-run --run-only=choreo,cocc "$TEST_DIR/multi_bin.co"

# -- Test 6: --sim=bad rejects invalid values ----------------------------
echo "--- --sim validation ---"

check "sim=bad is rejected (exit code 1)" "1" \
  bash -c '"$@" >/dev/null 2>&1; echo $?' _ bash "$LIT_SH" --sim=bad --dry-run "$TEST_DIR/no_req.co"

check_grep "sim=bad prints error message" \
  'Error: --sim must be off, on, or only' 1 \
  bash "$LIT_SH" --sim=bad --dry-run "$TEST_DIR/no_req.co"

# -- Test 7: combined flags ----------------------------------------------
echo "--- combined flags ---"

check_grep "combined --sim=only --run-only=choreo" \
  'SKIP(sim=only):' 1 \
  bash "$LIT_SH" --sim=only --dry-run --run-only=choreo "$TEST_DIR/nonsim_req.co"

# -- Summary --------------------------------------------------------------
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
exit 0
