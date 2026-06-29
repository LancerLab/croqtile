#!/usr/bin/env bash
# Test CROQ_PROJECT and CROQ_TARGET CMake options.
#
# Usage:
#   bash tests/cmake/test_croq_options.sh [--quick]
#
# Runs cmake configure (no build) with various option combinations and
# verifies that the expected targets and projects are enabled/disabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_BASE="/tmp/choreo-cmake-test-$$"
CMAKE="${CMAKE:-/usr/bin/cmake}"
PASS=0
FAIL=0
QUICK=0

[[ "${1:-}" == "--quick" ]] && QUICK=1

cleanup() { rm -rf "$BUILD_BASE"; }
trap cleanup EXIT

run_cmake() {
  local name="$1"; shift
  local build_dir="$BUILD_BASE/$name"
  mkdir -p "$build_dir"
  "$CMAKE" -S "$SRC_DIR" -B "$build_dir" -G Ninja "$@" 2>&1
}

assert_contains() {
  local label="$1" pattern="$2" output="$3"
  if echo "$output" | grep -qE "$pattern"; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    echo "FAIL [$label]: expected pattern '$pattern' not found"
  fi
}

assert_not_contains() {
  local label="$1" pattern="$2" output="$3"
  if echo "$output" | grep -qE "$pattern"; then
    FAIL=$((FAIL + 1))
    echo "FAIL [$label]: unexpected pattern '$pattern' found"
  else
    PASS=$((PASS + 1))
  fi
}

assert_file_exists() {
  local label="$1" path="$2"
  if [[ -f "$path" ]]; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    echo "FAIL [$label]: file '$path' not found"
  fi
}

assert_file_not_exists() {
  local label="$1" path="$2"
  if [[ -f "$path" ]]; then
    FAIL=$((FAIL + 1))
    echo "FAIL [$label]: file '$path' should not exist"
  else
    PASS=$((PASS + 1))
  fi
}

echo "=== CROQ_PROJECT / CROQ_TARGET CMake option tests ==="
echo "Source: $SRC_DIR"
echo ""

# ---------------------------------------------------------------
# Test 1: Default configuration (choreo;coir, all targets)
# ---------------------------------------------------------------
echo "--- Test 1: defaults (CROQ_PROJECT=choreo;coir, CROQ_TARGET=all) ---"
out=$(run_cmake t1_defaults)
assert_contains "t1-project" "CROQ_TARGET = all" "$out"
assert_contains "t1-targets" "Enabled targets:.*GPU.*CPU" "$out"
assert_not_contains "t1-skip" "skipped .not in CROQ_TARGET" "$out"

# ---------------------------------------------------------------
# Test 2: GPU-only target
# ---------------------------------------------------------------
echo "--- Test 2: CROQ_TARGET=gpu ---"
out=$(run_cmake t2_gpu_only -DCROQ_TARGET=gpu)
assert_contains "t2-enabled" "Enabled targets: GPU" "$out"
assert_contains "t2-skip-cpu" "CPU.*skipped" "$out"
assert_contains "t2-skip-hetero" "Hetero.*skipped" "$out"
assert_contains "t2-skip-amdgpu" "AMDGPU.*skipped" "$out"

# ---------------------------------------------------------------
# Test 3: Multiple targets via runtime names
# ---------------------------------------------------------------
echo "--- Test 3: CROQ_TARGET='cc;hip' ---"
out=$(run_cmake t3_mixed '-DCROQ_TARGET=cc;hip')
assert_contains "t3-cpu" "Enabled targets:.*CPU" "$out"
assert_contains "t3-amdgpu" "Enabled targets:.*AMDGPU" "$out"
assert_contains "t3-skip-gpu" "GPU.*skipped" "$out"
assert_contains "t3-skip-hetero" "Hetero.*skipped" "$out"

# ---------------------------------------------------------------
# Test 4: Unknown target names are ignored
# ---------------------------------------------------------------
echo "--- Test 4: unknown target name ---"
out=$(run_cmake t4_unknown '-DCROQ_TARGET=cpu;nonexistent')
assert_contains "t4-unknown-msg" "unknown target.*nonexistent.*ignoring" "$out"
assert_contains "t4-cpu-enabled" "Enabled targets: CPU" "$out"

# ---------------------------------------------------------------
# Test 5: GPU target gates CUDA/CUTE detection
# ---------------------------------------------------------------
echo "--- Test 5: non-GPU target disables CUDA/CUTE ---"
out=$(run_cmake t5_no_cuda -DCROQ_TARGET=cpu)
assert_contains "t5-cuda-off" "ENABLE_CUDA = OFF" "$out"
assert_contains "t5-cute-off" "ENABLE_CUTE = OFF" "$out"
assert_contains "t5-hip-off" "ENABLE_HIP = OFF" "$out"

# ---------------------------------------------------------------
# Test 6: CROQ_PROJECT=coir only
# ---------------------------------------------------------------
echo "--- Test 6: CROQ_PROJECT=coir ---"
out=$(run_cmake t6_coir_only '-DCROQ_PROJECT=coir')
assert_file_exists "t6-build-ninja" "$BUILD_BASE/t6_coir_only/build.ninja"

if [[ $QUICK -eq 0 ]]; then
# ---------------------------------------------------------------
# Test 7: Build with single target (cpu only) and verify binary
# ---------------------------------------------------------------
echo "--- Test 7: build with CROQ_TARGET=cpu (actual build) ---"
out=$(run_cmake t7_cpu_build -DCROQ_TARGET=cpu '-DCROQ_PROJECT=choreo')
ninja -C "$BUILD_BASE/t7_cpu_build" choreo 2>&1 | tail -3
assert_file_exists "t7-choreo" "$BUILD_BASE/t7_cpu_build/choreo"
# Verify cc target works
cc_out=$("$BUILD_BASE/t7_cpu_build/choreo" -n -t cc -gs <(echo 'buffer a[4]; buffer b[4]; dma(a, b);') 2>&1 | head -1 || true)
assert_contains "t7-cc-works" "#!/usr/bin/env bash" "$cc_out"

# ---------------------------------------------------------------
# Test 8: Build with gpu;cpu targets
# ---------------------------------------------------------------
echo "--- Test 8: build with CROQ_TARGET='gpu;cpu' ---"
out=$(run_cmake t8_gpu_cpu '-DCROQ_TARGET=gpu;cpu' '-DCROQ_PROJECT=choreo')
ninja -C "$BUILD_BASE/t8_gpu_cpu" choreo 2>&1 | tail -3
assert_file_exists "t8-choreo" "$BUILD_BASE/t8_gpu_cpu/choreo"

# ---------------------------------------------------------------
# Test 9: CROQ_PROJECT=choreo;co-mock
# ---------------------------------------------------------------
echo "--- Test 9: build co-mock ---"
out=$(run_cmake t9_comock '-DCROQ_PROJECT=choreo;co-mock' -DCROQ_TARGET=cpu)
ninja -C "$BUILD_BASE/t9_comock" co-mock 2>&1 | tail -3
assert_file_exists "t9-comock" "$BUILD_BASE/t9_comock/co-mock"

fi  # QUICK

echo ""
echo "======================================="
echo "Results: $PASS passed, $FAIL failed"
echo "======================================="
[[ $FAIL -eq 0 ]]
