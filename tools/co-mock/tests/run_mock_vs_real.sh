#!/bin/bash
# Compare co-mock output against cc-target compiled output.
# Usage: bash run_mock_vs_real.sh <test.co>
#
# Requires: build/co-mock, build/choreo, g++

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CO_MOCK="$REPO_ROOT/build/co-mock"
CHOREO="$REPO_ROOT/build/choreo"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <test.co> [target]"
  exit 1
fi

TEST_FILE="$1"
TARGET="${2:-cc}"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Mock interpreter output ==="
MOCK_OUT=$("$CO_MOCK" -t "$TARGET" "$TEST_FILE" 2>&1) || true
echo "$MOCK_OUT"

echo ""
echo "=== Compiled cc-target output ==="

# Extract function name from __co__ declaration
FUNC_NAME=$(grep -oP '__co__\s+\w+\s+(\w+)' "$TEST_FILE" | head -1 | awk '{print $NF}' || echo "test")

# Choreo generates source in a temp directory; capture that path
CHOREO_OUT=$("$CHOREO" -t cc "$TEST_FILE" -es -o "$TMPDIR/gen.cpp" 2>&1) || true

# Find choreo.h - check choreo's temp dir first, then runtime/
CHOREO_TMPDIR=$(echo "$CHOREO_OUT" | grep -oP '/tmp/\d+_\d+' | head -1 || true)
if [ -n "$CHOREO_TMPDIR" ] && [ -f "$CHOREO_TMPDIR/choreo.h" ]; then
  INCLUDE_DIR="$CHOREO_TMPDIR"
else
  INCLUDE_DIR="$REPO_ROOT/runtime"
fi

# Copy choreo.h and choreo_cc.h to our temp dir
cp "$INCLUDE_DIR"/choreo*.h "$TMPDIR/" 2>/dev/null || true
if [ ! -f "$TMPDIR/choreo.h" ]; then
  cp "$REPO_ROOT"/runtime/choreo*.h "$TMPDIR/" 2>/dev/null || true
fi

# Add main wrapper
cat >> "$TMPDIR/gen.cpp" << MAIN

int main() {
  ${FUNC_NAME}();
  return 0;
}
MAIN

g++ -std=c++17 -O2 -I"$TMPDIR" "$TMPDIR/gen.cpp" -o "$TMPDIR/a.out" -lpthread 2>&1
REAL_OUT=$("$TMPDIR/a.out" 2>&1) || true
echo "$REAL_OUT"

echo ""

if [ "$MOCK_OUT" = "$REAL_OUT" ]; then
  echo "PASS: outputs match exactly"
  exit 0
else
  echo "FAIL: outputs differ"
  diff <(echo "$MOCK_OUT") <(echo "$REAL_OUT") || true
  exit 1
fi
