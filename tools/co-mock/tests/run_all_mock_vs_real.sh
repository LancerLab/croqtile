#!/bin/bash
# Run all mock-vs-real comparison tests.
# Usage: bash run_all_mock_vs_real.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$SCRIPT_DIR/run_mock_vs_real.sh"
TEST_DIR="$SCRIPT_DIR/mock_vs_real"

passed=0
failed=0
errors=""

for f in "$TEST_DIR"/*.co; do
  name=$(basename "$f" .co)
  printf "%-20s " "$name"
  output=$(bash "$RUNNER" "$f" 2>&1) || true
  if echo "$output" | grep -q "PASS:"; then
    echo "PASS"
    ((passed++))
  else
    echo "FAIL"
    ((failed++))
    errors="$errors\n=== $name ===\n$output"
  fi
done

echo ""
echo "Results: $passed passed, $failed failed"

if [ $failed -gt 0 ]; then
  echo -e "\nFailure details:$errors"
  exit 1
fi
