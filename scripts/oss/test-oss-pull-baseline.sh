#!/usr/bin/env bash
# Integration tests for oss-pull.sh --baseline / --catchup / --last
# Operates on temporary test branches in the CURRENT repo.
# Safe: cleans up all test branches on exit.
#
# Usage: bash scripts/oss/test-oss-pull-baseline.sh

set -uo pipefail

CHOREO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$CHOREO_ROOT/scripts/oss"
PID=$$
TEST_MAIN="test-pull-main-$PID"
TEST_OSS="test-pull-oss-$PID"
TEST_BL="$(mktemp)"
PASS=0
FAIL=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()   { printf "\033[0;36m[TEST] %s\033[0m\n" "$*"; }
ok()    { printf "\033[0;32m  OK : %s\033[0m\n" "$*"; PASS=$((PASS+1)); }
fail()  { printf "\033[1;31m  FAIL: %s\033[0m\n" "$*"; FAIL=$((FAIL+1)); }
assert_contains() { local out="$1" pat="$2"
  if echo "$out" | grep -q "$pat"; then ok "$pat"; else fail "expected '$pat' in: $out"; fi; }
assert_not_contains() { local out="$1" pat="$2"
  if ! echo "$out" | grep -q "$pat"; then ok "absent '$pat'"; else fail "unexpected '$pat' in: $out"; fi; }

run_pull() {
  BASELINE_FILE="$TEST_BL" \
  OSS_BRANCH="$TEST_OSS" \
  MAIN_BRANCH="$TEST_MAIN" \
  bash "$SCRIPTS/oss-pull.sh" "$@" 2>&1 || true
}

# Add a commit to branch $1, message $2, file changes $3 (relative to repo root)
add_commit() {
  local branch="$1" msg="$2" file="$3" content="${4:-change $(date +%s)}"
  local prev_branch
  prev_branch="$(git symbolic-ref --short HEAD)"
  git checkout -q "$branch"
  mkdir -p "$(dirname "$CHOREO_ROOT/$file")"
  echo "$content" >> "$CHOREO_ROOT/$file"
  git -C "$CHOREO_ROOT" add "$file"
  git -C "$CHOREO_ROOT" commit -q -m "$msg"
  git checkout -q "$prev_branch"
}

# Return the SHA of the tip of $1
tip() { git rev-parse --short "$1"; }

# Check if $branch has commit with subject matching $pat
branch_has() { local br="$1" pat="$2"
  git log "$br" --oneline 2>/dev/null | grep -q "$pat"; }

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

cleanup() {
  echo ""
  log "Cleaning up test branches..."
  git checkout -q main 2>/dev/null || true
  git branch -D "$TEST_MAIN" 2>/dev/null || true
  git branch -D "$TEST_OSS"  2>/dev/null || true
  rm -f "$TEST_BL"
  echo ""
  if [[ $FAIL -eq 0 ]]; then
    printf "\033[0;32m=== ALL %d CHECKS PASSED ===\033[0m\n" "$PASS"
  else
    printf "\033[1;31m=== %d FAILED, %d PASSED ===\033[0m\n" "$FAIL" "$PASS"
    exit 1
  fi
}
trap cleanup EXIT

cd "$CHOREO_ROOT"
# Both test branches start from current main HEAD
git checkout -q main
git branch "$TEST_MAIN" main
git branch "$TEST_OSS"  main

log "Test branches created: $TEST_MAIN (main-equiv), $TEST_OSS (oss-equiv)"
log "Baseline file: $TEST_BL (empty / temp)"

# Count commits on $TEST_MAIN added since the test started (main HEAD)
counts_ahead() { git rev-list --count "$TEST_MAIN" ^main 2>/dev/null; }

# True if file exists on $TEST_MAIN
has_file() { git cat-file -e "$TEST_MAIN:$1" 2>/dev/null; }

# ============================================================
# A: --show-baseline with no baseline file → merge-base fallback
# ============================================================
log ""
log "=== SCENARIO A: show-baseline with no file ==="
> "$TEST_BL"   # empty
OUT="$(run_pull --show-baseline)"
log "Output: $OUT"
# merge-base of TEST_MAIN and TEST_OSS equals main (same SHA)
assert_contains "$OUT" "Baseline:"

# ============================================================
# B: --set-baseline writes oss HEAD; subsequent --catchup empty
# ============================================================
log ""
log "=== SCENARIO B: set-baseline → catchup empty ==="
OUT="$(run_pull --set-baseline)"
log "set-baseline output: $OUT"
assert_contains "$OUT" "Baseline set:"

# Now add a new public commit to TEST_OSS (fresh file, no conflict context)
add_commit "$TEST_OSS" "feat: public adds scenario-B file" "public-B.txt"
OSS_TIP="$(tip "$TEST_OSS")"
log "Added public commit $OSS_TIP to $TEST_OSS"

OUT="$(run_pull --catchup)"
log "catchup output: $OUT"
assert_contains "$OUT" "Catchup: 1 commit"
assert_contains "$OUT" "1 pulled"
has_file "public-B.txt" && ok "public-B.txt cherry-picked to $TEST_MAIN" || fail "public-B.txt missing on $TEST_MAIN"

# ============================================================
# C: oss-push marker is skipped; only public-native landed
# ============================================================
log ""
log "=== SCENARIO C: oss-push marker filtering ==="
# Re-set baseline to current TEST_OSS tip so we start fresh for this scenario
OUT="$(run_pull --set-baseline "$(git rev-parse "$TEST_OSS")")"
log "re-set baseline: $OUT"

add_commit "$TEST_OSS" \
  "(oss) sync: cherry picked from abc000 on main" \
  "public-C-internal.txt" "internal-sync"
add_commit "$TEST_OSS" \
  "fix: public fix typo in CONTRIBUTING" \
  "public-C-fix.txt" "public typo fix"

PUBLIC_TIP="$(tip "$TEST_OSS")"
log "Added 2 commits: oss-push marker + public fix; oss tip=$PUBLIC_TIP"

OUT="$(run_pull --catchup)"
log "catchup output: $OUT"
assert_contains "$OUT" "Catchup: 1 commit"
assert_not_contains "$OUT" "cherry picked from abc000"
assert_contains "$OUT" "fix: public fix typo in CONTRIBUTING"
assert_contains "$OUT" "1 pulled"
has_file "public-C-fix.txt" && ok "public-C-fix.txt on $TEST_MAIN" || fail "public-C-fix.txt missing"
! has_file "public-C-internal.txt" && ok "oss-push marker commit correctly skipped" || fail "internal commit should not be on $TEST_MAIN"

# ============================================================
# D: --max cap + re-run gets next batch
# ============================================================
log ""
log "=== SCENARIO D: --max 2 cap then re-run ==="
# Reset: set baseline to current TEST_OSS tip
OUT="$(run_pull --set-baseline "$(git rev-parse "$TEST_OSS")")"
log "re-set baseline: $OUT"

CNT_BEFORE="$(counts_ahead)"

# Add 4 public commits, each touching its own file to avoid context issues
for i in 1 2 3 4; do
  add_commit "$TEST_OSS" "public: feature-$i" "public-D-$i.txt" "feature $i"
done

OUT="$(run_pull --catchup --max 2)"
log "1st run (--max 2): $OUT"
assert_contains "$OUT" "Catchup: 2 commit"
assert_contains "$OUT" "re-run --catchup for more"
assert_contains "$OUT" "2 pulled"
CNT_AFTER="$(counts_ahead)"
[[ $((CNT_AFTER - CNT_BEFORE)) -eq 2 ]] && ok "2 commits added to $TEST_MAIN" || fail "Expected 2 new commits, got $((CNT_AFTER - CNT_BEFORE))"

OUT2="$(run_pull --catchup --max 2)"
log "2nd run: $OUT2"
assert_contains "$OUT2" "Catchup: 2 commit"
assert_contains "$OUT2" "2 pulled"
CNT_FINAL="$(counts_ahead)"
[[ $((CNT_FINAL - CNT_BEFORE)) -eq 4 ]] && ok "4 total commits on $TEST_MAIN" || fail "Expected 4 total, got $((CNT_FINAL - CNT_BEFORE))"

# ============================================================
# E: --last mode finds newest unpulled commit
# ============================================================
log ""
log "=== SCENARIO E: --last finds newest unpulled ==="
OUT="$(run_pull --set-baseline "$(git rev-parse "$TEST_OSS")")"
log "reset baseline"

add_commit "$TEST_OSS" "docs: public adds CONTRIBUTING" "public-E-1.txt" "contributing"
NEWEST="$(tip "$TEST_OSS")"
add_commit "$TEST_OSS" "docs: public adds CHANGELOG" "public-E-2.txt" "changelog"
NEWEST_OF_2="$(tip "$TEST_OSS")"
log "Added 2 commits; newest=$NEWEST_OF_2"

OUT="$(run_pull --last)"
log "--last output: $OUT"
assert_contains "$OUT" "Last unpulled:"
assert_contains "$OUT" "$NEWEST_OF_2"
assert_not_contains "$OUT" "$NEWEST "
ok "--last returned newest (latest) unpulled commit"

# ============================================================
# F: --catchup properly excludes private-only commits
# ============================================================
log ""
log "=== SCENARIO F: private-only commit skipped ==="
OUT="$(run_pull --set-baseline "$(git rev-parse "$TEST_OSS")")"
add_commit "$TEST_OSS" "internal: update scripts/oss tooling" "scripts/oss/oss_exclude_paths.txt" "some change"
ALL_PRIVATE_TIP="$(tip "$TEST_OSS")"
log "Added private-path-only commit $ALL_PRIVATE_TIP"

OUT="$(run_pull --catchup)"
log "catchup: $OUT"
assert_contains "$OUT" "Nothing to pull"
ok "private-only commit correctly skipped (no public files)"
