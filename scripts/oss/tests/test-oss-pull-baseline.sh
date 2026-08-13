#!/usr/bin/env bash
# Integration tests for oss-pull.sh --baseline / --catchup / --last
# Runs in a temporary sandbox clone (never mutates the caller's repo).
#
# Usage: bash scripts/oss/tests/test-oss-pull-baseline.sh
#        bash scripts/oss/tests/test-oss-pull-baseline.sh --quick

set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CHOREO_ROOT=""
SCRIPTS=""
PID=$$
TEST_MAIN="test-pull-main-$PID"
TEST_OSS="test-pull-oss-$PID"
TEST_BL="$(mktemp)"
SANDBOX_DIR="$(mktemp -d)"
PASS=0
FAIL=0
QUICK=0

if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
elif [[ $# -gt 0 ]]; then
  echo "Unknown option: $1" >&2
  echo "Usage: $0 [--quick]" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()   { printf "\033[0;36m[TEST] %s\033[0m\n" "$*"; }
ok()    { printf "\033[0;32m  OK : %s\033[0m\n" "$*"; PASS=$((PASS+1)); }
fail()  { printf "\033[1;31m  FAIL: %s\033[0m\n" "$*"; FAIL=$((FAIL+1)); }
assert_contains() { local out="$1" pat="$2"
  if grep -Fq "$pat" <<<"$out"; then ok "$pat"; else fail "expected '$pat' in: $out"; fi; }
assert_not_contains() { local out="$1" pat="$2"
  if ! grep -Fq "$pat" <<<"$out"; then ok "absent '$pat'"; else fail "unexpected '$pat' in: $out"; fi; }

create_sandbox() {
  local sandbox_repo="$SANDBOX_DIR/repo"
  local rel=""
  local source_path=""
  local target_path=""
  declare -A overlay_seen=()

  # Shared-object clone is fast and keeps source repo untouched.
  git clone --quiet --shared "$SOURCE_ROOT" "$sandbox_repo"

  # Overlay only source-side edits under scripts/oss so we test current
  # in-progress work without copying unrelated files.
  while IFS= read -r rel; do
    [[ -n "$rel" ]] && overlay_seen["$rel"]=1
  done < <(git -C "$SOURCE_ROOT" diff --name-only -- scripts/oss)
  while IFS= read -r rel; do
    [[ -n "$rel" ]] && overlay_seen["$rel"]=1
  done < <(git -C "$SOURCE_ROOT" ls-files --others --exclude-standard -- scripts/oss)

  # Always overlay this test file itself.
  overlay_seen["scripts/oss/tests/test-oss-pull-baseline.sh"]=1

  git -C "$sandbox_repo" config user.name "OSS Pull Test"
  git -C "$sandbox_repo" config user.email "oss-pull-test@example.com"

  # Apply overlay file-by-file and keep sandbox clean so oss-pull.sh does not
  # need stash/pop during scenarios.
  for rel in "${!overlay_seen[@]}"; do
    source_path="$SOURCE_ROOT/$rel"
    target_path="$sandbox_repo/$rel"
    if [[ -f "$source_path" ]]; then
      mkdir -p "$(dirname "$target_path")"
      cp -a "$source_path" "$target_path"
      git -C "$sandbox_repo" add "$rel"
    else
      rm -f "$target_path"
      git -C "$sandbox_repo" add -A "$rel"
    fi
  done

  if ! git -C "$sandbox_repo" diff --cached --quiet; then
    git -C "$sandbox_repo" commit -q -m "test: overlay local scripts/oss edits"
  fi

  CHOREO_ROOT="$sandbox_repo"
  SCRIPTS="$CHOREO_ROOT/scripts/oss"
}

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
  log "Cleaning up sandbox..."
  rm -f "$TEST_BL"
  rm -rf "$SANDBOX_DIR"
  echo ""
  if [[ $FAIL -eq 0 ]]; then
    printf "\033[0;32m=== ALL %d CHECKS PASSED ===\033[0m\n" "$PASS"
  else
    printf "\033[1;31m=== %d FAILED, %d PASSED ===\033[0m\n" "$FAIL" "$PASS"
    exit 1
  fi
}
trap cleanup EXIT

create_sandbox

cd "$CHOREO_ROOT"
# Both test branches start from current main HEAD
git checkout -q main
git branch "$TEST_MAIN" main
git branch "$TEST_OSS"  main

log "Sandbox repo: $CHOREO_ROOT"
log "Test branches created: $TEST_MAIN (main-equiv), $TEST_OSS (oss-equiv)"
log "Baseline file: $TEST_BL (empty / temp)"
[[ $QUICK -eq 1 ]] && log "Quick mode enabled: running core regression scenarios only"

# Count commits on $TEST_MAIN added since the test started (main HEAD)
counts_ahead() { git rev-list --count "$TEST_MAIN" ^main 2>/dev/null; }

# True if file exists on $TEST_MAIN
has_file() { git cat-file -e "$TEST_MAIN:$1" 2>/dev/null; }

# ============================================================
# A: --show-baseline with no baseline file → merge-base fallback
# ============================================================
if [[ $QUICK -eq 0 ]]; then
log ""
log "=== SCENARIO A: show-baseline with no file ==="
> "$TEST_BL"   # empty
OUT="$(run_pull --show-baseline)"
log "Output: $OUT"
# merge-base of TEST_MAIN and TEST_OSS equals main (same SHA)
assert_contains "$OUT" "Baseline:"
fi

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
if [[ $QUICK -eq 0 ]]; then
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
fi

# ============================================================
# D: --max cap + re-run gets next batch
# ============================================================
log ""
log "=== SCENARIO D: --max 2 cap then re-run ==="
# Reset: set baseline to current TEST_OSS tip
OUT="$(run_pull --set-baseline "$(git rev-parse "$TEST_OSS")")"
log "re-set baseline: $OUT"

CNT_BEFORE="$(counts_ahead)"

# Add public commits, each touching its own file to avoid context issues.
# Quick mode uses fewer commits for faster regression checks.
FEATURE_COUNT=4
CATCHUP_MAX=2
if [[ $QUICK -eq 1 ]]; then
  FEATURE_COUNT=2
  CATCHUP_MAX=1
fi

for ((i=1; i<=FEATURE_COUNT; i++)); do
  add_commit "$TEST_OSS" "public: feature-$i" "public-D-$i.txt" "feature $i"
done

OUT="$(run_pull --catchup --max "$CATCHUP_MAX")"
log "1st run (--max $CATCHUP_MAX): $OUT"
assert_contains "$OUT" "Catchup: $CATCHUP_MAX commit"
if [[ $FEATURE_COUNT -gt $CATCHUP_MAX ]]; then
  assert_contains "$OUT" "re-run --catchup for more"
fi
assert_contains "$OUT" "$CATCHUP_MAX pulled"
CNT_AFTER="$(counts_ahead)"
[[ $((CNT_AFTER - CNT_BEFORE)) -eq $CATCHUP_MAX ]] && ok "$CATCHUP_MAX commits added to $TEST_MAIN" || fail "Expected $CATCHUP_MAX new commits, got $((CNT_AFTER - CNT_BEFORE))"

if [[ $QUICK -eq 0 ]]; then
  OUT2="$(run_pull --catchup --max "$CATCHUP_MAX")"
  log "2nd run: $OUT2"
  assert_contains "$OUT2" "Catchup: $CATCHUP_MAX commit"
  assert_contains "$OUT2" "$CATCHUP_MAX pulled"
  CNT_FINAL="$(counts_ahead)"
  [[ $((CNT_FINAL - CNT_BEFORE)) -eq $FEATURE_COUNT ]] && ok "$FEATURE_COUNT total commits on $TEST_MAIN" || fail "Expected $FEATURE_COUNT total, got $((CNT_FINAL - CNT_BEFORE))"
fi

# ============================================================
# E: --last mode finds newest unpulled commit
# ============================================================
if [[ $QUICK -eq 0 ]]; then
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
fi

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

# ============================================================
# G: conflict-zone file (Makefile) applies cleanly when main has
#    no private divergence (regression for false-positive DIVERGED)
# ============================================================
log ""
log "=== SCENARIO G: conflict-zone file no false DIVERGED ==="
OUT="$(run_pull --set-baseline "$(git rev-parse "$TEST_OSS")")"
log "reset baseline"

# A public commit touching a conflict-zone file that main has NOT
# independently changed must pull cleanly (main matches the commit's
# parent), not be flagged DIVERGED against the oss/main tip.
add_commit "$TEST_OSS" "build(makefile): tweak coir-only helper" "Makefile" "# test: coir-only helper tweak"
log "Added Makefile-modifying commit $(tip "$TEST_OSS")"

OUT="$(run_pull --catchup)"
log "catchup: $OUT"
assert_contains "$OUT" "1 pulled"
assert_not_contains "$OUT" "DIVERGED"
has_file "Makefile" && ok "conflict-zone Makefile change pulled without false DIVERGED" || fail "Makefile change not pulled"

# ============================================================
# H: conflict-zone file with genuine private divergence is still
#    flagged (guard against an over-permissive scan)
# ============================================================
log ""
log "=== SCENARIO H: conflict-zone file genuine DIVERGED ==="
OUT="$(run_pull --set-baseline "$(git rev-parse "$TEST_OSS")")"
log "reset baseline"

# Independent private (main-only) change, then a public change to the
# same conflict-zone file.  main differs from the public commit's
# parent, so the scan must flag DIVERGED and halt.
add_commit "$TEST_MAIN" "internal: private Makefile tweak" "Makefile" "# test: private tweak"
add_commit "$TEST_OSS" "build: public Makefile tweak" "Makefile" "# test: public tweak"
log "Added private main tweak + public oss tweak to same file"

OUT="$(run_pull --catchup)"
log "catchup: $OUT"
assert_contains "$OUT" "DIVERGED"
assert_contains "$OUT" "Sync halted"
ok "genuine conflict-zone divergence still flagged and halted"

# ============================================================
# I: duplicate public patch already on main (different SHA, no
#    oss-push trailer) is skipped via public-file patch-id dedup
# ============================================================
log ""
log "=== SCENARIO I: duplicate patch already on main skipped ==="
OUT="$(run_pull --set-baseline "$(git rev-parse "$TEST_OSS")")"
log "reset baseline"

# Equivalent public change lands on main natively (no trailer, different
# message so its SHA differs), then the same public change arrives on
# oss/main under its own SHA followed by a later edit to the same file
# (so a naive tip-vs-tip file compare would NOT notice the duplicate).
add_commit "$TEST_MAIN" "feat: public feature I (main-native)" "public-I.txt" "feature-I line"
add_commit "$TEST_OSS" "feat: public feature I" "public-I.txt" "feature-I line"
add_commit "$TEST_OSS" "chore: follow-up tweak I" "public-I.txt" "feature-I follow-up"
log "main-native dup + oss dup + oss follow-up created"

OUT="$(run_pull --catchup)"
log "catchup: $OUT"
assert_contains "$OUT" "Catchup: 1 commit"
assert_contains "$OUT" "chore: follow-up tweak I"
assert_not_contains "$OUT" "feat: public feature I"
assert_contains "$OUT" "1 pulled"
git show "$TEST_MAIN:public-I.txt" | grep -q "feature-I follow-up" \
  && ok "follow-up content cherry-picked to $TEST_MAIN" \
  || fail "follow-up content missing on $TEST_MAIN"
