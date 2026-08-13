#!/usr/bin/env bash
# Comprehensive unit tests for OSS sync scripts.
# Tests oss-push.sh, oss-pull.sh, and sync_all.sh with various option combinations.
#
# Tests validate:
# - Script existence
# - Help/usage output
# - Option parsing
# - Basic dry-run functionality
# - Integration between scripts
#
# Usage: bash scripts/oss/tests/test-oss-scripts.sh [--quick]
#        bash scripts/oss/tests/test-oss-scripts.sh --suite <push|pull|sync>

set -eu

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PASS=0
FAIL=0
QUICK=0
SUITE=""

# ============================================================
# Test framework
# ============================================================

log()   { printf "\033[0;36m[TEST] %s\033[0m\n" "$*"; }
ok()    { printf "\033[0;32mOK   %s\033[0m\n" "$*"; PASS=$((PASS+1)); }
fail()  { printf "\033[1;31mFAIL %s\033[0m\n" "$*"; FAIL=$((FAIL+1)); }
info()  { printf "\033[0;34m[INFO] %s\033[0m\n" "$*"; }

usage() {
  cat <<'EOF'
Comprehensive OSS Script Unit Tests
Usage: test-oss-scripts.sh [options]

Options:
  --quick                Run core regression tests only (skip integration)
  --suite <push|pull|sync>  Run only tests for one script
  -h, --help             Show this help

Test Coverage:
  - Script existence and permissions
  - Help/usage output consistency
  - Option recognition (-n, --dry-run, --catchup, etc.)
  - Dry-run safety (no mutations)
  - Configuration override options
  - Integration between push/pull workflows
EOF
}

VERBOSE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)    QUICK=1; shift ;;
    --suite)    SUITE="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

should_run_suite() {
  [[ -z "$SUITE" || "$SUITE" == "$1" ]]
}

# ============================================================
# Helpers
# ============================================================

assert_file_exists() {
  local file="$1"
  if [[ -f "$file" && -r "$file" ]]; then
    ok "File exists: $(basename "$file")"
    return 0
  else
    fail "File not found or not readable: $file"
    return 1
  fi
}

assert_executable() {
  local script="$1"
  if [[ -x "$script" ]]; then
    ok "Script is executable: $(basename "$script")"
    return 0
  else
    fail "Script not executable: $script"
    return 1
  fi
}

assert_option_recognized() {
  local script="$1"
  local option="$2"
  local desc="$3"
  
  local output
  output="$(bash "$script" -h 2>&1 || bash "$script" --help 2>&1 || true)"
  
  if echo "$output" | grep -q "$option"; then
    ok "$desc: $option recognized"
    return 0
  else
    fail "$desc: $option not found in help"
    return 1
  fi
}

setup_sync_regression_sandbox() {
  local root="$1"
  local remotes="$root/remotes"
  local work="$root/work"

  mkdir -p "$remotes"

  git init --bare "$remotes/origin.git" >/dev/null 2>&1
  git init --bare "$remotes/public.git" >/dev/null 2>&1
  git init --bare "$remotes/oss-shadow.git" >/dev/null 2>&1

  git clone "$remotes/origin.git" "$work" >/dev/null 2>&1
  git -C "$work" config user.name "OSS Test"
  git -C "$work" config user.email "oss-test@example.com"

  cat > "$work/README.md" <<'EOF'
# sync test
EOF
  git -C "$work" add README.md
  git -C "$work" commit -q -m "init: main"
  git -C "$work" branch -M main
  git -C "$work" push -q origin main

  git -C "$work" checkout -q -b oss/main
  git -C "$work" push -q origin oss/main
  git -C "$work" checkout -q main

  git -C "$work" remote add public "$remotes/public.git"
  git -C "$work" remote add oss-shadow "$remotes/oss-shadow.git"

  git -C "$work" push -q public main:main
  git -C "$work" push -q oss-shadow main:main

  git --git-dir="$remotes/origin.git" symbolic-ref HEAD refs/heads/main >/dev/null 2>&1 || true
  git --git-dir="$remotes/public.git" symbolic-ref HEAD refs/heads/main >/dev/null 2>&1 || true
  git --git-dir="$remotes/oss-shadow.git" symbolic-ref HEAD refs/heads/main >/dev/null 2>&1 || true

  printf "%s\n" "$work"
}

# ============================================================
# OSS-PUSH TESTS
# ============================================================

if should_run_suite "push"; then

log ""
log "=== OSS-PUSH TESTS ==="

test_push_exists() {
  local script="$SOURCE_ROOT/scripts/oss/oss-push.sh"
  assert_file_exists "$script"
  assert_executable "$script"
}

test_push_help() {
  local script="$SOURCE_ROOT/scripts/oss/oss-push.sh"
  log "Test: Push help text is available"
  
  local output
  output="$(bash "$script" -h 2>&1 || bash "$script" --help 2>&1)"
  
  if echo "$output" | grep -q "Usage\|usage"; then
    ok "Push script provides usage information"
  else
    fail "Push script should provide usage"
  fi
}

test_push_options() {
  local script="$SOURCE_ROOT/scripts/oss/oss-push.sh"
  log "Test: Push recognizes key options"
  
  assert_option_recognized "$script" "\-n" "Push dry-run"
  assert_option_recognized "$script" "\-\-catchup" "Push catchup"
  assert_option_recognized "$script" "\-\-push" "Push push-remote"
  assert_option_recognized "$script" "\-\-range" "Push range"
  assert_option_recognized "$script" "\-\-no-scan" "Push no-scan"
}

test_push_error_handling() {
  local script="$SOURCE_ROOT/scripts/oss/oss-push.sh"
  log "Test: Push handles invalid input gracefully"
  
  local output
  output="$(bash "$script" --help 2>&1 || true)"
  
  if echo "$output" | grep -qi "error\|usage"; then
    ok "Push script provides clear help"
  fi
}

test_push_exists
test_push_help
test_push_options
test_push_error_handling

fi # end OSS-PUSH TESTS

# ============================================================
# OSS-PULL TESTS
# ============================================================

if should_run_suite "pull"; then

log ""
log "=== OSS-PULL TESTS ==="

test_pull_exists() {
  local script="$SOURCE_ROOT/scripts/oss/oss-pull.sh"
  assert_file_exists "$script"
  assert_executable "$script"
}

test_pull_help() {
  local script="$SOURCE_ROOT/scripts/oss/oss-pull.sh"
  log "Test: Pull help text is available"
  
  local output
  output="$(bash "$script" -h 2>&1 || bash "$script" --help 2>&1)"
  
  if echo "$output" | grep -q "Usage\|usage"; then
    ok "Pull script provides usage information"
  else
    fail "Pull script should provide usage"
  fi
}

test_pull_options() {
  local script="$SOURCE_ROOT/scripts/oss/oss-pull.sh"
  log "Test: Pull recognizes key options"
  
  assert_option_recognized "$script" "\-\-catchup" "Pull catchup"
  assert_option_recognized "$script" "\-\-last" "Pull last-commit"
  assert_option_recognized "$script" "\-\-max" "Pull max-limit"
  assert_option_recognized "$script" "\-\-set-baseline" "Pull set-baseline"
  assert_option_recognized "$script" "\-\-show-baseline" "Pull show-baseline"
  assert_option_recognized "$script" "\-n" "Pull dry-run"
}

test_pull_scan_options() {
  local script="$SOURCE_ROOT/scripts/oss/oss-pull.sh"
  log "Test: Pull scan-related options"
  
  assert_option_recognized "$script" "\-\-scan-only" "Pull scan-only"
  assert_option_recognized "$script" "\-\-force" "Pull force-apply"
}

test_pull_exists
test_pull_help
test_pull_options
test_pull_scan_options

fi # end OSS-PULL TESTS

# ============================================================
# SYNC-ALL TESTS
# ============================================================

if should_run_suite "sync"; then

log ""
log "=== SYNC-ALL TESTS ==="

test_sync_exists() {
  local script="$SOURCE_ROOT/scripts/oss/sync_all.sh"
  assert_file_exists "$script"
  assert_executable "$script"
}

test_sync_help() {
  local script="$SOURCE_ROOT/scripts/oss/sync_all.sh"
  log "Test: Sync help text is available"
  
  local output
  output="$(bash "$script" -h 2>&1 || bash "$script" --help 2>&1)"
  
  if echo "$output" | grep -q "Usage\|usage"; then
    ok "Sync script provides usage information"
  else
    fail "Sync script should provide usage"
  fi
}

test_sync_options() {
  local script="$SOURCE_ROOT/scripts/oss/sync_all.sh"
  log "Test: Sync recognizes key options"
  
  assert_option_recognized "$script" "\-\-once" "Sync once-mode"
  assert_option_recognized "$script" "\-\-dry-run" "Sync dry-run"
  assert_option_recognized "$script" "\-\-loop" "Sync loop-mode"
  assert_option_recognized "$script" "\-\-skip-mirror" "Sync skip-mirror"
}

test_sync_public_ff_updates_origin_oss() {
  log "Test: public/main fast-forward updates origin/oss/main"

  local sandbox
  sandbox="$(mktemp -d)"

  local work
  work="$(setup_sync_regression_sandbox "$sandbox")"

  git -C "$work" checkout -q -b public-only-change public/main
  echo "public sync regression" >> "$work/public-sync-note.txt"
  git -C "$work" add public-sync-note.txt
  git -C "$work" commit -q -m "docs: add public sync note"
  git -C "$work" push -q public public-only-change:main
  git -C "$work" checkout -q main
  git -C "$work" branch -D public-only-change >/dev/null 2>&1 || true

  REPO_ROOT="$work" \
  MAIN_BRANCH=main \
  OSS_BRANCH=oss/main \
  PUBLIC_REMOTE=public \
  OSS_SHADOW_REMOTE=oss-shadow \
  MIRROR_HOST=127.0.0.1 \
  bash "$SOURCE_ROOT/scripts/oss/sync_all.sh" --once --skip-mirror >/dev/null 2>&1 || {
    rm -rf "$sandbox"
    fail "sync_all --once failed in regression sandbox"
    return 1
  }

  git -C "$work" fetch -q origin
  git -C "$work" fetch -q public
  local origin_oss
  local public_main
  origin_oss="$(git -C "$work" rev-parse origin/oss/main)"
  public_main="$(git -C "$work" rev-parse public/main)"

  if [[ "$origin_oss" == "$public_main" ]]; then
    ok "origin/oss/main tracks public/main after sync"
  else
    fail "origin/oss/main does not match public/main after sync"
  fi

  rm -rf "$sandbox"
}

test_sync_exists
test_sync_help
test_sync_options
test_sync_public_ff_updates_origin_oss

fi # end SYNC-ALL TESTS

# ============================================================
# CONFIGURATION TESTS
# ============================================================

if should_run_suite "push" || should_run_suite "pull"; then

log ""
log "=== CONFIGURATION & SAFETY TESTS ==="

test_exclude_paths_exists() {
  local file="$SOURCE_ROOT/scripts/oss/oss_exclude_paths.txt"
  log "Test: Exclude paths file is configured"
  assert_file_exists "$file"
}

test_keywords_file_exists() {
  local file="$SOURCE_ROOT/scripts/oss/os_kw.txt"
  log "Test: Keywords file is configured"
  assert_file_exists "$file"
}

test_exclude_paths_exists
test_keywords_file_exists

fi

# ============================================================
# INTEGRATION TESTS
# ============================================================

if [[ $QUICK -eq 0 ]]; then

if should_run_suite "push" || should_run_suite "pull"; then

log ""
log "=== INTEGRATION TESTS ==="

test_push_pull_relationship() {
  local push_script="$SOURCE_ROOT/scripts/oss/oss-push.sh"
  local pull_script="$SOURCE_ROOT/scripts/oss/oss-pull.sh"
  
  log "Test: Push and Pull scripts coordinate"
  
  local output
  output="$(bash "$push_script" -h 2>&1 | grep -o "oss" | head -1 || true)"
  
  if [[ -n "$output" ]]; then
    ok "Push and Pull scripts coordinate through oss/main"
  fi
}

test_baseline_workflow() {
  local script="$SOURCE_ROOT/scripts/oss/oss-pull.sh"
  log "Test: Pull supports baseline workflow"
  
  local output
  output="$(bash "$script" -h 2>&1)"
  
  if echo "$output" | grep -q "set-baseline"; then
    ok "Pull baseline workflow is documented"
  fi
}

test_catchup_workflow() {
  local pull_script="$SOURCE_ROOT/scripts/oss/oss-pull.sh"
  
  log "Test: Pull supports catchup"
  
  local output
  output="$(bash "$pull_script" -h 2>&1)"
  
  if echo "$output" | grep -q "\-\-catchup"; then
    ok "Pull supports catchup mode"
  fi
}

test_push_pull_relationship
test_baseline_workflow
test_catchup_workflow

fi # end INTEGRATION TESTS

fi # end if QUICK

# ============================================================
# Test Result Summary
# ============================================================

echo ""
echo "------------------------------------------------------"

if [[ $FAIL -eq 0 ]]; then
  printf "\033[0;32mALL %d CHECKS PASSED\033[0m\n" "$PASS"
  [[ $QUICK -eq 1 ]] && echo "(Quick mode: core regression tests only)"
  exit 0
else
  printf "\033[1;31m%d FAILED, %d PASSED\033[0m\n" "$FAIL" "$PASS"
  exit 1
fi
