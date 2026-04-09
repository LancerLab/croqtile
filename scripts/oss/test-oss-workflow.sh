#!/usr/bin/env bash
set -euo pipefail

# Self-contained integration tests for the OSS sync workflow.
# Creates temporary git repos, populates them, and validates
# oss-push, oss-pull, and oss-scan end-to-end.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPBASE=""
PASS=0
FAIL=0
TESTS_RUN=0

# -------- test framework --------

setup_env() {
  TMPBASE="$(mktemp -d /tmp/oss_workflow_test.XXXXXX)"
  echo "Test environment: $TMPBASE"
}

teardown_env() {
  if [[ -n "$TMPBASE" && -d "$TMPBASE" ]]; then
    rm -rf "$TMPBASE"
    echo "Cleaned up $TMPBASE"
  fi
}
trap teardown_env EXIT

assert_eq() {
  local label="$1" expected="$2" actual="$3"
  if [[ "$expected" == "$actual" ]]; then
    return 0
  else
    echo "  ASSERT FAILED ($label): expected='$expected' actual='$actual'"
    return 1
  fi
}

assert_file_exists() {
  local label="$1" path="$2"
  if [[ -e "$path" ]]; then return 0; fi
  echo "  ASSERT FAILED ($label): file not found: $path"
  return 1
}

assert_file_not_exists() {
  local label="$1" path="$2"
  if [[ ! -e "$path" ]]; then return 0; fi
  echo "  ASSERT FAILED ($label): file should not exist: $path"
  return 1
}

assert_file_contains() {
  local label="$1" path="$2" pattern="$3"
  if grep -qE -- "$pattern" "$path" 2>/dev/null; then return 0; fi
  echo "  ASSERT FAILED ($label): '$path' does not contain pattern '$pattern'"
  return 1
}

assert_exit_code() {
  local label="$1" expected="$2" actual="$3"
  if [[ "$expected" == "$actual" ]]; then return 0; fi
  echo "  ASSERT FAILED ($label): expected exit=$expected got exit=$actual"
  return 1
}

run_test() {
  local name="$1"
  TESTS_RUN=$((TESTS_RUN + 1))
  echo ""
  echo "=== TEST: $name ==="
  if "$name"; then
    echo "--- PASS: $name ---"
    PASS=$((PASS + 1))
  else
    echo "--- FAIL: $name ---"
    FAIL=$((FAIL + 1))
  fi
}

# -------- repo scaffolding --------

# Create a "main" working repo with shared + GCU + GPU files,
# an oss/main branch, and copy the oss scripts into it.
create_test_repos() {
  local main_bare="$TMPBASE/origin.git"
  local public_bare="$TMPBASE/public.git"
  WORK="$TMPBASE/work"

  git init --bare -b main "$main_bare" >/dev/null 2>&1
  git init --bare -b main "$public_bare" >/dev/null 2>&1

  git clone "$main_bare" "$WORK" >/dev/null 2>&1
  cd "$WORK"
  git checkout -b main 2>/dev/null || true
  git remote add public "$public_bare"

  # Seed shared files
  mkdir -p lib/Target/GPU runtime tests/check
  echo "# Choreo" > README.md
  echo 'int main() { return 0; }' > lib/shared.cpp
  echo 'void gpu() {}' > lib/Target/GPU/cute_target.cpp
  echo '#include <cuda.h>' > runtime/choreo.h
  echo '// test' > tests/check/basic.co

  # Seed GCU-specific files (should be excluded from public)
  mkdir -p lib/Target/GCU tests/gcu samples/topscc benchmark
  echo 'void gcu_init() {}' > lib/Target/GCU/gcu_target.cpp
  echo '// gcu test' > tests/gcu/smoke.co
  echo '// sample' > samples/topscc/matmul.co
  echo '# bench' > benchmark/perf.sh

  # Seed internal files
  mkdir -p Documents/internal scripts .vscode
  echo 'internal doc' > Documents/internal/notes.md
  echo '#!/bin/bash' > scripts/run.sh
  echo '{}' > .vscode/settings.json
  echo '.gitlab config' > .gitlab-ci.yml

  # Copy OSS scripts from the real repo
  mkdir -p scripts/oss
  cp "$SCRIPT_DIR/oss-setup.sh" scripts/oss/
  cp "$SCRIPT_DIR/oss-push.sh" scripts/oss/
  cp "$SCRIPT_DIR/oss-pull.sh" scripts/oss/
  cp "$SCRIPT_DIR/oss-scan.sh" scripts/oss/
  cp "$SCRIPT_DIR/oss_exclude_paths.txt" scripts/oss/
  cp "$SCRIPT_DIR/os_kw.txt" scripts/oss/
  chmod +x scripts/oss/*.sh

  git add -A
  git commit -m "initial: shared + GCU + GPU seed" >/dev/null

  # Push to "origin"
  git push origin main >/dev/null 2>&1

  # Create a minimal oss/main branch with only shared/GPU content
  git checkout --orphan oss/main >/dev/null 2>&1
  git rm -rf . >/dev/null 2>&1

  echo "# Choreo" > README.md
  mkdir -p lib/Target/GPU runtime tests/check
  echo 'int main() { return 0; }' > lib/shared.cpp
  echo 'void gpu() {}' > lib/Target/GPU/cute_target.cpp
  echo '#include <cuda.h>' > runtime/choreo.h
  echo '// test' > tests/check/basic.co

  git add -A
  git commit -m "initial oss seed" >/dev/null
  git push public oss/main:main >/dev/null 2>&1

  git checkout main >/dev/null 2>&1
}

# -------- test cases --------

test_scan_clean_tree() {
  cd "$WORK"
  # Scan the oss/main branch tree from main (scripts live on main only)
  local rc=0
  scripts/oss/oss-scan.sh --tree oss/main -k scripts/oss/os_kw.txt >/dev/null 2>&1 || rc=$?
  assert_exit_code "clean tree scan" "0" "$rc"
}

test_scan_detects_keyword_in_file() {
  cd "$WORK"
  # Stage a file with a keyword on main and scan (branch doesn't matter for --staged)
  echo '// This mentions GCU hardware' > runtime/bad_comment.h
  git add runtime/bad_comment.h

  local rc=0
  scripts/oss/oss-scan.sh --staged -k scripts/oss/os_kw.txt >/dev/null 2>&1 || rc=$?

  git reset HEAD -- runtime/bad_comment.h >/dev/null
  rm -f runtime/bad_comment.h

  assert_exit_code "keyword in staged file" "1" "$rc"
}

test_scan_detects_keyword_in_path() {
  cd "$WORK"
  # Stage a file at a path that contains a keyword
  mkdir -p lib/dtu_stuff
  echo 'clean content' > lib/dtu_stuff/oops.cpp
  git add lib/dtu_stuff/oops.cpp

  local rc=0
  scripts/oss/oss-scan.sh --staged -k scripts/oss/os_kw.txt >/dev/null 2>&1 || rc=$?

  git reset HEAD -- lib/dtu_stuff/oops.cpp >/dev/null
  rm -rf lib/dtu_stuff

  assert_exit_code "keyword in path" "1" "$rc"
}

test_scan_detects_keyword_in_commit_message() {
  cd "$WORK"

  # Create a commit with a sensitive keyword in the message
  echo 'safe content' > lib/temp_file.cpp
  git add lib/temp_file.cpp
  git commit -m "fix gcu hardware init" >/dev/null

  local commit_sha
  commit_sha="$(git rev-parse HEAD)"

  local rc=0
  scripts/oss/oss-scan.sh --diff "$commit_sha" -k scripts/oss/os_kw.txt >/dev/null 2>&1 || rc=$?

  # Cleanup: revert the commit
  git reset --hard HEAD~1 >/dev/null

  assert_exit_code "keyword in commit message" "1" "$rc"
}

test_scan_dir_mode() {
  cd "$WORK"
  local testdir="$TMPBASE/scan_dir_test"
  mkdir -p "$testdir/subdir"
  echo 'clean code' > "$testdir/clean.cpp"
  echo '// eflame internal' > "$testdir/subdir/leak.h"

  local rc=0
  scripts/oss/oss-scan.sh --dir "$testdir" -k scripts/oss/os_kw.txt >/dev/null 2>&1 || rc=$?
  rm -rf "$testdir"

  assert_exit_code "keyword in dir scan" "1" "$rc"
}

test_push_filters_gcu_files() {
  cd "$WORK"

  # Create a mixed commit: shared + GCU changes
  echo 'int main() { return 1; }' > lib/shared.cpp
  echo 'void gcu_v2() {}' > lib/Target/GCU/gcu_target.cpp
  git add -A
  git commit -m "update shared and GCU code" >/dev/null

  local commit_sha
  commit_sha="$(git rev-parse HEAD)"

  # Push to oss/main (should include shared.cpp, exclude GCU)
  scripts/oss/oss-push.sh --no-scan -b oss/main "$commit_sha" >/dev/null 2>&1

  # Verify on oss/main branch
  git checkout oss/main >/dev/null 2>&1

  assert_file_contains "shared.cpp updated" lib/shared.cpp "return 1" || { git checkout main >/dev/null 2>&1; return 1; }
  assert_file_not_exists "GCU file absent" lib/Target/GCU/gcu_target.cpp || { git checkout main >/dev/null 2>&1; return 1; }

  git checkout main >/dev/null 2>&1
}

test_push_skips_all_excluded_commit() {
  cd "$WORK"

  # Create a commit touching only GCU files
  echo '// updated' >> lib/Target/GCU/gcu_target.cpp
  git add -A
  git commit -m "gcu only change" >/dev/null

  local commit_sha
  commit_sha="$(git rev-parse HEAD)"

  local output
  output="$(scripts/oss/oss-push.sh --no-scan -b oss/main "$commit_sha" 2>&1)"

  echo "$output" | grep -q "SKIP" || {
    echo "  ASSERT FAILED: expected SKIP for all-excluded commit"
    return 1
  }
}

test_push_blocks_keyword_in_content() {
  cd "$WORK"

  # Create a commit with a keyword in a shared file
  echo '// TODO: port from eflame SDK' > lib/shared_note.cpp
  git add lib/shared_note.cpp
  git commit -m "add shared note" >/dev/null

  local commit_sha
  commit_sha="$(git rev-parse HEAD)"

  local rc=0
  scripts/oss/oss-push.sh -b oss/main "$commit_sha" >/dev/null 2>&1 || rc=$?

  # Should fail due to keyword
  assert_exit_code "keyword blocks push" "1" "$rc" || return 1

  # Cleanup
  git reset --hard HEAD~1 >/dev/null 2>&1 || true
}

test_push_dry_run() {
  cd "$WORK"

  echo '// new feature' > lib/feature.cpp
  git add lib/feature.cpp
  git commit -m "add feature" >/dev/null

  local commit_sha
  commit_sha="$(git rev-parse HEAD)"

  local output
  output="$(scripts/oss/oss-push.sh -n -b oss/main "$commit_sha" 2>&1)"

  echo "$output" | grep -q "dry-run" || {
    echo "  ASSERT FAILED: expected dry-run output"
    return 1
  }

  # Verify oss/main branch was NOT changed (still at same tip)
  local oss_tip_before oss_tip_after
  oss_tip_before="$(git rev-parse oss/main)"

  # Run again in dry-run and check tip didn't change
  scripts/oss/oss-push.sh -n -b oss/main "$commit_sha" >/dev/null 2>&1 || true
  oss_tip_after="$(git rev-parse oss/main)"

  assert_eq "oss branch unchanged in dry-run" "$oss_tip_before" "$oss_tip_after"
}

test_pull_cherry_picks_back() {
  cd "$WORK"
  git checkout oss/main >/dev/null 2>&1

  # Create a community contribution on oss/main
  echo '// community fix' > lib/community_fix.cpp
  git add lib/community_fix.cpp
  git commit -m "community: fix edge case" >/dev/null

  local oss_commit
  oss_commit="$(git rev-parse HEAD)"

  git checkout main >/dev/null 2>&1

  # Pull it back
  scripts/oss/oss-pull.sh -b oss/main -t main "$oss_commit" >/dev/null 2>&1

  assert_file_exists "community fix on main" lib/community_fix.cpp || return 1
  assert_file_contains "fix content" lib/community_fix.cpp "community fix" || return 1
}

test_push_range() {
  cd "$WORK"

  # Create two commits
  echo '// range test 1' > lib/range1.cpp
  git add lib/range1.cpp
  git commit -m "range commit 1" >/dev/null

  echo '// range test 2' > lib/range2.cpp
  git add lib/range2.cpp
  git commit -m "range commit 2" >/dev/null

  local range_start range_end
  range_end="$(git rev-parse HEAD)"
  range_start="$(git rev-parse HEAD~2)"

  scripts/oss/oss-push.sh --no-scan -b oss/main --range "${range_start}..${range_end}" >/dev/null 2>&1

  git checkout oss/main >/dev/null 2>&1
  assert_file_exists "range1.cpp on oss" lib/range1.cpp || { git checkout main >/dev/null 2>&1; return 1; }
  assert_file_exists "range2.cpp on oss" lib/range2.cpp || { git checkout main >/dev/null 2>&1; return 1; }
  git checkout main >/dev/null 2>&1
}

test_scan_range_messages() {
  cd "$WORK"

  local before_sha
  before_sha="$(git rev-parse HEAD)"

  echo 'x' > lib/tmp1.cpp; git add .; git commit -m "clean message one" >/dev/null
  echo 'y' > lib/tmp2.cpp; git add .; git commit -m "fix draco regression" >/dev/null
  echo 'z' > lib/tmp3.cpp; git add .; git commit -m "clean message three" >/dev/null

  local rc=0
  scripts/oss/oss-scan.sh --range "${before_sha}..HEAD" -k scripts/oss/os_kw.txt >/dev/null 2>&1 || rc=$?

  # Cleanup
  git reset --hard "$before_sha" >/dev/null

  assert_exit_code "keyword in commit message range" "1" "$rc"
}

test_scan_detects_nonascii_in_staged() {
  cd "$WORK"
  # Stage a file with non-ASCII (Chinese characters)
  printf '// \xe4\xb8\xad\xe6\x96\x87\xe6\xb3\xa8\xe9\x87\x8a\n' > lib/nonascii.cpp
  git add lib/nonascii.cpp

  local rc=0
  scripts/oss/oss-scan.sh --staged -k scripts/oss/os_kw.txt >/dev/null 2>&1 || rc=$?

  git reset HEAD -- lib/nonascii.cpp >/dev/null
  rm -f lib/nonascii.cpp

  assert_exit_code "non-ASCII in staged" "1" "$rc"
}

test_scan_nonascii_in_dir() {
  cd "$WORK"
  local testdir="$TMPBASE/nonascii_dir_test"
  mkdir -p "$testdir"
  printf '// \xc3\xa9\xc3\xa0\xc3\xbc accent chars\n' > "$testdir/accents.h"
  echo 'clean ascii only' > "$testdir/clean.h"

  local rc=0
  scripts/oss/oss-scan.sh --dir "$testdir" -k scripts/oss/os_kw.txt >/dev/null 2>&1 || rc=$?
  rm -rf "$testdir"

  assert_exit_code "non-ASCII in dir" "1" "$rc"
}

test_push_preserves_author() {
  cd "$WORK"

  # Create a commit with a specific author
  echo '// authored feature' > lib/authored.cpp
  git add lib/authored.cpp
  GIT_AUTHOR_NAME="Jane Contributor" GIT_AUTHOR_EMAIL="jane@example.com" \
    git commit -m "feature from Jane" >/dev/null

  local commit_sha
  commit_sha="$(git rev-parse HEAD)"

  scripts/oss/oss-push.sh --no-scan -b oss/main "$commit_sha" >/dev/null 2>&1

  git checkout oss/main >/dev/null 2>&1

  local author
  author="$(git log -1 --format='%an <%ae>')"
  assert_eq "author preserved" "Jane Contributor <jane@example.com>" "$author" || { git checkout main >/dev/null 2>&1; return 1; }

  git checkout main >/dev/null 2>&1
}

test_scan_ghost_include() {
  cd "$WORK"
  # Create a commit that includes a GCU header from shared code
  echo '#include "Target/GCU/gcu_target.hpp"' > lib/shared_with_ghost.cpp
  git add lib/shared_with_ghost.cpp
  git commit -m "add ghost include" >/dev/null

  local commit_sha
  commit_sha="$(git rev-parse HEAD)"

  local rc=0
  scripts/oss/oss-scan.sh --diff "$commit_sha" -k scripts/oss/os_kw.txt >/dev/null 2>&1 || rc=$?

  git reset --hard HEAD~1 >/dev/null

  assert_exit_code "ghost include detected" "1" "$rc"
}

test_scan_coupled_changes() {
  cd "$WORK"
  # Create a commit modifying both shared and GCU files
  echo '// updated shared' >> lib/shared.cpp
  echo '// updated gcu' >> lib/Target/GCU/gcu_target.cpp
  git add -A
  git commit -m "coupled change" >/dev/null

  local commit_sha
  commit_sha="$(git rev-parse HEAD)"

  local output
  output="$(scripts/oss/oss-scan.sh --diff "$commit_sha" -k scripts/oss/os_kw.txt -v 2>&1 || true)"

  git reset --hard HEAD~1 >/dev/null

  echo "$output" | grep -q "coupled" || {
    echo "  ASSERT FAILED: expected coupled-change warning"
    return 1
  }
}

test_pull_preserves_author() {
  cd "$WORK"
  git checkout oss/main >/dev/null 2>&1

  echo '// external dev fix' > lib/external_fix.cpp
  GIT_AUTHOR_NAME="External Dev" GIT_AUTHOR_EMAIL="ext@community.org" \
    git add lib/external_fix.cpp
  GIT_AUTHOR_NAME="External Dev" GIT_AUTHOR_EMAIL="ext@community.org" \
    git commit -m "community: external dev patch" >/dev/null

  local oss_commit
  oss_commit="$(git rev-parse HEAD)"

  git checkout main >/dev/null 2>&1

  scripts/oss/oss-pull.sh -b oss/main -t main "$oss_commit" >/dev/null 2>&1

  local author
  author="$(git log -1 --format='%an <%ae>')"
  assert_eq "pull author preserved" "External Dev <ext@community.org>" "$author"
}

# -------- main --------

echo "OSS Workflow Integration Tests"
echo "=============================="

setup_env
create_test_repos

run_test test_scan_clean_tree
run_test test_scan_detects_keyword_in_file
run_test test_scan_detects_keyword_in_path
run_test test_scan_detects_keyword_in_commit_message
run_test test_scan_dir_mode
run_test test_scan_detects_nonascii_in_staged
run_test test_scan_nonascii_in_dir
run_test test_scan_ghost_include
run_test test_scan_coupled_changes
run_test test_push_filters_gcu_files
run_test test_push_skips_all_excluded_commit
run_test test_push_blocks_keyword_in_content
run_test test_push_dry_run
run_test test_pull_cherry_picks_back
run_test test_pull_preserves_author
run_test test_push_range
run_test test_scan_range_messages
run_test test_push_preserves_author

echo ""
echo "=============================="
echo "Results: $PASS passed, $FAIL failed (of $TESTS_RUN)"
echo "=============================="

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
