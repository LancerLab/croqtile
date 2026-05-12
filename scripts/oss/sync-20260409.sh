#!/usr/bin/env bash
set -euo pipefail

# Batch sync from main to oss/main.
# Sync point: 836bbd64 (2026-04-02, "speed-up skips")
# Range: 836bbd64..main (35 commits total)
#
# Each commit is classified as CLEAN, FIXABLE, MANUAL, or SKIP.
# They MUST be applied in topological order (as on main).
#
# Usage:
#   bash scripts/oss/sync-20260409.sh --dry-run    # preview
#   bash scripts/oss/sync-20260409.sh               # execute
#   bash scripts/oss/sync-20260409.sh --auto-only   # stop before first MANUAL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
DRY_RUN=0
AUTO_ONLY=0
case "${1:-}" in
  --dry-run)   DRY_RUN=1 ;;
  --auto-only) AUTO_ONLY=1 ;;
esac

cd "$REPO_ROOT"

SYNC_BASE="836bbd64"
OSS_BRANCH="oss/main"

# Ordered commit list: SHA TYPE
# Types: SKIP, CLEAN, FIXABLE, MANUAL
read -r -d '' COMMIT_LIST << 'COMMITEOF' || true
4e68c8d2 SKIP
12955bab FIXABLE
ede4098d SKIP
21ed09bb CLEAN
0ae04902 MANUAL
6cbf0a4a CLEAN
108dbc36 CLEAN
428caf63 CLEAN
9c60d2a3 SKIP
27738359 SKIP
a5a1046f CLEAN
3d7b11d4 CLEAN
710a32d0 SKIP
4b25f079 SKIP
6d52adf4 CLEAN
8937953e CLEAN
0b020d9e MANUAL
79f9c461 CLEAN
8d170764 MANUAL
e38eccfd FIXABLE
a0426c3b CLEAN
d78d0c09 CLEAN
9c732055 SKIP
5324cfb3 CLEAN
983e86dd CLEAN
0fb0f409 CLEAN
7c48b154 CLEAN
6e5c6141 MANUAL
674c69c9 CLEAN
c262ea58 CLEAN
cad1e3c9 SKIP
ad9da7e0 CLEAN
5046e4a8 CLEAN
1be0eabb CLEAN
c5a73843 SKIP
COMMITEOF

total=0; skip=0; clean=0; fixable=0; manual=0
while IFS=' ' read -r sha type; do
  [[ -z "$sha" ]] && continue
  total=$((total + 1))
  case "$type" in
    SKIP)    skip=$((skip + 1)) ;;
    CLEAN)   clean=$((clean + 1)) ;;
    FIXABLE) fixable=$((fixable + 1)) ;;
    MANUAL)  manual=$((manual + 1)) ;;
  esac
done <<< "$COMMIT_LIST"

echo "=== Choreo -> OSS Sync Batch ($(date +%Y-%m-%d)) ==="
echo "Sync base: $SYNC_BASE"
echo "Target:    $OSS_BRANCH"
echo "Commits:   $total total ($clean clean, $fixable fixable, $manual manual, $skip skip)"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
  echo "--- DRY RUN (topological order) ---"
  echo ""
  while IFS=' ' read -r sha type; do
    [[ -z "$sha" ]] && continue
    short=$(git rev-parse --short "$sha" 2>/dev/null || echo "$sha")
    msg=$(git log -1 --format="%s" "$sha" 2>/dev/null || echo "???")
    author=$(git log -1 --format="%an" "$sha" 2>/dev/null || echo "???")
    case "$type" in
      SKIP)    printf "  SKIP    %-10s [%-20s] %s\n" "$short" "$author" "$msg" ;;
      CLEAN)   printf "  CLEAN   %-10s [%-20s] %s\n" "$short" "$author" "$msg" ;;
      FIXABLE) printf "  FIXABLE %-10s [%-20s] %s\n" "$short" "$author" "$msg" ;;
      MANUAL)  printf "  MANUAL  %-10s [%-20s] %s\n" "$short" "$author" "$msg" ;;
    esac
  done <<< "$COMMIT_LIST"
  echo ""
  echo "Run without --dry-run to execute."
  exit 0
fi

# --- Execution mode ---

is_excluded() {
  local fpath="$1"
  case "$fpath" in
    lib/Target/GCU/*|tests/gcu/*|benchmark/*|Documents/internal/*|Documents/Documentation/target/*|Documents/GPU-Examples/*|extern/*|scripts/*.sh|scripts/*.md|scripts/hooks/*|scripts/oss/*|samples/*|.gitlab-ci.yml|.gitlab/*|runtime/catz/*|.gitignore|.gitmodules|.gitattributes|.vscode/*|results.tsv|.claude/*|.codex/*|.github/skills/*|.cursor/*|performance/*|AGENTS.md|.clang-format)
      return 0 ;;
    *) return 1 ;;
  esac
}

fix_nonascii() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  sed -i 's/\xe2\x80\x94/--/g' "$f" 2>/dev/null || true
  sed -i 's/\xe2\x80\x93/-/g' "$f" 2>/dev/null || true
  sed -i 's/\xe2\x86\x92/->/g' "$f" 2>/dev/null || true
  sed -i 's/\xe2\x86\x90/<-/g' "$f" 2>/dev/null || true
  sed -i 's/\xe2\x80\xa6/.../g' "$f" 2>/dev/null || true
  sed -i 's/\xc3\x97/x/g' "$f" 2>/dev/null || true
}

# Copy tools to survive branch switch
TOOL_TMPDIR="$(mktemp -d)"
cp scripts/oss/oss-push.sh scripts/oss/oss-scan.sh scripts/oss/os_kw.txt \
   scripts/oss/oss_exclude_paths.txt "$TOOL_TMPDIR/"
chmod +x "$TOOL_TMPDIR"/*.sh

ORIG_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD)

cleanup() {
  local cur=$(git symbolic-ref --short HEAD 2>/dev/null || true)
  if [[ "$cur" != "$ORIG_BRANCH" ]]; then
    echo "Returning to $ORIG_BRANCH..."
    git checkout "$ORIG_BRANCH" 2>/dev/null || true
  fi
  rm -rf "$TOOL_TMPDIR"
}
trap cleanup EXIT

git checkout "$OSS_BRANCH" 2>/dev/null

pushed=0; skipped=0; failed=0; stopped=0

while IFS=' ' read -r sha type; do
  [[ -z "$sha" ]] && continue
  short=$(git rev-parse --short "$sha" 2>/dev/null || echo "$sha")
  msg=$(git log -1 --format="%s" "$sha" 2>/dev/null || echo "???")
  author=$(git log -1 --format='%an <%ae>' "$sha")
  adate=$(git log -1 --format='%ai' "$sha")

  case "$type" in
    SKIP)
      echo "SKIP    $short: $msg"
      skipped=$((skipped + 1))
      continue
      ;;

    MANUAL)
      if [[ $AUTO_ONLY -eq 1 ]]; then
        echo ""
        echo "STOPPED before MANUAL commit $short: $msg"
        echo "  Author: $author"
        echo "  This commit needs human review. Handle it manually, then re-run."
        stopped=$((stopped + 1))
        break
      fi
      echo ""
      echo "=== MANUAL $short: $msg ==="
      echo "  Author: $author"
      echo "  This commit has keyword violations in shared code."
      echo "  Please handle manually:"
      echo "    git cherry-pick --no-commit $sha"
      echo "    # fix violations in the staged files"
      echo "    bash $TOOL_TMPDIR/oss-scan.sh --staged -k $TOOL_TMPDIR/os_kw.txt"
      echo "    GIT_AUTHOR_DATE='$adate' git commit --author='$author' -m '<sanitized msg>'"
      echo ""
      echo "  Waiting... press Enter when done, or type 'skip' to skip this commit:"
      read -r response
      if [[ "$response" == "skip" ]]; then
        echo "  Skipped."
        skipped=$((skipped + 1))
      else
        echo "  Assuming manual commit was completed."
        pushed=$((pushed + 1))
      fi
      continue
      ;;
  esac

  # CLEAN or FIXABLE -- automated processing
  echo -n "PUSH    $short: $msg ... "

  # Get included files
  included_files=()
  for f in $(git diff-tree --no-commit-id -r --name-only "$sha"); do
    if ! is_excluded "$f"; then
      included_files+=("$f")
    fi
  done

  if [[ ${#included_files[@]} -eq 0 ]]; then
    echo "SKIP (all files excluded)"
    skipped=$((skipped + 1))
    continue
  fi

  # Generate filtered patch
  patch_file="$(mktemp)"
  git diff --binary "${sha}^" "$sha" -- "${included_files[@]}" > "$patch_file"

  if [[ ! -s "$patch_file" ]]; then
    echo "SKIP (empty patch)"
    rm -f "$patch_file"
    skipped=$((skipped + 1))
    continue
  fi

  # Apply patch
  if ! git apply --index --3way "$patch_file" 2>/dev/null; then
    echo "FAIL (patch conflict)"
    rm -f "$patch_file"
    git reset --hard HEAD >/dev/null 2>&1
    failed=$((failed + 1))
    continue
  fi
  rm -f "$patch_file"

  # Auto-fix non-ASCII for FIXABLE commits
  if [[ "$type" == "FIXABLE" ]]; then
    for f in "${included_files[@]}"; do
      fix_nonascii "$f"
      git add "$f" 2>/dev/null || true
    done
  fi

  # Keyword + non-ASCII scan
  local_rc=0
  OSS_SCAN_REPO_ROOT="$REPO_ROOT" "$TOOL_TMPDIR/oss-scan.sh" --staged \
    -k "$TOOL_TMPDIR/os_kw.txt" >/dev/null 2>&1 || local_rc=$?

  if [[ $local_rc -ne 0 ]]; then
    echo "FAIL (scan violation)"
    git reset --hard HEAD >/dev/null 2>&1
    failed=$((failed + 1))
    continue
  fi

  # Commit with original author
  GIT_AUTHOR_DATE="$adate" git commit \
    --author="$author" \
    -m "${msg}

(cherry picked from ${sha} on main)" >/dev/null 2>&1

  new_sha=$(git rev-parse --short HEAD)
  echo "OK -> $new_sha [$author]"
  pushed=$((pushed + 1))

done <<< "$COMMIT_LIST"

echo ""
echo "=== Summary ==="
echo "Pushed:  $pushed"
echo "Skipped: $skipped"
echo "Failed:  $failed"
if [[ $stopped -gt 0 ]]; then
  echo "Stopped: at manual commit (re-run after handling it)"
fi
echo ""
echo "Next steps:"
echo "  1. Handle any MANUAL/FAILED commits"
echo "  2. bash scripts/oss/oss-scan.sh --tree oss/main   # final safety check"
echo "  3. git push oss-shadow oss/main:main               # publish"
