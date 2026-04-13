#!/usr/bin/env bash
set -euo pipefail

# Cherry-pick commits from the oss/ branch back to main.
#
# Unlike oss-push, the public repo is NOT a subset of main -- it may
# have its own .gitignore, CI configs, README, skills, etc.  This
# script scans incoming commits for:
#
#   1. Files that collide with private-only paths (would overwrite
#      internal config).
#   2. Content that introduces keywords or non-ASCII into main.
#
# When violations are found, the script HALTS and reports them.
# The user must fix the violations (possibly by pushing a fix to
# oss/main first) before resuming the sync.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

EXCLUDE_FILE="$SCRIPT_DIR/oss_exclude_paths.txt"
KW_FILE="$SCRIPT_DIR/os_kw.txt"
OSS_BRANCH="oss/main"
TARGET_BRANCH="main"
FORCE=0
SCAN_ONLY=0

# Files that are managed separately per repo -- incoming changes to
# these must be reviewed, not blindly cherry-picked.
CONFLICT_FILES=(.gitignore .gitmodules .gitattributes .clang-format
                Makefile CMakeLists.txt README.md LICENSE mkdocs.yml)

usage() {
  cat <<'EOF'
Usage: oss-pull.sh [options] <commit> [<commit>...]
       oss-pull.sh [options] --range <from>..<to>

Cherry-pick commits from the oss/ branch back to main, with violation
scanning. Halts on any issue that could break the private repo.

Options:
  -b <branch>   Source branch (default: oss/main)
  -t <branch>   Target branch (default: main)
  --scan-only   Scan commits but do not cherry-pick (report only)
  --force       Apply even if violations are found (use with caution)
  --range <r>   Expand a revision range via git rev-list --reverse
  -h            Show help

Violations detected:
  CONFLICT-FILE  Commit modifies a repo-config file managed separately
  PRIVATE-PATH   Commit creates files in a private-only directory
  KEYWORD        Commit introduces a forbidden keyword into main

When violations are found (without --force):
  1. Script halts and shows what needs fixing
  2. Fix on oss/main first if needed (push fix to public)
  3. Re-run this script after the fix
EOF
}

COMMITS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b)          OSS_BRANCH="$2"; shift 2 ;;
    -t)          TARGET_BRANCH="$2"; shift 2 ;;
    --scan-only) SCAN_ONLY=1; shift ;;
    --force)     FORCE=1; shift ;;
    --range)
      mapfile -t range_commits < <(git -C "$REPO_ROOT" rev-list --reverse "$2")
      COMMITS+=("${range_commits[@]}")
      shift 2 ;;
    -h)          usage; exit 0 ;;
    -*)          echo "Error: unknown option $1" >&2; usage; exit 2 ;;
    *)           COMMITS+=("$1"); shift ;;
  esac
done

if [[ ${#COMMITS[@]} -eq 0 ]]; then
  echo "Error: no commits specified" >&2
  usage
  exit 2
fi

cd "$REPO_ROOT"

# -------- load exclude patterns for private-path detection --------

PRIVATE_PREFIXES=()
PRIVATE_EXACT=()

if [[ -f "$EXCLUDE_FILE" ]]; then
  while IFS= read -r pat; do
    [[ -z "$pat" || "$pat" =~ ^[[:space:]]*# ]] && continue
    if [[ "$pat" == */ || "$pat" == *'/*' ]]; then
      local_prefix="${pat%\*}"
      local_prefix="${local_prefix%/}/"
      PRIVATE_PREFIXES+=("$local_prefix")
    elif [[ "$pat" != *'*'* && "$pat" != *'?'* ]]; then
      PRIVATE_EXACT+=("$pat")
    fi
  done < "$EXCLUDE_FILE"
fi

is_private() {
  local fpath="$1"
  for pfx in "${PRIVATE_PREFIXES[@]}"; do
    [[ "$fpath" == "$pfx"* ]] && return 0
  done
  for ex in "${PRIVATE_EXACT[@]}"; do
    [[ "$fpath" == "$ex" ]] && return 0
  done
  return 1
}

is_conflict_file() {
  local fpath="$1"
  for cf in "${CONFLICT_FILES[@]}"; do
    [[ "$fpath" == "$cf" ]] && return 0
  done
  return 1
}

# -------- load keywords for content scan --------
KW_PATTERNS=()
if [[ -f "$KW_FILE" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    KW_PATTERNS+=("$line")
  done < "$KW_FILE"
fi

# -------- pre-scan all commits --------

echo "=== Scanning ${#COMMITS[@]} commit(s) for violations ==="
echo ""

TOTAL=${#COMMITS[@]}
TOTAL_VIOLATIONS=0
BLOCKED_AT=""
VIOLATION_REPORT=""

for commit in "${COMMITS[@]}"; do
  git rev-parse --verify "$commit^{commit}" >/dev/null 2>&1 \
    || { echo "WARNING: cannot resolve '$commit', skipping" >&2; continue; }

  short="$(git rev-parse --short "$commit")"
  msg="$(git log -1 --format='%s' "$commit" | head -c 60)"
  mapfile -t files < <(git diff-tree --no-commit-id -r --name-only "$commit")

  violations=()

  for f in "${files[@]}"; do
    [[ -z "$f" ]] && continue

    if is_private "$f"; then
      violations+=("PRIVATE-PATH: $f")
    fi

    if is_conflict_file "$f"; then
      violations+=("CONFLICT-FILE: $f (managed separately in private repo)")
    fi
  done

  # Content scan: check diff for keywords
  if [[ ${#KW_PATTERNS[@]} -gt 0 ]]; then
    diff_content="$(git diff "$commit^..$commit" -- | grep -E '^\+' | grep -v '^+++' || true)"
    for kw in "${KW_PATTERNS[@]}"; do
      if echo "$diff_content" | grep -qiE "$kw"; then
        hits="$(echo "$diff_content" | grep -ciE "$kw" || true)"
        violations+=("KEYWORD: '$kw' found ($hits hit(s) in diff)")
      fi
    done
  fi

  # Non-ASCII check on added lines
  diff_added="$(git diff "$commit^..$commit" -- | grep -E '^\+' | grep -v '^+++' || true)"
  if echo "$diff_added" | grep -Pq '[^\x00-\x7F]' 2>/dev/null; then
    non_ascii_count="$(echo "$diff_added" | grep -Pc '[^\x00-\x7F]' 2>/dev/null || echo "?")"
    violations+=("NON-ASCII: $non_ascii_count line(s) with non-ASCII characters")
  fi

  if [[ ${#violations[@]} -gt 0 ]]; then
    echo "VIOLATION $short $msg"
    for v in "${violations[@]}"; do
      echo "  $v"
    done
    TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + ${#violations[@]}))
    if [[ -z "$BLOCKED_AT" ]]; then
      BLOCKED_AT="$short"
    fi
    VIOLATION_REPORT+="$short: ${#violations[@]} issue(s)"$'\n'
  else
    echo "CLEAN     $short $msg"
  fi
done

echo ""

# -------- decision gate --------

if [[ $TOTAL_VIOLATIONS -gt 0 && $FORCE -eq 0 ]]; then
  echo "========================================"
  echo "BLOCKED: $TOTAL_VIOLATIONS violation(s) found."
  echo ""
  echo "Sync is halted to protect the private repo."
  echo ""
  echo "Options:"
  echo "  1. Fix the violations on oss/main and push the fix to public"
  echo "  2. Re-run with --force to apply anyway (use with caution)"
  echo "  3. Cherry-pick manually with: git cherry-pick --no-commit <sha>"
  echo ""
  echo "Violation summary:"
  echo "$VIOLATION_REPORT"
  exit 1
fi

if [[ $SCAN_ONLY -eq 1 ]]; then
  if [[ $TOTAL_VIOLATIONS -eq 0 ]]; then
    echo "CLEAN: all $TOTAL commit(s) passed scan."
  fi
  exit 0
fi

# -------- apply commits --------

if [[ $TOTAL_VIOLATIONS -gt 0 ]]; then
  echo "WARNING: applying with --force despite $TOTAL_VIOLATIONS violation(s)."
  echo ""
fi

CURRENT="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD)"
if [[ "$CURRENT" != "$TARGET_BRANCH" ]]; then
  echo "Switching to '$TARGET_BRANCH'..."
  git checkout "$TARGET_BRANCH"
fi

cleanup() {
  if [[ "$(git symbolic-ref --short HEAD 2>/dev/null || true)" != "$CURRENT" ]]; then
    echo "Returning to '$CURRENT'..."
    git checkout "$CURRENT" 2>/dev/null || true
  fi
}
trap cleanup EXIT

PULLED=0
FAILED=0

for commit in "${COMMITS[@]}"; do
  git rev-parse --verify "$commit^{commit}" >/dev/null 2>&1 \
    || { echo "ERROR: cannot resolve '$commit'" >&2; FAILED=$((FAILED+1)); continue; }

  short="$(git rev-parse --short "$commit")"
  echo "Cherry-picking $short..."

  if git cherry-pick --no-commit "$commit"; then
    orig_author="$(git log -1 --format='%an <%ae>' "$commit")"
    orig_date="$(git log -1 --format='%ai' "$commit")"
    orig_msg="$(git log -1 --format=%B "$commit")"
    # Strip AI tool markers
    clean_msg="$(echo "$orig_msg" | sed '/^Made-with:/d; /^Generated-by:/d')"
    GIT_AUTHOR_DATE="$orig_date" git commit \
      --author="$orig_author" \
      -m "$clean_msg"
    new_sha="$(git rev-parse --short HEAD)"
    echo "OK $short -> $new_sha (author: $orig_author)"
    PULLED=$((PULLED+1))
  else
    echo "ERROR $short: cherry-pick conflict."
    echo "  Resolve: edit files, git add, git cherry-pick --continue"
    echo "  Or abort: git cherry-pick --abort"
    FAILED=$((FAILED+1))
    break
  fi
done

echo ""
echo "Summary: $PULLED pulled, $FAILED failed (of $TOTAL)"
if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
