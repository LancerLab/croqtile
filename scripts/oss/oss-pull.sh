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
# shellcheck source=oss-config.sh
source "$SCRIPT_DIR/oss-config.sh"

TARGET_BRANCH="$MAIN_BRANCH"
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
     oss-pull.sh [options] --last
     oss-pull.sh [options] --catchup

Cherry-pick commits from the oss/ branch back to main, with violation
scanning. Halts on any issue that could break the private repo.

If the public remote or oss/main branch is missing, the script will
set them up automatically (equivalent to running oss-setup.sh).

Options:
  -b <branch>   Source branch (default: oss/main)
  -t <branch>   Target branch (default: main)
  --last        Pull the most recent oss/main commit not yet on target
  --catchup     Pull ALL unpulled oss/main commits to target (oldest first)
  --scan-only   Scan commits but do not cherry-pick (report only)
  --force       Apply even if violations are found (use with caution)
  --range <r>   Expand a revision range via git rev-list --reverse
  -n            Dry run: scan only, do not apply
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
PULL_LAST=0
PULL_CATCHUP=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
  -b)          OSS_BRANCH="$2"; shift 2 ;;
  -t)          TARGET_BRANCH="$2"; shift 2 ;;
  --last)      PULL_LAST=1; shift ;;
  --catchup)   PULL_CATCHUP=1; shift ;;
  --scan-only) SCAN_ONLY=1; shift ;;
  --force)     FORCE=1; shift ;;
  -n)          DRY_RUN=1; SCAN_ONLY=1; shift ;;
  --range)
    mapfile -t range_commits < <(git -C "$REPO_ROOT" rev-list --reverse "$2")
    COMMITS+=("${range_commits[@]}")
    shift 2 ;;
  -h)          usage; exit 0 ;;
  -*)          echo "Error: unknown option $1" >&2; usage; exit 2 ;;
  *)           COMMITS+=("$1"); shift ;;
  esac
done

if [[ $PULL_LAST -eq 0 && $PULL_CATCHUP -eq 0 && ${#COMMITS[@]} -eq 0 ]]; then
  echo "Error: no commits specified (or use --last)" >&2
  usage
  exit 2
fi

cd "$REPO_ROOT"

# -------- auto-setup: ensure oss/main branch and remotes exist --------

ensure_remote() {
  local name="$1" url="$2"
  if ! git remote get-url "$name" >/dev/null 2>&1; then
  echo "Remote '$name' not found -- adding -> $url"
  git remote add "$name" "$url"
  git fetch "$name" 2>/dev/null || echo "  Warning: fetch from '$name' failed (network issue?)"
  fi
}

ensure_oss_branch() {
  if git show-ref --verify --quiet "refs/heads/$OSS_BRANCH"; then
  return 0
  fi

  echo "Branch '$OSS_BRANCH' not found -- setting up..."
  ensure_remote "$OSS_SHADOW_REMOTE" "$OSS_SHADOW_URL"
  ensure_remote "$PUBLIC_REMOTE" "$PUBLIC_URL"

  if git show-ref --verify --quiet "refs/remotes/$OSS_SHADOW_REMOTE/main"; then
  git fetch "$OSS_SHADOW_REMOTE"
  git branch "$OSS_BRANCH" "$OSS_SHADOW_REMOTE/main"
  git branch -u "$OSS_SHADOW_REMOTE/main" "$OSS_BRANCH"
  echo "  Created '$OSS_BRANCH' tracking $OSS_SHADOW_REMOTE/main"
  elif git show-ref --verify --quiet "refs/remotes/$PUBLIC_REMOTE/main"; then
  git branch "$OSS_BRANCH" "$PUBLIC_REMOTE/main"
  echo "  Created '$OSS_BRANCH' from $PUBLIC_REMOTE/main"
  else
  echo "Error: cannot find a remote oss branch to create '$OSS_BRANCH' from." >&2
  echo "  Tried: $OSS_SHADOW_REMOTE/main, $PUBLIC_REMOTE/main" >&2
  echo "  Check network connectivity or run: make oss-setup" >&2
  exit 1
  fi
}

ensure_oss_branch

# -------- --last mode: find the newest unpulled oss/main commit --------

if [[ $PULL_LAST -eq 1 ]]; then
  # Build set of oss/main SHAs already cherry-picked to main
  # (detected via cherry-pick trailers in main's commit messages).
  declare -A PULLED_SHAS=()
  while IFS= read -r trailer_sha; do
  [[ -z "$trailer_sha" ]] && continue
  full="$(git rev-parse --verify "$trailer_sha^{commit}" 2>/dev/null || true)"
  [[ -n "$full" ]] && PULLED_SHAS["$full"]=1
  PULLED_SHAS["$trailer_sha"]=1
  done < <(git log "$TARGET_BRANCH" --format=%B 2>/dev/null \
       | grep -oP '(?<=cherry picked from )\w+' || true)

  # Stash working tree so trial cherry-picks don't clobber local changes
  stash_needed=0
  if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet HEAD 2>/dev/null; then
  git stash push -q -m "oss-pull-last: save working tree"
  stash_needed=1
  fi

  # Walk oss/main from newest to oldest.  Skip:
  #   - commits that are cherry-picks FROM main (oss-push generated)
  #   - commits already pulled to main (by trailer)
  #   - commits whose content is already reflected on main (no net diff)
  found_last=""
  while IFS= read -r oss_sha; do
  # Skip oss-push-generated cherry-picks (trailer: "on main")
  if git log -1 --format=%B "$oss_sha" 2>/dev/null \
     | grep -qP 'cherry picked from \w+ on main'; then
    continue
  fi
  # Skip if already pulled
  [[ -n "${PULLED_SHAS[$oss_sha]:-}" ]] && continue
  # Skip if content already reflected (trial cherry-pick produces no diff)
  if git cherry-pick --no-commit "$oss_sha" 2>/dev/null; then
    if git diff --cached --quiet HEAD 2>/dev/null; then
    git reset --hard HEAD >/dev/null 2>&1
    continue
    fi
    git reset --hard HEAD >/dev/null 2>&1
  else
    git cherry-pick --abort 2>/dev/null || git reset --hard HEAD 2>/dev/null || true
  fi
  found_last="$oss_sha"
  break
  done < <(git rev-list "$OSS_BRANCH")

  # Restore working tree
  if [[ $stash_needed -eq 1 ]]; then
  git stash pop -q 2>/dev/null || true
  fi

  if [[ -z "$found_last" ]]; then
  echo "main is fully caught up with $OSS_BRANCH. Nothing to pull."
  exit 0
  fi

  short="$(git rev-parse --short "$found_last")"
  msg="$(git log -1 --format='%s' "$found_last" | head -c 60)"
  echo "Last unpulled commit: $short $msg"
  echo ""
  COMMITS+=("$found_last")
fi

# -------- --catchup mode: find ALL unpulled oss/main commits --------

if [[ $PULL_CATCHUP -eq 1 ]]; then
  declare -A PULLED_SHAS=()
  while IFS= read -r trailer_sha; do
  [[ -z "$trailer_sha" ]] && continue
  full="$(git rev-parse --verify "$trailer_sha^{commit}" 2>/dev/null || true)"
  [[ -n "$full" ]] && PULLED_SHAS["$full"]=1
  PULLED_SHAS["$trailer_sha"]=1
  done < <(git log "$TARGET_BRANCH" --format=%B 2>/dev/null \
       | grep -oP '(?<=cherry picked from )\w+' || true)

  stash_needed=0
  if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet HEAD 2>/dev/null; then
  git stash push -q -m "oss-pull-catchup: save working tree"
  stash_needed=1
  fi

  found_all=()
  while IFS= read -r oss_sha; do
  if git log -1 --format=%B "$oss_sha" 2>/dev/null \
     | grep -qP 'cherry picked from \w+ on main'; then
    continue
  fi
  [[ -n "${PULLED_SHAS[$oss_sha]:-}" ]] && continue
  if git cherry-pick --no-commit "$oss_sha" 2>/dev/null; then
    if git diff --cached --quiet HEAD 2>/dev/null; then
    git reset --hard HEAD >/dev/null 2>&1
    continue
    fi
    git reset --hard HEAD >/dev/null 2>&1
  else
    git cherry-pick --abort 2>/dev/null || git reset --hard HEAD 2>/dev/null || true
  fi
  found_all+=("$oss_sha")
  done < <(git rev-list "$OSS_BRANCH")

  if [[ $stash_needed -eq 1 ]]; then
  git stash pop -q 2>/dev/null || true
  fi

  if [[ ${#found_all[@]} -eq 0 ]]; then
  echo "main is fully caught up with $OSS_BRANCH. Nothing to pull."
  exit 0
  fi

  # Reverse to oldest-first for correct cherry-pick ordering
  for ((i=${#found_all[@]}-1; i>=0; i--)); do
  COMMITS+=("${found_all[$i]}")
  done

  echo "Catchup: ${#COMMITS[@]} unpulled commit(s) from $OSS_BRANCH (oldest first)."
  for c in "${COMMITS[@]}"; do
  echo "  $(git rev-parse --short "$c") $(git log -1 --format='%s' "$c" | head -c 60)"
  done
  echo ""
fi

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
  # Strip (?i) prefix -- we use grep -i for case-insensitive matching
  line="${line#'(?i)'}"
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
