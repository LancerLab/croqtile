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

CONFLICT_PREFIXES=()
CONFLICT_EXACT=()

if [[ -f "$CONFLICT_FILE" ]]; then
  while IFS= read -r pat; do
  [[ -z "$pat" || "$pat" =~ ^[[:space:]]*# ]] && continue
  if [[ "$pat" == */ || "$pat" == *'/*' ]]; then
    local_prefix="${pat%\*}"
    local_prefix="${local_prefix%/}/"
    CONFLICT_PREFIXES+=("$local_prefix")
  elif [[ "$pat" != *'*'* && "$pat" != *'?'* ]]; then
    CONFLICT_EXACT+=("$pat")
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
  for pfx in "${CONFLICT_PREFIXES[@]}"; do
  [[ "$fpath" == "$pfx"* ]] && return 0
  done
  for ex in "${CONFLICT_EXACT[@]}"; do
  [[ "$fpath" == "$ex" ]] && return 0
  done
  return 1
}

# -------- pre-scan: delegate to oss-pull-scan.sh --------
PULL_SCAN_CMD="$SCRIPT_DIR/oss-pull-scan.sh"
TOTAL=${#COMMITS[@]}

scan_rc=0
"$PULL_SCAN_CMD" -b "$OSS_BRANCH" -e "$EXCLUDE_FILE" "${COMMITS[@]}" || scan_rc=$?

if [[ $SCAN_ONLY -eq 1 ]]; then
  exit $scan_rc
fi

if [[ $scan_rc -ne 0 && $FORCE -eq 0 ]]; then
  echo ""
  echo "Sync halted. Fix violations on oss/main first, or use --force."
  exit 1
fi

# -------- apply commits --------

if [[ $scan_rc -ne 0 ]]; then
  echo "WARNING: applying with --force despite violations."
  echo ""
fi

CURRENT="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD)"
if [[ "$CURRENT" != "$TARGET_BRANCH" ]]; then
  echo "Switching to '$TARGET_BRANCH'..."
  git checkout "$TARGET_BRANCH"
fi

# Stash any uncommitted work so cherry-pick doesn't clobber it
STASH_SHA=""
if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet HEAD 2>/dev/null; then
  STASH_SHA="$(git stash create 'oss-pull stash' 2>/dev/null || true)"
  if [[ -n "$STASH_SHA" ]]; then
  git stash store -m "oss-pull stash" "$STASH_SHA" >/dev/null 2>&1 || true
  git reset --hard HEAD >/dev/null 2>&1
  echo "Stashed local changes (will restore after pull)."
  fi
fi

cleanup() {
  if [[ -n "$STASH_SHA" ]]; then
  echo "Restoring stashed local changes..."
  git stash pop --index 2>/dev/null || git stash pop 2>/dev/null || \
    echo "WARNING: could not restore stash -- run: git stash pop" >&2
  fi
  if [[ "$(git symbolic-ref --short HEAD 2>/dev/null || true)" != "$CURRENT" ]]; then
  echo "Returning to '$CURRENT'..."
  git checkout "$CURRENT" 2>/dev/null || true
  fi
}
trap cleanup EXIT

PULLED=0
SKIPPED=0
FAILED=0

for commit in "${COMMITS[@]}"; do
  git rev-parse --verify "$commit^{commit}" >/dev/null 2>&1 \
  || { echo "ERROR: cannot resolve '$commit'" >&2; FAILED=$((FAILED+1)); continue; }

  short="$(git rev-parse --short "$commit")"
  echo "Cherry-picking $short..."

  # Cherry-pick without committing so we can strip excluded-path changes
  # before they land on main.  Use || true: conflicts on excluded files
  # are expected and resolved below.
  git cherry-pick --no-commit "$commit" 2>/dev/null || true

  # Unstage/revert every change (add, modify, delete, conflict) on
  # excluded paths.  Deletions of excluded files come from oss/main
  # housekeeping and must never propagate to main.
  mapfile -t staged_files   < <(git diff --cached --name-only HEAD 2>/dev/null)
  mapfile -t conflict_files < <(git diff --name-only --diff-filter=U 2>/dev/null)

  declare -A seen_excl=()
  for f in "${staged_files[@]}" "${conflict_files[@]}"; do
  [[ -z "$f" ]] && continue
  if is_private "$f" || is_conflict_file "$f"; then
    seen_excl["$f"]=1
  fi
  done

  for f in "${!seen_excl[@]}"; do
  git reset HEAD -- "$f" >/dev/null 2>&1 || true
  if git cat-file -e "HEAD:$f" 2>/dev/null; then
    git checkout HEAD -- "$f" 2>/dev/null || true
  else
    rm -f "$f" 2>/dev/null || true
  fi
  done

  # Fail if any conflicts remain on included (public) files
  mapfile -t remaining_conflicts < <(git diff --name-only --diff-filter=U 2>/dev/null)
  if [[ ${#remaining_conflicts[@]} -gt 0 && -n "${remaining_conflicts[0]}" ]]; then
  echo "ERROR $short: conflicts in public files:"
  printf '    %s\n' "${remaining_conflicts[@]}"
  echo "  Resolve manually, then: git add <files> && git cherry-pick --continue"
  git cherry-pick --abort 2>/dev/null || git reset --hard HEAD 2>/dev/null || true
  FAILED=$((FAILED+1))
  break
  fi

  # Skip if nothing public changed (oss/main-only housekeeping commit)
  if git diff --cached --quiet HEAD 2>/dev/null; then
  echo "SKIP $short: no public changes after stripping excluded files"
  git reset --hard HEAD >/dev/null 2>&1
  SKIPPED=$((SKIPPED+1))
  continue
  fi

  orig_author="$(git log -1 --format='%an <%ae>' "$commit")"
  orig_date="$(git log -1 --format='%ai' "$commit")"
  orig_msg="$(git log -1 --format=%B "$commit")"
  clean_msg="$(echo "$orig_msg" | sed '/^Made-with:/d; /^Generated-by:/d')"
  GIT_AUTHOR_DATE="$orig_date" git commit \
  --author="$orig_author" \
  -m "$clean_msg"
  new_sha="$(git rev-parse --short HEAD)"
  echo "OK $short -> $new_sha (author: $orig_author)"
  PULLED=$((PULLED+1))
done

echo ""
echo "Summary: $PULLED pulled, $SKIPPED skipped, $FAILED failed (of $TOTAL)"
if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
