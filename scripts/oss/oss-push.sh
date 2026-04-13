#!/usr/bin/env bash
set -euo pipefail

# Cherry-pick commits from main to the oss/ branch, filtering out
# excluded paths and running a full oss-scan before committing.
#
# By default, commits are applied to the LOCAL oss/main branch only.
# The script does NOT push to any remote unless --push is given.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

EXCLUDE_FILE="$SCRIPT_DIR/oss_exclude_paths.txt"
KW_FILE="$SCRIPT_DIR/os_kw.txt"
OSS_BRANCH="oss/main"
DRY_RUN=0
SKIP_SCAN=0
DO_PUSH=0
PUSH_REMOTE="origin"

usage() {
  cat <<'EOF'
Usage: oss-push.sh [options] <commit> [<commit>...]
       oss-push.sh [options] --range <from>..<to>

Cherry-pick commits to the local oss/ branch, filtering excluded paths
and running a full oss-scan gate before each commit.

By default, changes stay LOCAL. Nothing is pushed to any remote.

Options:
  -b <branch>     Target branch (default: oss/main)
  -e <file>       Exclude-paths file
  -k <file>       Keyword file
  -n              Dry run: show what would happen without committing
  --no-scan       Skip keyword scan (use with caution)
  --push [remote] After all commits, push oss/main to remote (default: origin)
  --range <r>     Expand a revision range via git rev-list --reverse
  -h              Show help

Workflow:
  1. Cherry-pick each commit (path-filtered)
  2. Run oss-scan on staged changes (keyword, non-ASCII, ghost-ref)
  3. Run oss-scan --tree on the full oss/main tree after commit
  4. If scan fails -> reset and report error
  5. After all commits, show summary
  6. Only push if --push is given; otherwise prompt

Each commit's original author, date, and message are preserved, with a
trailer noting the source SHA. Made-with:/Generated-by: trailers are
stripped automatically.
EOF
}

COMMITS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b)        OSS_BRANCH="$2"; shift 2 ;;
    -e)        EXCLUDE_FILE="$2"; shift 2 ;;
    -k)        KW_FILE="$2"; shift 2 ;;
    -n)        DRY_RUN=1; shift ;;
    --no-scan) SKIP_SCAN=1; shift ;;
    --push)
      DO_PUSH=1
      if [[ "${2:-}" != "" && "${2:-}" != -* && "${2:-}" != "" ]]; then
        PUSH_REMOTE="$2"; shift
      fi
      shift ;;
    --range)
      mapfile -t range_commits < <(git -C "$REPO_ROOT" rev-list --reverse "$2")
      COMMITS+=("${range_commits[@]}")
      shift 2 ;;
    -h)        usage; exit 0 ;;
    -*)        echo "Error: unknown option $1" >&2; usage; exit 2 ;;
    *)         COMMITS+=("$1"); shift ;;
  esac
done

if [[ ${#COMMITS[@]} -eq 0 ]]; then
  echo "Error: no commits specified" >&2
  usage
  exit 2
fi

[[ -f "$EXCLUDE_FILE" ]] || { echo "Error: exclude file not found: $EXCLUDE_FILE" >&2; exit 2; }

# -------- load exclude patterns --------

PREFIX_PATS=()
EXACT_PATS=()
GLOB_REGEXES=()

while IFS= read -r pat; do
  [[ -z "$pat" || "$pat" =~ ^[[:space:]]*# ]] && continue
  if [[ "$pat" == */ || "$pat" == *'/*' ]]; then
    local_prefix="${pat%\*}"
    local_prefix="${local_prefix%/}/"
    PREFIX_PATS+=("$local_prefix")
  elif [[ "$pat" != *'*'* && "$pat" != *'?'* ]]; then
    EXACT_PATS+=("$pat")
  else
    local_re="$(echo "$pat" | sed -e 's/[.+[\](){}^$|]/\\&/g' -e 's/\*/[^\/]*/g' -e 's/?/[^\/]/g')"
    GLOB_REGEXES+=("^${local_re}$")
  fi
done < "$EXCLUDE_FILE"

is_excluded() {
  local fpath="$1"
  for pfx in "${PREFIX_PATS[@]}"; do
    [[ "$fpath" == "$pfx"* || "$fpath/" == "$pfx" ]] && return 0
  done
  for ex in "${EXACT_PATS[@]}"; do
    [[ "$fpath" == "$ex" ]] && return 0
  done
  for re in "${GLOB_REGEXES[@]}"; do
    echo "$fpath" | grep -qE "$re" && return 0
  done
  return 1
}

# -------- main --------

cd "$REPO_ROOT"

# Copy scan tools to a temp directory so they survive branch switches
TOOL_TMPDIR="$(mktemp -d)"
cp "$SCRIPT_DIR/oss-scan.sh" "$TOOL_TMPDIR/"
chmod +x "$TOOL_TMPDIR/oss-scan.sh"
if [[ -f "$KW_FILE" ]]; then
  cp "$KW_FILE" "$TOOL_TMPDIR/os_kw.txt"
  KW_FILE="$TOOL_TMPDIR/os_kw.txt"
fi
if [[ -f "$EXCLUDE_FILE" ]]; then
  cp "$EXCLUDE_FILE" "$TOOL_TMPDIR/oss_exclude_paths.txt"
fi
SCAN_CMD="$TOOL_TMPDIR/oss-scan.sh"

ORIG_BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD)"

if [[ "$ORIG_BRANCH" != "$OSS_BRANCH" ]]; then
  if ! git show-ref --verify --quiet "refs/heads/$OSS_BRANCH"; then
    echo "Error: branch '$OSS_BRANCH' does not exist. Run oss-setup.sh first." >&2
    rm -rf "$TOOL_TMPDIR"
    exit 1
  fi
  echo "Switching to '$OSS_BRANCH'..."
  git checkout "$OSS_BRANCH"
fi

cleanup() {
  if [[ "$(git symbolic-ref --short HEAD 2>/dev/null || true)" != "$ORIG_BRANCH" ]]; then
    echo "Returning to '$ORIG_BRANCH'..."
    git checkout "$ORIG_BRANCH" 2>/dev/null || true
  fi
  rm -rf "$TOOL_TMPDIR"
}
trap cleanup EXIT

TOTAL=${#COMMITS[@]}
PUSHED=0
SKIPPED=0
FAILED=0

for commit in "${COMMITS[@]}"; do
  git rev-parse --verify "$commit^{commit}" >/dev/null 2>&1 \
    || { echo "ERROR: cannot resolve '$commit'" >&2; FAILED=$((FAILED+1)); continue; }

  short="$(git rev-parse --short "$commit")"
  orig_msg="$(git -C "$REPO_ROOT" log -1 --format=%B "$commit")"
  orig_author="$(git -C "$REPO_ROOT" log -1 --format='%an <%ae>' "$commit")"
  orig_date="$(git -C "$REPO_ROOT" log -1 --format='%ai' "$commit")"
  parent="${commit}^"

  mapfile -t all_files < <(git diff-tree --no-commit-id -r --name-only "$commit")

  included=()
  excluded=()
  for f in "${all_files[@]}"; do
    [[ -z "$f" ]] && continue
    if is_excluded "$f"; then
      excluded+=("$f")
    else
      included+=("$f")
    fi
  done

  if [[ ${#included[@]} -eq 0 ]]; then
    echo "SKIP $short: all ${#all_files[@]} file(s) are excluded"
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  echo "---- $short: ${#included[@]} included, ${#excluded[@]} excluded ----"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] would include:"
    printf '    %s\n' "${included[@]}"
    if [[ ${#excluded[@]} -gt 0 ]]; then
      echo "  [dry-run] would exclude:"
      printf '    %s\n' "${excluded[@]}"
    fi
    PUSHED=$((PUSHED+1))
    continue
  fi

  # Cherry-pick with --no-commit so we can strip excluded files.
  # This uses git's 3-way merge (parent as base), which handles divergent
  # file content between main and oss/main far better than raw diff+apply.
  cherry_ok=0
  git cherry-pick --no-commit "$commit" 2>/dev/null && cherry_ok=1 || cherry_ok=0

  # Regardless of cherry-pick exit code (conflicts are expected for excluded
  # files), unstage all excluded files and resolve their conflicts.
  mapfile -t staged_files < <(git diff --cached --name-only HEAD 2>/dev/null)
  mapfile -t conflict_files < <(git diff --name-only --diff-filter=U 2>/dev/null)

  # Combine both lists for exclusion processing
  declare -A seen_excl=()
  for f in "${staged_files[@]}" "${conflict_files[@]}"; do
    [[ -z "$f" ]] && continue
    if is_excluded "$f"; then
      seen_excl["$f"]=1
    fi
  done

  if [[ ${#seen_excl[@]} -gt 0 ]]; then
    for f in "${!seen_excl[@]}"; do
      # Restore to oss/main HEAD state (unstage + revert worktree)
      git reset HEAD -- "$f" >/dev/null 2>&1 || true
      git checkout HEAD -- "$f" 2>/dev/null || rm -f "$f" 2>/dev/null || true
    done
  fi

  # Check for remaining conflicts on INCLUDED files
  mapfile -t remaining_conflicts < <(git diff --name-only --diff-filter=U 2>/dev/null)
  if [[ ${#remaining_conflicts[@]} -gt 0 && -n "${remaining_conflicts[0]}" ]]; then
    echo "ERROR $short: merge conflicts in included files:"
    printf '    %s\n' "${remaining_conflicts[@]}"
    echo "  Resolve conflicts manually on '$OSS_BRANCH', then commit."
    git cherry-pick --abort 2>/dev/null || git reset --hard HEAD 2>/dev/null || true
    FAILED=$((FAILED+1))
    continue
  fi

  # Verify there are actually staged changes for included files
  if git diff --cached --quiet HEAD 2>/dev/null; then
    echo "SKIP $short: no effective changes for included files after filtering"
    git reset --hard HEAD >/dev/null 2>&1
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  # ---- GATE 1: scan staged changes for keyword/non-ASCII violations ----
  if [[ $SKIP_SCAN -eq 0 && -f "$KW_FILE" ]]; then
    echo "  Scanning staged changes..."
    if ! OSS_SCAN_REPO_ROOT="$REPO_ROOT" "$SCAN_CMD" --staged -k "$KW_FILE"; then
      echo "ERROR $short: keyword/non-ASCII violation in staged changes. Resetting."
      git reset --hard HEAD >/dev/null
      FAILED=$((FAILED+1))
      continue
    fi
  fi

  # Strip AI-tool trailers
  clean_msg="$(echo "$orig_msg" | sed '/^Made-with:/d; /^Generated-by:/d')"

  # Commit preserving original author, date, and message
  GIT_AUTHOR_DATE="$orig_date" git commit \
    --author="$orig_author" \
    -m "$(printf '%s\n(cherry picked from %s on main)' "$clean_msg" "$commit")"

  # ---- GATE 2: full tree scan of oss/main after commit ----
  if [[ $SKIP_SCAN -eq 0 ]]; then
    echo "  Scanning full oss/main tree..."
    if ! OSS_SCAN_REPO_ROOT="$REPO_ROOT" "$SCAN_CMD" --tree HEAD -k "$KW_FILE" 2>&1; then
      echo "ERROR $short: full tree scan failed after commit."
      echo "  The commit is on oss/main but has violations."
      echo "  Fix with: git checkout oss/main && <fix> && git commit --amend"
      echo "  Or revert: git revert HEAD"
      FAILED=$((FAILED+1))
      # Don't reset -- the commit is already made; user must fix or revert
      continue
    fi
  fi

  new_sha="$(git rev-parse --short HEAD)"
  echo "OK $short -> $new_sha"
  PUSHED=$((PUSHED+1))
done

echo ""
echo "========================================"
echo "Summary: $PUSHED committed, $SKIPPED skipped, $FAILED failed (of $TOTAL)"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry-run mode -- nothing was committed)"
  exit 0
fi

if [[ $FAILED -gt 0 ]]; then
  echo ""
  echo "WARNING: Some commits failed. Fix violations before pushing."
  echo "  Inspect: git log oss/main --oneline -$((PUSHED + 5))"
  exit 1
fi

if [[ $PUSHED -eq 0 ]]; then
  echo "Nothing to push."
  exit 0
fi

# ---- Push decision ----
if [[ $DO_PUSH -eq 1 ]]; then
  echo ""
  echo "Pushing oss/main to '$PUSH_REMOTE'..."
  git push "$PUSH_REMOTE" "$OSS_BRANCH" 2>&1
  echo "Pushed successfully."
else
  echo ""
  echo "Commits applied to LOCAL oss/main only (not pushed)."
  echo ""
  echo "To push to internal remote:"
  echo "  git push origin oss/main"
  echo ""
  echo "To push to internal mirror (choreo-open):"
  echo "  git push oss-shadow oss/main:main"
  echo ""
  echo "Or re-run with --push:"
  echo "  oss-push.sh --push <same args>"
fi
