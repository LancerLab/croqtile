#!/usr/bin/env bash
set -euo pipefail

# Cherry-pick commits from main to the oss/ branch, filtering out
# excluded paths and running a full oss-scan before committing.
#
# By default, commits are applied to the LOCAL oss/main branch only.
# The script does NOT push to any remote unless --push is given.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=oss-config.sh
source "$SCRIPT_DIR/oss-config.sh"

DRY_RUN=0
SKIP_SCAN=0
DO_PUSH=0
INTERACTIVE=0
PUSH_REMOTE="origin"

usage() {
  cat <<'EOF'
Usage: oss-push.sh [options] <commit> [<commit>...]
     oss-push.sh [options] --range <from>..<to>
     oss-push.sh [options] --catchup

Cherry-pick commits to the local oss/ branch, filtering excluded paths
and running a full oss-scan gate before each commit.

By default, changes stay LOCAL. Nothing is pushed to any remote.

Options:
  -b <branch>     Target branch (default: oss/main)
  -e <file>       Exclude-paths file
  -k <file>       Keyword file
  -n              Dry run: show what would happen without committing
  --no-scan       Skip keyword scan (use with caution)
  --interactive   When code is clean but commit message has violations,
                  open an editor to fix the message instead of failing
  --catchup       Auto-detect unsynced commits on main and push them all
  --push [remote] After all commits, push oss/main to remote (default: origin)
  --range <r>     Expand a revision range via git rev-list --reverse
  -h              Show help

Workflow:
  1. Build set of already-synced SHAs from oss/main cherry-pick trailers
  2. Skip commits already synced (avoids duplicate/conflict errors)
  3. Cherry-pick each new commit (path-filtered)
  4. Run oss-scan on staged changes (keyword, non-ASCII, ghost-ref)
  5. Run oss-scan --tree on the full oss/main tree after commit
  6. If scan fails -> reset and report error
  7. After all commits, show summary
  8. Only push if --push is given; otherwise prompt

Each commit's original author, date, and message are preserved, with a
trailer noting the source SHA. Made-with:/Generated-by: trailers are
stripped automatically.
EOF
}

CATCHUP=0
COMMITS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
  -b)        OSS_BRANCH="$2"; shift 2 ;;
  -e)        EXCLUDE_FILE="$2"; shift 2 ;;
  -k)        KW_FILE="$2"; shift 2 ;;
  -n)        DRY_RUN=1; shift ;;
  --no-scan) SKIP_SCAN=1; shift ;;
  --interactive) INTERACTIVE=1; shift ;;
  --catchup) CATCHUP=1; shift ;;
  --push)
    DO_PUSH=1
    if [[ "${2:-}" != "" && "${2:-}" != -* && "${2:-}" != "" ]]; then
    PUSH_REMOTE="$2"; shift
    fi
    shift ;;
  --range)
    range_commits=()
    mapfile -t range_commits < <(git -C "$REPO_ROOT" rev-list --reverse "$2")
    COMMITS+=(${range_commits[@]+"${range_commits[@]}"})
    shift 2 ;;
  -h)        usage; exit 0 ;;
  -*)        echo "Error: unknown option $1" >&2; usage; exit 2 ;;
  *)         COMMITS+=("$1"); shift ;;
  esac
done

if [[ $CATCHUP -eq 0 && ${#COMMITS[@]} -eq 0 ]]; then
  echo "Error: no commits specified (or use --catchup)" >&2
  usage
  exit 2
fi

# Resolve symbolic refs (HEAD, branch names) to full SHAs while still on
# the current branch.  Without this, "HEAD" would resolve against the
# target branch after the checkout below.
if [[ ${#COMMITS[@]} -gt 0 ]]; then
  RESOLVED=()
  for c in "${COMMITS[@]}"; do
  full="$(git rev-parse --verify "$c^{commit}" 2>/dev/null)" \
    || { echo "Error: cannot resolve commit '$c'" >&2; exit 2; }
  RESOLVED+=("$full")
  done
  COMMITS=("${RESOLVED[@]}")
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

# -------- build set of already-synced main SHAs from oss/main trailers --------

declare -A SYNCED_SHAS=()
LATEST_SYNCED_SHA=""

build_synced_set() {
  local found_latest=0
  while IFS= read -r line; do
  local sha
  sha="$(echo "$line" | grep -oP '(?<=cherry picked from )\w+' || true)"
  if [[ -n "$sha" ]]; then
    local full
    # Use --verify to ensure the object actually exists in the repo;
    # without it, git rev-parse accepts any 40-hex string even if the
    # commit was rebased away or GC'd, leading to "bad object" errors
    # in downstream rev-list / rev-parse --short calls.
    full="$(git rev-parse --verify "$sha^{commit}" 2>/dev/null || true)"
    if [[ -n "$full" ]]; then
    SYNCED_SHAS["$full"]=1
    if [[ $found_latest -eq 0 ]]; then
      LATEST_SYNCED_SHA="$full"
      found_latest=1
    fi
    fi
    SYNCED_SHAS["$sha"]=1
  fi
  done < <(git log "$OSS_BRANCH" --format=%B)
}

echo "Building synced-commit index from $OSS_BRANCH trailers..."
build_synced_set
echo "  ${#SYNCED_SHAS[@]} main commit(s) already synced."
if [[ -n "$LATEST_SYNCED_SHA" ]]; then
  echo "  Latest sync point: $(git rev-parse --short "$LATEST_SYNCED_SHA") ($(git log -1 --format='%s' "$LATEST_SYNCED_SHA" | head -c 60))"
fi

# -------- catchup mode: find unsynced commits since last sync point --------

if [[ $CATCHUP -eq 1 ]]; then
  echo ""

  if [[ -z "$LATEST_SYNCED_SHA" ]]; then
  # Check .git/sync-all state file as fallback
  local_state="$REPO_ROOT/.git/sync-all/last-oss-push-sha"
  if [[ -f "$local_state" ]]; then
    LATEST_SYNCED_SHA="$(cat "$local_state")"
    echo "Using sync-all state file: $(git rev-parse --short "$LATEST_SYNCED_SHA")"
  else
    echo "ERROR: cannot determine last sync point."
    echo "  No cherry-pick trailers found on $OSS_BRANCH and no state file."
    echo "  Use explicit commits or --range instead, or set the state file:"
    echo "    echo <sha> > $local_state"
    exit 1
  fi
  fi

  echo "Catchup mode: scanning main since $(git rev-parse --short "$LATEST_SYNCED_SHA")..."

  catchup_commits=()
  mapfile -t catchup_commits < <(git rev-list --reverse "$LATEST_SYNCED_SHA..main")

  # Filter out already-synced ones (there may be a few between sync point
  # and HEAD that were synced via different paths)
  for sha in ${catchup_commits[@]+"${catchup_commits[@]}"}; do
  if [[ -z "${SYNCED_SHAS[$sha]:-}" ]]; then
    COMMITS+=("$sha")
  fi
  done

  echo "  ${#catchup_commits[@]} commit(s) since sync point, ${#COMMITS[@]} unsynced."
  echo ""

  if [[ ${#COMMITS[@]} -eq 0 ]]; then
  echo "oss/main is fully caught up with main. Nothing to do."
  exit 0
  fi
fi

TOTAL=${#COMMITS[@]}
PUSHED=0
SKIPPED=0
FAILED=0
ALREADY=0

for commit in "${COMMITS[@]}"; do
  git rev-parse --verify "$commit^{commit}" >/dev/null 2>&1 \
  || { echo "ERROR: cannot resolve '$commit'" >&2; FAILED=$((FAILED+1)); continue; }

  full_sha="$(git rev-parse "$commit")"
  short="$(git rev-parse --short "$commit")"

  # Skip already-synced commits
  if [[ -n "${SYNCED_SHAS[$full_sha]:-}" ]]; then
  ALREADY=$((ALREADY+1))
  continue
  fi

  orig_msg="$(git -C "$REPO_ROOT" log -1 --format=%B "$commit")"
  orig_author="$(git -C "$REPO_ROOT" log -1 --format='%an <%ae>' "$commit")"
  orig_date="$(git -C "$REPO_ROOT" log -1 --format='%ai' "$commit")"
  parent="${commit}^"

  all_files=()
  mapfile -t all_files < <(git diff-tree --no-commit-id -r --name-only "$commit")

  included=()
  excluded=()
  for f in ${all_files[@]+"${all_files[@]}"}; do
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
  # Actually simulate the cherry-pick to detect already-reflected commits
  git cherry-pick --no-commit "$commit" 2>/dev/null || true
  # Strip excluded files from staging
  for f in $(git diff --cached --name-only HEAD 2>/dev/null) \
       $(git diff --name-only --diff-filter=U 2>/dev/null); do
    [[ -z "$f" ]] && continue
    if is_excluded "$f"; then
    git reset HEAD -- "$f" >/dev/null 2>&1 || true
    git checkout HEAD -- "$f" 2>/dev/null || rm -f "$f" 2>/dev/null || true
    fi
  done
  if git diff --cached --quiet HEAD 2>/dev/null; then
    echo "  [dry-run] already reflected on $OSS_BRANCH (no net changes)"
    git reset --hard HEAD >/dev/null 2>&1
    SKIPPED=$((SKIPPED+1))
    continue
  fi
  echo "  [dry-run] would include:"
  git diff --cached --name-only HEAD 2>/dev/null | while read -r f; do
    echo "    $f"
  done
  git reset --hard HEAD >/dev/null 2>&1
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
  staged_files=()
  conflict_files=()
  mapfile -t staged_files < <(git diff --cached --name-only HEAD 2>/dev/null)
  mapfile -t conflict_files < <(git diff --name-only --diff-filter=U 2>/dev/null)

  # Combine both lists for exclusion processing
  declare -A seen_excl=()
  for f in ${staged_files[@]+"${staged_files[@]}"} ${conflict_files[@]+"${conflict_files[@]}"}; do
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
  remaining_conflicts=()
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
  echo "SKIP $short: already reflected on $OSS_BRANCH (changes produce no net diff)"
  git reset --hard HEAD >/dev/null 2>&1
  SKIPPED=$((SKIPPED+1))
  continue
  fi

  # ---- GATE 1: scan staged changes for keyword/non-ASCII violations ----
  # Strip AI-tool trailers early so the message scan sees the final text.
  clean_msg="$(echo "$orig_msg" | sed '/^Made-with:/d; /^Generated-by:/d')"
  if [[ $SKIP_SCAN -eq 0 && -f "$KW_FILE" ]]; then
  # Step A: scan staged code files only (no --msg-file)
  echo "  Scanning staged code files..."
  if ! OSS_SCAN_REPO_ROOT="$REPO_ROOT" "$SCAN_CMD" --staged -k "$KW_FILE"; then
    echo "ERROR $short: keyword/non-ASCII violation in staged code. Resetting."
    git reset --hard HEAD >/dev/null
    FAILED=$((FAILED+1))
    continue
  fi

  # Step B: scan commit message (with cherry-pick trailer appended)
  _msg_tmp="$(mktemp)"
  _full_msg_tmp="$(mktemp)"
  printf '%s' "$clean_msg" > "$_msg_tmp"
  printf '%s\n(cherry picked from %s on main)' "$clean_msg" "$commit" > "$_full_msg_tmp"

  _msg_ok=0
  if OSS_SCAN_REPO_ROOT="$REPO_ROOT" "$SCAN_CMD" --msg-only --msg-file "$_full_msg_tmp" -k "$KW_FILE" 2>&1; then
    _msg_ok=1
  fi

  if [[ $_msg_ok -eq 0 ]]; then
    if [[ $INTERACTIVE -eq 1 ]]; then
    echo ""
    echo "  Commit message has keyword violations (code files are clean)."
    echo "  Edit the message to remove forbidden keywords. Save and quit to continue."
    _max_edits=2
    _edit=0
    _msg_fixed=0
    while [[ $_edit -lt $_max_edits ]]; do
      _edit=$((_edit + 1))
      echo "  Opening \${EDITOR:-vi} (attempt $_edit of $_max_edits)..."
      ${EDITOR:-vi} "$_msg_tmp"
      printf '%s\n(cherry picked from %s on main)' "$(cat "$_msg_tmp")" "$commit" > "$_full_msg_tmp"
      echo "  Re-scanning commit message..."
      if OSS_SCAN_REPO_ROOT="$REPO_ROOT" "$SCAN_CMD" --msg-only --msg-file "$_full_msg_tmp" -k "$KW_FILE" 2>&1; then
      echo "  Message is clean."
      _msg_fixed=1
      break
      fi
      echo "  Message still has violations."
      if [[ $_edit -lt $_max_edits ]]; then
      echo "  (attempt $_edit of $_max_edits -- one more chance)"
      fi
    done
    if [[ $_msg_fixed -eq 0 ]]; then
      echo "ERROR $short: commit message still has violations after $_max_edits edit(s). Resetting."
      rm -f "$_msg_tmp" "$_full_msg_tmp"
      git reset --hard HEAD >/dev/null 2>&1
      FAILED=$((FAILED+1))
      continue
    fi
    clean_msg="$(cat "$_msg_tmp")"
    else
    rm -f "$_msg_tmp" "$_full_msg_tmp"
    echo "ERROR $short: keyword violation in commit message. Resetting."
    git reset --hard HEAD >/dev/null
    FAILED=$((FAILED+1))
    continue
    fi
  fi
  rm -f "$_msg_tmp" "$_full_msg_tmp"
  fi

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

  # ---- GATE 3: verify CMake configure_file inputs exist ----
  cmake_ok=1
  while IFS= read -r cmake_file; do
  [[ -z "$cmake_file" ]] && continue
  while IFS= read -r input_path; do
    [[ -z "$input_path" ]] && continue
    # Resolve ${CMAKE_SOURCE_DIR} to repo root
    resolved="${input_path/\$\{CMAKE_SOURCE_DIR\}\//}"
    resolved="${resolved/\$\{CMAKE_SOURCE_DIR\}/}"
    if [[ "$resolved" != "$input_path" ]] && ! git cat-file -e "HEAD:$resolved" 2>/dev/null; then
    echo "  WARNING: CMake configure_file input missing: $resolved"
    cmake_ok=0
    fi
  done < <(git show HEAD:"$cmake_file" 2>/dev/null | grep -oP 'configure_file\(\s*\K\$\{CMAKE_SOURCE_DIR\}/[^\s]+' || true)
  done < <(git ls-tree -r --name-only HEAD | grep 'CMakeLists.txt$')

  if [[ $cmake_ok -eq 0 ]]; then
  echo "ERROR $short: oss/main build would fail (missing CMake inputs)."
  echo "  Reverting commit..."
  git reset --hard HEAD~1 >/dev/null 2>&1
  FAILED=$((FAILED+1))
  continue
  fi

  new_sha="$(git rev-parse --short HEAD)"
  echo "OK $short -> $new_sha"
  PUSHED=$((PUSHED+1))
done

echo ""
echo "========================================"
echo "Summary: $PUSHED committed, $SKIPPED skipped, $FAILED failed, $ALREADY already-synced (of $TOTAL)"

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
