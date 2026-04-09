#!/usr/bin/env bash
set -euo pipefail

# Cherry-pick commits from main to the oss/ branch, filtering out
# excluded paths and scanning for forbidden keywords before committing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

EXCLUDE_FILE="$SCRIPT_DIR/oss_exclude_paths.txt"
KW_FILE="$SCRIPT_DIR/os_kw.txt"
OSS_BRANCH="oss/main"
DRY_RUN=0
SKIP_SCAN=0

usage() {
  cat <<'EOF'
Usage: oss-push.sh [options] <commit> [<commit>...]
       oss-push.sh [options] --range <from>..<to>

Cherry-pick commits to the oss/ branch, filtering excluded paths
and running keyword scan before each commit.

Options:
  -b <branch>  Target branch (default: oss/main)
  -e <file>    Exclude-paths file (default: scripts/oss/oss_exclude_paths.txt)
  -k <file>    Keyword file (default: scripts/oss/os_kw.txt)
  -n           Dry run: show what would happen without committing
  --no-scan    Skip keyword scan (use with caution)
  --range <r>  Expand a revision range via git rev-list --reverse
  -h           Show help

Each commit's original author, date, and message are preserved, with a
trailer noting the source SHA. The committer identity is whoever runs
the script (i.e. you), so `git log --format=fuller` shows both.
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
# (scripts/oss/ may not exist on the target branch).
TOOL_TMPDIR="$(mktemp -d)"
cp "$SCRIPT_DIR/oss-scan.sh" "$TOOL_TMPDIR/"
chmod +x "$TOOL_TMPDIR/oss-scan.sh"
if [[ -f "$KW_FILE" ]]; then
  cp "$KW_FILE" "$TOOL_TMPDIR/os_kw.txt"
  KW_FILE="$TOOL_TMPDIR/os_kw.txt"
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

  # Get changed files in this commit
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

  # Generate patch for only the included files
  patch_file="$(mktemp)"
  git diff --binary "$parent" "$commit" -- "${included[@]}" > "$patch_file"

  if [[ ! -s "$patch_file" ]]; then
    echo "SKIP $short: no effective changes for included files"
    rm -f "$patch_file"
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  # Apply the filtered patch
  if ! git apply --index --3way "$patch_file" 2>&1; then
    echo "ERROR $short: patch failed to apply."
    echo "  Resolve conflicts manually on '$OSS_BRANCH', then commit."
    echo "  Patch saved: $patch_file"
    FAILED=$((FAILED+1))
    continue
  fi
  rm -f "$patch_file"

  # Keyword scan on staged changes
  if [[ $SKIP_SCAN -eq 0 && -f "$KW_FILE" ]]; then
    if ! OSS_SCAN_REPO_ROOT="$REPO_ROOT" "$SCAN_CMD" --staged -k "$KW_FILE"; then
      echo "ERROR $short: keyword violation detected. Resetting staged changes."
      git reset --hard HEAD >/dev/null
      FAILED=$((FAILED+1))
      continue
    fi
  fi

  # Strip AI-tool trailers (Made-with: Cursor, Generated-by:, etc.)
  clean_msg="$(echo "$orig_msg" | sed '/^Made-with:/d; /^Generated-by:/d')"

  # Commit preserving original author, date, and message
  GIT_AUTHOR_DATE="$orig_date" git commit \
    --author="$orig_author" \
    -m "$(cat <<EOF
${clean_msg}
(cherry picked from ${commit} on main)
EOF
)"

  new_sha="$(git rev-parse --short HEAD)"
  echo "OK $short -> $new_sha"
  PUSHED=$((PUSHED+1))
done

echo ""
echo "Summary: $PUSHED pushed, $SKIPPED skipped, $FAILED failed (of $TOTAL)"
if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
