#!/usr/bin/env bash
set -uo pipefail

# Scan for sensitive/forbidden keywords and non-ASCII characters in
# paths and file contents. Exit 0 = clean, 1 = violations, 2 = usage error.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow parent scripts (e.g. oss-push.sh) to override REPO_ROOT when
# this script is copied to a temp dir outside the repo.
REPO_ROOT="${OSS_SCAN_REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel 2>/dev/null || true)}"

KW_FILE="$SCRIPT_DIR/os_kw.txt"
MODE="worktree"
TARGET=""
VERBOSE=0

usage() {
  cat <<'EOF'
Usage: oss-scan.sh [options]

Scan for sensitive/forbidden keywords in file paths and contents.

Modes (pick one):
  --tree <branch>    Scan entire branch tree (git ls-tree + git grep)
  --staged           Scan staged changes only (git diff --cached)
  --diff <rev>       Scan a single commit's diff (git diff-tree)
  --range <a>..<b>   Scan commit messages in a range
  --dir <path>       Scan a directory on disk
  (default)          Scan current worktree tracked files

Options:
  -k <file>    Keyword file (default: scripts/oss/os_kw.txt)
  -v           Verbose: print each violation immediately
  -h           Show help

Exit codes:
  0  Clean (no violations found)
  1  Violations found
  2  Usage / config error
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree)   MODE="tree";   TARGET="${2:-}"; shift 2 || { echo "Error: --tree needs <branch>"; exit 2; } ;;
    --staged) MODE="staged"; shift ;;
    --diff)   MODE="diff";   TARGET="${2:-}"; shift 2 || { echo "Error: --diff needs <rev>"; exit 2; } ;;
    --range)  MODE="range";  TARGET="${2:-}"; shift 2 || { echo "Error: --range needs <a>..<b>"; exit 2; } ;;
    --dir)    MODE="dir";    TARGET="${2:-}"; shift 2 || { echo "Error: --dir needs <path>"; exit 2; } ;;
    -k)       KW_FILE="${2:-}"; shift 2 || { echo "Error: -k needs <file>"; exit 2; } ;;
    -v)       VERBOSE=1; shift ;;
    -h)       usage; exit 0 ;;
    -*)       echo "Error: unknown option $1" >&2; usage; exit 2 ;;
    *)        echo "Error: unexpected argument $1" >&2; usage; exit 2 ;;
  esac
done

# -------- load keyword patterns --------

if [[ ! -f "$KW_FILE" ]]; then
  echo "Error: keyword file not found: $KW_FILE" >&2
  exit 2
fi

# Parse patterns: handle (?i) prefix by splitting into pattern + case flag.
# Builds COMBINED_PATTERN (a single ERE alternation) and CASE_FLAG.
CASE_FLAG=""
RAW_PATTERNS=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  if [[ "$line" == '(?i)'* ]]; then
    CASE_FLAG="-i"
    RAW_PATTERNS+=("${line#'(?i)'}")
  else
    RAW_PATTERNS+=("$line")
  fi
done < "$KW_FILE"

if [[ ${#RAW_PATTERNS[@]} -eq 0 ]]; then
  echo "Warning: no patterns loaded from $KW_FILE" >&2
  exit 0
fi

COMBINED_PATTERN="$(IFS='|'; echo "${RAW_PATTERNS[*]}")"

VIOLATIONS=0
VIOLATION_LOG=""

record() {
  local ctx="$1"; shift
  local msg="$*"
  VIOLATIONS=$((VIOLATIONS + 1))
  VIOLATION_LOG+="  [$ctx] $msg"$'\n'
  if [[ $VERBOSE -eq 1 ]]; then
    echo "VIOLATION [$ctx] $msg" >&2
  fi
}

# -------- scan helpers --------

scan_paths() {
  local ctx="$1"
  while IFS= read -r fpath; do
    [[ -z "$fpath" ]] && continue
    if echo "$fpath" | grep -qE $CASE_FLAG -- "$COMBINED_PATTERN"; then
      record "$ctx/path" "$fpath"
    fi
  done
}

FILTER_TREEISH_PREFIX=""

filter_excluded_lines() {
  # Input: lines from git grep.
  # With treeish: "main:path:lineno:content" or "oss/main:path:lineno:content"
  # Without:      "path:lineno:content"
  # FILTER_TREEISH_PREFIX should be set to "main:" or "oss/main:" etc. when
  # scanning a branch, or "" for working tree scans.
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local stripped="$line"
    if [[ -n "$FILTER_TREEISH_PREFIX" && "$stripped" == "$FILTER_TREEISH_PREFIX"* ]]; then
      stripped="${stripped#"$FILTER_TREEISH_PREFIX"}"
    fi
    local fpath="${stripped%%:*}"
    is_excluded_path "$fpath" && continue
    echo "$line"
  done
}

scan_content_via_git_grep() {
  local treeish="$1"
  local ctx="$2"
  local raw_matches
  raw_matches="$(git -C "$REPO_ROOT" grep -I -nE $CASE_FLAG -- "$COMBINED_PATTERN" "$treeish" 2>/dev/null || true)"
  [[ -z "$raw_matches" ]] && return
  local matches
  matches="$(echo "$raw_matches" | filter_excluded_lines)"
  if [[ -n "$matches" ]]; then
    local count
    count="$(echo "$matches" | wc -l)"
    record "$ctx/content" "$count match(es) in tree $treeish"
    VIOLATION_LOG+="$(echo "$matches" | head -40 | sed 's/^/    /')"$'\n'
    if [[ $count -gt 40 ]]; then
      VIOLATION_LOG+="    ... and $((count - 40)) more"$'\n'
    fi
  fi
}

scan_content_from_diff() {
  local diff_text="$1"
  local ctx="$2"
  local added_lines
  added_lines="$(echo "$diff_text" | grep -E '^\+' | grep -v '^+++' || true)"
  if [[ -z "$added_lines" ]]; then return; fi

  local matches
  matches="$(echo "$added_lines" | grep -nE $CASE_FLAG -- "$COMBINED_PATTERN" 2>/dev/null || true)"
  if [[ -n "$matches" ]]; then
    local count
    count="$(echo "$matches" | wc -l)"
    record "$ctx/content" "$count added line(s) match keyword patterns"
    VIOLATION_LOG+="$(echo "$matches" | head -20 | sed 's/^/    /')"$'\n'
  fi
}

scan_file_on_disk() {
  local filepath="$1"
  local ctx="$2"
  if file --brief --mime-type "$filepath" 2>/dev/null | grep -q '^text/'; then
    local matches
    matches="$(grep -nE $CASE_FLAG -- "$COMBINED_PATTERN" "$filepath" 2>/dev/null || true)"
    if [[ -n "$matches" ]]; then
      local count
      count="$(echo "$matches" | wc -l)"
      record "$ctx/content" "$filepath: $count line(s)"
      VIOLATION_LOG+="$(echo "$matches" | head -10 | sed 's/^/    /')"$'\n'
    fi
  fi
}

# -------- non-ASCII scan helpers --------
# Strict rule: no non-ASCII bytes allowed in public code.

NON_ASCII_RE='[^\x00-\x7F]'

scan_nonascii_via_git_grep() {
  local treeish="$1"
  local ctx="$2"
  local raw_matches
  raw_matches="$(git -C "$REPO_ROOT" grep -I -nP -- "$NON_ASCII_RE" "$treeish" 2>/dev/null || true)"
  [[ -z "$raw_matches" ]] && return
  local matches
  matches="$(echo "$raw_matches" | filter_excluded_lines)"
  if [[ -n "$matches" ]]; then
    local count
    count="$(echo "$matches" | wc -l)"
    record "$ctx/non-ascii" "$count file(s) contain non-ASCII characters"
    VIOLATION_LOG+="$(echo "$matches" | head -20 | sed 's/^/    /')"$'\n'
    if [[ $count -gt 20 ]]; then
      VIOLATION_LOG+="    ... and $((count - 20)) more"$'\n'
    fi
  fi
}

scan_nonascii_from_diff() {
  local diff_text="$1"
  local ctx="$2"
  local added_lines
  added_lines="$(echo "$diff_text" | grep -E '^\+' | grep -v '^+++' || true)"
  [[ -z "$added_lines" ]] && return
  local matches
  matches="$(echo "$added_lines" | grep -nP -- "$NON_ASCII_RE" 2>/dev/null || true)"
  if [[ -n "$matches" ]]; then
    local count
    count="$(echo "$matches" | wc -l)"
    record "$ctx/non-ascii" "$count added line(s) contain non-ASCII characters"
    VIOLATION_LOG+="$(echo "$matches" | head -10 | sed 's/^/    /')"$'\n'
  fi
}

scan_nonascii_on_disk() {
  local dir="$1"
  local ctx="$2"
  local matches
  matches="$(grep -rnP -- "$NON_ASCII_RE" "$dir" 2>/dev/null || true)"
  if [[ -n "$matches" ]]; then
    local count
    count="$(echo "$matches" | wc -l)"
    record "$ctx/non-ascii" "$count line(s) contain non-ASCII characters"
    VIOLATION_LOG+="$(echo "$matches" | head -20 | sed 's/^/    /')"$'\n'
  fi
}

# -------- ghost-reference detection --------
# Detects references to excluded paths/content in non-excluded files.
# This catches structural dependencies that keyword scanning misses.

EXCLUDE_FILE="${OSS_SCAN_EXCLUDE_FILE:-$SCRIPT_DIR/oss_exclude_paths.txt}"

# Load exclude patterns from the shared file into arrays for matching.
EXCL_DIRS=()
EXCL_GLOBS=()
EXCL_EXACT=()

if [[ -f "$EXCLUDE_FILE" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    line="${line%"${line##*[![:space:]]}"}"  # trim trailing whitespace
    if [[ "$line" == */ || "$line" == */* && "$line" == *\* ]]; then
      # Directory prefix: strip trailing / or /*
      _d="${line%/}"
      _d="${_d%/\*}"
      EXCL_DIRS+=("$_d/")
    elif [[ "$line" == *'*'* || "$line" == *'?'* ]]; then
      EXCL_GLOBS+=("$line")
    else
      EXCL_EXACT+=("$line")
    fi
  done < "$EXCLUDE_FILE"
fi

is_excluded_path() {
  local fpath="$1"
  # Directory prefix match
  for d in "${EXCL_DIRS[@]}"; do
    [[ "$fpath" == "$d"* ]] && return 0
  done
  # Glob match (fnmatch-style, no /)
  for g in "${EXCL_GLOBS[@]}"; do
    # shellcheck disable=SC2254
    case "$fpath" in
      $g) return 0 ;;
    esac
  done
  # Exact match
  for e in "${EXCL_EXACT[@]}"; do
    [[ "$fpath" == "$e" ]] && return 0
  done
  return 1
}

scan_ghost_refs_from_diff() {
  local diff_text="$1"
  local ctx="$2"

  local added_lines
  added_lines="$(echo "$diff_text" | grep -E '^\+' | grep -v '^+++' || true)"
  [[ -z "$added_lines" ]] && return

  # Check #include directives referencing excluded directories
  local inc_hits
  inc_hits="$(echo "$added_lines" | grep -nE '^\+.*#include.*["<](Target/GCU/|catz/)' 2>/dev/null || true)"
  if [[ -n "$inc_hits" ]]; then
    local count
    count="$(echo "$inc_hits" | wc -l)"
    record "$ctx/ghost-include" "$count #include(s) reference excluded paths"
    VIOLATION_LOG+="$(echo "$inc_hits" | head -10 | sed 's/^/    /')"$'\n'
  fi
}

scan_coupled_changes() {
  local rev="$1"
  local ctx="$2"

  local all_files
  mapfile -t all_files < <(git -C "$REPO_ROOT" diff-tree --no-commit-id -r --name-only "$rev")

  local has_excluded=0 has_included=0
  local excluded_dirs=""
  for f in "${all_files[@]}"; do
    [[ -z "$f" ]] && continue
    if is_excluded_path "$f"; then
      has_excluded=1
      local d="${f%/*}"
      case "$excluded_dirs" in
        *"$d"*) ;;
        *) excluded_dirs="$excluded_dirs $d" ;;
      esac
    else
      has_included=1
    fi
  done

  if [[ $has_excluded -eq 1 && $has_included -eq 1 ]]; then
    record "$ctx/coupled" "commit modifies both public and excluded files (dirs:$excluded_dirs)"
    VIOLATION_LOG+="    Review included files for dependencies on excluded code."$'\n'
    VIOLATION_LOG+="    Excluded dirs touched:$excluded_dirs"$'\n'
  fi
}

# -------- mode implementations --------

filter_excluded_paths() {
  while IFS= read -r fpath; do
    [[ -z "$fpath" ]] && continue
    is_excluded_path "$fpath" && continue
    echo "$fpath"
  done
}

mode_tree() {
  local branch="$1"
  git -C "$REPO_ROOT" rev-parse --verify "$branch^{commit}" >/dev/null 2>&1 \
    || { echo "Error: cannot resolve '$branch'" >&2; exit 2; }

  echo "Scanning tree: $branch (excluding ${#EXCL_DIRS[@]} dir rules, ${#EXCL_GLOBS[@]} glob rules)"

  FILTER_TREEISH_PREFIX="$branch:"

  # Scan paths, filtering out excluded ones
  scan_paths "tree" < <(git -C "$REPO_ROOT" ls-tree -r --name-only "$branch" | filter_excluded_paths)

  # Scan file contents using git grep (filter_excluded_lines handles exclusion)
  scan_content_via_git_grep "$branch" "tree"

  # Strict non-ASCII check (filter_excluded_lines handles exclusion)
  scan_nonascii_via_git_grep "$branch" "tree"
}

mode_staged() {
  echo "Scanning staged changes..."

  # Scan paths of staged files (process substitution to avoid subshell)
  scan_paths "staged" < <(git -C "$REPO_ROOT" diff --cached --name-only)

  # Scan added lines in staged diff
  local diff_text
  diff_text="$(git -C "$REPO_ROOT" diff --cached)"
  scan_content_from_diff "$diff_text" "staged"

  # Strict non-ASCII check on staged additions
  scan_nonascii_from_diff "$diff_text" "staged"

  # Ghost-reference detection
  scan_ghost_refs_from_diff "$diff_text" "staged"
}

mode_diff() {
  local rev="$1"
  git -C "$REPO_ROOT" rev-parse --verify "$rev^{commit}" >/dev/null 2>&1 \
    || { echo "Error: cannot resolve '$rev'" >&2; exit 2; }

  echo "Scanning diff of: $rev"

  # Scan changed file paths (process substitution to avoid subshell)
  scan_paths "diff" < <(git -C "$REPO_ROOT" diff-tree --no-commit-id -r --name-only "$rev")

  # Scan commit message
  local msg
  msg="$(git -C "$REPO_ROOT" log -1 --format=%B "$rev")"
  if echo "$msg" | grep -qE $CASE_FLAG -- "$COMBINED_PATTERN"; then
    record "diff/message" "commit message of $rev"
    VIOLATION_LOG+="$(echo "$msg" | grep -nE $CASE_FLAG -- "$COMBINED_PATTERN" | head -5 | sed 's/^/    /')"$'\n'
  fi

  # Scan added lines in the diff
  local diff_text
  diff_text="$(git -C "$REPO_ROOT" diff-tree -p "$rev")"
  scan_content_from_diff "$diff_text" "diff"

  # Strict non-ASCII check on diff additions
  scan_nonascii_from_diff "$diff_text" "diff"

  # Ghost-reference detection
  scan_ghost_refs_from_diff "$diff_text" "diff"
  scan_coupled_changes "$rev" "diff"
}

mode_range() {
  local range="$1"
  echo "Scanning commit messages in range: $range"

  while IFS= read -r sha; do
    [[ -z "$sha" ]] && continue
    local msg
    msg="$(git -C "$REPO_ROOT" log -1 --format=%B "$sha")"
    if echo "$msg" | grep -qE $CASE_FLAG -- "$COMBINED_PATTERN"; then
      local short
      short="$(git -C "$REPO_ROOT" rev-parse --short "$sha")"
      record "range/message" "commit $short"
      VIOLATION_LOG+="$(echo "$msg" | grep -nE $CASE_FLAG -- "$COMBINED_PATTERN" | head -3 | sed 's/^/    /')"$'\n'
    fi
  done < <(git -C "$REPO_ROOT" rev-list "$range")
}

mode_dir() {
  local dir="$1"
  [[ -d "$dir" ]] || { echo "Error: directory not found: $dir" >&2; exit 2; }

  echo "Scanning directory: $dir"

  # Scan paths (process substitution to avoid subshell)
  while IFS= read -r fpath; do
    local rel="${fpath#"$dir"/}"
    if echo "$rel" | grep -qE $CASE_FLAG -- "$COMBINED_PATTERN"; then
      record "dir/path" "$rel"
    fi
  done < <(find "$dir" -type f)

  # Scan file contents (process substitution to avoid subshell)
  while IFS= read -r fpath; do
    scan_file_on_disk "$fpath" "dir"
  done < <(find "$dir" -type f)

  # Strict non-ASCII check
  scan_nonascii_on_disk "$dir" "dir"
}

mode_worktree() {
  echo "Scanning worktree tracked files (excluding ${#EXCL_DIRS[@]} dir rules, ${#EXCL_GLOBS[@]} glob rules)..."
  FILTER_TREEISH_PREFIX=""

  # Scan paths, filtering excluded
  scan_paths "worktree" < <(git -C "$REPO_ROOT" ls-files | filter_excluded_paths)

  # Scan contents using git grep, filtering excluded
  local raw_matches
  raw_matches="$(git -C "$REPO_ROOT" grep -I -nE $CASE_FLAG -- "$COMBINED_PATTERN" 2>/dev/null || true)"
  if [[ -n "$raw_matches" ]]; then
    local matches
    matches="$(echo "$raw_matches" | filter_excluded_lines)"
    if [[ -n "$matches" ]]; then
      local count
      count="$(echo "$matches" | wc -l)"
      record "worktree/content" "$count match(es) in working tree"
      VIOLATION_LOG+="$(echo "$matches" | head -40 | sed 's/^/    /')"$'\n'
      if [[ $count -gt 40 ]]; then
        VIOLATION_LOG+="    ... and $((count - 40)) more"$'\n'
      fi
    fi
  fi

  # Strict non-ASCII check on worktree, filtering excluded
  local raw_na
  raw_na="$(git -C "$REPO_ROOT" grep -I -nP -- "$NON_ASCII_RE" 2>/dev/null || true)"
  if [[ -n "$raw_na" ]]; then
    local na_matches
    na_matches="$(echo "$raw_na" | filter_excluded_lines)"
    if [[ -n "$na_matches" ]]; then
      local na_count
      na_count="$(echo "$na_matches" | wc -l)"
      record "worktree/non-ascii" "$na_count line(s) contain non-ASCII characters"
      VIOLATION_LOG+="$(echo "$na_matches" | head -20 | sed 's/^/    /')"$'\n'
      if [[ $na_count -gt 20 ]]; then
        VIOLATION_LOG+="    ... and $((na_count - 20)) more"$'\n'
      fi
    fi
  fi
}

# -------- dispatch --------

case "$MODE" in
  tree)     mode_tree "$TARGET" ;;
  staged)   mode_staged ;;
  diff)     mode_diff "$TARGET" ;;
  range)    mode_range "$TARGET" ;;
  dir)      mode_dir "$TARGET" ;;
  worktree) mode_worktree ;;
esac

# -------- report --------

echo ""
if [[ $VIOLATIONS -eq 0 ]]; then
  echo "CLEAN: no forbidden keywords, non-ASCII, or ghost references found."
  exit 0
else
  echo "VIOLATIONS: $VIOLATIONS issue(s) found."
  echo ""
  echo "$VIOLATION_LOG"
  exit 1
fi
