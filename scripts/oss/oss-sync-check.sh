#!/usr/bin/env bash
set -euo pipefail

# Verify file-level consistency between main and oss/main.
#
# Compares every non-excluded, non-conflict file that exists on both
# branches and reports any content differences. Files that only exist
# on one branch (main-only or oss-only) are listed separately.
#
# Exit codes:
#   0  All non-excluded/conflict files are identical
#   1  Divergences found
#   2  Usage error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=oss-config.sh
source "$SCRIPT_DIR/oss-config.sh"

VERBOSE=0

usage() {
  cat <<'EOF'
Usage: oss-sync-check.sh [options]

Compare non-excluded, non-conflict files between main and oss/main.
Reports any content divergences that indicate missed or incomplete syncs.

Options:
  -v, --verbose   Show per-file diff stats for divergent files
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"

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

# -------- load conflict patterns --------

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
  done < "$CONFLICT_FILE"
fi

is_conflict() {
  local fpath="$1"
  for pfx in "${CONFLICT_PREFIXES[@]}"; do
    [[ "$fpath" == "$pfx"* ]] && return 0
  done
  for ex in "${CONFLICT_EXACT[@]}"; do
    [[ "$fpath" == "$ex" ]] && return 0
  done
  return 1
}

# -------- compare branches --------

divergent=()
main_only=()
oss_only=()
identical=0
excluded=0
conflict=0

while IFS= read -r fpath; do
  [[ -z "$fpath" ]] && continue

  if is_excluded "$fpath"; then
    excluded=$((excluded + 1))
    continue
  fi
  if is_conflict "$fpath"; then
    conflict=$((conflict + 1))
    continue
  fi

  main_blob=$(git rev-parse "$MAIN_BRANCH:$fpath" 2>/dev/null) || main_blob=""
  oss_blob=$(git rev-parse "$OSS_BRANCH:$fpath" 2>/dev/null) || oss_blob=""

  if [[ -n "$main_blob" && -z "$oss_blob" ]]; then
    main_only+=("$fpath")
  elif [[ -z "$main_blob" && -n "$oss_blob" ]]; then
    oss_only+=("$fpath")
  elif [[ "$main_blob" != "$oss_blob" ]]; then
    divergent+=("$fpath")
  else
    identical=$((identical + 1))
  fi
done < <(git diff "$MAIN_BRANCH" "$OSS_BRANCH" --name-only 2>/dev/null)

# -------- report --------

echo "OSS Sync Check: $MAIN_BRANCH vs $OSS_BRANCH"
echo "  Excluded:   $excluded (skipped)"
echo "  Conflict:   $conflict (skipped)"

has_issues=0

if [[ ${#divergent[@]} -gt 0 ]]; then
  has_issues=1
  echo ""
  echo "${_C_RED}DIVERGENT (${#divergent[@]} file(s)):${_C_RESET}"
  for f in "${divergent[@]}"; do
    if [[ $VERBOSE -eq 1 ]]; then
      stat=$(git diff "$MAIN_BRANCH:$f" "$OSS_BRANCH:$f" --stat 2>/dev/null | tail -1)
      echo "  $f  ($stat)"
    else
      echo "  $f"
    fi
  done
fi

if [[ ${#main_only[@]} -gt 0 ]]; then
  has_issues=1
  echo ""
  echo "${_C_YELLOW}MAIN-ONLY (${#main_only[@]} file(s) missing from $OSS_BRANCH):${_C_RESET}"
  for f in "${main_only[@]}"; do
    echo "  $f"
  done
fi

if [[ ${#oss_only[@]} -gt 0 ]]; then
  has_issues=1
  echo ""
  echo "${_C_YELLOW}OSS-ONLY (${#oss_only[@]} file(s) missing from $MAIN_BRANCH):${_C_RESET}"
  for f in "${oss_only[@]}"; do
    echo "  $f"
  done
fi

if [[ $has_issues -eq 0 ]]; then
  echo ""
  echo "OK: all non-excluded/conflict files are identical."
  exit 0
else
  echo ""
  echo "FAIL: divergences found. Run with -v for diff stats."
  exit 1
fi
