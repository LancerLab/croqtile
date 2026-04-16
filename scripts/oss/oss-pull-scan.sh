#!/usr/bin/env bash
set -euo pipefail

# Scan incoming public commits (on oss/main or a fetched public branch)
# for conflicts before pulling them into main.
#
# Detects:
#   1. File-path conflicts -- public commit touches a path that exists
#      in our private exclude list (e.g. CI config, .gitignore).
#   2. New private-zone files -- public commit creates files in dirs
#      we consider private (lib/Target/GCU/, scripts/, etc.).
#   3. Structural overlap -- public commit modifies a file that has
#      diverged between main and oss/main (content conflict risk).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=oss-config.sh
source "$SCRIPT_DIR/oss-config.sh"
RANGE=""
QUIET=0
JSON=0

usage() {
  cat <<'EOF'
Usage: oss-pull-scan.sh [options] [<commit>...] [--range <from>..<to>]

Scan incoming commits on the public side for conflicts before pulling.

Modes:
  <commit>             Scan specific commit(s)
  --range <from>..<to> Scan a range of commits
  --new                Scan all new commits on public/main not yet on oss/main
  --fetch              Fetch public remote first, then scan new commits

Options:
  -b <branch>   Local oss branch (default: oss/main)
  -r <remote>   Public remote name (default: public)
  -e <file>     Exclude-paths file (default: scripts/oss/oss_exclude_paths.txt)
  -q            Quiet mode (exit code only: 0=clean, 1=conflicts)
  --json        Output results as JSON
  -h            Show help

Exit codes:
  0  Clean -- no conflicts detected
  1  Conflicts found
  2  Usage error
EOF
}

COMMITS=()
FETCH=0
SCAN_NEW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  -b)       OSS_BRANCH="$2"; shift 2 ;;
  -r)       PUBLIC_REMOTE="$2"; shift 2 ;;
  -e)       EXCLUDE_FILE="$2"; shift 2 ;;
  -q)       QUIET=1; shift ;;
  --json)   JSON=1; shift ;;
  --fetch)  FETCH=1; SCAN_NEW=1; shift ;;
  --new)    SCAN_NEW=1; shift ;;
  --range)
    mapfile -t range_commits < <(git -C "$REPO_ROOT" rev-list --reverse "$2")
    COMMITS+=("${range_commits[@]}")
    shift 2 ;;
  -h)       usage; exit 0 ;;
  -*)       echo "Error: unknown option $1" >&2; usage; exit 2 ;;
  *)        COMMITS+=("$1"); shift ;;
  esac
done

cd "$REPO_ROOT"

if [[ $FETCH -eq 1 ]]; then
  timeout 30 git fetch "$PUBLIC_REMOTE" 2>/dev/null || {
  echo "Warning: could not fetch '$PUBLIC_REMOTE' (network issue or timeout)" >&2
  }
fi

if [[ $SCAN_NEW -eq 1 && ${#COMMITS[@]} -eq 0 ]]; then
  if git show-ref --verify --quiet "refs/remotes/$PUBLIC_REMOTE/main"; then
  mapfile -t COMMITS < <(git rev-list --reverse "$OSS_BRANCH..$PUBLIC_REMOTE/main")
  if [[ ${#COMMITS[@]} -eq 0 ]]; then
    [[ $QUIET -eq 0 ]] && echo "No new commits on $PUBLIC_REMOTE/main."
    exit 0
  fi
  else
  echo "Error: $PUBLIC_REMOTE/main not found. Run: git fetch $PUBLIC_REMOTE" >&2
  exit 2
  fi
fi

if [[ ${#COMMITS[@]} -eq 0 ]]; then
  echo "Error: no commits to scan." >&2
  usage
  exit 2
fi

[[ -f "$EXCLUDE_FILE" ]] || {
  echo "Error: exclude file not found: $EXCLUDE_FILE" >&2
  exit 2
}

# -------- load private-only path patterns --------
# These are paths that exist in our private repo but NOT in public.
# If a public commit creates/modifies these, it's a conflict.

PRIVATE_PREFIXES=()
PRIVATE_EXACT=()

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

# Files that are typically repo-specific config and should never collide
CONFLICT_ZONE_FILES=(
  ".github/workflows/"
  ".gitlab-ci.yml"
  ".gitlab/"
  "Makefile"
  "CMakeLists.txt"
  ".gitignore"
  ".gitmodules"
  ".gitattributes"
  ".clang-format"
  "README.md"
  "LICENSE"
  "mkdocs.yml"
)

is_private_path() {
  local fpath="$1"
  for pfx in "${PRIVATE_PREFIXES[@]}"; do
  [[ "$fpath" == "$pfx"* ]] && return 0
  done
  for ex in "${PRIVATE_EXACT[@]}"; do
  [[ "$fpath" == "$ex" ]] && return 0
  done
  return 1
}

is_conflict_zone() {
  local fpath="$1"
  for zone in "${CONFLICT_ZONE_FILES[@]}"; do
  if [[ "$zone" == */ ]]; then
    [[ "$fpath" == "$zone"* ]] && return 0
  else
    [[ "$fpath" == "$zone" ]] && return 0
  fi
  done
  return 1
}

# -------- scan commits --------

TOTAL=${#COMMITS[@]}
TOTAL_ISSUES=0
COMMIT_RESULTS=()

for commit in "${COMMITS[@]}"; do
  git rev-parse --verify "$commit^{commit}" >/dev/null 2>&1 \
  || { echo "Warning: cannot resolve '$commit', skipping" >&2; continue; }

  short="$(git rev-parse --short "$commit")"
  msg="$(git log -1 --format='%s' "$commit" | head -c 60)"
  mapfile -t files < <(git diff-tree --diff-filter=d --no-commit-id -r --name-only "$commit")

  private_hits=()
  conflict_zone_hits=()
  diverged_hits=()

  for f in "${files[@]}"; do
  [[ -z "$f" ]] && continue

  if is_private_path "$f"; then
    private_hits+=("$f")
  fi

  if is_conflict_zone "$f"; then
    # Check if the file has diverged between main and oss/main
    main_hash="$(git rev-parse "main:$f" 2>/dev/null || echo "MISSING")"
    oss_hash="$(git rev-parse "$OSS_BRANCH:$f" 2>/dev/null || echo "MISSING")"
    if [[ "$main_hash" != "$oss_hash" ]]; then
    diverged_hits+=("$f (main≠oss)")
    else
    conflict_zone_hits+=("$f")
    fi
  fi
  done

  issues=0
  result_line="$short $msg"

  MAX_SHOW=5

  if [[ ${#private_hits[@]} -gt 0 ]]; then
  issues=$((issues + ${#private_hits[@]}))
  if [[ $QUIET -eq 0 && $JSON -eq 0 ]]; then
    echo "PRIVATE-PATH $short: ${#private_hits[@]} file(s) touch private-only paths"
    for ((i=0; i<${#private_hits[@]} && i<MAX_SHOW; i++)); do
    echo "  ${private_hits[$i]}"
    done
    (( ${#private_hits[@]} > MAX_SHOW )) && echo "  ... and $((${#private_hits[@]} - MAX_SHOW)) more"
  fi
  fi

  if [[ ${#diverged_hits[@]} -gt 0 ]]; then
  issues=$((issues + ${#diverged_hits[@]}))
  if [[ $QUIET -eq 0 && $JSON -eq 0 ]]; then
    echo "DIVERGED     $short: ${#diverged_hits[@]} file(s) diverged between main and oss"
    printf '  %s\n' "${diverged_hits[@]}"
  fi
  fi

  if [[ ${#conflict_zone_hits[@]} -gt 0 ]]; then
  if [[ $QUIET -eq 0 && $JSON -eq 0 ]]; then
    echo "CONFIG-ZONE  $short: ${#conflict_zone_hits[@]} repo-config file(s) modified (review recommended)"
    printf '  %s\n' "${conflict_zone_hits[@]}"
  fi
  fi

  if [[ $issues -eq 0 && $QUIET -eq 0 && $JSON -eq 0 ]]; then
  echo "CLEAN        $short $msg"
  fi

  TOTAL_ISSUES=$((TOTAL_ISSUES + issues))
done

if [[ $JSON -eq 1 ]]; then
  echo "{\"commits_scanned\": $TOTAL, \"issues\": $TOTAL_ISSUES}"
elif [[ $QUIET -eq 0 ]]; then
  echo ""
  if [[ $TOTAL_ISSUES -eq 0 ]]; then
  echo "CLEAN: $TOTAL commit(s) scanned, no conflicts."
  else
  echo "FOUND: $TOTAL_ISSUES issue(s) across $TOTAL commit(s)."
  echo "Review before pulling: private-path and diverged files need manual resolution."
  fi
fi

[[ $TOTAL_ISSUES -eq 0 ]]
