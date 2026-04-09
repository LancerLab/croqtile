#!/usr/bin/env bash
set -euo pipefail

# Cherry-pick commits from the oss/ branch back to main.
# No filtering needed (public -> private is always safe).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

OSS_BRANCH="oss/main"
TARGET_BRANCH="main"

usage() {
  cat <<'EOF'
Usage: oss-pull.sh [options] <commit> [<commit>...]
       oss-pull.sh [options] --range <from>..<to>

Cherry-pick commits from the oss/ branch back to main.
No path filtering or keyword scanning is applied (public to private
is always safe).

Options:
  -b <branch>  Source branch (default: oss/main)
  -t <branch>  Target branch (default: main)
  --range <r>  Expand a revision range via git rev-list --reverse
  -h           Show help
EOF
}

COMMITS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b)      OSS_BRANCH="$2"; shift 2 ;;
    -t)      TARGET_BRANCH="$2"; shift 2 ;;
    --range)
      mapfile -t range_commits < <(git -C "$REPO_ROOT" rev-list --reverse "$2")
      COMMITS+=("${range_commits[@]}")
      shift 2 ;;
    -h)      usage; exit 0 ;;
    -*)      echo "Error: unknown option $1" >&2; usage; exit 2 ;;
    *)       COMMITS+=("$1"); shift ;;
  esac
done

if [[ ${#COMMITS[@]} -eq 0 ]]; then
  echo "Error: no commits specified" >&2
  usage
  exit 2
fi

cd "$REPO_ROOT"

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

TOTAL=${#COMMITS[@]}
PULLED=0
FAILED=0

for commit in "${COMMITS[@]}"; do
  git rev-parse --verify "$commit^{commit}" >/dev/null 2>&1 \
    || { echo "ERROR: cannot resolve '$commit'" >&2; FAILED=$((FAILED+1)); continue; }

  short="$(git rev-parse --short "$commit")"
  echo "Cherry-picking $short from $OSS_BRANCH..."

  if git cherry-pick --no-commit "$commit"; then
    orig_author="$(git log -1 --format='%an <%ae>' "$commit")"
    orig_date="$(git log -1 --format='%ai' "$commit")"
    orig_msg="$(git log -1 --format=%B "$commit")"
    GIT_AUTHOR_DATE="$orig_date" git commit \
      --author="$orig_author" \
      -m "$orig_msg"
    new_sha="$(git rev-parse --short HEAD)"
    echo "OK $short -> $new_sha (author: $orig_author)"
    PULLED=$((PULLED+1))
  else
    echo "ERROR $short: cherry-pick failed (conflict?)."
    echo "  Resolve the conflict, then: git cherry-pick --continue"
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
