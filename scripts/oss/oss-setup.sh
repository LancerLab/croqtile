#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

PUBLIC_REMOTE="public"
PUBLIC_URL="git@github.com:LancerLab/croqtile.git"
OSS_BRANCH="oss/main"

usage() {
  cat <<'EOF'
Usage: oss-setup.sh [-r <remote-name>] [-u <remote-url>] [-b <branch>] [-h]

One-time setup: adds the public GitHub remote and creates a local branch
tracking its main.

Options:
  -r <name>   Remote name (default: public)
  -u <url>    Remote URL (default: git@github.com:LancerLab/choreo.git)
  -b <branch> Local branch name (default: oss/main)
             Use oss/<name> namespace for releases, e.g. oss/release/v1.0
  -h          Show help
EOF
}

while getopts ":r:u:b:h" opt; do
  case "$opt" in
    r) PUBLIC_REMOTE="$OPTARG" ;;
    u) PUBLIC_URL="$OPTARG" ;;
    b) OSS_BRANCH="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Error: -$OPTARG needs an argument"; usage; exit 2 ;;
    \?) echo "Error: unknown option -$OPTARG"; usage; exit 2 ;;
  esac
done

cd "$REPO_ROOT"

if git remote get-url "$PUBLIC_REMOTE" >/dev/null 2>&1; then
  existing_url="$(git remote get-url "$PUBLIC_REMOTE")"
  echo "Remote '$PUBLIC_REMOTE' already exists -> $existing_url"
  if [[ "$existing_url" != "$PUBLIC_URL" ]]; then
    echo "WARNING: existing URL differs from requested: $PUBLIC_URL"
    echo "  To update: git remote set-url $PUBLIC_REMOTE $PUBLIC_URL"
  fi
else
  echo "Adding remote '$PUBLIC_REMOTE' -> $PUBLIC_URL"
  git remote add "$PUBLIC_REMOTE" "$PUBLIC_URL"
fi

echo "Fetching from '$PUBLIC_REMOTE'..."
git fetch "$PUBLIC_REMOTE"

if git show-ref --verify --quiet "refs/heads/$OSS_BRANCH"; then
  echo "Branch '$OSS_BRANCH' already exists."
  echo "  Tip: $(git log -1 --oneline "$OSS_BRANCH")"
else
  if git show-ref --verify --quiet "refs/remotes/$PUBLIC_REMOTE/main"; then
    echo "Creating branch '$OSS_BRANCH' tracking '$PUBLIC_REMOTE/main'..."
    git branch "$OSS_BRANCH" "$PUBLIC_REMOTE/main"
    git branch -u "$PUBLIC_REMOTE/main" "$OSS_BRANCH"
  else
    echo "Remote '$PUBLIC_REMOTE/main' not found."
    echo "Creating orphan '$OSS_BRANCH' for initial bootstrap..."
    local_branch="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
    git checkout --orphan "$OSS_BRANCH"
    git rm -rf . >/dev/null 2>&1 || true
    git commit --allow-empty -m "Initial oss branch"
    if [[ -n "$local_branch" ]]; then
      git checkout "$local_branch"
    fi
  fi
fi

echo ""
echo "Setup complete."
echo "  Remote:  $PUBLIC_REMOTE -> $(git remote get-url "$PUBLIC_REMOTE")"
echo "  Branch:  $OSS_BRANCH"
echo ""
echo "Workflow:"
echo "  Push to public:   scripts/oss/oss-push.sh <commit>"
echo "  Pull from public: scripts/oss/oss-pull.sh <commit>"
echo "  Scan for leaks:   scripts/oss/oss-scan.sh --tree $OSS_BRANCH"
