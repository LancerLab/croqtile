#!/usr/bin/env bash
# Force a remote branch to match the local branch exactly.
#
# This is a DESTRUCTIVE operation: the remote branch is overwritten with the
# local version via force-push. Use only when normal fast-forward sync fails
# due to diverged histories (e.g., after a rebase or history rewrite on the
# authoritative side).
#
# Safety: requires explicit double-confirmation with branch details shown.
#
# Typical use cases:
#   - Mirror's oss/main diverged from local oss/main
#   - A remote was accidentally force-pushed and needs to be reset
#   - Bootstrapping a new remote with a specific branch state
#
# This script runs from the WSL sync hub (central authority for all branches).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BRANCH=""
REMOTE=""
REMOTE_BRANCH=""
DRY_RUN=0
SKIP_CONFIRM=0
ALL_REMOTES=0

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

usage() {
  cat <<'EOF'
Usage: force_sync_branch.sh [options] <local-branch> <remote>[:<remote-branch>]

Force a remote branch to match the local branch via force-push (--force-with-lease).

Arguments:
  <local-branch>           Local branch to use as the source of truth
  <remote>                 Remote name (e.g., origin, oss-shadow, public)
  <remote>:<remote-branch> Optional: map to a different remote branch name
                           (e.g., oss-shadow:main pushes oss/main -> main)

Options:
  --all-remotes     Force-sync the branch to ALL known remotes for it
  --dry-run, -n     Show what would happen without pushing
  --yes             Skip interactive confirmation (for scripts; still shows details)
  -h, --help        Show this help

Examples:
  # Force mirror's oss/main to match local
  force_sync_branch.sh oss/main origin

  # Force GitHub's main to match local oss/main
  force_sync_branch.sh oss/main public:main

  # Force oss/main to all 3 remotes (origin, oss-shadow, public)
  force_sync_branch.sh --all-remotes oss/main

  # Dry run to see what would happen
  force_sync_branch.sh -n main origin

EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-remotes)  ALL_REMOTES=1; shift ;;
    --dry-run|-n)   DRY_RUN=1; shift ;;
    --yes)          SKIP_CONFIRM=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    -*)             echo "Unknown option: $1" >&2; usage; exit 2 ;;
    *)              POSITIONAL+=("$1"); shift ;;
  esac
done

lgit() { git -C "$REPO_ROOT" "$@"; }

# -------- parse arguments --------

if [[ ${#POSITIONAL[@]} -lt 1 ]]; then
  echo "Error: local branch required" >&2
  usage
  exit 2
fi

BRANCH="${POSITIONAL[0]}"

if [[ $ALL_REMOTES -eq 0 ]]; then
  if [[ ${#POSITIONAL[@]} -lt 2 ]]; then
    echo "Error: remote required (or use --all-remotes)" >&2
    usage
    exit 2
  fi

  remote_spec="${POSITIONAL[1]}"
  if [[ "$remote_spec" == *:* ]]; then
    REMOTE="${remote_spec%%:*}"
    REMOTE_BRANCH="${remote_spec#*:}"
  else
    REMOTE="$remote_spec"
    REMOTE_BRANCH="$BRANCH"
  fi
fi

# -------- validate local branch --------

cd "$REPO_ROOT"

if ! lgit show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo -e "${RED}Error: local branch '$BRANCH' does not exist${NC}" >&2
  exit 1
fi

LOCAL_SHA="$(lgit rev-parse "$BRANCH")"
LOCAL_SHORT="$(lgit rev-parse --short "$BRANCH")"
LOCAL_MSG="$(lgit log -1 --format='%s' "$BRANCH" | head -c 72)"
LOCAL_DATE="$(lgit log -1 --format='%ci' "$BRANCH")"
LOCAL_COUNT="$(lgit rev-list --count "$BRANCH")"

# -------- build target list --------

declare -a TARGETS=()  # "remote:remote_branch" pairs

build_all_remote_targets() {
  local branch="$1"

  # For oss/main, the known mapping is:
  #   origin      -> oss/main
  #   oss-shadow  -> main
  #   public      -> main
  # For main:
  #   origin      -> main

  for r in $(lgit remote); do
    case "$branch" in
      oss/main)
        case "$r" in
          origin)      TARGETS+=("$r:$branch") ;;
          oss-shadow)  TARGETS+=("$r:main") ;;
          public)      TARGETS+=("$r:main") ;;
        esac
        ;;
      main)
        case "$r" in
          origin)      TARGETS+=("$r:$branch") ;;
        esac
        ;;
      *)
        # Generic: push to origin with same name
        if [[ "$r" == "origin" ]]; then
          TARGETS+=("$r:$branch")
        fi
        ;;
    esac
  done
}

if [[ $ALL_REMOTES -eq 1 ]]; then
  build_all_remote_targets "$BRANCH"
  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "Error: no known remote targets for branch '$BRANCH'" >&2
    exit 1
  fi
else
  TARGETS+=("$REMOTE:$REMOTE_BRANCH")
fi

# -------- gather remote state for each target --------

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║            FORCE SYNC BRANCH -- DESTRUCTIVE OPERATION       ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Source (local):${NC}"
echo -e "  Branch:  ${GREEN}$BRANCH${NC}"
echo -e "  SHA:     $LOCAL_SHORT ($LOCAL_SHA)"
echo -e "  Message: $LOCAL_MSG"
echo -e "  Date:    $LOCAL_DATE"
echo -e "  Commits: $LOCAL_COUNT"
echo ""

echo -e "${BOLD}Targets:${NC}"
echo ""

declare -a REMOTE_SHAS=()
declare -a REMOTE_STATUS=()

for target in "${TARGETS[@]}"; do
  r="${target%%:*}"
  rb="${target#*:}"

  # Fetch latest from remote
  lgit fetch "$r" 2>/dev/null || true

  remote_ref="$r/$rb"
  if lgit show-ref --verify --quiet "refs/remotes/$remote_ref" 2>/dev/null; then
    rsah="$(lgit rev-parse "$remote_ref")"
    rshort="$(lgit rev-parse --short "$remote_ref")"
    rmsg="$(lgit log -1 --format='%s' "$remote_ref" | head -c 72)"
    rdate="$(lgit log -1 --format='%ci' "$remote_ref")"
    rcount="$(lgit rev-list --count "$remote_ref")"

    REMOTE_SHAS+=("$rsah")

    if [[ "$rsah" == "$LOCAL_SHA" ]]; then
      REMOTE_STATUS+=("up-to-date")
      echo -e "  ${GREEN}✓ $r:$rb${NC} -- already matches"
      echo "    SHA: $rshort  ($rmsg)"
    elif lgit merge-base --is-ancestor "$rsah" "$LOCAL_SHA"; then
      REMOTE_STATUS+=("behind")
      behind="$(lgit rev-list --count "$rsah..$LOCAL_SHA")"
      echo -e "  ${YELLOW}↑ $r:$rb${NC} -- behind by $behind commit(s), fast-forward possible"
      echo "    Remote: $rshort  ($rmsg)"
      echo "    Date:   $rdate"
    elif lgit merge-base --is-ancestor "$LOCAL_SHA" "$rsah"; then
      REMOTE_STATUS+=("ahead")
      ahead="$(lgit rev-list --count "$LOCAL_SHA..$rsah")"
      echo -e "  ${RED}↓ $r:$rb${NC} -- remote is AHEAD by $ahead commit(s)"
      echo "    Remote: $rshort  ($rmsg)"
      echo "    Date:   $rdate"
      echo -e "    ${RED}WARNING: force-push will DISCARD $ahead remote commit(s)!${NC}"
    else
      REMOTE_STATUS+=("diverged")
      echo -e "  ${RED}✗ $r:$rb${NC} -- DIVERGED"
      echo "    Remote: $rshort  ($rmsg)"
      echo "    Date:   $rdate"
      local_only="$(lgit rev-list --count "$rsah..$LOCAL_SHA" 2>/dev/null || echo '?')"
      remote_only="$(lgit rev-list --count "$LOCAL_SHA..$rsah" 2>/dev/null || echo '?')"
      echo -e "    ${RED}Local has $local_only unique, remote has $remote_only unique commit(s)${NC}"
      echo -e "    ${RED}Force-push will DISCARD the $remote_only remote-only commit(s)!${NC}"
    fi
  else
    REMOTE_SHAS+=("")
    REMOTE_STATUS+=("new")
    echo -e "  ${YELLOW}+ $r:$rb${NC} -- remote branch does not exist (will create)"
  fi
  echo ""
done

# Check if anything needs doing
needs_push=0
for status in "${REMOTE_STATUS[@]}"; do
  [[ "$status" != "up-to-date" ]] && needs_push=1
done

if [[ $needs_push -eq 0 ]]; then
  echo -e "${GREEN}All targets already match. Nothing to do.${NC}"
  exit 0
fi

# -------- confirmation --------

if [[ $DRY_RUN -eq 1 ]]; then
  echo -e "${YELLOW}[DRY RUN] Would force-push the above targets. No changes made.${NC}"
  exit 0
fi

if [[ $SKIP_CONFIRM -eq 0 ]]; then
  echo -e "${BOLD}${RED}This will FORCE-PUSH local '$BRANCH' to the above target(s).${NC}"
  echo -e "${RED}Any commits on the remote that are not in the local branch will be LOST.${NC}"
  echo ""
  read -rp "Type the branch name to confirm [$BRANCH]: " confirm1

  if [[ "$confirm1" != "$BRANCH" ]]; then
    echo "Aborted. (You typed '$confirm1', expected '$BRANCH')"
    exit 1
  fi

  echo ""
  echo -e "${BOLD}Second confirmation:${NC}"

  for i in "${!TARGETS[@]}"; do
    target="${TARGETS[$i]}"
    status="${REMOTE_STATUS[$i]}"
    r="${target%%:*}"
    rb="${target#*:}"

    case "$status" in
      up-to-date) continue ;;
      behind)     echo -e "  ${YELLOW}$r:$rb${NC} -- will fast-forward (safe)" ;;
      ahead)      echo -e "  ${RED}$r:$rb${NC} -- will DISCARD remote commits (destructive)" ;;
      diverged)   echo -e "  ${RED}$r:$rb${NC} -- will OVERWRITE diverged history (destructive)" ;;
      new)        echo -e "  ${YELLOW}$r:$rb${NC} -- will create new branch" ;;
    esac
  done

  echo ""
  read -rp "Type YES (uppercase) to proceed: " confirm2

  if [[ "$confirm2" != "YES" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

# -------- execute force-push --------

echo ""
echo -e "${BOLD}Executing force-push...${NC}"
echo ""

errors=0
for i in "${!TARGETS[@]}"; do
  target="${TARGETS[$i]}"
  status="${REMOTE_STATUS[$i]}"
  r="${target%%:*}"
  rb="${target#*:}"

  if [[ "$status" == "up-to-date" ]]; then
    echo -e "  ${GREEN}$r:$rb${NC} -- skipped (already matches)"
    continue
  fi

  echo -n "  $r:$rb -- "

  if lgit push --force-with-lease "$r" "$BRANCH:$rb" 2>&1; then
    echo -e "${GREEN}OK${NC}"
  else
    echo -e "${RED}FAILED${NC}"
    errors=$((errors + 1))
  fi
done

echo ""
if [[ $errors -gt 0 ]]; then
  echo -e "${RED}$errors push(es) failed. Check output above.${NC}"
  exit 1
else
  echo -e "${GREEN}All targets synced successfully.${NC}"
fi

# -------- post-push verification --------

echo ""
echo -e "${BOLD}Verification:${NC}"

for target in "${TARGETS[@]}"; do
  r="${target%%:*}"
  rb="${target#*:}"

  lgit fetch "$r" 2>/dev/null || true
  remote_ref="$r/$rb"

  if lgit show-ref --verify --quiet "refs/remotes/$remote_ref" 2>/dev/null; then
    rsah="$(lgit rev-parse "$remote_ref")"
    if [[ "$rsah" == "$LOCAL_SHA" ]]; then
      echo -e "  ${GREEN}✓ $r:$rb${NC} = $LOCAL_SHORT"
    else
      echo -e "  ${RED}✗ $r:$rb${NC} = $(lgit rev-parse --short "$remote_ref") (MISMATCH!)"
    fi
  else
    echo -e "  ${RED}✗ $r:$rb${NC} -- ref not found after push"
  fi
done
echo ""
