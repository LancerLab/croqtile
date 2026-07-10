#!/usr/bin/env bash
# Sync croqtile-skills from its origin remote.
#
# Lives in scripts/oss/ inside the choreo repo, so it always works
# even when croqtile-skills is not yet installed.
#
# Usage:
#   bash scripts/oss/sync-skills.sh
#   bash scripts/oss/sync-skills.sh --once
#   bash scripts/oss/sync-skills.sh --loop --poll-interval 300

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHOREO_ROOT="$(realpath "$SCRIPT_DIR/../..")"

# Auto-detect skills repo root.
if [[ -d "$CHOREO_ROOT/extern/croqtile-skills/.git" ]]; then
  SKILLS_ROOT="$CHOREO_ROOT/extern/croqtile-skills"
elif [[ -d "$HOME/dev/croqtile-skills/.git" ]]; then
  SKILLS_ROOT="$HOME/dev/croqtile-skills"
else
  echo "FATAL: cannot locate croqtile-skills repo" >&2
  exit 1
fi

REMOTE="${SKILLS_REMOTE:-origin}"
BRANCH="${SKILLS_BRANCH:-main}"

LOOP=0
PAUSE_SECS=300
DRY_RUN=0
QUIET=0
ONCE=1

usage() {
  cat <<'EOF'
Usage: sync-skills.sh [options]

Sync croqtile-skills from its origin remote.

Options:
  --once              Run one cycle and exit (default)
  --loop              Run continuously
  --poll-interval N   Seconds between cycles (default: 300)
  --dry-run, -n       Preview only
  -q, --quiet         Suppress non-error output
  -h, --help          This help

Environment:
  SKILLS_REMOTE       Remote name (default: origin)
  SKILLS_BRANCH       Branch name (default: main)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)           ONCE=1; LOOP=0; shift ;;
    --loop)           LOOP=1; ONCE=0; shift ;;
    --poll-interval)  PAUSE_SECS="$2"; shift 2 ;;
    --dry-run|-n)     DRY_RUN=1; shift ;;
    -q|--quiet)       QUIET=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)                echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

lgit() { git -C "$SKILLS_ROOT" "$@"; }

timestamp() { date '+%F %T'; }
log()     { [[ $QUIET -eq 0 ]] && echo "[$(timestamp)] $*"; }

run_cycle() {
  local errors=0
  log "=== Skills sync cycle start ==="

  # Fetch from remote
  log "Fetching $REMOTE..."
  if ! lgit fetch "$REMOTE" --prune 2>&1; then
    log "ERROR: fetch $REMOTE failed"
    return 1
  fi

  local local_tip remote_tip
  local_tip="$(lgit rev-parse HEAD)"
  remote_tip="$(lgit rev-parse "$REMOTE/$BRANCH" 2>/dev/null || echo "")"

  if [[ -z "$remote_tip" ]]; then
    log "ERROR: no $REMOTE/$BRANCH ref"
    return 1
  fi

  if [[ "$local_tip" == "$remote_tip" ]]; then
    log "Already up to date ($(lgit log -1 --oneline HEAD))"
    log "=== Skills sync cycle end (no changes) ==="
    return 0
  fi

  if lgit merge-base --is-ancestor "$local_tip" "$remote_tip"; then
    log "Fast-forwarding to $REMOTE/$BRANCH..."
    if [[ $DRY_RUN -eq 1 ]]; then
      log "  [dry-run] would rebase onto $REMOTE/$BRANCH"
    else
      lgit rebase "$REMOTE/$BRANCH" 2>&1 || {
        log "ERROR: rebase failed, attempting abort"
        lgit rebase --abort 2>/dev/null || true
        errors=1
      }
    fi
  elif lgit merge-base --is-ancestor "$remote_tip" "$local_tip"; then
    log "Local ahead of $REMOTE/$BRANCH."
    if [[ $DRY_RUN -eq 1 ]]; then
      log "  [dry-run] would push to $REMOTE"
    else
      lgit push "$REMOTE" "$BRANCH:$BRANCH" 2>&1 || {
        log "ERROR: push failed"
        errors=1
      }
    fi
  else
    log "ERROR: local and $REMOTE/$BRANCH have diverged"
    log "  Local:  $(lgit log -1 --oneline HEAD)"
    log "  Remote: $(lgit log -1 --oneline "$REMOTE/$BRANCH")"
    log "  Manual resolution required."
    errors=1
  fi

  if [[ $errors -eq 0 ]]; then
    log "Synced to $(lgit log -1 --oneline HEAD)"
  fi
  log "=== Skills sync cycle end (errors=$errors) ==="
  return $errors
}

main() {
  cd "$SKILLS_ROOT"
  log "sync-skills.sh starting (repo=$SKILLS_ROOT, remote=$REMOTE, branch=$BRANCH)"

  if [[ $LOOP -eq 1 ]]; then
    log "Starting sync daemon (interval=${PAUSE_SECS}s)..."
    while true; do
      set +e; run_cycle; rc=$?; set -e
      log "Next cycle in ${PAUSE_SECS}s..."
      sleep "$PAUSE_SECS"
    done
  else
    run_cycle
  fi
}

main
