#!/usr/bin/env bash
# Unified sync orchestrator: combines sync_remotes (origin <-> mirror) with
# oss sync (main <-> oss/main <-> GitHub) in a single ordered cycle.
#
# Topology:
#   oss-shadow (GitHub) <-> local:oss/main | local:main <-> origin <-> mirror
#
# Each cycle (strict order):
#   1. sync_remotes --once  (origin <-> mirror, all branches)
#   2. main -> oss/main     (cherry-pick new commits, scan gated)
#   3. oss/main -> origin + oss-shadow + public  (push if ahead)
#   4. public -> main       (scan + auto-pull clean commits)
#   5. main -> origin       (push if ahead from pull)
#
# Designed to run on a dedicated sync machine (clean worktree, no dev work).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

POLL_INTERVAL=${POLL_INTERVAL:-120}
RUN_ONCE=0
DRY_RUN=0
LOG_ENABLED=0
LOG_FILE=""
SKIP_MIRROR=0

OSS_BRANCH="oss/main"
OSS_REMOTE="oss-shadow"
PUBLIC_REMOTE="public"
MAIN_BRANCH="main"

ALERT_EMAIL=${ALERT_EMAIL:-xiaofeng.guan@enflame-tech.com}

usage() {
  cat <<'EOF'
Usage: sync_all.sh [options]

Unified sync orchestrator: origin <-> mirror + main <-> oss/main <-> GitHub.

Options:
  --once             Run exactly one sync cycle, then exit
  --dry-run          Show what would happen without making changes
  --poll-interval N  Seconds between cycles (default: 120)
  --skip-mirror      Skip the sync_remotes mirror step
  --log              Enable file logging
  --log-file PATH    Log to specific file
  --alert-email ADDR Alert email address
  -h, --help         Show this help

Environment variables:
  POLL_INTERVAL, ALERT_EMAIL, MIRROR_HOST, LOCAL_REPO, MIRROR_REPO
  (all sync_remotes.sh variables are forwarded)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)           RUN_ONCE=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --poll-interval)  POLL_INTERVAL="$2"; shift 2 ;;
    --skip-mirror)    SKIP_MIRROR=1; shift ;;
    --log)            LOG_ENABLED=1; shift ;;
    --log-file)       LOG_ENABLED=1; LOG_FILE="$2"; shift 2 ;;
    --alert-email)    ALERT_EMAIL="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *)                echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# -------- state directory --------

STATE_DIR="$REPO_ROOT/.git/sync-all"
mkdir -p "$STATE_DIR"

LAST_OSS_PUSH_FILE="$STATE_DIR/last-oss-push-sha"
CYCLE_LOG="$STATE_DIR/last-cycle.log"

if [[ $LOG_ENABLED -eq 1 ]]; then
  [[ -z "$LOG_FILE" ]] && LOG_FILE="$STATE_DIR/sync-all.log"
  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

# -------- helpers --------

timestamp() { date '+%F %T'; }
log()       { echo "[$(timestamp)] $*"; }
log_section() { echo ""; log "======== $* ========"; }

notify() {
  local title="$1" msg="$2"
  if command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /C "msg %USERNAME% \"$title: $(echo "$msg" | tr '\n' ' ')\"" 2>/dev/null || true
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send "$title" "$msg" 2>/dev/null || true
  fi
  echo "[notify] $title: $msg" >&2
}

alert() {
  local subject="$1" body="$2"
  notify "$subject" "$body"
  if command -v mail >/dev/null 2>&1; then
    printf '%b\n' "$body" | mail -s "$subject" "$ALERT_EMAIL" 2>/dev/null || true
  fi
}

lgit() { git -C "$REPO_ROOT" "$@"; }

ensure_branch() {
  local branch="$1"
  lgit show-ref --verify --quiet "refs/heads/$branch" || {
    log "ERROR: branch '$branch' does not exist"
    return 1
  }
}

ensure_oss_branch() {
  if lgit show-ref --verify --quiet "refs/heads/$OSS_BRANCH"; then
    return 0
  fi

  log "Branch '$OSS_BRANCH' does not exist -- attempting to create it"

  # Try fetching from oss-shadow first
  if lgit remote get-url "$OSS_REMOTE" >/dev/null 2>&1; then
    lgit fetch "$OSS_REMOTE" 2>/dev/null || true
  fi

  if lgit show-ref --verify --quiet "refs/remotes/$OSS_REMOTE/main"; then
    lgit branch "$OSS_BRANCH" "$OSS_REMOTE/main"
    lgit branch -u "$OSS_REMOTE/main" "$OSS_BRANCH"
    log "Created $OSS_BRANCH tracking $OSS_REMOTE/main"
    return 0
  fi

  # Try public remote
  if lgit remote get-url "$PUBLIC_REMOTE" >/dev/null 2>&1; then
    lgit fetch "$PUBLIC_REMOTE" 2>/dev/null || true
  fi

  if lgit show-ref --verify --quiet "refs/remotes/$PUBLIC_REMOTE/main"; then
    lgit branch "$OSS_BRANCH" "$PUBLIC_REMOTE/main"
    log "Created $OSS_BRANCH from $PUBLIC_REMOTE/main"
    return 0
  fi

  # Try oss-setup.sh
  local setup_script="$SCRIPT_DIR/oss/oss-setup.sh"
  if [[ -f "$setup_script" ]]; then
    log "Running oss-setup.sh to bootstrap..."
    (cd "$REPO_ROOT" && bash "$setup_script") 2>&1 || true

    if lgit show-ref --verify --quiet "refs/heads/$OSS_BRANCH"; then
      log "oss-setup.sh created $OSS_BRANCH"
      return 0
    fi
  fi

  log "ERROR: could not create $OSS_BRANCH (no remote data available)"
  return 1
}

# -------- oss tool staging --------
# oss-push/pull/scan live under scripts/oss/ which is excluded from oss/main.
# We copy them to a temp dir so they work regardless of which branch is checked out.

TOOL_TMPDIR=""

stage_oss_tools() {
  TOOL_TMPDIR="$(mktemp -d)"
  local oss_dir="$SCRIPT_DIR/oss"
  for f in oss-push.sh oss-pull.sh oss-scan.sh oss-pull-scan.sh; do
    if [[ -f "$oss_dir/$f" ]]; then
      cp "$oss_dir/$f" "$TOOL_TMPDIR/"
      chmod +x "$TOOL_TMPDIR/$f"
    fi
  done
  for f in os_kw.txt oss_exclude_paths.txt; do
    [[ -f "$oss_dir/$f" ]] && cp "$oss_dir/$f" "$TOOL_TMPDIR/"
  done
}

cleanup_oss_tools() {
  [[ -n "$TOOL_TMPDIR" && -d "$TOOL_TMPDIR" ]] && rm -rf "$TOOL_TMPDIR"
  TOOL_TMPDIR=""
}

# -------- Step 1: sync_remotes --------

step_sync_remotes() {
  log_section "Step 1: sync_remotes (origin <-> mirror)"

  if [[ $SKIP_MIRROR -eq 1 ]]; then
    log "Skipped (--skip-mirror)"
    return 0
  fi

  local sr_script="$SCRIPT_DIR/sync_remotes.sh"
  if [[ ! -x "$sr_script" ]]; then
    log "WARNING: sync_remotes.sh not found at $sr_script; skipping mirror sync"
    return 0
  fi

  local sr_args=(--once)
  [[ $DRY_RUN -eq 1 ]] && sr_args+=(--dry-run)

  if ! bash "$sr_script" "${sr_args[@]}" 2>&1; then
    alert "sync-all: mirror sync failed" "sync_remotes.sh exited with error"
    return 1
  fi

  log "Mirror sync completed."
}

# -------- Step 2: main -> oss/main (cherry-pick with scan) --------

get_last_synced_sha() {
  if [[ -f "$LAST_OSS_PUSH_FILE" ]]; then
    cat "$LAST_OSS_PUSH_FILE"
  else
    lgit log "$OSS_BRANCH" -1 --format='%B' | grep -oP '(?<=cherry picked from )\w+' | head -1 || \
    lgit merge-base "$MAIN_BRANCH" "$OSS_BRANCH" 2>/dev/null || \
    echo ""
  fi
}

save_last_synced_sha() {
  echo "$1" > "$LAST_OSS_PUSH_FILE"
}

step_push_to_oss() {
  log_section "Step 2: main -> oss/main (cherry-pick new commits)"

  ensure_branch "$MAIN_BRANCH" || return 1
  ensure_oss_branch || return 1

  local last_sha
  last_sha="$(get_last_synced_sha)"

  if [[ -z "$last_sha" ]]; then
    log "WARNING: cannot determine last synced SHA. Set manually:"
    log "  echo <sha> > $LAST_OSS_PUSH_FILE"
    return 1
  fi

  if ! lgit rev-parse --verify "$last_sha^{commit}" >/dev/null 2>&1; then
    log "WARNING: last synced SHA '$last_sha' is not a valid commit"
    return 1
  fi

  local new_commits
  mapfile -t new_commits < <(lgit rev-list --reverse "$last_sha..$MAIN_BRANCH")

  if [[ ${#new_commits[@]} -eq 0 ]]; then
    log "No new commits on $MAIN_BRANCH since $last_sha"
    return 0
  fi

  log "${#new_commits[@]} new commit(s) to process"

  local pushed=0 skipped=0 failed=0
  local orig_branch
  orig_branch="$(lgit symbolic-ref --short HEAD 2>/dev/null || lgit rev-parse --short HEAD)"

  stage_oss_tools

  local push_args=()
  [[ $DRY_RUN -eq 1 ]] && push_args+=(-n)
  push_args+=(-k "$TOOL_TMPDIR/os_kw.txt")
  push_args+=(-e "$TOOL_TMPDIR/oss_exclude_paths.txt")

  for sha in "${new_commits[@]}"; do
    local short
    short="$(lgit rev-parse --short "$sha")"
    local msg
    msg="$(lgit log -1 --format='%s' "$sha" | head -c 60)"

    log "Processing $short: $msg"

    if [[ $DRY_RUN -eq 1 ]]; then
      log "  [dry-run] would cherry-pick $short"
      pushed=$((pushed + 1))
      save_last_synced_sha "$sha"
      continue
    fi

    local result
    if OSS_SCAN_REPO_ROOT="$REPO_ROOT" bash "$TOOL_TMPDIR/oss-push.sh" \
        -k "$TOOL_TMPDIR/os_kw.txt" \
        -e "$TOOL_TMPDIR/oss_exclude_paths.txt" \
        "$sha" 2>&1; then
      result="ok"
    else
      result="fail"
    fi

    # Parse output: check if it was skipped (all excluded) vs failed
    if [[ "$result" == "ok" ]]; then
      pushed=$((pushed + 1))
      save_last_synced_sha "$sha"
    else
      # Check if it was just skipped (all files excluded)
      local push_output
      push_output="$(OSS_SCAN_REPO_ROOT="$REPO_ROOT" bash "$TOOL_TMPDIR/oss-push.sh" \
          -n -k "$TOOL_TMPDIR/os_kw.txt" \
          -e "$TOOL_TMPDIR/oss_exclude_paths.txt" \
          "$sha" 2>&1 || true)"

      if echo "$push_output" | grep -q "SKIP.*all.*excluded"; then
        log "  SKIP $short: all files excluded"
        skipped=$((skipped + 1))
        save_last_synced_sha "$sha"
      else
        log "  FAILED $short: scan or apply error"
        failed=$((failed + 1))
        alert "sync-all: oss-push failed" "Commit $short ($msg) failed scan/apply on oss/main"
        break
      fi
    fi
  done

  # Return to original branch
  local cur_branch
  cur_branch="$(lgit symbolic-ref --short HEAD 2>/dev/null || true)"
  if [[ "$cur_branch" != "$orig_branch" ]]; then
    lgit checkout "$orig_branch" 2>/dev/null || true
  fi

  cleanup_oss_tools

  log "Push summary: $pushed pushed, $skipped skipped, $failed failed"
  [[ $failed -gt 0 ]] && return 1
  return 0
}

# -------- Step 3: push oss/main to all remotes (origin + oss-shadow) --------

push_oss_to_one_remote() {
  local remote="$1" remote_branch="$2"

  local remote_ref="$remote/$remote_branch"
  if ! lgit show-ref --verify --quiet "refs/remotes/$remote_ref" 2>/dev/null; then
    lgit fetch "$remote" 2>/dev/null || {
      log "  WARNING: cannot fetch from '$remote'; skipping"
      return 0
    }
  fi

  local local_sha remote_sha
  local_sha="$(lgit rev-parse "$OSS_BRANCH")"
  remote_sha="$(lgit rev-parse "$remote_ref" 2>/dev/null || echo "")"

  if [[ "$local_sha" == "$remote_sha" ]]; then
    log "  $remote: up-to-date"
    return 0
  fi

  if [[ -z "$remote_sha" ]]; then
    log "  $remote: no remote ref yet; will push"
  elif ! lgit merge-base --is-ancestor "$remote_sha" "$local_sha"; then
    log "  $remote: WARNING diverged -- manual resolution needed"
    alert "sync-all: oss diverged from $remote" \
      "oss/main and $remote_ref have diverged"
    return 1
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    log "  $remote: [dry-run] would push"
    return 0
  fi

  log "  $remote: pushing oss/main -> $remote_branch..."
  if lgit push "$remote" "$OSS_BRANCH:$remote_branch" 2>&1; then
    log "  $remote: pushed successfully."
  else
    alert "sync-all: oss push to $remote failed" \
      "Failed to push oss/main to $remote"
    return 1
  fi
}

step_push_oss_to_remotes() {
  log_section "Step 3: push oss/main to origin + oss-shadow + public"

  ensure_oss_branch || return 1

  local errors=0

  # Push to origin (internal GitLab) -- oss/main -> oss/main
  push_oss_to_one_remote origin "$OSS_BRANCH" || errors=$((errors + 1))

  # Push to oss-shadow (internal mirror of public) -- oss/main -> main
  push_oss_to_one_remote "$OSS_REMOTE" main || errors=$((errors + 1))

  # Push to public (GitHub) -- oss/main -> main
  if [[ "$PUBLIC_REMOTE" != "$OSS_REMOTE" ]]; then
    push_oss_to_one_remote "$PUBLIC_REMOTE" main || errors=$((errors + 1))
  fi

  [[ $errors -gt 0 ]] && return 1
  return 0
}

# -------- Step 4: public -> main (scan + auto-pull) --------

step_pull_from_public() {
  log_section "Step 4: public -> main (scan + auto-pull)"

  # Fetch from public/oss-shadow
  log "Fetching from $OSS_REMOTE..."
  lgit fetch "$OSS_REMOTE" 2>/dev/null || {
    log "WARNING: cannot fetch from $OSS_REMOTE; skipping pull"
    return 0
  }

  # Also try the 'public' remote if different
  if [[ "$PUBLIC_REMOTE" != "$OSS_REMOTE" ]]; then
    lgit fetch "$PUBLIC_REMOTE" 2>/dev/null || true
  fi

  ensure_branch "$MAIN_BRANCH" || return 1
  ensure_oss_branch || return 1

  # Find commits on oss-shadow/main that aren't on oss/main
  local remote_ref="$OSS_REMOTE/main"
  lgit show-ref --verify --quiet "refs/remotes/$remote_ref" || {
    log "No remote ref $remote_ref; nothing to pull"
    return 0
  }

  local oss_tip remote_tip
  oss_tip="$(lgit rev-parse "$OSS_BRANCH")"
  remote_tip="$(lgit rev-parse "$remote_ref")"

  if [[ "$oss_tip" == "$remote_tip" ]]; then
    log "No new public commits."
    return 0
  fi

  if lgit merge-base --is-ancestor "$remote_tip" "$oss_tip"; then
    log "oss/main is already ahead of $remote_ref; no incoming commits"
    return 0
  fi

  # New commits from public
  local incoming
  mapfile -t incoming < <(lgit rev-list --reverse "$oss_tip..$remote_tip")

  if [[ ${#incoming[@]} -eq 0 ]]; then
    log "No new incoming commits."
    return 0
  fi

  log "${#incoming[@]} incoming commit(s) from public"

  stage_oss_tools

  # Scan all incoming commits
  local has_violations=0
  for sha in "${incoming[@]}"; do
    local short
    short="$(lgit rev-parse --short "$sha")"
    local msg
    msg="$(lgit log -1 --format='%s' "$sha" | head -c 60)"

    # Quick keyword + non-ASCII scan on diff
    local diff_content
    diff_content="$(lgit diff "$sha^..$sha" -- | grep -E '^\+' | grep -v '^+++' || true)"

    local violations=0

    # Keyword check
    if [[ -f "$TOOL_TMPDIR/os_kw.txt" ]]; then
      while IFS= read -r kw; do
        [[ -z "$kw" || "$kw" =~ ^[[:space:]]*# ]] && continue
        kw="${kw#'(?i)'}"
        if echo "$diff_content" | grep -qiE "$kw" 2>/dev/null; then
          log "  VIOLATION $short: keyword '$kw'"
          violations=$((violations + 1))
        fi
      done < "$TOOL_TMPDIR/os_kw.txt"
    fi

    # Non-ASCII check
    if echo "$diff_content" | grep -Pq '[^\x00-\x7F]' 2>/dev/null; then
      log "  VIOLATION $short: non-ASCII content"
      violations=$((violations + 1))
    fi

    if [[ $violations -gt 0 ]]; then
      log "  BLOCKED $short: $violations violation(s)"
      has_violations=1
    else
      log "  CLEAN $short: $msg"
    fi
  done

  cleanup_oss_tools

  if [[ $has_violations -gt 0 ]]; then
    alert "sync-all: incoming violations" \
      "${#incoming[@]} public commit(s) have violations. Manual fix needed on oss/main."
    return 1
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    log "[dry-run] would fast-forward oss/main to $remote_ref and pull to main"
    return 0
  fi

  # Fast-forward oss/main to remote tip
  log "Fast-forwarding oss/main to $remote_ref..."
  lgit branch -f "$OSS_BRANCH" "$remote_tip"

  # Cherry-pick new commits to main
  local orig_branch
  orig_branch="$(lgit symbolic-ref --short HEAD 2>/dev/null || lgit rev-parse --short HEAD)"

  if [[ "$orig_branch" != "$MAIN_BRANCH" ]]; then
    lgit checkout "$MAIN_BRANCH" 2>/dev/null || true
  fi

  local pulled=0 pull_failed=0
  for sha in "${incoming[@]}"; do
    local short
    short="$(lgit rev-parse --short "$sha")"

    if lgit cherry-pick --no-commit "$sha" 2>/dev/null; then
      local orig_author orig_date orig_msg clean_msg
      orig_author="$(lgit log -1 --format='%an <%ae>' "$sha")"
      orig_date="$(lgit log -1 --format='%ai' "$sha")"
      orig_msg="$(lgit log -1 --format=%B "$sha")"
      clean_msg="$(echo "$orig_msg" | sed '/^Made-with:/d; /^Generated-by:/d')"

      GIT_AUTHOR_DATE="$orig_date" lgit commit \
        --author="$orig_author" \
        -m "$clean_msg" 2>/dev/null

      log "  OK $short -> $(lgit rev-parse --short HEAD)"
      pulled=$((pulled + 1))
    else
      log "  ERROR $short: cherry-pick conflict"
      lgit cherry-pick --abort 2>/dev/null || lgit reset --hard HEAD 2>/dev/null || true
      pull_failed=$((pull_failed + 1))
      alert "sync-all: pull conflict" "Cherry-pick $short to main failed with conflict"
      break
    fi
  done

  # Restore branch
  local cur
  cur="$(lgit symbolic-ref --short HEAD 2>/dev/null || true)"
  if [[ "$cur" != "$orig_branch" ]]; then
    lgit checkout "$orig_branch" 2>/dev/null || true
  fi

  log "Pull summary: $pulled pulled, $pull_failed failed"

  # Update the last-synced SHA since oss/main advanced
  if [[ $pulled -gt 0 ]]; then
    save_last_synced_sha "$(lgit rev-parse "$MAIN_BRANCH")"
  fi

  [[ $pull_failed -gt 0 ]] && return 1
  return 0
}

# -------- Step 5: push main to origin if ahead --------

step_push_main_to_origin() {
  log_section "Step 5: push main to origin (if ahead)"

  local local_sha origin_sha
  local_sha="$(lgit rev-parse "$MAIN_BRANCH")"
  origin_sha="$(lgit rev-parse "origin/$MAIN_BRANCH" 2>/dev/null || echo "")"

  if [[ "$local_sha" == "$origin_sha" ]]; then
    log "main is up-to-date with origin."
    return 0
  fi

  if [[ -z "$origin_sha" ]]; then
    log "No origin/$MAIN_BRANCH ref; skipping"
    return 0
  fi

  if ! lgit merge-base --is-ancestor "$origin_sha" "$local_sha"; then
    log "WARNING: main has diverged from origin/main"
    alert "sync-all: main diverged" "main and origin/main have diverged"
    return 1
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    log "[dry-run] would push main to origin"
    return 0
  fi

  log "Pushing main to origin..."
  if lgit push origin "$MAIN_BRANCH" 2>&1; then
    log "Pushed successfully."
  else
    alert "sync-all: push main failed" "Failed to push main to origin"
    return 1
  fi
}

# -------- cycle orchestrator --------

run_cycle() {
  local cycle_start cycle_errors=0

  cycle_start="$(date +%s)"
  log_section "SYNC CYCLE START"

  # Ensure we are on main for safety
  local cur_branch
  cur_branch="$(lgit symbolic-ref --short HEAD 2>/dev/null || true)"
  if [[ "$cur_branch" != "$MAIN_BRANCH" ]]; then
    log "Switching to $MAIN_BRANCH..."
    lgit checkout "$MAIN_BRANCH" 2>/dev/null || true
  fi

  # Refresh from origin
  log "Fetching from origin..."
  lgit fetch origin --prune --tags 2>/dev/null || true

  step_sync_remotes || cycle_errors=$((cycle_errors + 1))
  step_push_to_oss  || cycle_errors=$((cycle_errors + 1))
  step_push_oss_to_remotes || cycle_errors=$((cycle_errors + 1))
  step_pull_from_public    || cycle_errors=$((cycle_errors + 1))
  step_push_main_to_origin || cycle_errors=$((cycle_errors + 1))

  local elapsed=$(( $(date +%s) - cycle_start ))
  log_section "SYNC CYCLE END ($elapsed seconds, $cycle_errors error(s))"

  if [[ $cycle_errors -gt 0 ]]; then
    log "WARNING: cycle completed with $cycle_errors error(s)"
  fi

  return $cycle_errors
}

# -------- main --------

log "sync_all.sh starting (poll_interval=${POLL_INTERVAL}s, dry_run=$DRY_RUN)"
log "repo: $REPO_ROOT"
log "state: $STATE_DIR"

if [[ $RUN_ONCE -eq 1 ]]; then
  run_cycle
  exit $?
fi

while true; do
  set +e
  run_cycle
  status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    log "Cycle had errors; will retry next interval"
  fi

  log "Sleeping ${POLL_INTERVAL}s..."
  sleep "$POLL_INTERVAL"
done
