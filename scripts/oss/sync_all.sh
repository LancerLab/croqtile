#!/usr/bin/env bash
set -uo pipefail

# Unified sync daemon for the Choreo open-source workflow.
#
# Priority chain for divergence resolution:
#   public > origin > mirror > oss-shadow
# A higher-priority remote is NEVER force-pushed. When diverged,
# the lower-priority repo is rebased onto the higher and force-pushed.
#
# Topology:
#   public       = GitHub (LancerLab/croqtile) -- external contributions
#   origin       = internal GitLab fork (authority for main + oss/main)
#   mirror       = SSH-accessible bundle-mirror (synced via sync_remotes.sh)
#   oss-shadow   = internal mirror of public (era-dev/choreo-open) -- least priority
#
# Cycle (strict order):
#   Phase 0: Ensure remotes (public, oss-shadow) + oss/main branch
#   Phase 1: public -> oss/main (verify scan, warn on violations, rebase)
#            also pushes oss/main to public when local is ahead
#   Phase 2: oss/main public-native commits -> main (via oss-pull.sh --catchup)
#   Phase 3: Run sync_remotes --local-wins to sync origin <-> bundle-mirror
#   Phase 4: FF local branches to origin; sync oss-shadow;
#            if mirror brought changes, re-trigger immediately

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=oss-config.sh
source "$SCRIPT_DIR/oss-config.sh"

SYNC_REMOTES_SCRIPT="$SCRIPT_DIR/sync_remotes.sh"

LOOP=0
DRY_RUN=0
PAUSE_SECS=120
LOG_FILE=""
LOG_ENABLED=0
SKIP_MIRROR=0

STATE_DIR="$REPO_ROOT/.git/sync-all"

export MIRROR_HOST MIRROR_REPO LOCAL_REMOTE MIRROR_REMOTE

usage() {
  cat <<'EOF'
Usage: sync_all.sh [options]

Unified sync daemon: public <-> oss/main <-> origin <-> bundle-mirror.

Priority: public > origin > mirror > oss-shadow.
On divergence, lower-priority is rebased onto higher and force-pushed.

Options:
  --once              Run one cycle and exit (default)
  --loop              Run continuously with pause between cycles
  --log               Like --loop, log to .git/sync-all/sync-all.log
  --log-file PATH     Log to specific file (implies --loop)
  --dry-run, -n       Preview only, no mutations
  --poll-interval N   Pause N seconds between cycles (default: 120)
  --skip-mirror       Skip phase 3 (sync_remotes)
  --mirror-host HOST  Override MIRROR_HOST
  --mirror-repo PATH  Override MIRROR_REPO
  --alert-email ADDR  Alert email address
  -h, --help          Show this help

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)           shift ;;
    --loop)           LOOP=1; shift ;;
    --log)            LOOP=1; LOG_ENABLED=1; shift ;;
    --log-file)       LOOP=1; LOG_ENABLED=1; LOG_FILE="$2"; shift 2 ;;
    --dry-run|-n)     DRY_RUN=1; shift ;;
    --poll-interval)  PAUSE_SECS="$2"; shift 2 ;;
    --skip-mirror)    SKIP_MIRROR=1; shift ;;
    --mirror-host)    MIRROR_HOST="$2"; export MIRROR_HOST; shift 2 ;;
    --mirror-repo)    MIRROR_REPO="$2"; export MIRROR_REPO; shift 2 ;;
    --alert-email)    ALERT_EMAIL="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *)                echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# -------- state & logging --------

mkdir -p "$STATE_DIR"

if [[ $LOG_ENABLED -eq 1 ]]; then
  [[ -z "$LOG_FILE" ]] && LOG_FILE="$STATE_DIR/sync-all.log"
  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

# -------- helpers --------

lgit() { git -C "$REPO_ROOT" "$@"; }

timestamp() { date '+%F %T'; }
log()       { echo "[$(timestamp)] $*"; }
log_phase() { echo ""; log "$(printf '=%.0s' {1..60})"; log "  PHASE $1: $2"; log "$(printf '=%.0s' {1..60})"; }

is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

notify() {
  local title="$1" body="$2"

  if [[ -n "${OSS_WATCH_HOOK:-}" && -x "$OSS_WATCH_HOOK" ]]; then
    "$OSS_WATCH_HOOK" "$title" "$body" && return 0
  fi

  if is_wsl; then
    local ps_path
    ps_path="$(command -v powershell.exe 2>/dev/null || true)"
    if [[ -n "$ps_path" ]]; then
      "$ps_path" -NoProfile -Command "
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
        \$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        \$xml.LoadXml('<toast><visual><binding template=\"ToastText02\"><text id=\"1\">$title</text><text id=\"2\">$body</text></binding></visual></toast>')
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Choreo Sync').Show(\$xml)
      " 2>/dev/null && return 0
    fi
    if command -v wsl-notify-send &>/dev/null; then
      wsl-notify-send --category "Choreo" "$title: $body" && return 0
    fi
  fi

  if command -v notify-send &>/dev/null; then
    notify-send -u critical "$title" "$body" 2>/dev/null && return 0
  fi

  printf '\a' 2>/dev/null || true
  echo "*** [$title] $body ***" >&2
}

warn_highlighted() {
  local title="$1" body="$2"
  echo "" >&2
  echo "${_C_YELLOW}=== WARNING: $title ===${_C_RESET}" >&2
  while IFS= read -r line; do
    echo "${_C_YELLOW}  $line${_C_RESET}" >&2
  done <<< "$body"
  echo "" >&2
}

alert() {
  local subject="$1" body="$2"
  notify "$subject" "$body"
  if [[ -n "$ALERT_EMAIL" ]] && command -v mail >/dev/null 2>&1; then
    printf '%b\n' "$body" | mail -s "$subject" "$ALERT_EMAIL" 2>/dev/null || true
  fi
}

FATAL=0

fatal_error() {
  local title="$1" body="$2"
  FATAL=1
  echo "" >&2
  echo "${_C_RED}!!! FATAL: $title !!!${_C_RESET}" >&2
  while IFS= read -r line; do
    echo "${_C_RED}  $line${_C_RESET}" >&2
  done <<< "$body"
  echo "${_C_RED}  Sync halted. Fix the issue; the next cycle will retry.${_C_RESET}" >&2
  echo "" >&2
  notify "FATAL: $title" "$body"
  alert "sync-all FATAL: $title" "$body"
}

# Push to a remote. Normal push for high-priority (public); on failure → fatal.
safe_push() {
  local remote="$1" refspec="$2" desc="${3:-push}"
  if [[ $DRY_RUN -eq 1 ]]; then
    log "  [dry-run] would $desc"
    return 0
  fi
  local out rc=0
  out="$(lgit push "$remote" "$refspec" 2>&1)" || rc=$?
  if [[ $rc -ne 0 ]]; then
    fatal_error "$desc failed" "git push $remote $refspec\n$out"
    return 1
  fi
  log "  $desc OK"
}

# Force-push to a lower-priority remote; on failure → fatal.
safe_force_push() {
  local remote="$1" refspec="$2" desc="${3:-force-push}"
  if [[ $DRY_RUN -eq 1 ]]; then
    log "  [dry-run] would $desc"
    return 0
  fi
  local out rc=0
  out="$(lgit push --force-with-lease "$remote" "$refspec" 2>&1)" || rc=$?
  if [[ $rc -ne 0 ]]; then
    fatal_error "$desc failed" "git push --force-with-lease $remote $refspec\n$out"
    return 1
  fi
  log "  $desc OK"
}

ensure_on_branch() {
  local target="$1"
  local cur
  cur="$(lgit symbolic-ref --short HEAD 2>/dev/null || true)"
  if [[ "$cur" != "$target" ]]; then
    local out rc=0
    out="$(lgit checkout "$target" 2>&1)" || rc=$?
    if [[ $rc -ne 0 ]]; then
      fatal_error "Cannot checkout $target" "$out"
      return 1
    fi
  fi
}

save_state() {
  mkdir -p "$STATE_DIR"
  echo "$2" > "$STATE_DIR/$1"
}
load_state() {
  local f="$STATE_DIR/$1"
  [[ -f "$f" ]] && cat "$f" || echo ""
}

# Stage oss tools to tmpdir so they survive branch switches
TOOL_TMPDIR=""
stage_oss_tools() {
  [[ -n "$TOOL_TMPDIR" && -d "$TOOL_TMPDIR" ]] && return 0
  TOOL_TMPDIR="$(mktemp -d)"
  for f in oss-push.sh oss-pull.sh oss-scan.sh oss-pull-scan.sh; do
    [[ -f "$SCRIPT_DIR/$f" ]] && cp "$SCRIPT_DIR/$f" "$TOOL_TMPDIR/" && chmod +x "$TOOL_TMPDIR/$f"
  done
  for f in os_kw.txt oss_exclude_paths.txt oss-pull-baseline.txt; do
    [[ -f "$SCRIPT_DIR/$f" ]] && cp "$SCRIPT_DIR/$f" "$TOOL_TMPDIR/"
  done
}
cleanup_oss_tools() {
  [[ -n "$TOOL_TMPDIR" && -d "$TOOL_TMPDIR" ]] && rm -rf "$TOOL_TMPDIR"
  TOOL_TMPDIR=""
}

# Fetch a remote with up to MAX_FETCH_ATTEMPTS retries and exponential
# backoff. Shows the actual error on each failure instead of suppressing
# it, so transient SSH/network glitches can be diagnosed and retried.
MAX_FETCH_ATTEMPTS=${MAX_FETCH_ATTEMPTS:-3}
FETCH_RETRY_BASE_SECS=${FETCH_RETRY_BASE_SECS:-5}

fetch_with_retry() {
  local remote="$1" attempt=0 out rc=0
  while [[ $attempt -lt $MAX_FETCH_ATTEMPTS ]]; do
    attempt=$((attempt + 1))
    out="$(timeout 30 git -C "$REPO_ROOT" fetch "$remote" --prune 2>&1)" && return 0
    rc=$?
    log "  Attempt $attempt/$MAX_FETCH_ATTEMPTS failed (exit $rc): $(echo "$out" | head -2)"
    [[ $attempt -lt $MAX_FETCH_ATTEMPTS ]] && sleep $((attempt * FETCH_RETRY_BASE_SECS))
  done
  return $rc
}

sync_origin_oss_branch() {
  local local_tip origin_tip
  local_tip="$(lgit rev-parse "$OSS_BRANCH" 2>/dev/null || echo "")"
  [[ -z "$local_tip" ]] && { log "Local $OSS_BRANCH not found. Skipping origin sync."; return 0; }

  if ! lgit show-ref --verify --quiet "refs/remotes/origin/$OSS_BRANCH"; then
    log "origin/$OSS_BRANCH missing. Publishing local $OSS_BRANCH to origin..."
    safe_push origin "$OSS_BRANCH:$OSS_BRANCH" "Publish $OSS_BRANCH to origin" || return 1
    return 0
  fi

  origin_tip="$(lgit rev-parse "origin/$OSS_BRANCH" 2>/dev/null || echo "")"
  if [[ "$origin_tip" == "$local_tip" ]]; then
    log "origin/$OSS_BRANCH already up to date."
    return 0
  fi

  if lgit merge-base --is-ancestor "$origin_tip" "$local_tip"; then
    log "Updating origin/$OSS_BRANCH to latest public-aligned tip..."
    safe_push origin "$OSS_BRANCH:$OSS_BRANCH" "Push updated $OSS_BRANCH to origin" || return 1
    return 0
  fi

  if lgit merge-base --is-ancestor "$local_tip" "$origin_tip"; then
    # origin is still ahead of local -- phase0 should have FF'd but didn't
    # (e.g. dry-run mode). Do not force-push stale local over origin.
    log "origin/$OSS_BRANCH is ahead of local; nothing to sync back."
    return 0
  fi

  log "origin/$OSS_BRANCH diverged. Forcing to public-aligned local $OSS_BRANCH (public > origin)."
  safe_force_push origin "$OSS_BRANCH:$OSS_BRANCH" "Force-sync $OSS_BRANCH to origin" || return 1
}

# ======================================================================
# PHASE 0: Ensure remotes and oss/main branch
# ======================================================================

phase0_setup() {
  log_phase 0 "Ensure remotes and oss/main branch"

  local repo_check
  repo_check="$(lgit rev-parse --is-inside-work-tree 2>&1)" || true
  if [[ "$repo_check" != "true" ]]; then
    fatal_error "Not a git repository" "$REPO_ROOT: $repo_check"
    return 1
  fi

  if ! lgit remote get-url "$PUBLIC_REMOTE" >/dev/null 2>&1; then
    log "Adding remote '$PUBLIC_REMOTE' -> $PUBLIC_URL"
    [[ $DRY_RUN -eq 0 ]] && lgit remote add "$PUBLIC_REMOTE" "$PUBLIC_URL"
  else
    log "Remote '$PUBLIC_REMOTE' OK -> $(lgit remote get-url "$PUBLIC_REMOTE")"
  fi

  if ! lgit remote get-url "$OSS_SHADOW_REMOTE" >/dev/null 2>&1; then
    log "Adding remote '$OSS_SHADOW_REMOTE' -> $OSS_SHADOW_URL"
    [[ $DRY_RUN -eq 0 ]] && lgit remote add "$OSS_SHADOW_REMOTE" "$OSS_SHADOW_URL"
  else
    log "Remote '$OSS_SHADOW_REMOTE' OK -> $(lgit remote get-url "$OSS_SHADOW_REMOTE")"
  fi

  for r in origin "$PUBLIC_REMOTE" "$OSS_SHADOW_REMOTE"; do
    if lgit remote get-url "$r" >/dev/null 2>&1; then
      log "Fetching $r..."
      if fetch_with_retry "$r"; then
        log "  $r OK"
      elif [[ "$r" == "$PUBLIC_REMOTE" ]]; then
        warn_highlighted "fetch $PUBLIC_REMOTE failed" \
          "Could not reach $PUBLIC_REMOTE after $MAX_FETCH_ATTEMPTS attempts.\nNew commits pushed to public will NOT be ingested this cycle."
      else
        log "  WARNING: fetch $r failed -- continuing with stale refs"
      fi
    fi
  done

  if lgit show-ref --verify --quiet "refs/heads/$OSS_BRANCH"; then
    log "Branch '$OSS_BRANCH' OK -> $(lgit log -1 --oneline "$OSS_BRANCH")"
    # Fast-forward local oss/main from origin when origin is strictly ahead.
    # Without this, phase1 would see a stale local==public and then
    # sync_origin_oss_branch would force-push the stale tip to origin,
    # wiping any commits that were pushed to origin but not yet local.
    if lgit show-ref --verify --quiet "refs/remotes/origin/$OSS_BRANCH"; then
      local _lsha _osha
      _lsha="$(lgit rev-parse "$OSS_BRANCH")"
      _osha="$(lgit rev-parse "origin/$OSS_BRANCH")"
      if [[ "$_lsha" != "$_osha" ]] && lgit merge-base --is-ancestor "$_lsha" "$_osha"; then
        log "  Fast-forwarding local $OSS_BRANCH -> origin/$OSS_BRANCH"
        lgit branch -f "$OSS_BRANCH" "origin/$OSS_BRANCH"
      fi
    fi
    return 0
  fi

  log "Branch '$OSS_BRANCH' not found -- creating..."
  [[ $DRY_RUN -eq 1 ]] && { log "[dry-run] would create $OSS_BRANCH"; return 0; }

  if lgit show-ref --verify --quiet "refs/remotes/origin/$OSS_BRANCH"; then
    lgit branch "$OSS_BRANCH" "origin/$OSS_BRANCH"
    lgit branch -u "origin/$OSS_BRANCH" "$OSS_BRANCH" 2>/dev/null || true
    log "  Created from origin/$OSS_BRANCH"
  elif lgit show-ref --verify --quiet "refs/remotes/$OSS_SHADOW_REMOTE/main"; then
    lgit branch "$OSS_BRANCH" "$OSS_SHADOW_REMOTE/main"
    log "  Created from $OSS_SHADOW_REMOTE/main"
  elif lgit show-ref --verify --quiet "refs/remotes/$PUBLIC_REMOTE/main"; then
    lgit branch "$OSS_BRANCH" "$PUBLIC_REMOTE/main"
    log "  Created from $PUBLIC_REMOTE/main"
  else
    fatal_error "Cannot create $OSS_BRANCH" "No source branch found on origin, $OSS_SHADOW_REMOTE, or $PUBLIC_REMOTE."
    return 1
  fi
}

# ======================================================================
# PHASE 1: Ingest public repo changes (public > origin)
# ======================================================================

phase1_public_ingest() {
  log_phase 1 "Ingest public repo changes (public has highest priority)"

  if ! lgit show-ref --verify --quiet "refs/remotes/$PUBLIC_REMOTE/main"; then
    log "No $PUBLIC_REMOTE/main ref. Skipping."; return 0
  fi

  # 1.1: Scan any new commits from public (warn only, never blocks sync)
  local new_commits=()
  mapfile -t new_commits < <(lgit rev-list --reverse "$OSS_BRANCH..$PUBLIC_REMOTE/main" 2>/dev/null)

  if [[ ${#new_commits[@]} -gt 0 ]]; then
    log "Found ${#new_commits[@]} new commit(s) on $PUBLIC_REMOTE/main."
    if [[ $DRY_RUN -eq 1 ]]; then
      for c in "${new_commits[@]}"; do log "  [dry-run] $(lgit log -1 --oneline "$c")"; done
    else
      stage_oss_tools
      local scan_rc=0
      local scan_out=""
      scan_out="$(REPO_ROOT="$REPO_ROOT" "$TOOL_TMPDIR/oss-pull-scan.sh" -b "$OSS_BRANCH" -r "$PUBLIC_REMOTE" \
        "${new_commits[@]}" 2>&1)" || scan_rc=$?
      if [[ $scan_rc -eq 1 ]]; then
        local summary=""
        for c in "${new_commits[@]}"; do
          summary+="  $(lgit log -1 --oneline "$c")"$'\n'
        done
        warn_highlighted "Scan violations on public" \
          "$(printf '%d new commit(s) triggered scan warnings:\n%sSync proceeds -- review manually.' \
            "${#new_commits[@]}" "$summary")"
        alert "sync-all: scan violations" \
          "${#new_commits[@]} commit(s) from public triggered scan warnings."
      elif [[ $scan_rc -ne 0 ]]; then
        warn_highlighted "Public scan failed (tooling/infrastructure)" \
          "$(printf 'oss-pull-scan exited %d. Sync proceeds, but review scanner health.\n%s' \
            "$scan_rc" "${scan_out:-<no output>}")"
      elif [[ -n "$scan_out" ]]; then
        echo "$scan_out" | while IFS= read -r line; do
          [[ -n "$line" ]] && log "  [pull-scan] $line"
        done
      fi
    fi
  else
    log "No new commits on $PUBLIC_REMOTE/main."
  fi

  # 1.2: Sync oss/main with public
  local oss_tip public_tip
  oss_tip="$(lgit rev-parse "$OSS_BRANCH")"
  public_tip="$(lgit rev-parse "$PUBLIC_REMOTE/main")"

  if [[ "$oss_tip" == "$public_tip" ]]; then
    log "oss/main matches public. OK."
    sync_origin_oss_branch || return 1
    return 0
  fi

  if lgit merge-base --is-ancestor "$public_tip" "$oss_tip"; then
    # Local is ahead of public: normal push (NEVER force-push to public)
    log "oss/main ahead of public. Pushing to public..."
    safe_push "$PUBLIC_REMOTE" "$OSS_BRANCH:main" "Push oss/main to public" || return 1
    sync_origin_oss_branch || return 1
    return 0
  fi

  if lgit merge-base --is-ancestor "$oss_tip" "$public_tip"; then
    log "Fast-forwarding oss/main to public/main..."
    local saved
    saved="$(lgit symbolic-ref --short HEAD 2>/dev/null || lgit rev-parse --short HEAD)"
    ensure_on_branch "$OSS_BRANCH" || return 1
    lgit merge --ff-only "$PUBLIC_REMOTE/main" 2>/dev/null || lgit reset --hard "$PUBLIC_REMOTE/main"
    ensure_on_branch "$saved" || return 1
    sync_origin_oss_branch || return 1
    return 0
  fi

  # Diverged: rebase origin's oss/main extras onto public (public wins)
  log "oss/main diverged from public. Rebasing onto public (public > origin)..."
  local merge_base saved
  merge_base="$(lgit merge-base "$OSS_BRANCH" "$PUBLIC_REMOTE/main")"
  saved="$(lgit symbolic-ref --short HEAD 2>/dev/null || lgit rev-parse --short HEAD)"
  ensure_on_branch "$OSS_BRANCH" || return 1

  local rebase_ok=0
  if lgit rebase --onto "$PUBLIC_REMOTE/main" "$merge_base" "$OSS_BRANCH" 2>/dev/null; then
    log "  Rebase successful."
    rebase_ok=1
  else
    # rebase --abort fully restores oss/main to its pre-rebase state.
    # Do NOT fall back to reset --hard + cherry-pick: that would discard
    # oss/main's commits before we know all cherry-picks will succeed.
    lgit rebase --abort 2>/dev/null || true
    ensure_on_branch "$saved" 2>/dev/null || true
    fatal_error "Rebase conflict: $OSS_BRANCH vs $PUBLIC_REMOTE/main" \
      "Cannot rebase $OSS_BRANCH onto $PUBLIC_REMOTE/main. Manual resolution required:\n  git -C \"$REPO_ROOT\" checkout $OSS_BRANCH\n  git rebase $PUBLIC_REMOTE/main"
    return 1
  fi

  if [[ $rebase_ok -eq 1 ]]; then
    # Force-push to origin (lower priority) is allowed
    safe_force_push origin "$OSS_BRANCH" "Force-push rebased oss/main to origin" || {
      ensure_on_branch "$saved" 2>/dev/null || true; return 1; }
    # Normal push to public (NEVER force-push to highest priority)
    safe_push "$PUBLIC_REMOTE" "$OSS_BRANCH:main" "Push rebased oss/main to public" || {
      ensure_on_branch "$saved" 2>/dev/null || true; return 1; }
  fi

  ensure_on_branch "$saved" || return 1
  log "Phase 1 complete."
}

# ======================================================================
# PHASE 2: Pull public-native commits to main
# ======================================================================

phase2_pull_to_main() {
  log_phase 2 "Pull public-native commits to main (via oss-pull.sh --catchup)"
  ensure_on_branch "$MAIN_BRANCH" || return 1

  local oss_pull="$SCRIPT_DIR/oss-pull.sh"
  if [[ ! -x "$oss_pull" ]]; then
    fatal_error "oss-pull.sh not found" "Expected at $oss_pull"
    return 1
  fi

  local pre_sha
  pre_sha="$(lgit rev-parse HEAD)"

  # Auto-init baseline on first sync (skips all pre-existing oss/main history)
  local baseline_file="$SCRIPT_DIR/oss-pull-baseline.txt"
  if [[ ! -f "$baseline_file" ]]; then
    log "No pull baseline found -- setting baseline to current oss/main HEAD..."
    bash "$oss_pull" --set-baseline 2>&1 | while IFS= read -r line; do log "  $line"; done || true
  fi

  local pull_args=(-b "$OSS_BRANCH" -t "$MAIN_BRANCH" --catchup --max 5)
  [[ $DRY_RUN -eq 1 ]] && pull_args+=(-n)

  local output rc=0
  output="$(bash "$oss_pull" "${pull_args[@]}" 2>&1)" || rc=$?

  echo "$output" | while IFS= read -r line; do log "  $line"; done

  if echo "$output" | grep -q "Nothing to pull\|fully caught up"; then
    log "main is caught up with oss/main."
  elif [[ $rc -eq 128 ]]; then
    fatal_error "oss-pull.sh fatal (git)" "$(echo "$output" | grep -i 'fatal:' | head -3)"
    return 1
  elif [[ $rc -ne 0 ]]; then
    warn_highlighted "oss-pull.sh failure" \
      "$(printf 'Exit %d. Manual intervention may be needed.\n%s' \
        "$rc" "$(echo "$output" | tail -5)")"
  fi

  if [[ $DRY_RUN -eq 0 ]]; then
    local post_sha
    post_sha="$(lgit rev-parse HEAD)"
    if [[ "$pre_sha" != "$post_sha" ]]; then
      local n; n="$(lgit rev-list --count "$pre_sha..$post_sha")"
      log "Pushing main to origin ($n commit(s) pulled)..."
      safe_push origin "$MAIN_BRANCH" "Push main to origin" || return 1
    fi
  fi
  log "Phase 2 complete."
}

# ======================================================================
# PHASE 3: Sync origin <-> bundle-mirror (origin > mirror)
# ======================================================================

MIRROR_REACHABLE=1

check_mirror_reachable() {
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$MIRROR_HOST" true 2>/dev/null
}

phase3_sync_mirror() {
  log_phase 3 "Sync origin <-> bundle-mirror (origin > mirror)"

  if [[ $SKIP_MIRROR -eq 1 ]]; then log "Skipped (--skip-mirror)"; return 0; fi
  if [[ ! -x "$SYNC_REMOTES_SCRIPT" ]]; then
    fatal_error "sync_remotes.sh not found" "Expected at $SYNC_REMOTES_SCRIPT"
    return 1
  fi

  if ! check_mirror_reachable; then
    log "Mirror host '$MIRROR_HOST' unreachable. Skipping mirror sync."
    MIRROR_REACHABLE=0
    return 0
  fi
  MIRROR_REACHABLE=1

  save_state "pre-sync-main" "$(lgit rev-parse "origin/$MAIN_BRANCH" 2>/dev/null || echo "")"
  save_state "pre-sync-oss" "$(lgit rev-parse "origin/$OSS_BRANCH" 2>/dev/null || echo "")"

  local sr_args=(--once --local-wins)
  [[ $DRY_RUN -eq 1 ]] && sr_args+=(--dry-run)
  log "Running sync_remotes.sh ${sr_args[*]}..."
  local sr_rc=0
  bash "$SYNC_REMOTES_SCRIPT" "${sr_args[@]}" 2>&1 | while IFS= read -r line; do
    [[ -n "$line" ]] && log "  [sync_remotes] $line"
  done
  sr_rc=${PIPESTATUS[0]}
  if [[ $sr_rc -ne 0 ]]; then
    if [[ $sr_rc -eq 128 ]]; then
      fatal_error "sync_remotes.sh fatal" "sync_remotes exited 128. Check the [sync_remotes] log lines above for the first fatal/error details."
      return 1
    fi
    log "  WARNING: sync_remotes.sh exited $sr_rc"
  fi

  [[ $DRY_RUN -eq 1 ]] && { log "Phase 3 complete (dry-run)."; return 0; }

  lgit fetch origin --prune 2>/dev/null || true
  log "Phase 3 complete."
}

# ======================================================================
# PHASE 4: FF local worktree to match origin (already updated by
#           sync_remotes); sync oss-shadow; retrigger if new commits.
# Returns 0 if re-trigger needed, 1 otherwise
# ======================================================================

phase4_post_sync() {
  log_phase 4 "Fast-forward local worktree, sync oss-shadow"

  if [[ $SKIP_MIRROR -eq 1 || $MIRROR_REACHABLE -eq 0 ]]; then
    log "Mirror skipped or unreachable. Syncing oss-shadow only."
    sync_oss_shadow
    return 1
  fi

  lgit fetch origin --prune 2>/dev/null || true

  local pre_main pre_oss post_main post_oss
  pre_main="$(load_state "pre-sync-main")"
  pre_oss="$(load_state "pre-sync-oss")"
  post_main="$(lgit rev-parse "origin/$MAIN_BRANCH" 2>/dev/null || echo "")"
  post_oss="$(lgit rev-parse "origin/$OSS_BRANCH" 2>/dev/null || echo "")"

  local mirror_brought_changes=0

  if [[ -n "$pre_main" && -n "$post_main" && "$pre_main" != "$post_main" ]]; then
    local n; n="$(lgit rev-list --count "$pre_main..$post_main" 2>/dev/null || echo "?")"
    log "Mirror brought $n new commit(s) to origin/main."
    mirror_brought_changes=1
  fi

  if [[ -n "$pre_oss" && -n "$post_oss" && "$pre_oss" != "$post_oss" ]]; then
    local n; n="$(lgit rev-list --count "$pre_oss..$post_oss" 2>/dev/null || echo "?")"
    log "Mirror brought $n new commit(s) to origin/oss/main."
    mirror_brought_changes=1
  fi

  if [[ $mirror_brought_changes -eq 0 ]]; then
    log "No new mirror changes."
    sync_oss_shadow
    return 1
  fi

  [[ $DRY_RUN -eq 1 ]] && {
    log "[dry-run] Would fast-forward local branches and re-trigger."
    return 0
  }

  # Fast-forward local main
  if [[ -n "$pre_main" && -n "$post_main" && "$pre_main" != "$post_main" ]]; then
    ensure_on_branch "$MAIN_BRANCH" || return 1
    log "Fast-forwarding local main to origin/main..."
    local ff_out ff_rc=0
    ff_out="$(lgit pull --ff-only origin "$MAIN_BRANCH" 2>&1)" || ff_rc=$?
    if [[ $ff_rc -ne 0 ]]; then
      log "FF failed (local diverged from origin). Resetting local main to origin/main (origin > local)."
      local reset_out reset_rc=0
      reset_out="$(lgit reset --hard "origin/$MAIN_BRANCH" 2>&1)" || reset_rc=$?
      if [[ $reset_rc -ne 0 ]]; then
        fatal_error "Reset local main to origin failed" "git reset --hard origin/$MAIN_BRANCH\n$reset_out"
        return 1
      fi
      log "  Local main reset to origin/main OK."
    fi
  fi

  # Fast-forward local oss/main
  if [[ -n "$pre_oss" && -n "$post_oss" && "$pre_oss" != "$post_oss" ]]; then
    local saved
    saved="$(lgit symbolic-ref --short HEAD 2>/dev/null || lgit rev-parse --short HEAD)"
    ensure_on_branch "$OSS_BRANCH" || return 1
    log "Fast-forwarding local oss/main to origin/oss/main..."
    local ff_out ff_rc=0
    ff_out="$(lgit pull --ff-only origin "$OSS_BRANCH" 2>&1)" || ff_rc=$?
    if [[ $ff_rc -ne 0 ]]; then
      log "FF failed (local diverged from origin). Resetting local oss/main to origin/oss/main (origin > local)."
      local reset_out reset_rc=0
      reset_out="$(lgit reset --hard "origin/$OSS_BRANCH" 2>&1)" || reset_rc=$?
      if [[ $reset_rc -ne 0 ]]; then
        fatal_error "Reset local oss/main to origin failed" "git reset --hard origin/$OSS_BRANCH\n$reset_out"
        return 1
      fi
      log "  Local oss/main reset to origin/oss/main OK."
    fi
    ensure_on_branch "$saved" || return 1
  fi

  sync_oss_shadow
  log "Phase 4: Mirror changes ingested. Triggering new round."
  return 0
}

sync_oss_shadow() {
  [[ $DRY_RUN -eq 1 ]] && { log "[dry-run] Would sync oss-shadow to match public."; return 0; }

  local public_sha shadow_sha
  public_sha="$(lgit rev-parse "$PUBLIC_REMOTE/main" 2>/dev/null || echo "")"
  shadow_sha="$(lgit rev-parse "$OSS_SHADOW_REMOTE/main" 2>/dev/null || echo "")"

  if [[ -z "$public_sha" || -z "$shadow_sha" ]]; then return 0; fi
  if [[ "$public_sha" == "$shadow_sha" ]]; then
    log "oss-shadow matches public. OK."
    return 0
  fi

  if lgit merge-base --is-ancestor "$shadow_sha" "$public_sha" 2>/dev/null; then
    log "Syncing oss-shadow to match public (fast-forward)..."
    lgit push "$OSS_SHADOW_REMOTE" "$PUBLIC_REMOTE/main:main" 2>&1 || log "  WARNING: oss-shadow push failed"
  else
    log "oss-shadow diverged from public. Force-pushing (public > oss-shadow)..."
    lgit push --force-with-lease "$OSS_SHADOW_REMOTE" "$PUBLIC_REMOTE/main:main" 2>&1 || log "  WARNING: oss-shadow force-push failed"
  fi
}

# ======================================================================
# Main cycle
# ======================================================================

run_cycle() {
  local cycle_start cycle_errors=0
  cycle_start="$(date +%s)"
  FATAL=0

  log "$(printf '#%.0s' {1..60})"
  log "  SYNC CYCLE START  $(timestamp)"
  log "$(printf '#%.0s' {1..60})"

  ensure_on_branch "$MAIN_BRANCH" || { [[ $FATAL -eq 1 ]] && return 2; return 1; }

  phase0_setup || cycle_errors=$((cycle_errors + 1))
  [[ $FATAL -eq 1 ]] && { log "FATAL error in Phase 0. Halting cycle."; return 2; }

  phase1_public_ingest || cycle_errors=$((cycle_errors + 1))
  [[ $FATAL -eq 1 ]] && { log "FATAL error in Phase 1. Halting cycle."; return 2; }

  phase2_pull_to_main || cycle_errors=$((cycle_errors + 1))
  [[ $FATAL -eq 1 ]] && { log "FATAL error in Phase 2. Halting cycle."; return 2; }

  phase3_sync_mirror || cycle_errors=$((cycle_errors + 1))
  [[ $FATAL -eq 1 ]] && { log "FATAL error in Phase 3. Halting cycle."; return 2; }

  local retrigger=0
  phase4_post_sync && retrigger=1
  [[ $FATAL -eq 1 ]] && { log "FATAL error in Phase 4. Halting cycle."; return 2; }

  local elapsed=$(( $(date +%s) - cycle_start ))
  log ""
  log "$(printf '#%.0s' {1..60})"
  log "  SYNC CYCLE END  ($elapsed seconds, $cycle_errors error(s))"
  log "$(printf '#%.0s' {1..60})"

  [[ $cycle_errors -gt 0 ]] && log "WARNING: cycle had $cycle_errors error(s)"
  [[ $retrigger -eq 1 ]] && return 0
  return 1
}

# ======================================================================
# Entry point
# ======================================================================

cd "$REPO_ROOT"

log "sync_all.sh starting (pause=${PAUSE_SECS}s, dry_run=$DRY_RUN)"
log "repo: $REPO_ROOT"
log "mirror: $MIRROR_HOST:$MIRROR_REPO"
log "priority: public > origin > mirror > oss-shadow"
log "state: $STATE_DIR"

if [[ $LOOP -eq 1 ]]; then
  log "Starting sync daemon..."
  while true; do
    set +e
    run_cycle
    rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
      log "Re-triggering immediately (mirror brought changes)."
      continue
    elif [[ $rc -eq 2 ]]; then
      log "FATAL error occurred. Sleeping ${PAUSE_SECS}s before retry..."
      log "  Fix the issue manually. The daemon will retry next cycle."
      sleep "$PAUSE_SECS"
    else
      log "Next cycle in ${PAUSE_SECS}s..."
      sleep "$PAUSE_SECS"
    fi
  done
else
  run_cycle
  rc=$?
  [[ $rc -eq 2 ]] && exit 1
  exit 0
fi
