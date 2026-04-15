#!/usr/bin/env bash
set -euo pipefail

# Periodic watcher: fetches public remote, runs pull-scan, and notifies
# on violations. Designed to run as a background daemon in WSL or Linux.
#
# Notification cascade (first available wins):
#   1. powershell.exe toast (WSL -> Windows notification center)
#   2. wsl-notify-send (WSL bridge)
#   3. notify-send (Linux desktop / WSLg)
#   4. Terminal bell + stderr message (always)
#
# Usage:
#   scripts/oss/oss-watch.sh              # run once
#   scripts/oss/oss-watch.sh --loop       # poll every INTERVAL seconds
#   scripts/oss/oss-watch.sh --loop 600   # poll every 10 minutes
#   scripts/oss/oss-watch.sh --daemon     # daemonize (nohup + background)
#
# To stop the daemon:
#   kill $(cat /tmp/oss-watch.pid)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=oss-config.sh
source "$SCRIPT_DIR/oss-config.sh"

PULL_SCAN="$SCRIPT_DIR/oss-pull-scan.sh"
OSS_SCAN="$SCRIPT_DIR/oss-scan.sh"
LOG_FILE="${OSS_WATCH_LOG:-/tmp/oss-watch.log}"
PID_FILE="/tmp/oss-watch.pid"
INTERVAL=300
LOOP=0
DAEMON=0

usage() {
  cat <<'EOF'
Usage: oss-watch.sh [options]

Periodic sync watcher for the oss/main branch.

Options:
  --loop [N]      Poll every N seconds (default: 300 = 5 min)
  --daemon        Run as background daemon
  --remote <r>    Public remote name (default: public)
  --branch <b>    OSS branch (default: oss/main)
  --log <file>    Log file (default: /tmp/oss-watch.log)
  -h              Show help

Environment:
  OSS_WATCH_LOG   Override log file path
  OSS_WATCH_HOOK  Path to a custom notification script (receives message as $1)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --loop)
    LOOP=1
    if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
    INTERVAL="$2"; shift
    fi
    shift ;;
  --daemon)  DAEMON=1; LOOP=1; shift ;;
  --remote)  PUBLIC_REMOTE="$2"; shift 2 ;;
  --branch)  OSS_BRANCH="$2"; shift 2 ;;
  --log)     LOG_FILE="$2"; shift 2 ;;
  -h)        usage; exit 0 ;;
  *)         echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# -------- notification helpers --------

is_wsl() {
  grep -qi microsoft /proc/version 2>/dev/null
}

notify() {
  local title="$1"
  local body="$2"

  # Custom hook takes priority
  if [[ -n "${OSS_WATCH_HOOK:-}" && -x "$OSS_WATCH_HOOK" ]]; then
  "$OSS_WATCH_HOOK" "$title" "$body" && return 0
  fi

  # WSL: PowerShell toast notification (appears in Windows notification center)
  if is_wsl; then
  local ps_path
  ps_path="$(command -v powershell.exe 2>/dev/null || true)"
  if [[ -n "$ps_path" ]]; then
    "$ps_path" -NoProfile -Command "
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
    \$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    \$xml.LoadXml('<toast><visual><binding template=\"ToastText02\"><text id=\"1\">$title</text><text id=\"2\">$body</text></binding></visual></toast>')
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Choreo OSS Watch').Show(\$xml)
    " 2>/dev/null && return 0
  fi

  # Fallback: wsl-notify-send
  if command -v wsl-notify-send &>/dev/null; then
    wsl-notify-send --category "Choreo" "$title: $body" && return 0
  fi
  fi

  # Linux desktop notification
  if command -v notify-send &>/dev/null; then
  notify-send -u critical "$title" "$body" 2>/dev/null && return 0
  fi

  # Terminal bell + stderr (always works)
  printf '\a' 2>/dev/null || true
  echo "*** [$title] $body ***" >&2
}

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" >> "$LOG_FILE"
  echo "[$ts] $*"
}

# -------- main check --------

run_check() {
  cd "$REPO_ROOT"

  log "Fetching $PUBLIC_REMOTE..."
  if ! timeout 30 git fetch "$PUBLIC_REMOTE" 2>/dev/null; then
  log "WARNING: could not fetch $PUBLIC_REMOTE (network issue or timeout)"
  return 0
  fi

  # Count new commits on public that aren't on oss/main
  local new_count
  new_count="$(git rev-list --count "$OSS_BRANCH..$PUBLIC_REMOTE/main" 2>/dev/null || echo 0)"

  if [[ "$new_count" -eq 0 ]]; then
  log "No new commits on $PUBLIC_REMOTE/main."
  return 0
  fi

  log "Found $new_count new commit(s) on $PUBLIC_REMOTE/main."

  # Run pull-scan
  local scan_output
  scan_output="$("$PULL_SCAN" --new -r "$PUBLIC_REMOTE" -b "$OSS_BRANCH" 2>&1)" || true
  local scan_exit=$?

  echo "$scan_output" >> "$LOG_FILE"

  if [[ $scan_exit -ne 0 ]]; then
  # Extract summary line
  local summary
  summary="$(echo "$scan_output" | tail -1)"
  log "VIOLATION: $summary"
  notify "OSS Sync Conflict" "$new_count new commit(s): $summary"
  return 1
  fi

  # Also run oss-scan on the current oss/main tree
  local tree_output
  tree_output="$("$OSS_SCAN" --tree "$OSS_BRANCH" 2>&1)" || true
  local tree_exit=$?

  if [[ $tree_exit -ne 0 ]]; then
  local tree_summary
  tree_summary="$(echo "$tree_output" | grep -c 'VIOLATION\|FAIL\|ERROR' || echo "issues")"
  log "TREE VIOLATION: oss/main has $tree_summary issue(s)"
  notify "OSS Tree Issue" "oss/main tree scan found issues"
  return 1
  fi

  log "All clear. $new_count new commit(s) ready to pull."
  return 0
}

# -------- daemon / loop --------

if [[ $DAEMON -eq 1 ]]; then
  echo "Starting oss-watch daemon (interval=${INTERVAL}s, log=$LOG_FILE)..."
  nohup "$0" --loop "$INTERVAL" --remote "$PUBLIC_REMOTE" --branch "$OSS_BRANCH" --log "$LOG_FILE" \
  >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "PID: $(cat "$PID_FILE")"
  echo "Stop with: kill \$(cat $PID_FILE)"
  exit 0
fi

if [[ $LOOP -eq 1 ]]; then
  log "Starting watch loop (interval=${INTERVAL}s)..."
  echo $$ > "$PID_FILE"
  while true; do
  run_check || true
  log "Next check in ${INTERVAL}s..."
  sleep "$INTERVAL"
  done
else
  run_check
fi
