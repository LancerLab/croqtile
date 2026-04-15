#!/usr/bin/env bash
# Shared configuration for all OSS sync scripts.
# Source this file instead of duplicating remote/branch/URL definitions.

# -------- repo root detection --------
# Every script that sources this must set SCRIPT_DIR before sourcing.
# We detect the repo root via git, so moving the oss/ directory works.
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(git -C "${SCRIPT_DIR:-.}" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "FATAL: not inside a git repository (from ${SCRIPT_DIR:-?})" >&2
    exit 128
  }
fi

# -------- branch names --------
OSS_BRANCH="${OSS_BRANCH:-oss/main}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"

# -------- remotes --------
PUBLIC_REMOTE="${PUBLIC_REMOTE:-public}"
PUBLIC_URL="${PUBLIC_URL:-git@github.com:LancerLab/croqtile.git}"
OSS_SHADOW_REMOTE="${OSS_SHADOW_REMOTE:-oss-shadow}"
OSS_SHADOW_URL="${OSS_SHADOW_URL:-git@git.enflame.cn:era-dev/choreo-open.git}"

# -------- mirror / bundle-sync --------
MIRROR_HOST="${MIRROR_HOST:-garfee@10.0.16.52}"
MIRROR_REPO="${MIRROR_REPO:-dev/choreo-sync/}"
LOCAL_REMOTE="${LOCAL_REMOTE:-origin}"
MIRROR_REMOTE="${MIRROR_REMOTE:-origin}"
LOCAL_REMOTE_URL="${LOCAL_REMOTE_URL:-git@git.enflame.cn:xiaofeng.guan/choreo.git}"
MIRROR_REMOTE_URL="${MIRROR_REMOTE_URL:-git@10.0.16.44:lancerlab/choreo.git}"

# -------- alerting --------
ALERT_EMAIL="${ALERT_EMAIL:-}"

# -------- data files (relative to SCRIPT_DIR) --------
EXCLUDE_FILE="${EXCLUDE_FILE:-$SCRIPT_DIR/oss_exclude_paths.txt}"
KW_FILE="${KW_FILE:-$SCRIPT_DIR/os_kw.txt}"

# -------- terminal helpers --------
supports_color() {
  [[ -t 2 ]] && [[ "${TERM:-dumb}" != "dumb" ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]
}

_C_YELLOW="" _C_RED="" _C_RESET=""
if supports_color; then
  _C_YELLOW=$'\033[1;33m'
  _C_RED=$'\033[1;31m'
  _C_RESET=$'\033[0m'
fi
