#!/usr/bin/env bash
# Bundle-based bi-directional sync between two remote repositories.
#
# Local side:
#   - working repo at LOCAL_REPO
#   - authoritative remote named LOCAL_REMOTE (default: origin)
#
# Mirror side:
#   - working repo on MIRROR_HOST at MIRROR_REPO
#   - authoritative remote named MIRROR_REMOTE (default: origin)
#
# High-level workflow per cycle:
#   1. refresh local repo from LOCAL_REMOTE
#   2. refresh mirror repo from MIRROR_REMOTE
#   3. optionally propagate proven branch deletions
#   4. bundle local remote-tracking refs/tags and copy to mirror
#   5. bundle mirror origin-tracking refs/tags and copy back
#   6. compare remote histories; fast-forward the older remote when possible
#   7. refresh both repos again so their visible state tracks the synced remotes
#
# Notes:
#   - branch deletions are only propagated with --sync-deletions
#   - tag deletions are NOT propagated automatically
#   - diverged refs are reported and left for manual resolution

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=oss-config.sh
source "$SCRIPT_DIR/oss-config.sh"

LOCAL_REPO=${LOCAL_REPO:-$REPO_ROOT}

RUN_ONCE=0
INIT_MODE=0
SYNC_DELETIONS=0
LOG_ENABLED=0
DRY_RUN=0
LOCAL_WINS=0

POLL_INTERVAL=${POLL_INTERVAL:-120}
REMOTE_TMP_DIR=${REMOTE_TMP_DIR:-/tmp/$(id -u)/choreo-sync}
LOG_FILE=${LOG_FILE:-}
EXCLUDE_BRANCHES=${EXCLUDE_BRANCHES:-}

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

setup_logging() {
  if [[ $LOG_ENABLED -ne 1 ]]; then
    return 0
  fi

  if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="$STATE_DIR/sync.log"
  else
    LOG_FILE=$(resolve_state_path "$LOG_FILE")
  fi

  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1
  log "file logging enabled: $LOG_FILE"
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --once                     run exactly one sync cycle, then exit
  --init                     clone local and/or mirror repos if missing
  --sync-deletions           propagate branch deletions when safely inferred
  --dry-run                  show planned changes without pushing or deleting
  --local-wins               on divergence, force-push local version to mirror
  --log                      enable file logging to the default log path
  --log-file PATH            enable file logging to PATH
  --exclude-branch NAME      exclude a branch from deletion sync only (repeatable)
  --local-repo PATH          local repo path (default: $DEFAULT_LOCAL_REPO)
  --mirror-repo PATH         mirror repo path on MIRROR_HOST (default: $DEFAULT_MIRROR_REPO)
  --mirror-host HOST         mirror ssh target (default: $MIRROR_HOST)
  --local-remote NAME        local remote name (default: $LOCAL_REMOTE)
  --mirror-remote NAME       mirror remote name (default: $MIRROR_REMOTE)
  --local-url URL            local clone URL for --init (default: $DEFAULT_LOCAL_REMOTE_URL)
  --mirror-url URL           mirror clone URL for --init (default: $DEFAULT_MIRROR_REMOTE_URL)
  --poll-interval SEC        loop interval in seconds (default: $POLL_INTERVAL)
  --alert-email ADDRESS      email target for warnings (default: $ALERT_EMAIL)
  -h, --help                 show this help

Examples:
  $0 --once
  $0 --once --log
  $0 --sync-deletions --log-file logs/sync.log
  $0 --dry-run --exclude-branch main --exclude-branch release/pre-beta
  $0 --local-repo ~/dev/choreo --mirror-host garfee@10.0.16.52 --mirror-repo ~/dev/choreo/
  $0 --init --local-repo ~/work/choreo --mirror-repo ~/dev/choreo/
EOF
}

notify() {
  local title="$1"
  local msg="$2"
  local one_line

  one_line=$(printf '%s' "$msg" | tr '\n' ' ')

  if command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /C "msg %USERNAME% \"$title: $one_line\"" >/dev/null 2>&1 || true
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send "$title" "$msg" || true
  fi

  echo "[notify] $title: $msg" >&2
}

mail_alert() {
  local subject="$1"
  local body="$2"

  if command -v mail >/dev/null 2>&1; then
    printf '%b\n' "$body" | mail -s "$subject" "$ALERT_EMAIL" || true
  elif command -v sendmail >/dev/null 2>&1; then
    sendmail "$ALERT_EMAIL" <<EOF || true
Subject: $subject

$body
EOF
  else
    echo "[mail_alert] $subject: $body" >&2
  fi
}

require_command() {
  local cmd="$1"

  command -v "$cmd" >/dev/null 2>&1 || {
    echo "missing required command: $cmd" >&2
    exit 1
  }
}

make_abs_path() {
  local path="$1"

  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  elif [[ "$path" == ~* ]]; then
    eval "printf '%s\\n' \"$path\""
  else
    printf '%s/%s\n' "$PWD" "$path"
  fi
}

resolve_state_path() {
  local path="$1"

  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$LOCAL_REPO" "$path"
  fi
}

warn() {
  local subject="$1"
  local body="$2"
  local key
  local stamp

  key=$(printf '%s\n%s' "$subject" "$body" | sha1sum | awk '{print $1}')
  stamp="$ALERT_STATE_DIR/$key"

  if [[ -f "$stamp" ]]; then
    return 0
  fi

  notify "$subject" "$body"
  mail_alert "$subject" "$body"
  : > "$stamp"
}

lgit() {
  git -C "$LOCAL_REPO" "$@"
}

ensure_state_dirs() {
  STATE_DIR=${STATE_DIR:-$LOCAL_REPO/.git/sync-remotes}
  STATE_DIR=$(resolve_state_path "$STATE_DIR")
  ALERT_STATE_DIR="$STATE_DIR/alerts"
  SNAPSHOT_LOCAL_BRANCHES="$STATE_DIR/prev-local-branches.txt"
  SNAPSHOT_MIRROR_BRANCHES="$STATE_DIR/prev-mirror-branches.txt"
  CURRENT_LOCAL_BRANCHES="$STATE_DIR/current-local-branches.txt"
  CURRENT_MIRROR_BRANCHES="$STATE_DIR/current-mirror-branches.txt"
  LOCAL_OUT_BUNDLE="$STATE_DIR/local-to-mirror.bundle"
  LOCAL_IN_BUNDLE="$STATE_DIR/mirror-to-local.bundle"
  REMOTE_IN_BUNDLE="$REMOTE_TMP_DIR/local-to-mirror.bundle"
  REMOTE_OUT_BUNDLE="$REMOTE_TMP_DIR/mirror-to-local.bundle"

  mkdir -p "$STATE_DIR" "$ALERT_STATE_DIR"
}

current_local_branch_names() {
  lgit for-each-ref --format='%(refname:strip=3)' "refs/remotes/$LOCAL_REMOTE" | grep -v '^HEAD$' | sort -u
}

current_mirror_branch_names() {
  ssh "$MIRROR_HOST" bash -s -- "$MIRROR_REPO" "$MIRROR_REMOTE" <<'EOF'
set -euo pipefail

repo_dir="$1"
remote_name="$2"

if [[ "$repo_dir" == '~' ]]; then
  repo_dir="$HOME"
elif [[ "$repo_dir" == ~/* ]]; then
  repo_dir="$HOME/${repo_dir#~/}"
elif [[ "$repo_dir" != /* ]]; then
  repo_dir="$HOME/$repo_dir"
fi

git -C "$repo_dir" for-each-ref --format='%(refname:strip=3)' "refs/remotes/$remote_name" | grep -v '^HEAD$' | sort -u
EOF
}

save_branch_snapshots() {
  current_local_branch_names > "$SNAPSHOT_LOCAL_BRANCHES"
  current_mirror_branch_names > "$SNAPSHOT_MIRROR_BRANCHES"
}

collect_current_branch_views() {
  current_local_branch_names > "$CURRENT_LOCAL_BRANCHES"
  current_mirror_branch_names > "$CURRENT_MIRROR_BRANCHES"
}

snapshot_has_branch() {
  local file="$1"
  local branch="$2"

  [[ -f "$file" ]] && grep -Fxq -- "$branch" "$file"
}

branch_previously_shared() {
  local branch="$1"

  snapshot_has_branch "$SNAPSHOT_LOCAL_BRANCHES" "$branch" && snapshot_has_branch "$SNAPSHOT_MIRROR_BRANCHES" "$branch"
}

branch_is_excluded() {
  local branch="$1"
  local item

  for item in $EXCLUDE_BRANCHES; do
    [[ -n "$item" ]] || continue
    [[ "$branch" == "$item" ]] && return 0
  done

  return 1
}

maybe_run_local_push() {
  local source_ref="$1"
  local target_ref="$2"
  local description="$3"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "dry-run: would push $description to local remote '$LOCAL_REMOTE' as '$target_ref'"
    return 0
  fi

  if ! lgit push "$LOCAL_REMOTE" "$source_ref:$target_ref"; then
    warn "git-sync push warning" "Failed to push $description to local remote '$LOCAL_REMOTE'."
  fi
}

maybe_run_local_delete() {
  local branch="$1"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "dry-run: would delete branch '$branch' from local remote '$LOCAL_REMOTE'"
    return 0
  fi

  if ! lgit push "$LOCAL_REMOTE" ":refs/heads/$branch"; then
    warn "git-sync delete warning" "Failed to delete branch '$branch' from local remote '$LOCAL_REMOTE'."
  fi
}

ensure_local_repo() {
  if lgit rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi

  if [[ $INIT_MODE -ne 1 ]]; then
    echo "local repo is missing or invalid: $LOCAL_REPO" >&2
    exit 1
  fi

  log "initializing local repo at '$LOCAL_REPO'"
  mkdir -p "$(dirname "$LOCAL_REPO")"
  git clone "$LOCAL_REMOTE_URL" "$LOCAL_REPO"
}

ensure_mirror_repo() {
  ssh "$MIRROR_HOST" bash -s -- \
    "$MIRROR_REPO" \
    "$MIRROR_REMOTE" \
    "$MIRROR_REMOTE_URL" \
    "$INIT_MODE" <<'EOF'
set -euo pipefail

repo_dir="$1"
remote_name="$2"
remote_url="$3"
init_mode="$4"

if [[ "$repo_dir" == '~' ]]; then
  repo_dir="$HOME"
elif [[ "$repo_dir" == ~/* ]]; then
  repo_dir="$HOME/${repo_dir#~/}"
elif [[ "$repo_dir" != /* ]]; then
  repo_dir="$HOME/$repo_dir"
fi

if git -C "$repo_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  exit 0
fi

if [[ "$init_mode" != 1 ]]; then
  echo "mirror repo is missing or invalid: $repo_dir" >&2
  exit 1
fi

mkdir -p "$(dirname "$repo_dir")"
git clone "$remote_url" "$repo_dir"

if [[ "$remote_name" != origin ]]; then
  git -C "$repo_dir" remote rename origin "$remote_name"
fi
EOF
}

fast_forward_local_current_branch() {
  local branch
  local upstream_ref
  local branch_ref

  branch=$(lgit symbolic-ref --quiet --short HEAD 2>/dev/null || true)
  [[ -n "$branch" ]] || return 0

  upstream_ref="refs/remotes/$LOCAL_REMOTE/$branch"
  branch_ref="refs/heads/$branch"
  lgit show-ref --verify --quiet "$upstream_ref" || return 0

  if ! lgit diff --quiet || ! lgit diff --cached --quiet; then
    log "local branch '$branch' has uncommitted changes; skip fast-forward"
    return 0
  fi

  if [[ $(lgit rev-parse "$branch_ref") == $(lgit rev-parse "$upstream_ref") ]]; then
    return 0
  fi

  if lgit merge-base --is-ancestor "$branch_ref" "$upstream_ref"; then
    log "fast-forwarding local branch '$branch' to '$LOCAL_REMOTE/$branch'"
    lgit merge --ff-only "$LOCAL_REMOTE/$branch"
  elif lgit merge-base --is-ancestor "$upstream_ref" "$branch_ref"; then
    log "local branch '$branch' is ahead of '$LOCAL_REMOTE/$branch'; leave as-is"
  else
    warn "git-sync local branch conflict" "Local checked-out branch '$branch' diverged from '$LOCAL_REMOTE/$branch'."
  fi
}

refresh_local_repo() {
  log "refreshing local repo '$LOCAL_REPO' from remote '$LOCAL_REMOTE'"
  lgit fetch "$LOCAL_REMOTE" --prune --tags
  fast_forward_local_current_branch
}

refresh_mirror_repo() {
  log "refreshing mirror repo '$MIRROR_REPO' from '$MIRROR_HOST:$MIRROR_REMOTE'"
  ssh "$MIRROR_HOST" bash -s -- "$MIRROR_REPO" "$MIRROR_REMOTE" <<'EOF'
set -euo pipefail

repo_dir="$1"
remote_name="$2"

if [[ "$repo_dir" == '~' ]]; then
  repo_dir="$HOME"
elif [[ "$repo_dir" == ~/* ]]; then
  repo_dir="$HOME/${repo_dir#~/}"
elif [[ "$repo_dir" != /* ]]; then
  repo_dir="$HOME/$repo_dir"
fi

git -C "$repo_dir" fetch "$remote_name" --prune --tags

branch=$(git -C "$repo_dir" symbolic-ref --quiet --short HEAD 2>/dev/null || true)
if [[ -z "$branch" ]]; then
  exit 0
fi

upstream_ref="refs/remotes/$remote_name/$branch"
branch_ref="refs/heads/$branch"
git -C "$repo_dir" show-ref --verify --quiet "$upstream_ref" || exit 0

if ! git -C "$repo_dir" diff --quiet || ! git -C "$repo_dir" diff --cached --quiet; then
  echo "mirror branch '$branch' has uncommitted changes; skip fast-forward" >&2
  exit 0
fi

if [[ $(git -C "$repo_dir" rev-parse "$branch_ref") == $(git -C "$repo_dir" rev-parse "$upstream_ref") ]]; then
  exit 0
fi

if git -C "$repo_dir" merge-base --is-ancestor "$branch_ref" "$upstream_ref"; then
  git -C "$repo_dir" merge --ff-only "$remote_name/$branch"
elif git -C "$repo_dir" merge-base --is-ancestor "$upstream_ref" "$branch_ref"; then
  echo "mirror branch '$branch' is ahead of '$remote_name/$branch'; leave as-is" >&2
else
  echo "mirror branch '$branch' diverged from '$remote_name/$branch'; leave as-is" >&2
fi
EOF
}

clear_local_namespace() {
  local prefix="$1"

  while IFS= read -r ref; do
    lgit update-ref -d "$ref"
  done < <(lgit for-each-ref --format='%(refname)' "$prefix")
}

create_local_bundle() {
  local output="$1"
  local refs=()

  mapfile -t refs < <(
    {
      lgit for-each-ref --format='%(refname)' "refs/remotes/$LOCAL_REMOTE"
      lgit for-each-ref --format='%(refname)' refs/tags
    } | grep -v '/HEAD$'
  )

  if [[ ${#refs[@]} -eq 0 ]]; then
    warn "git-sync error" "No refs found for local remote '$LOCAL_REMOTE'."
    return 1
  fi

  rm -f "$output"
  lgit bundle create "$output" "${refs[@]}"
}

prepare_mirror_bundle() {
  ssh "$MIRROR_HOST" bash -s -- "$MIRROR_REPO" "$MIRROR_REMOTE" "$REMOTE_TMP_DIR" "$REMOTE_OUT_BUNDLE" <<'EOF'
set -euo pipefail

repo_dir="$1"
mirror_remote="$2"
tmp_dir="$3"
out_bundle="$4"

if [[ "$repo_dir" == '~' ]]; then
  repo_dir="$HOME"
elif [[ "$repo_dir" == ~/* ]]; then
  repo_dir="$HOME/${repo_dir#~/}"
elif [[ "$repo_dir" != /* ]]; then
  repo_dir="$HOME/$repo_dir"
fi

mkdir -p "$tmp_dir"
git -C "$repo_dir" fetch "$mirror_remote" --prune --tags

refs=()
mapfile -t refs < <(
  {
    git -C "$repo_dir" for-each-ref --format='%(refname)' "refs/remotes/$mirror_remote"
    git -C "$repo_dir" for-each-ref --format='%(refname)' refs/tags
  } | grep -v '/HEAD$'
)

if [[ ${#refs[@]} -eq 0 ]]; then
  echo "ERROR::no refs found for mirror remote '$mirror_remote'" >&2
  exit 1
fi

rm -f "$out_bundle"
git -C "$repo_dir" bundle create "$out_bundle" "${refs[@]}"
EOF
}

import_mirror_bundle() {
  clear_local_namespace "refs/sync/mirror"
  lgit fetch --no-tags "$LOCAL_IN_BUNDLE" \
    "refs/remotes/$MIRROR_REMOTE/*:refs/sync/mirror/remotes/$MIRROR_REMOTE/*" \
    "refs/tags/*:refs/sync/mirror/tags/*"
}

branch_names_for_sync() {
  {
    lgit for-each-ref --format='%(refname:strip=3)' "refs/remotes/$LOCAL_REMOTE"
    lgit for-each-ref --format='%(refname:strip=5)' "refs/sync/mirror/remotes/$MIRROR_REMOTE"
  } | grep -v '^HEAD$' | sort -u
}

tag_names_for_sync() {
  {
    lgit for-each-ref --format='%(refname:strip=2)' refs/tags
    lgit for-each-ref --format='%(refname:strip=4)' refs/sync/mirror/tags
  } | sort -u
}

push_to_local_remote() {
  local source_ref="$1"
  local target_ref="$2"
  local description="$3"

  maybe_run_local_push "$source_ref" "$target_ref" "$description"
}

push_delete_local_branch() {
  local branch="$1"

  maybe_run_local_delete "$branch"
}

push_delete_mirror_branch() {
  local branch="$1"
  local output

  if [[ $DRY_RUN -eq 1 ]]; then
    log "dry-run: would delete branch '$branch' from mirror remote '$MIRROR_REMOTE'"
    return 0
  fi

  output=$(ssh "$MIRROR_HOST" bash -s -- "$MIRROR_REPO" "$MIRROR_REMOTE" "$branch" <<'EOF'
set -euo pipefail

repo_dir="$1"
remote_name="$2"
branch="$3"

if [[ "$repo_dir" == '~' ]]; then
  repo_dir="$HOME"
elif [[ "$repo_dir" == ~/* ]]; then
  repo_dir="$HOME/${repo_dir#~/}"
elif [[ "$repo_dir" != /* ]]; then
  repo_dir="$HOME/$repo_dir"
fi

if ! git -C "$repo_dir" push "$remote_name" ":refs/heads/$branch"; then
  echo "WARN::Failed to delete branch '$branch' from mirror remote '$remote_name'."
fi
EOF
  )

  [[ -n "$output" ]] && printf '%s\n' "$output"
  while IFS= read -r line; do
    [[ "$line" == WARN::* ]] || continue
    warn "git-sync delete warning" "${line#WARN::}"
  done <<< "$output"
}

sync_branch_deletions() {
  local branch
  local local_has
  local mirror_has

  if [[ $SYNC_DELETIONS -ne 1 ]]; then
    return 0
  fi

  if [[ ! -f "$SNAPSHOT_LOCAL_BRANCHES" || ! -f "$SNAPSHOT_MIRROR_BRANCHES" ]]; then
    log "no prior branch snapshot found; skip branch deletion sync this cycle"
    return 0
  fi

  collect_current_branch_views

  while IFS= read -r branch; do
    [[ -n "$branch" ]] || continue
    branch_is_excluded "$branch" && continue
    branch_previously_shared "$branch" || continue

    local_has=0
    mirror_has=0
    snapshot_has_branch "$CURRENT_LOCAL_BRANCHES" "$branch" && local_has=1
    snapshot_has_branch "$CURRENT_MIRROR_BRANCHES" "$branch" && mirror_has=1

    if [[ $local_has -eq 1 && $mirror_has -eq 0 ]]; then
      log "branch '$branch' was deleted on mirror; deleting it from local remote '$LOCAL_REMOTE'"
      push_delete_local_branch "$branch"
    elif [[ $local_has -eq 0 && $mirror_has -eq 1 ]]; then
      log "branch '$branch' was deleted locally; deleting it from mirror remote '$MIRROR_REMOTE'"
      push_delete_mirror_branch "$branch"
    fi
  done < <(
    {
      cat "$CURRENT_LOCAL_BRANCHES"
      cat "$CURRENT_MIRROR_BRANCHES"
      cat "$SNAPSHOT_LOCAL_BRANCHES"
      cat "$SNAPSHOT_MIRROR_BRANCHES"
    } | sort -u
  )
}

sync_local_remote_from_mirror() {
  local branch
  local local_ref
  local mirror_ref
  local local_sha
  local mirror_sha
  local tag
  local local_tag_ref
  local mirror_tag_ref
  local local_tag_sha
  local mirror_tag_sha

  while IFS= read -r branch; do
    [[ -n "$branch" ]] || continue

    local_ref="refs/remotes/$LOCAL_REMOTE/$branch"
    mirror_ref="refs/sync/mirror/remotes/$MIRROR_REMOTE/$branch"
    local_sha=""
    mirror_sha=""

    lgit show-ref --verify --quiet "$local_ref" && local_sha=$(lgit rev-parse "$local_ref")
    lgit show-ref --verify --quiet "$mirror_ref" && mirror_sha=$(lgit rev-parse "$mirror_ref")

    if [[ -z "$local_sha" && -n "$mirror_sha" ]]; then
      log "local remote '$LOCAL_REMOTE' is missing branch '$branch'; pushing mirror copy"
      push_to_local_remote "$mirror_ref" "refs/heads/$branch" "branch '$branch'"
    elif [[ -n "$local_sha" && -n "$mirror_sha" ]]; then
      if [[ "$local_sha" == "$mirror_sha" ]]; then
        continue
      elif lgit merge-base --is-ancestor "$local_ref" "$mirror_ref"; then
        log "branch '$branch': mirror is ahead; pushing to '$LOCAL_REMOTE'"
        push_to_local_remote "$mirror_ref" "refs/heads/$branch" "branch '$branch'"
      elif lgit merge-base --is-ancestor "$mirror_ref" "$local_ref"; then
        log "branch '$branch': local '$LOCAL_REMOTE' is already ahead"
      else
        if [[ $LOCAL_WINS -eq 1 ]]; then
          log "branch '$branch': DIVERGED -- local wins, will force-push to mirror"
        else
          warn "git-sync conflict" "Diverged branch: $branch\nlocal($LOCAL_REMOTE): $local_sha\nmirror($MIRROR_REMOTE): $mirror_sha"
        fi
      fi
    fi
  done < <(branch_names_for_sync)

  while IFS= read -r tag; do
    [[ -n "$tag" ]] || continue

    local_tag_ref="refs/tags/$tag"
    mirror_tag_ref="refs/sync/mirror/tags/$tag"
    local_tag_sha=""
    mirror_tag_sha=""

    lgit show-ref --verify --quiet "$local_tag_ref" && local_tag_sha=$(lgit rev-parse "$local_tag_ref")
    lgit show-ref --verify --quiet "$mirror_tag_ref" && mirror_tag_sha=$(lgit rev-parse "$mirror_tag_ref")

    if [[ -z "$local_tag_sha" && -n "$mirror_tag_sha" ]]; then
      log "local remote '$LOCAL_REMOTE' is missing tag '$tag'; pushing mirror tag"
      push_to_local_remote "$mirror_tag_ref" "refs/tags/$tag" "tag '$tag'"
    elif [[ -n "$local_tag_sha" && -n "$mirror_tag_sha" && "$local_tag_sha" != "$mirror_tag_sha" ]]; then
      warn "git-sync conflict" "Diverged tag: $tag\nlocal($LOCAL_REMOTE): $local_tag_sha\nmirror($MIRROR_REMOTE): $mirror_tag_sha"
    fi
  done < <(tag_names_for_sync)
}

push_local_bundle_to_mirror() {
  scp "$LOCAL_OUT_BUNDLE" "$MIRROR_HOST:$REMOTE_IN_BUNDLE"
}

pull_mirror_bundle_to_local() {
  scp "$MIRROR_HOST:$REMOTE_OUT_BUNDLE" "$LOCAL_IN_BUNDLE"
}

sync_mirror_remote_from_local() {
  local output

    if [[ $DRY_RUN -eq 1 ]]; then
        output=$(ssh "$MIRROR_HOST" bash -s -- \
        "$MIRROR_REPO" \
        "$MIRROR_REMOTE" \
        "$LOCAL_REMOTE" \
        "$REMOTE_IN_BUNDLE" \
        "$DRY_RUN" \
        "$LOCAL_WINS" <<'EOF'
set -euo pipefail

repo_dir="$1"
mirror_remote="$2"
local_remote="$3"
incoming_bundle="$4"
dry_run="$5"
local_wins="$6"

if [[ "$repo_dir" == '~' ]]; then
  repo_dir="$HOME"
elif [[ "$repo_dir" == ~/* ]]; then
  repo_dir="$HOME/${repo_dir#~/}"
elif [[ "$repo_dir" != /* ]]; then
  repo_dir="$HOME/$repo_dir"
fi

while IFS= read -r ref; do
  git -C "$repo_dir" update-ref -d "$ref"
done < <(git -C "$repo_dir" for-each-ref --format='%(refname)' refs/sync/local)

git -C "$repo_dir" fetch --no-tags "$incoming_bundle" \
  "refs/remotes/$local_remote/*:refs/sync/local/remotes/$local_remote/*" \
  "refs/tags/*:refs/sync/local/tags/*"

while IFS= read -r branch; do
  [[ -n "$branch" ]] || continue

  mirror_ref="refs/remotes/$mirror_remote/$branch"
  local_ref="refs/sync/local/remotes/$local_remote/$branch"
  mirror_sha=""
  local_sha=""

  git -C "$repo_dir" show-ref --verify --quiet "$mirror_ref" && mirror_sha=$(git -C "$repo_dir" rev-parse "$mirror_ref")
  git -C "$repo_dir" show-ref --verify --quiet "$local_ref" && local_sha=$(git -C "$repo_dir" rev-parse "$local_ref")

  if [[ -z "$mirror_sha" && -n "$local_sha" ]]; then
    echo "dry-run: would push branch '$branch' to mirror remote '$mirror_remote'"
  elif [[ -n "$mirror_sha" && -n "$local_sha" ]]; then
    if [[ "$mirror_sha" == "$local_sha" ]]; then
      continue
    elif git -C "$repo_dir" merge-base --is-ancestor "$mirror_ref" "$local_ref"; then
      echo "dry-run: would push branch '$branch' to mirror remote '$mirror_remote'"
    elif [[ "$local_wins" == "1" ]]; then
      echo "dry-run: would FORCE-push branch '$branch' to mirror remote '$mirror_remote' (local wins)"
    fi
  fi
done < <(
  {
    git -C "$repo_dir" for-each-ref --format='%(refname:strip=3)' "refs/remotes/$mirror_remote"
    git -C "$repo_dir" for-each-ref --format='%(refname:strip=5)' "refs/sync/local/remotes/$local_remote"
  } | grep -v '^HEAD$' | sort -u
)

while IFS= read -r tag; do
  [[ -n "$tag" ]] || continue

  mirror_tag_ref="refs/tags/$tag"
  local_tag_ref="refs/sync/local/tags/$tag"
  mirror_tag_sha=""
  local_tag_sha=""

  git -C "$repo_dir" show-ref --verify --quiet "$mirror_tag_ref" && mirror_tag_sha=$(git -C "$repo_dir" rev-parse "$mirror_tag_ref")
  git -C "$repo_dir" show-ref --verify --quiet "$local_tag_ref" && local_tag_sha=$(git -C "$repo_dir" rev-parse "$local_tag_ref")

  if [[ -z "$mirror_tag_sha" && -n "$local_tag_sha" ]]; then
    echo "dry-run: would push tag '$tag' to mirror remote '$mirror_remote'"
  fi
done < <(
  {
    git -C "$repo_dir" for-each-ref --format='%(refname:strip=2)' refs/tags
    git -C "$repo_dir" for-each-ref --format='%(refname:strip=4)' refs/sync/local/tags
  } | sort -u
)
EOF
        )

        printf '%s\n' "$output"
        return 0
    fi

  output=$(ssh "$MIRROR_HOST" bash -s -- \
    "$MIRROR_REPO" \
    "$MIRROR_REMOTE" \
    "$LOCAL_REMOTE" \
    "$REMOTE_IN_BUNDLE" \
    "$LOCAL_WINS" <<'EOF'
set -euo pipefail

repo_dir="$1"
mirror_remote="$2"
local_remote="$3"
incoming_bundle="$4"
local_wins="$5"

if [[ "$repo_dir" == '~' ]]; then
  repo_dir="$HOME"
elif [[ "$repo_dir" == ~/* ]]; then
  repo_dir="$HOME/${repo_dir#~/}"
elif [[ "$repo_dir" != /* ]]; then
  repo_dir="$HOME/$repo_dir"
fi

while IFS= read -r ref; do
  git -C "$repo_dir" update-ref -d "$ref"
done < <(git -C "$repo_dir" for-each-ref --format='%(refname)' refs/sync/local)

git -C "$repo_dir" fetch --no-tags "$incoming_bundle" \
  "refs/remotes/$local_remote/*:refs/sync/local/remotes/$local_remote/*" \
  "refs/tags/*:refs/sync/local/tags/*"

while IFS= read -r branch; do
  [[ -n "$branch" ]] || continue

  mirror_ref="refs/remotes/$mirror_remote/$branch"
  local_ref="refs/sync/local/remotes/$local_remote/$branch"
  mirror_sha=""
  local_sha=""

  git -C "$repo_dir" show-ref --verify --quiet "$mirror_ref" && mirror_sha=$(git -C "$repo_dir" rev-parse "$mirror_ref")
  git -C "$repo_dir" show-ref --verify --quiet "$local_ref" && local_sha=$(git -C "$repo_dir" rev-parse "$local_ref")

  if [[ -z "$mirror_sha" && -n "$local_sha" ]]; then
    if ! git -C "$repo_dir" push "$mirror_remote" "$local_ref:refs/heads/$branch"; then
      echo "WARN::Failed to push branch '$branch' to mirror remote '$mirror_remote'."
    fi
  elif [[ -n "$mirror_sha" && -n "$local_sha" ]]; then
    if [[ "$mirror_sha" == "$local_sha" ]]; then
      continue
    elif git -C "$repo_dir" merge-base --is-ancestor "$mirror_ref" "$local_ref"; then
      if ! git -C "$repo_dir" push "$mirror_remote" "$local_ref:refs/heads/$branch"; then
        echo "WARN::Failed to push branch '$branch' to mirror remote '$mirror_remote'."
      fi
    elif [[ "$local_wins" == "1" ]]; then
      echo "INFO::branch '$branch': DIVERGED -- force-pushing local to mirror (local wins)"
      if ! git -C "$repo_dir" push --force "$mirror_remote" "$local_ref:refs/heads/$branch"; then
        echo "WARN::Failed to force-push branch '$branch' to mirror remote '$mirror_remote'."
      fi
    fi
  fi
done < <(
  {
    git -C "$repo_dir" for-each-ref --format='%(refname:strip=3)' "refs/remotes/$mirror_remote"
    git -C "$repo_dir" for-each-ref --format='%(refname:strip=5)' "refs/sync/local/remotes/$local_remote"
  } | grep -v '^HEAD$' | sort -u
)

while IFS= read -r tag; do
  [[ -n "$tag" ]] || continue

  mirror_tag_ref="refs/tags/$tag"
  local_tag_ref="refs/sync/local/tags/$tag"
  mirror_tag_sha=""
  local_tag_sha=""

  git -C "$repo_dir" show-ref --verify --quiet "$mirror_tag_ref" && mirror_tag_sha=$(git -C "$repo_dir" rev-parse "$mirror_tag_ref")
  git -C "$repo_dir" show-ref --verify --quiet "$local_tag_ref" && local_tag_sha=$(git -C "$repo_dir" rev-parse "$local_tag_ref")

  if [[ -z "$mirror_tag_sha" && -n "$local_tag_sha" ]]; then
    if ! git -C "$repo_dir" push "$mirror_remote" "$local_tag_ref:refs/tags/$tag"; then
      echo "WARN::Failed to push tag '$tag' to mirror remote '$mirror_remote'."
    fi
  fi
done < <(
  {
    git -C "$repo_dir" for-each-ref --format='%(refname:strip=2)' refs/tags
    git -C "$repo_dir" for-each-ref --format='%(refname:strip=4)' refs/sync/local/tags
  } | sort -u
)

git -C "$repo_dir" fetch "$mirror_remote" --prune --tags
EOF
  )

  printf '%s\n' "$output"

  while IFS= read -r line; do
    [[ "$line" == WARN::* ]] || continue
    warn "git-sync mirror push warning" "${line#WARN::}"
  done <<< "$output"
}

run_cycle() {
  refresh_local_repo
  refresh_mirror_repo

  sync_branch_deletions
  refresh_local_repo
  refresh_mirror_repo

  log "creating local bundle from '$LOCAL_REPO'"
  create_local_bundle "$LOCAL_OUT_BUNDLE"

  log "preparing mirror bundle from '$MIRROR_HOST:$MIRROR_REPO'"
  prepare_mirror_bundle

  log "copying local bundle to mirror"
  push_local_bundle_to_mirror

  log "copying mirror bundle to local"
  pull_mirror_bundle_to_local

  log "importing mirror bundle locally"
  import_mirror_bundle

  log "syncing mirror remote '$MIRROR_REMOTE' back to local remote '$LOCAL_REMOTE'"
  sync_local_remote_from_mirror

  log "syncing local remote '$LOCAL_REMOTE' to mirror remote '$MIRROR_REMOTE'"
  sync_mirror_remote_from_local

  refresh_local_repo
  refresh_mirror_repo
  save_branch_snapshots
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --once)
        RUN_ONCE=1
        ;;
      --init)
        INIT_MODE=1
        ;;
      --sync-deletions)
        SYNC_DELETIONS=1
        ;;
      --dry-run)
        DRY_RUN=1
        ;;
      --local-wins)
        LOCAL_WINS=1
        ;;
      --log)
        LOG_ENABLED=1
        ;;
      --log-file)
        LOG_ENABLED=1
        LOG_FILE="$2"
        shift
        ;;
      --exclude-branch)
        if [[ -n "$EXCLUDE_BRANCHES" ]]; then
          EXCLUDE_BRANCHES="$EXCLUDE_BRANCHES $2"
        else
          EXCLUDE_BRANCHES="$2"
        fi
        shift
        ;;
      --local-repo)
        LOCAL_REPO="$2"
        shift
        ;;
      --mirror-repo)
        MIRROR_REPO="$2"
        shift
        ;;
      --mirror-host)
        MIRROR_HOST="$2"
        shift
        ;;
      --local-remote)
        LOCAL_REMOTE="$2"
        shift
        ;;
      --mirror-remote)
        MIRROR_REMOTE="$2"
        shift
        ;;
      --local-url)
        LOCAL_REMOTE_URL="$2"
        shift
        ;;
      --mirror-url)
        MIRROR_REMOTE_URL="$2"
        shift
        ;;
      --poll-interval)
        POLL_INTERVAL="$2"
        shift
        ;;
      --alert-email)
        ALERT_EMAIL="$2"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
    shift
  done
}

check_mirror_reachable() {
  if ssh -o ConnectTimeout=5 -o BatchMode=yes "$MIRROR_HOST" true 2>/dev/null; then
    return 0
  fi
  return 1
}

MIRROR_REACHABLE=1

main() {
  require_command git
  require_command ssh
  require_command scp
  require_command sha1sum
  require_command awk
  require_command grep

  parse_args "$@"

  LOCAL_REPO=$(make_abs_path "$LOCAL_REPO")
  ensure_local_repo
  ensure_state_dirs
  setup_logging

  if ! check_mirror_reachable; then
    log "WARNING: mirror host '$MIRROR_HOST' is unreachable -- skipping mirror sync"
    MIRROR_REACHABLE=0
    return 0
  fi

  ensure_mirror_repo

  if [[ $RUN_ONCE -eq 1 ]]; then
    run_cycle
    return 0
  fi

  while true; do
    set +e
    run_cycle
    status=$?
    set -e

    if [[ $status -ne 0 ]]; then
      warn "git-sync error" "Sync cycle failed. Check terminal output for details."
    fi
    sleep "$POLL_INTERVAL"
  done
}

main "$@"
