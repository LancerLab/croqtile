#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sync_patch.sh [options] <path-to-repo-A>

Options:
  -r <rev>     Commit-ish in repo B to sync (default: HEAD)
  -o <file>    Output patch file path (default: ./sync_patch_<rev>_<ts>.patch in repo B)
  -k <file>    Forbidden keyword file (one pattern per line). If omitted, no keyword check.
  -i           Interactive mode: ask before creating new dirs/files or creating move target dirs
  -f           Force mode: create new dirs/files and move targets without prompting
  -h           Show help

Behavior:
  M: only apply if file exists in A (otherwise skip)
  A: if file already exists in A -> ABORT
     else:
       - if parent dir exists in A: default create
       - if parent dir missing in A: default skip
       - -i prompts in both cases; -f always creates
  D: if file exists in A -> delete it; else skip
  R*: rename/move: requires old path exists in A, otherwise ABORT
     - if new parent dir exists in A: do it
     - if new parent dir missing in A: -i prompt, -f create, else ABORT
  Other statuses: ABORT

Output:
  - Patch file (git diff --binary --find-renames) against A's HEAD (after stashing A local changes).
  - Commit message saved alongside as: <patch>.b_commit_msg.txt

Apply manually in repo A:
  git apply --index <patch>   # optional --index
  # or just:
  git apply <patch>
EOF
}

# -------- option parsing --------
REV="HEAD"
INTERACTIVE=0
FORCE=0
OUT_PATCH=""
KW_FILE=""

while getopts ":r:o:k:ifh" opt; do
  case "$opt" in
    r) REV="$OPTARG" ;;
    o) OUT_PATCH="$OPTARG" ;;
    k) KW_FILE="$OPTARG" ;;
    i) INTERACTIVE=1 ;;
    f) FORCE=1 ;;
    h) usage; exit 0 ;;
    :) echo "Error: -$OPTARG needs an argument"; usage; exit 2 ;;
    \?) echo "Error: unknown option -$OPTARG"; usage; exit 2 ;;
  esac
done
shift $((OPTIND-1))

if [[ $# -ne 1 ]]; then
  echo "Error: missing <path-to-choreo-open-repo>"
  usage
  exit 2
fi

CO_OPEN_REPO="$(cd "$1" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repo B root from git (don’t assume current working directory)
CHOREO_REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$CHOREO_REPO" ]]; then
  echo "Error: cannot find repo B root (is this script inside a git repo?)"
  exit 1
fi

if [[ -z "$KW_FILE" ]] && [[ -f $SCRIPT_DIR/os_kw.txt ]]; then
  KW_FILE=$SCRIPT_DIR/os_kw.txt
fi

# Validate A and B
git -C "$CHOREO_REPO" rev-parse --is-inside-work-tree >/dev/null
git -C "$CO_OPEN_REPO" rev-parse --is-inside-work-tree >/dev/null

# Default output patch path in repo B
if [[ -z "$OUT_PATCH" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  safe_rev="$(echo "$REV" | tr '/:' '__')"
  OUT_PATCH="$CHOREO_REPO/sync_patch_${safe_rev}_${ts}.patch"
fi

OUT_MSG="${OUT_PATCH}.b_commit_msg.txt"

# -------- helpers --------
prompt_yn() {
  local q="$1"
  local ans
  while true; do
    read -r -p "$q [y/n] " ans
    case "$ans" in
      y|Y) return 0 ;;
      n|N) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

abort() {
  echo "ABORT: $*" >&2
  exit 1
}

# Write file content from B at REV to A path (no checkout needed)
write_file_from_B_rev() {
  local rel="$1"
  local dst="$CO_OPEN_REPO/$rel"
  mkdir -p "$(dirname "$dst")"
  # Use git show to extract blob at REV:path
  git -C "$CHOREO_REPO" show "${REV}:${rel}" > "$dst"
}

dir_exists_in_A() {
  local rel_dir="$1"
  [[ -d "$CO_OPEN_REPO/$rel_dir" ]]
}

file_exists_in_A() {
  local rel="$1"
  [[ -e "$CO_OPEN_REPO/$rel" ]]
}

# Forbidden keyword checking: patch content (A changeset) + B commit message
keyword_check() {
  local patch_file="$1"
  local msg_file="$2"
  local kw_file="$3"

  [[ -z "$kw_file" ]] && return 0
  [[ -f "$kw_file" ]] || abort "keyword file not found: $kw_file"

  # Build a single extended regex from non-empty, non-comment lines
  local pat
  pat="$(grep -v '^[[:space:]]*$' "$kw_file" | grep -v '^[[:space:]]*#' || true)"
  [[ -z "$pat" ]] && return 0

  # Use grep -F? You likely want patterns. We'll treat each line as a regex.
  # Check patch (A changeset)
  while IFS= read -r re; do
    [[ -z "$re" ]] && continue
    if grep -nE -- "$re" "$patch_file" >/dev/null 2>&1; then
      echo "Forbidden keyword/pattern matched in PATCH (A changeset): $re" >&2
      echo "Showing first matches:" >&2
      grep -nE -- "$re" "$patch_file" | head -n 20 >&2
      exit 1
    fi
    if grep -nE -- "$re" "$msg_file" >/dev/null 2>&1; then
      echo "Forbidden keyword/pattern matched in B COMMIT MESSAGE: $re" >&2
      echo "Showing first matches:" >&2
      grep -nE -- "$re" "$msg_file" | head -n 20 >&2
      exit 1
    fi
  done < <(printf '%s\n' "$pat")
}

# -------- main flow --------

# Ensure REV exists in B
git -C "$CHOREO_REPO" rev-parse --verify "$REV^{commit}" >/dev/null

echo "Repo B: $CHOREO_REPO"
echo "Repo A: $CO_OPEN_REPO"
echo "B commit: $REV"
echo "Output patch: $OUT_PATCH"
[[ -n "$KW_FILE" ]] && echo "Keyword file: $KW_FILE"

# Collect commit message from B (for later keyword check + for user review)
git -C "$CHOREO_REPO" show -s --format=%B "$REV" > "$OUT_MSG"

# Get name-status with renames detected
# Example lines:
#   M\tpath
#   A\tpath
#   R100\told\tnew
mapfile -t CHANGES < <(git -C "$CHOREO_REPO" diff-tree --no-commit-id --name-status -r -M "$REV")

# Stash A local changes (sandbox)
A_WAS_DIRTY=0
STASH_CREATED=0
STASH_REF=""

if [[ -n "$(git -C "$CO_OPEN_REPO" status --porcelain)" ]]; then
  A_WAS_DIRTY=1
  echo "Repo A has local changes -> stashing (including untracked)..."
  # record current stash top
  BEFORE_TOP="$(git -C "$CO_OPEN_REPO" rev-parse -q --verify refs/stash 2>/dev/null || true)"
  git -C "$CO_OPEN_REPO" stash push -u -m "sync_patch sandbox stash $(date +%F_%T)" >/dev/null
  AFTER_TOP="$(git -C "$CO_OPEN_REPO" rev-parse -q --verify refs/stash 2>/dev/null || true)"
  if [[ "$AFTER_TOP" != "$BEFORE_TOP" && -n "$AFTER_TOP" ]]; then
    STASH_CREATED=1
    STASH_REF="stash@{0}"
  else
    # Very rare, but be defensive
    STASH_CREATED=0
  fi
fi

# Apply B changes into A worktree according to your rules
echo "Cooking changes into repo A working tree (sandbox)..."
for line in "${CHANGES[@]}"; do
  # split by tabs safely
  IFS=$'\t' read -r status p1 p2 <<<"$line"

  case "$status" in
    M)
      rel="$p1"
      if file_exists_in_A "$rel"; then
        echo "M  $rel  -> apply (exists in A)"
        write_file_from_B_rev "$rel"
      else
        echo "M  $rel  -> skip (missing in A)"
      fi
      ;;
    A)
      rel="$p1"
      if file_exists_in_A "$rel"; then
        abort "A  $rel but file already exists in A (rule: abort)"
      fi

      parent="$(dirname "$rel")"
      if dir_exists_in_A "$parent"; then
        # default create
        if [[ $INTERACTIVE -eq 1 ]]; then
          if prompt_yn "New file '$rel' (parent exists in A). Create?"; then
            echo "A  $rel  -> create"
            write_file_from_B_rev "$rel"
          else
            echo "A  $rel  -> skip (user chose no)"
          fi
        else
          # default yes, regardless of -f
          echo "A  $rel  -> create (default)"
          write_file_from_B_rev "$rel"
        fi
      else
        # default skip unless -i or -f
        if [[ $FORCE -eq 1 ]]; then
          echo "A  $rel  -> create (force, parent missing in A)"
          write_file_from_B_rev "$rel"
        elif [[ $INTERACTIVE -eq 1 ]]; then
          if prompt_yn "New file '$rel' (parent missing in A). Create directory+file?"; then
            echo "A  $rel  -> create (user approved)"
            write_file_from_B_rev "$rel"
          else
            echo "A  $rel  -> skip (user chose no)"
          fi
        else
          echo "A  $rel  -> skip (default, parent missing in A)"
        fi
      fi
      ;;

    D)
      rel="$p1"
      if file_exists_in_A "$rel"; then
        echo "D  $rel  -> delete (exists in A)"
        rm -f -- "$CO_OPEN_REPO/$rel"
      else
        echo "D  $rel  -> skip (missing in A)"
      fi
      ;;

    R* )
      old="$p1"
      new="$p2"
      [[ -z "$new" ]] && abort "Malformed rename line: $line"

      # Rule: moving a file existed in both A and B
      if ! file_exists_in_A "$old"; then
        abort "Rename $old -> $new but old path not present in A (rule 6 / abort)"
      fi

      new_parent="$(dirname "$new")"
      if dir_exists_in_A "$new_parent"; then
        echo "$status  $old -> $new  -> move (target dir exists in A)"
        mkdir -p "$CO_OPEN_REPO/$new_parent"
        rm -f "$CO_OPEN_REPO/$new" 2>/dev/null || true
        mkdir -p "$(dirname "$CO_OPEN_REPO/$new")"
        git -C "$CHOREO_REPO" show "${REV}:${new}" > "$CO_OPEN_REPO/$new"
        rm -f "$CO_OPEN_REPO/$old"
      else
        # new dir missing in A: interactive query (or force), otherwise abort
        if [[ $FORCE -eq 1 ]]; then
          echo "$status  $old -> $new  -> move (force create dir)"
          mkdir -p "$CO_OPEN_REPO/$new_parent"
          git -C "$CHOREO_REPO" show "${REV}:${new}" > "$CO_OPEN_REPO/$new"
          rm -f "$CO_OPEN_REPO/$old"
        elif [[ $INTERACTIVE -eq 1 ]]; then
          if prompt_yn "Move '$old' -> '$new' but target dir missing in A. Create dir and move?"; then
            mkdir -p "$CO_OPEN_REPO/$new_parent"
            git -C "$CHOREO_REPO" show "${REV}:${new}" > "$CO_OPEN_REPO/$new"
            rm -f "$CO_OPEN_REPO/$old"
          else
            abort "User declined move into missing dir (rule 6 / abort)"
          fi
        else
          abort "Move '$old' -> '$new' targets dir missing in A; use -i or -f (rule 5)"
        fi
      fi
      ;;
    *)
      abort "Unhandled change status '$status' (line: $line)"
      ;;
  esac
done

# Now create patch of what we did in A (changeset of A)
echo "Generating patch from repo A working tree..."
git -C "$CO_OPEN_REPO" diff --binary --find-renames > "$OUT_PATCH"

# Keyword check (patch content + B commit message)
if [[ -n "$KW_FILE" ]]; then
  echo "Running forbidden keyword check on: patch(A changeset) + B commit message..."
  keyword_check "$OUT_PATCH" "$OUT_MSG" "$KW_FILE"
fi

# Cleanup A sandbox: reset + clean + restore stash
echo "Cleaning up repo A sandbox (reset/clean)..."
git -C "$CO_OPEN_REPO" reset --hard >/dev/null
git -C "$CO_OPEN_REPO" clean -fd >/dev/null

if [[ $STASH_CREATED -eq 1 ]]; then
  echo "Restoring repo A stashed changes..."
  # pop as you requested; if conflicts, stop and tell user.
  if ! git -C "$CO_OPEN_REPO" stash pop >/dev/null; then
    echo "WARNING: stash pop had conflicts. Your stash is still in the stash list; resolve conflicts manually."
  fi
fi

echo
echo "Done."
echo "Patch:       $OUT_PATCH"
echo "B commit msg:$OUT_MSG"
echo
echo "Review:"
echo "  less -R '$OUT_PATCH'"
echo "Apply manually in choreo-open repo:"
echo "  (cd '$CO_OPEN_REPO' && git apply --index '$OUT_PATCH')"

