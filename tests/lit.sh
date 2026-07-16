#!/usr/bin/env bash

#================ a simple set implementation ====================
#
# Add an element to the set (no duplicates)
# Usage: set_add SET_NAME "element"
set_add() {
  local set_name="$1"
  shift  # Remove the set name from arguments

  # Handle case where no elements are provided
  if [ $# -eq 0 ]; then
    return 0
  fi

  local current_elements
  eval "current_elements=(\"\${${set_name}[@]}\")"

  # Process each new element
  for new_element in "$@"; do
    # Check if element already exists
    local found=0
    for existing_element in "${current_elements[@]}"; do
      if [ "$existing_element" = "$new_element" ]; then
        found=1
        break
      fi
    done

    # Add only if not found
    if [ $found -eq 0 ]; then
      current_elements+=("$new_element")
    fi
  done

  # Update the original array
  eval "$set_name=(\"\${current_elements[@]}\")"
}

# Check if element exists in set
# Usage: set_contains SET_NAME "element"
# Returns 0 if found, 1 if not found
set_contains() {
  local set_name="$1"
  local search_element="$2"
  local current_elements

  eval "current_elements=(\"\${${set_name}[@]}\")"

  # Handle empty array case
  if [ ${#current_elements[@]} -eq 0 ]; then
    return 1
  fi

  for element in "${current_elements[@]}"; do
    if [ "$element" = "$search_element" ]; then
      return 0
    fi
  done
  return 1
}

# Check if any element starts with the given prefix
# Usage: set_contains_prefix SET_NAME "prefix"
# Returns 0 if found, 1 if not found
set_contains_prefix() {
  local set_name="$1"
  local prefix="$2"
  local current_elements

  eval "current_elements=(\"\${${set_name}[@]}\")"

  if [ ${#current_elements[@]} -eq 0 ]; then
    return 1
  fi

  for element in "${current_elements[@]}"; do
    if [[ "$element" == "${prefix}"* ]]; then
      return 0
    fi
  done
  return 1
}

# Check if set is empty
# Usage: set_empty SET_NAME
# Returns 0 if empty, 1 if not empty
set_empty() {
    local set_name="$1"
    local current_elements

    eval "current_elements=(\"\${${set_name}[@]}\")"

    if [ ${#current_elements[@]} -eq 0 ]; then
        return 0  # Empty
    else
        return 1  # Not empty
    fi
}

# Get the size of the set
# Usage: set_size SET_NAME
# Echoes the number of elements
set_size() {
    local set_name="$1"
    local current_elements

    eval "current_elements=(\"\${${set_name}[@]}\")"
    echo ${#current_elements[@]}
}

set_clear() {
    local set_name="$1"
    eval "$set_name=()"
}

set_print() {
    local set_name="$1"
    local delimiter="${2:- }"  # Default to space
    local current_elements

    eval "current_elements=(\"\${${set_name}[@]}\")"

    # Handle empty set
    if [ ${#current_elements[@]} -eq 0 ]; then
        return 0  # Print nothing for empty set
    fi

    # Print elements with specified delimiter
    local first=1
    for element in "${current_elements[@]}"; do
        if [ $first -eq 1 ]; then
            printf '%s' "$element"
            first=0
        else
            printf '%s%s' "$delimiter" "$element"
        fi
    done
    printf '\n'
}

# ---- hook registry for target configure----

# Hooks stored as newline-separated "PHASE FUNC" pairs
HOOKS=""

register_hook() {
  # usage: register_hook <phase> <funcname>
  local phase="$1"
  local func="$2"

  # optional sanity: must look like a shell identifier
  case "$func" in
    ''|*[!a-zA-Z0-9_]*|[0-9]*)
      echo "test-runner: invalid hook function name: $func" >&2
      exit 2
      ;;
  esac

  # Ensure function exists *now* (config may define it before registering)
  command -v "$func" >/dev/null 2>&1 || {
    echo "lit.sh: hook function not found: $func (phase $phase)" >&2
    exit 2
  }

  HOOKS="${HOOKS}${HOOKS:+
}${phase} ${func}"
}

run_hooks() {
  local phase="$1"
  shift

  local line p func
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    p="${line%% *}"
    func="${line#* }"
    [[ "$p" == "$phase" ]] || continue

    # call hook in *current shell*
    "$func" "$@"
  done <<< "$HOOKS"
}

run_hooks_strict() {
  local phase="$1"
  shift

  local line p func

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue

    p="${line%% *}"
    func="${line#* }"

    [[ "$p" == "$phase" ]] || continue

    #echo "==> hook[$phase]: $func" >&2

    # Contain failure even under `set -e`
    if ! "$func" "$@"; then
      # echo "!! hook failed [$phase]: $func" >&2
      return 1
    fi
  done <<< "$HOOKS"

  return 0
}

# ---- Choreo lit.cfg marker validation ----
# Only source lit.cfg files whose first line starts with "# co-lit".
# This prevents accidentally sourcing configs from other tools (e.g. LLVM).
is_choreo_cfg() { local h; read -r h < "$1" 2>/dev/null && [[ "$h" == "# co-lit"* ]]; }

#===================== utilities =========================

get_terminal_width() {
    local width

    # Try COLUMNS environment variable first
    if [ -n "$COLUMNS" ]; then
        width=$COLUMNS
    # Try stty size
    elif width=$(stty size 2>/dev/null | cut -d' ' -f2) && [ -n "$width" ]; then
        :
    # Try stty -a (older systems)
    elif width=$(stty -a 2>/dev/null | grep -o 'columns [0-9]*' | cut -d' ' -f2) && [ -n "$width" ]; then
        :
    # Final fallback
    else
        width=80
    fi

    echo "$width"
}

supports_color_output() {
  [ -t 1 ] || return 1
  [ -n "${NO_COLOR:-}" ] && return 1
  [ "${TERM:-dumb}" = "dumb" ] && return 1

  if command -v tput >/dev/null 2>&1; then
    local colors
    colors=$(tput colors 2>/dev/null || echo 0)
    [[ "$colors" =~ ^[0-9]+$ ]] || colors=0
    [ "$colors" -ge 8 ] || return 1
  fi

  return 0
}

color_enabled=0
color_reset=""
color_bold=""
color_green=""
color_red=""
color_yellow=""
color_cyan=""

init_output_styles() {
  if ! supports_color_output; then
    return 0
  fi

  color_enabled=1
  color_reset=$'\033[0m'
  color_bold=$'\033[1m'
  color_green=$'\033[32m'
  color_red=$'\033[31m'
  color_yellow=$'\033[33m'
  color_cyan=$'\033[36m'
}

style_text() {
  local kind="$1"
  local text="$2"

  if [ "$color_enabled" -ne 1 ]; then
    printf '%s' "$text"
    return 0
  fi

  case "$kind" in
    PASS)
      printf '%b%s%b' "${color_bold}${color_green}" "$text" "$color_reset"
      ;;
    FAIL|UNEXPECTED*)
      printf '%b%s%b' "${color_bold}${color_red}" "$text" "$color_reset"
      ;;
    XFAIL)
      printf '%b%s%b' "${color_bold}${color_cyan}" "$text" "$color_reset"
      ;;
    SKIP*)
      printf '%b%s%b' "${color_bold}${color_yellow}" "$text" "$color_reset"
      ;;
    SUMMARY)
      printf '%b%s%b' "$color_bold" "$text" "$color_reset"
      ;;
    *)
      printf '%s' "$text"
      ;;
  esac
}

print_status_line() {
  local label="$1"
  shift

  printf '%s %s\n' "$(style_text "$label" "$label")" "$*"
}

print_timed_status_line() {
  local kind="$1"
  local file="$2"
  local count="$3"
  local total="$4"
  local elapsed_time="$5"
  local plain_label="${kind}:"
  local plain_rest=" $file ($count of $total)"
  local label_width=${#plain_label}
  local rest_width=$((max_text_width - label_width))

  [ "$rest_width" -lt 1 ] && rest_width=1
  printf '%s%-*s %s\n' \
    "$(style_text "$kind" "$plain_label")" \
    "$rest_width" \
    "$plain_rest" \
    "| Time: $elapsed_time"
}

expand_target_specs() {
  local out_name="$1"
  shift

  local -a saved_req_targets=("${REQ_TARGETS[@]}")
  local -a normalized_specs=()
  local spec
  for spec in "$@"; do
    normalized_specs+=("$(toupper "$spec")")
  done

  set_clear REQ_TARGETS
  if [ ${#normalized_specs[@]} -ne 0 ]; then
    run_hooks "set_archs" "${normalized_specs[@]}"
  fi

  eval "$out_name=(\"\${REQ_TARGETS[@]}\")"
  REQ_TARGETS=("${saved_req_targets[@]}")
}

target_specs_match_candidate() {
  local candidate="$1"
  shift

  local -a expanded_targets=()
  local expanded_target

  expand_target_specs expanded_targets "$@"
  for expanded_target in "${expanded_targets[@]}"; do
    if [[ "$expanded_target" == "$candidate" ]]; then
      return 0
    fi
  done

  return 1
}

target_specs_are_required() {
  local -a expanded_targets=()
  local expanded_target

  expand_target_specs expanded_targets "$@"
  [ ${#expanded_targets[@]} -ne 0 ] || return 1

  for expanded_target in "${expanded_targets[@]}"; do
    if ! set_contains REQ_TARGETS "$expanded_target"; then
      return 1
    fi
  done

  return 0
}


validate_cuda_home() {
  # Check if CUDA_HOME is set
  if [[ -z "${CUDA_HOME}" ]]; then
#    echo "Error: CUDA_HOME environment variable is not set" >&2
    return 1
  fi

  # Check if CUDA_HOME points to a directory
  if [[ ! -d "${CUDA_HOME}" ]]; then
    echo "Error: CUDA_HOME does not point to a valid directory: ${CUDA_HOME}" >&2
    return 1
  fi

  # Check for essential CUDA files/directories
  local required_paths=(
    "bin/nvcc"
    "lib64"
    "include/cuda.h"
  )

  for path in "${required_paths[@]}"; do
    if [[ ! -e "${CUDA_HOME}/${path}" ]]; then
      echo "Error: Missing required CUDA component: ${CUDA_HOME}/${path}" >&2
      return 1
    fi
  done

#  echo "CUDA_HOME is valid: ${CUDA_HOME}"
  return 0
}

CFG_SOURCED=""

already_sourced_cfg() {
  echo "$CFG_SOURCED" | grep -Fqx "$1" 2>/dev/null
}

mark_sourced_cfg() {
  if [ -z "$CFG_SOURCED" ]; then
    CFG_SOURCED="$1"
  else
    CFG_SOURCED="$CFG_SOURCED
$1"
  fi
}

abspath_dir_of() {
  p="$1"
  if [ -d "$p" ]; then
    (cd "$p" 2>/dev/null && pwd) || return 1
  else
    (cd "$(dirname "$p")" 2>/dev/null && pwd) || return 1
  fi
}

abspath_file() {
  echo "$(abspath_dir_of "$1")/$(basename "$1")"
}

# ---- include_dir discovery mechanism ----
#
# A lit.cfg may call include_dir("../path") to declare that another
# directory's tests should run under this cfg's target hooks.
# During expand_includes() discovery the calls are collected; at
# runtime (inside prepare()) include_dir() is a no-op.
_DISCOVERY_MODE=0    # set to 1 only inside expand_includes
_CURRENT_CFG_DIR=""  # cfg directory being sourced during discovery
_PENDING_INCLUDES=() # collected "cfg_dir|rel_dir" pairs

# Called from lit.cfg to declare a shared directory.
# No-op at runtime; recorded during expand_includes() discovery.
include_dir() {
  [ $_DISCOVERY_MODE -eq 0 ] && return 0
  _PENDING_INCLUDES+=("${_CURRENT_CFG_DIR}|$1")
}

# expand_includes GivenDir
# Sources every lit.cfg under GivenDir to collect include_dir() calls.
# For each declared directory, adds its .co/.cmt files to files_array as
# "abs_file|cfg_dir".  Files claimed by any include are suppressed from
# their own direct (hookless) run so each file runs exactly once per target.
expand_includes() {
  local given_dir="$1"
  local abs_dir
  abs_dir="$(abspath_dir_of "$given_dir")" || return 0

  # Source all lit.cfg files in discovery mode to collect include_dir calls.
  # HOOKS and CFG_SOURCED are reset afterward so per-file prepare() is unaffected.
  _DISCOVERY_MODE=1
  _PENDING_INCLUDES=()
  local cfg_file
  while IFS= read -r -d '' cfg_file; do
    _CURRENT_CFG_DIR="$(abspath_dir_of "$cfg_file")"
    CFG_SOURCED=""
    HOOKS=""
    . "$cfg_file" 2>/dev/null || true
  done < <(find "$abs_dir" -name 'lit.cfg' -print0)
  _DISCOVERY_MODE=0
  CFG_SOURCED=""
  HOOKS=""

  # Nothing to do if no include_dir declarations were found
  [ ${#_PENDING_INCLUDES[@]} -eq 0 ] && return 0

  local -a target_entries=()
  local -a suppressed_abs=()
  local inc_entry cfg_dir rel_dir resolved_dir
  for inc_entry in "${_PENDING_INCLUDES[@]}"; do
    cfg_dir="${inc_entry%%|*}"
    rel_dir="${inc_entry#*|}"
    resolved_dir="$(cd "$cfg_dir" && cd "$rel_dir" 2>/dev/null && pwd)" || continue
    while IFS= read -r -d '' f; do
      local abs_f
      abs_f="$(abspath_file "$f")"
      target_entries+=("${abs_f}|${cfg_dir}")
      suppressed_abs+=("$abs_f")
    done < <(find "$resolved_dir" -type f \( -name '*.co' -o -name '*.cmt' \) -print0)
  done

  [ ${#target_entries[@]} -eq 0 ] && return 0

  # Rebuild files_array: drop suppressed entries, then append per-target entries
  local -a new_array=()
  local entry abs_entry suppressed s
  for entry in "${files_array[@]}"; do
    abs_entry="$(abspath_file "$entry")"
    suppressed=0
    for s in "${suppressed_abs[@]}"; do
      if [ "$abs_entry" = "$s" ]; then
        suppressed=1
        break
      fi
    done
    [ $suppressed -eq 0 ] && new_array+=("$entry")
  done
  for entry in "${target_entries[@]}"; do
    new_array+=("$entry")
  done
  files_array=("${new_array[@]}")
}

load_cfg_chain() {
  # Walk UP from the test file's directory, collecting Choreo-marked
  # lit.cfg files.  Stop at repo_root (inclusive) or /.
  # Then source in reverse (parent-first) order.
  local test_path="$1"
  local cfg_name="${2:-lit.cfg}"
  local test_dir
  test_dir="$(abspath_dir_of "$test_path")"

  local -a _chain=()
  local d="$test_dir"
  while :; do
    if [[ -f "$d/$cfg_name" ]] && is_choreo_cfg "$d/$cfg_name"; then
      _chain+=("$d/$cfg_name")
    fi
    [[ "$d" == "$repo_root" ]] && break
    [[ "$d" == "/" || -z "$d" ]] && break
    d="${d%/*}"
    [[ -z "$d" ]] && d="/"
  done

  # Source in reverse order (outermost parent first).
  local _i cfg_abs
  for (( _i=${#_chain[@]}-1; _i>=0; _i-- )); do
    cfg_abs="${_chain[$_i]}"
    if ! already_sourced_cfg "$cfg_abs"; then
      mark_sourced_cfg "$cfg_abs"
      . "$cfg_abs"
    fi
  done
}

#=========================================================

# Get the directory where the script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Repo root is the cfg-chain walk-up boundary.  Prefer git; fall back to
# the script's parent directory so lit.sh works for tests anywhere in the
# repository (e.g. tools/co-mock/tests/).
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  repo_root="$(cd "${script_dir}/.." && pwd)"
  echo "Warning: not inside a git repository; assuming repo root = $repo_root" >&2
fi
repo_root="$(cd "$repo_root" && pwd)"
timestamp=$(date +%Y%m%d%H%M%S)

# Add repo root and its tooling directories to PATH
export PATH="$script_dir:${repo_root}:${repo_root}/extern/bin/:${repo_root}/extern/:${repo_root}/extern/llvm-project/bin/:$PATH"

# Check if FileCheck exists in the PATH
FILECHECK=$(which FileCheck \
            FileCheck-18 FileCheck-17 FileCheck-16 \
            FileCheck-15 FileCheck-14 \
            FileCheck-10 2>/dev/null | head -1)

GDB_BIN=$(which gdb gdb-multiarch 2>/dev/null | head -1)
CUDA_GDB_BIN=$(which cuda-gdb 2>/dev/null | head -1)

if [ -z "$FILECHECK" ]; then
  echo "-------------------------------------------------------"
  echo "ERROR: FileCheck utility not found!"
  echo "This project requires FileCheck (from LLVM) for testing."
  echo ""
  echo "On Ubuntu/Debian, install it with:"
  echo "  sudo apt update && sudo apt install llvm-dev"
  echo "-------------------------------------------------------"
  exit 1
fi

# choreo and copp are optional -- tests that don't use them (e.g. co-mock
# tests) can run without them.  The sed substitution in execute_command()
# handles missing binaries gracefully.
_choreo_bin="$(which choreo 2>/dev/null || true)"
_copp_bin="$(which copp 2>/dev/null || true)"
_cocc_bin="$(which cocc 2>/dev/null || true)"

if ! which not.sh &>/dev/null; then
  echo "Error: not.sh is not found in PATH."
  exit 1
fi

if ! which bc &>/dev/null; then
  echo "Error: bc is not found in PATH."
  exit 1
fi

echo "---------------------------------------"
echo "        Choreo SimpleLit - v0.34"
echo "---------------------------------------"
echo ""

init_output_styles

reproduce_commands=()
working_command=""
num_tested=0
num_passed=0
num_failed=0
num_uepass=0
num_xfails=0
num_skiped=0

is_in_docker=false
is_in_shell=false
REQ_TARGETS=()
requires_dynamic_shape=0
requires_gdb=0
requires_cudagdb=0
expect_fail=
expect_skip=
req_features=()

choreo_target=""
choreo_arch=""
declare -A FEATURE_CACHE=()

max_jobs=1
dry_run=0

sim_mode="off"
run_only=""
target_only=""

need_cute=0
need_cuda=0

if [ -f /.dockerenv ] || grep -qE "(docker|containerd)" /proc/1/cgroup; then
  is_in_docker=true
  is_in_shell=false
else
  is_in_docker=false
  is_in_shell=true
fi

# Check the hardware device availability and type
# Note: consider the machine only installed a single target
device_type="none"
mach=
simulator="none"

# Cache hardware detection results per sourced cfg chain.
# This avoids repeated detection for many files under the same lit config.
declare -A HW_DETECT_CACHE=()

# Cache cfg chain and hook registry, keyed by the set of lit.cfg files
# in the path (chain_key).  Different directories that share the same
# lit.cfg chain are served from a single cache entry, so each chain is
# sourced exactly once -- preventing side-effect re-initialization of
# target variables on a cache hit.
declare -A CFG_CHAIN_CACHE=()
declare -A HOOKS_CACHE=()

# Fast directory -> chain_key mapping so compute_cfg_files_key() is
# called at most once per unique directory.
declare -A DIR_TO_CHAIN_KEY=()

# Walk UP from dir to repo_root (or /), collecting Choreo-marked
# lit.cfg paths.  Returns the result via _cfg_files_key (no subshell).
# The key is ordered parent-first (reversed from the walk-up order)
# to match load_cfg_chain's sourcing order.
_cfg_files_key=""
compute_cfg_files_key() {
  local dir="$1"
  local -a _found=()
  local d="$dir"
  while :; do
    [[ -f "$d/lit.cfg" ]] && is_choreo_cfg "$d/lit.cfg" && _found+=("$d/lit.cfg")
    [[ "$d" == "$repo_root" ]] && break
    [[ "$d" == "/" || -z "$d" ]] && break
    d="${d%/*}"
    [[ -z "$d" ]] && d="/"
  done
  local key="" _i
  for (( _i=${#_found[@]}-1; _i>=0; _i-- )); do
    key="${key}${key:+;}${_found[$_i]}"
  done
  _cfg_files_key="${key:-__no_cfg__}"
}

make_cfg_cache_key() {
  local cfgs="$1"
  if [ -z "$cfgs" ]; then
    echo "__no_cfg__"
    return
  fi
  # Flatten multi-line cfg list into a stable key.
  echo "$cfgs" | tr '\n' ';'
}

save_hw_detect_cache() {
  local key="$1"
  HW_DETECT_CACHE["$key"]="${device_type}|${mach}|${simulator}"
}

restore_hw_detect_cache() {
  local key="$1"
  local cached="${HW_DETECT_CACHE["$key"]}"

  IFS='|' read -r device_type mach simulator <<< "$cached"
}

# Function to fill the target-specific variables
# prepare <file> [cfg_override_dir]
#   cfg_override_dir: when set, load the cfg chain rooted at this directory
#   instead of the directory that contains <file>.  Used by expand_includes
#   so that files in tests/check/ are processed with a target's lit.cfg.
prepare() {
  local file="$1"
  local cfg_override="${2:-}"

  # Validate input
  if [[ -z "$file" ]] || [[ ! -f "$file" ]]; then
    echo "Error: Invalid file specified" >&2
    return 1
  fi

  local ext="${file##*.}"
  if [[ "${ext}" == "co" ]]; then
    :
  elif [[ "${ext}" == "cmt" ]]; then
    :
  elif [[ "${ext}" == "mlir" ]]; then
    :
  else
    echo "Error: Invalid test file: $file" >&2
    return 1
  fi

  # now load the configure
  local cfg_root_dir
  if [[ -n "$cfg_override" ]]; then
    cfg_root_dir="$(abspath_dir_of "${cfg_override}/__sentinel__")"
  else
    cfg_root_dir="$(abspath_dir_of "$file")"
  fi

  local chain_key
  if [[ -n "${DIR_TO_CHAIN_KEY["$cfg_root_dir"]+x}" ]]; then
    chain_key="${DIR_TO_CHAIN_KEY["$cfg_root_dir"]}"
  else
    compute_cfg_files_key "$cfg_root_dir"
    chain_key="$_cfg_files_key"
    DIR_TO_CHAIN_KEY["$cfg_root_dir"]="$chain_key"
  fi

  if [[ -n "${CFG_CHAIN_CACHE["$chain_key"]+x}" ]]; then
    CFG_SOURCED="${CFG_CHAIN_CACHE["$chain_key"]}"
    HOOKS="${HOOKS_CACHE["$chain_key"]}"
  else
    CFG_SOURCED=""
    HOOKS=""
    if [[ -n "$cfg_override" ]]; then
      # Load the cfg chain for the override dir by passing a sentinel path
      # whose dirname resolves to that directory.
      load_cfg_chain "${cfg_override}/__sentinel__"
    else
      load_cfg_chain "$file"
    fi
    CFG_CHAIN_CACHE["$chain_key"]="$CFG_SOURCED"
    HOOKS_CACHE["$chain_key"]="$HOOKS"
  fi

  local cfg_cache_key
  cfg_cache_key="$(make_cfg_cache_key "$CFG_SOURCED")"

  # detect the hardware
  if [[ -n "${HW_DETECT_CACHE["$cfg_cache_key"]+x}" ]]; then
    restore_hw_detect_cache "$cfg_cache_key"
  else
    run_hooks "hw_detect"

    # target-specific preparation
    run_hooks "target_prepare"

    save_hw_detect_cache "$cfg_cache_key"
  fi


  # Reset target requirement
  requires_dynamic_shape=0
  requires_gdb=0
  requires_cudagdb=0
  need_cute=0
  need_cuda=0
  expect_fail=
  expect_skip=

  local requires=""
  local line=""
  local payload=""

  # Parse key directives with builtins to reduce process spawning.
  while IFS= read -r line; do
    if [[ "${ext}" == "co" ]]; then
      [[ "$line" == "//"* ]] || continue
      payload="${line#//}"
    else
      [[ "$line" == "#"* ]] || continue
      payload="${line#\#}"
    fi

    if [[ -z "$expect_fail" ]] && [[ "$payload" == *"XFAIL:"* ]]; then
      expect_fail="${payload#*XFAIL:}"
      expect_fail="$(trim_spaces "$expect_fail")"
    fi

    if [[ -z "$expect_skip" ]] && [[ "$payload" == *"SKIP:"* ]]; then
      expect_skip="$line"
    fi

    if [[ -z "$requires" ]] && [[ "$payload" == *"REQUIRES:"* ]]; then
      requires="${payload#*REQUIRES:}"
      requires="$(trim_spaces "$requires")"
    fi

    if [[ -n "$expect_skip" ]] && [[ -n "$requires" ]] && [[ -n "$expect_fail" ]]; then
      break
    fi
  done < "$file"

  set_clear REQ_TARGETS

  # If no REQUIRES line found, return early
  if [[ -z "$requires" ]]; then
    return 0
  fi

  # Extract components
  local tgts=()
  local libs=()
  req_features=()
  local token
  for token in $requires; do
    if [[ "$token" == TARGET-* ]]; then
      tgts+=("${token#TARGET-}")
    elif [[ "$token" == LIBRARY-* ]]; then
      libs+=("${token#LIBRARY-}")
    elif [[ "$token" == "DYNAMIC-SHAPE" ]]; then
      requires_dynamic_shape=1
    elif [[ "$token" == "GDB" ]] || [[ "$token" == "TOOL-GDB" ]]; then
      requires_gdb=1
    elif [[ "$token" == "CUDA-GDB" ]]; then
      requires_cudagdb=1
    else
      req_features+=("$(toupper "$token")")
    fi
  done

  # Process targets
  if [ ${#tgts[@]} -ne 0 ]; then
    run_hooks "set_archs" "${tgts[@]}"
  fi

  if set_empty REQ_TARGETS && [ ${#tgts[@]} -ne 0 ]; then
    echo "invalid target: ${tgts[*]}" >&2
  fi

  # has library requirement
  for token in "${libs[@]}"; do
    if [[ "$token" == "CUTE" ]]; then
      need_cute=1
      need_cuda=1
      break
    fi
  done
}

lock_file="/tmp/test_script_lock_${timestamp}"
counter_file="/tmp/test_counters_${timestamp}.txt"
reproduce_file="/tmp/reproduce_commands_${timestamp}.txt"
rm -f $counter_file
rm -f $reproduce_file
touch $reproduce_file

initialize_counters() {
  if [ ! -f "$counter_file" ]; then
    echo "num_tested=0" > "$counter_file"
    echo "num_failed=0" >> "$counter_file"
    echo "num_passed=0" >> "$counter_file"
    echo "num_uepass=0" >> "$counter_file"
    echo "num_xfails=0" >> "$counter_file"
  fi
}

read_counter() {
  local counter_name=$1
  grep -E "^$counter_name=" "$counter_file" | cut -d'=' -f2
}

increment_counter() {
  local counter_name=$1
  local value=$2
  local counter_lock_file="/tmp/${counter_name}_lock_${timestamp}"

  exec 200>"$counter_lock_file"
  flock -x 200

  local current_value
  current_value=$(read_counter "$counter_name")
  old_num_tested=$(read_counter "num_tested")

  new_value=$((current_value + value))
  new_num_tested=$((old_num_tested + 1))

  sed -i "s/^$counter_name=.*/$counter_name=$new_value/" "$counter_file"
  sed -i "s/^num_tested=.*/num_tested=$new_num_tested/" "$counter_file"

  exec 200>&-
}

append_reproduce_command() {
  local command=$1
  local reproduce_lock_file="/tmp/reproduce_lock_${timestamp}"

  exec 200>"$reproduce_lock_file"
  flock -x 200

  echo "$command" >> "$reproduce_file"

  exec 200>&-
}

# Function to replace placeholders and execute command
execute_command() {
  local file=$1
  local command=$2
  local count=$3
  local total=$4
  local env_set="$5"
  local env_unset="$6"
  local run_env="$7"

  # Replace 'choreo', 'copp' and 'FileCheck' with their absolute paths
  # Note: It must uses '-n' to remove comments inside host code.
  #       Or else FileCheck will check the line of "// CHECK:"
  # workaround: Use cuda_gdb instead of cuda-gdb.
  #             Or else it will be replaced to cuda-/bin/gdb
  # IMPORTANT: %s substitution (file path) happens AFTER these sed calls to
  # avoid the binary names (e.g. 'choreo') being matched inside the file path.
  local _co_mock_bin; _co_mock_bin="$(which co-mock 2>/dev/null || echo co-mock)"
  local _sed_args=()
  _sed_args+=(-e "s#\bco-mock\b#__CO_MOCK_PLACEHOLDER__#g")
  [[ -n "$_choreo_bin" ]] && _sed_args+=(-e "s#%choreo#${_choreo_bin} -n#g")
  [[ -n "$_choreo_bin" ]] && _sed_args+=(-e "s#\bchoreo\b#${_choreo_bin} -n#g")
  [[ -n "$_copp_bin" ]]   && _sed_args+=(-e "s#%copp#${_copp_bin}#g")
  [[ -n "$_copp_bin" ]]   && _sed_args+=(-e "s#\bcopp\b#${_copp_bin}#g")
  [[ -n "$_cocc_bin" ]]   && _sed_args+=(-e "s#%cocc#${_cocc_bin} -n#g")
  [[ -n "$_cocc_bin" ]]   && _sed_args+=(-e "s#\bcocc\b#${_cocc_bin} -n#g")
  _sed_args+=(-e "s#\bFileCheck\b#${FILECHECK}#g")
  _sed_args+=(-e "s#\bgdb\b#${GDB_BIN}#g")
  _sed_args+=(-e "s#\bcuda_gdb\b#${CUDA_GDB_BIN}#g")
  _sed_args+=(-e "s#__CO_MOCK_PLACEHOLDER__#${_co_mock_bin}#g")
  command=$(echo "$command" | sed "${_sed_args[@]}")
  local not_command=$(which not.sh | sed 's/[&/\]/\\&/g')
  command=$(echo "$command" | sed "s/\bnot \(.*\)/${not_command} \1/")
  run_hooks "target_cmd" "command"
  # Strip any unresolved %target (e.g. when no target hook was registered)
  command="${command//%target/}"
  # Replace %s with the filename (must be after sed so the path isn't mangled)
  command=${command//%s/"$file"}

  # num_tested=$(($num_tested + 1))
  # echo "num_tested before add " $(read_counter "num_tested")
  # echo "num_tested after add " $(read_counter "num_tested")

  # start timing
  local start_time_ns=$(date +%s%N)

  # execute the command
  command="${env_set} ${run_env} $command"
  working_command="$command"

  if [ $dry_run -eq 1 ]; then
    echo "DRYRUN: $file" >&2
    local exit_code=0
  else
    eval "$command" 2>/dev/null
    local exit_code=$?
  fi

  if [ ! -z "${env_unset}" ]; then
    eval "$env_unset" 2>/dev/null
  fi
  working_command=""

  # Calculate elapsed time in nanoseconds
  local end_time_ns=$(date +%s%N)
  local elapsed_ns=$((end_time_ns - start_time_ns))

  # Convert time to appropriate unit
  local elapsed_time
  if [[ $elapsed_ns -ge 1000000000 ]]; then
      elapsed_time="$(bc <<< "scale=3; $elapsed_ns / 1000000000") s"
  elif [[ $elapsed_ns -ge 1000000 ]]; then
      elapsed_time="$(bc <<< "scale=3; $elapsed_ns / 1000000") ms"
  else
      elapsed_time="$(bc <<< "scale=3; $elapsed_ns / 1000") us"
  fi

  local term_width=$(get_terminal_width)
  local max_text_width=$((term_width - 25))

  if [[ $exit_code -eq 0 ]]; then
    if [[ "$expect_fail" == "*"* ]]; then
      increment_counter num_uepass 1
      append_reproduce_command "$command"
      print_timed_status_line "UNEXPECTED PASS" "$file" "$count" "$total" "$elapsed_time"
    elif [[ ! -z "${expect_fail}" ]] &&
         [[ "$(toupper ${expect_fail})" ==  *"$(toupper ${mach})"* ]]; then
      increment_counter num_uepass 1
      append_reproduce_command "$command"
      print_timed_status_line "UNEXPECTED PASS" "$file" "$count" "$total" "$elapsed_time"
    else
      increment_counter num_passed 1
      print_timed_status_line "PASS" "$file" "$count" "$total" "$elapsed_time"
    fi
  else
    if [[ "${expect_fail}" == "*"* ]]; then
      increment_counter num_xfails 1
      print_timed_status_line "XFAIL" "$file" "$count" "$total" "$elapsed_time"
    elif [[ ! -z "${expect_fail}" ]] &&
         [[ "$(toupper ${expect_fail})" ==  *"$(toupper ${mach})"* ]]; then
      increment_counter num_xfails 1
      print_timed_status_line "XFAIL" "$file" "$count" "$total" "$elapsed_time"
    else
      increment_counter num_failed 1
      append_reproduce_command "$command"
      print_timed_status_line "FAIL" "$file" "$count" "$total" "$elapsed_time"
    fi
  fi
}

# ---------------------------------------"
#         Handle arguments
# ---------------------------------------"
if [ $# -lt 1 ]; then
    echo "Usage: $0 [-jN] [--dry-run] [--sim=off|on|only] [--run-only=BIN] [--target-only=TGT] <file_or_directory>"
    exit 1
fi

# Process arguments with while-case loop
while [[ $# -gt 0 ]]; do
  case $1 in
    -j*)
      # Handle -jN argument (extract the number after -j)
      num_jobs="${1#-j}"

      if [[ ! "$num_jobs" =~ ^[1-9][0-9]*$ ]]; then
          echo "Error: Invalid -j value '$num_jobs'. It must be a positive integer."
          exit 1
      fi

      max_jobs=$num_jobs
      shift
      ;;
    -l)
      save_log=true
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --sim=*)
      sim_mode="${1#*=}"
      if [[ "$sim_mode" != "off" && "$sim_mode" != "on" && "$sim_mode" != "only" ]]; then
        echo "Error: --sim must be off, on, or only"
        exit 1
      fi
      shift
      ;;
    --run-only=*)
      run_only="${1#*=}"
      shift
      ;;
    --target-only=*)
      target_only="${1#*=}"
      shift
      ;;
    -*)
      # Handle invalid option
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      # Handle the first positional argument (file or directory)
      if [ -d "$1" ]; then
          # If it's a directory, find all .co, .cmt(cmake test) files
          _given_dir="$1"
          files_array=($(find "$_given_dir" -type f -name '*.co' -o -name '*.cmt' -o -name '*.mlir'))
          expand_includes "$_given_dir"
      elif [ -f "$1" ]; then
          # If it's a file, add it to the array
          files_array=("$1")
      else
          # Invalid argument
          echo "Provided argument is not a valid file or directory."
          exit 1
      fi
      shift
      ;;
  esac
done

cleantmplocks() {
  rm -f $lock_file
  rm -f $counter_lock_file
  rm -f $reproduce_lock_file
}

showresult() {
  echo ""
  echo "$(style_text "SUMMARY" "------ Lit Test summary ------")"
  echo "Tested:  $(read_counter 'num_tested')"
  echo "$(style_text "PASS" "Passed:")  $(read_counter 'num_passed')"

  [ $num_skiped -ne 0 ] && echo "$(style_text "SKIP" "Skipped:") $num_skiped"
  [ $(read_counter 'num_failed') -ne 0 ] && echo "$(style_text "FAIL" "Failed:")  $(read_counter 'num_failed')"
  [ $(read_counter 'num_xfails') -ne 0 ] && echo "$(style_text "XFAIL" "Expected Failures:") $(read_counter 'num_xfails')"
  [ $(read_counter 'num_uepass') -ne 0 ] && echo "$(style_text "UNEXPECTED PASS" "Unexpected Passes:") $(read_counter 'num_uepass')"

  local succed=$(($(read_counter 'num_passed') + $(read_counter 'num_xfails')))
  local failed=$(($(read_counter 'num_failed') + $(read_counter 'num_uepass')))

  if [[ $failed -ne 0 ]]; then
    echo ""
    echo "Commands to reproduce failures:"

    while IFS= read -r com; do
        echo "$com"
    done < "$reproduce_file"

  fi

  if [[ "${working_command}" != "" ]]; then
    echo ""
    echo "Command in-work:"
    echo "${working_command}"
  fi

  return ${failed}
}

handlestatus() {
  if [ $? -ne 0 ]; then
    echo "Tests failed"
    exit $?
  fi
}

on_ctrl_c() {
  cleantmplocks
  showresult
  exit 1
}

tolower() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

toupper() {
  echo "$1" | tr '[:lower:]' '[:upper:]'
}

trim_spaces() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  echo "$s"
}

retrieve_run_config() {
  local line="$@"

  # reset
  run_command=
  run_environ=
  run_target=

  if [[ $line =~ [[:blank:]]*RUN:[[:blank:]]*(.+) ]]; then
    # Extract the command after "RUN:"
    run_command="${BASH_REMATCH[1]}"
    run_environ="shell"
  elif [[ $line =~ [[:blank:]]*RUN-([^-]+):[[:blank:]]*(.+) ]]; then
    run_target=$(echo "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')
    local matched_command="${BASH_REMATCH[2]}"
    if target_specs_match_candidate "$mach" "$run_target"; then
      run_command="${matched_command}"
      run_environ="shell"
    elif [[ "${run_target}" == "docker" ]]; then
      run_command="${matched_command}"
      run_environ="docker"
      run_target=
    fi
  elif [[ $line =~ [[:blank:]]*RUN-([^-]+)-([^-]+):[[:blank:]]*(.+) ]]; then
    run_target=$(echo "${BASH_REMATCH[2]}" | tr '[:upper:]' '[:lower:]')
    run_environ=$(echo "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')
    run_command="${BASH_REMATCH[3]}"
  fi
}

# detect the device supported features
initialize_counters

trap on_ctrl_c SIGINT

for _entry in "${files_array[@]}"; do
  # Decode optional cfg_override encoded as "file|cfg_dir"
  _cfg_override="${_entry#*|}"
  [ "$_cfg_override" = "$_entry" ] && _cfg_override=""
  file="${_entry%%|*}"

  # check requirement specified by the file
  prepare "$file" "$_cfg_override"

  if [ ! -z "$expect_skip" ]; then
    _skip_reason="${expect_skip#*SKIP:}"
    _skip_reason="$(trim_spaces "$_skip_reason")"
    print_status_line "SKIP(${_skip_reason}):" "$file"
    num_skiped=$(($num_skiped + 1));
    continue;
  fi

  if [ ${need_cuda} -eq 1 ]; then
    if ! validate_cuda_home; then
      print_status_line "SKIP(CUDA):" "$file"
      num_skiped=$(($num_skiped + 1));
      continue;
    elif [ "$device_type" != "gpu" ]; then
      print_status_line "SKIP(GPU):" "$file"
      num_skiped=$(($num_skiped + 1));
      continue;
    else
      run_env="CUDA_HOME=${CUDA_HOME}"
    fi
  fi

  if [ ${requires_gdb} -eq 1 ] && [ -z "${GDB_BIN}" ]; then
    print_status_line "SKIP(GDB):" "$file"
    num_skiped=$(($num_skiped + 1));
    continue;
  fi

  if [ ${requires_cudagdb} -eq 1 ] && [ -z "${CUDA_GDB_BIN}" ]; then
    print_status_line "SKIP(CUDA-GDB):" "$file"
    num_skiped=$(($num_skiped + 1));
    continue;
  fi

  exe_env=
  unset_env=

  if ! run_hooks_strict "target_noskip" ; then
    continue; #target requires skipping
  fi

  # If it is in the end2end folder, keep it blocking
  sequential=0
  [[ "$(dirname ${file})" == *"end2end"* ]] && sequential=1;

  if [[ -z "${CHOREO_ENABLE_GPU_MMA_TESTS}" ]] || [[ "${CHOREO_ENABLE_GPU_MMA_TESTS}" != "1" ]]; then
    if [[ "$(dirname ${file})" == *"gpu/end2end/wmma"* ]] || [[ "$(dirname ${file})" == *"gpu/end2end/ptx_mma"* ]]; then
      print_status_line "SKIP(GPU-MMA):" "${file}"
      num_skiped=$(($num_skiped + 1));
      continue;
    fi
  fi

  # Skip whole file early when required targets cannot match current machine/simulator.
  if ! set_empty REQ_TARGETS; then
    _sim_base="${simulator#sim-}"
    if [[ $device_type == "none" && "$simulator" == "none" ]] || ! { set_contains REQ_TARGETS "$mach" || set_contains REQ_TARGETS "$simulator" || set_contains REQ_TARGETS "$_sim_base";  }; then
      _all_skipped_targets=$(set_print REQ_TARGETS)
      print_status_line "SKIP($(toupper "${_all_skipped_targets}")):" "${file}"
      num_skiped=$(($num_skiped + 1));
      continue;
    fi
  fi

  # --sim=only: only run files that require a simulator target
  if [[ $sim_mode == "only" ]]; then
    if ! set_contains_prefix REQ_TARGETS "sim-"; then
      _all_skipped_targets=$(set_print REQ_TARGETS)
      print_status_line "SKIP(sim=only):" "${file}"
      num_skiped=$(($num_skiped + 1));
      continue;
    fi
  fi

  # --sim=off: skip files that exclusively require a simulator target.
  # Tests that also match a non-sim arch (e.g. TARGET-ARCH300+,TARGET-SIMARCH)
  # are allowed through since they can run without a simulator.
  if [[ $sim_mode == "off" ]]; then
    if set_contains_prefix REQ_TARGETS "sim-"; then
      _sim_base="${simulator#sim-}"
      if ! set_contains REQ_TARGETS "$mach" && ! set_contains REQ_TARGETS "$_sim_base"; then
        _all_skipped_targets=$(set_print REQ_TARGETS)
        print_status_line "SKIP(sim=off):" "${file}"
        num_skiped=$(($num_skiped + 1));
        continue;
      fi
    fi
  fi

  # Skip when the test requires compiler features not supported by the target.
  if [ ${#req_features[@]} -ne 0 ] && [ -n "$choreo_target" ]; then
    _feat_key="${choreo_target}:${choreo_arch}"
    if [[ -z "${FEATURE_CACHE["$_feat_key"]+x}" ]]; then
      _feat_args=("-t" "$choreo_target" "--print-features")
      [[ -n "$choreo_arch" ]] && _feat_args+=("-arch=$choreo_arch")
      FEATURE_CACHE["$_feat_key"]="$(choreo "${_feat_args[@]}" 2>/dev/null || true)"
    fi
    _avail="${FEATURE_CACHE["$_feat_key"]}"
    _feat_missing=""
    for _rf in "${req_features[@]}"; do
      if ! echo "$_avail" | grep -qxF "$_rf"; then
        _feat_missing="$_rf"
        break
      fi
    done
    if [[ -n "$_feat_missing" ]]; then
      echo "SKIP($_feat_missing): ${file}"
      num_skiped=$(($num_skiped + 1));
      continue;
    fi
  fi

  run_num=$(grep -E 'RUN(:|-.*:)' $file | wc -l)
  run_count=0

  ext="${file##*.}"
  # Read the file and search for lines starting with "// RUN:"
  while IFS= read -r line; do
    # check if it is valid line
    if [[ ${ext} == "co" || ${ext} == "mlir" ]]; then
      if [[ "${line}" =~ ^//[[:blank:]]*RUN(.+) ]]; then
        line=${line#//}
      else
       continue;
      fi
    elif [[ ${ext} == "cmt" ]]; then
      if [[ "${line}" =~ ^\#[[:blank:]]*RUN(.+) ]]; then
        line=${line#\#}
      else
       continue;
      fi
    fi

    retrieve_run_config $line

    [[ -z "$run_command" ]] && continue;

    # Either execute or skip
    run_count=$(($run_count + 1))

    # Check if the execution environment matches
    if [[ ("$run_environ" == "docker" && "$is_in_docker" == false) ||
          ("$run_environ" == "shell" && "$is_in_shell" == false) ]]; then
      # Skip when the environment does not match
      print_status_line "SKIP(${run_environ}):" "${file} ($run_count of $run_num)"
      num_skiped=$(($num_skiped + 1)); #simply skip the unmatched target
      continue;
    fi

    # There is a RUN-TARGET
    if [[ ! -z "$run_target" ]]; then
      # check if run-target violates the REQUIRES
      if ! target_specs_are_required "$run_target"; then
        echo "ERROR($file): run target ($run_target) is not listed as a test targets ($run_target)."
        exit 1
      fi
    fi

    # requires specific device to run
    if ! set_empty REQ_TARGETS; then
      #echo "device: $device_type, reqs: $(set_print REQ_TARGETS), mach: $mach"
      if [[ $device_type == "none" && "$simulator" == "none" ]] || ! { set_contains REQ_TARGETS "$mach" || set_contains REQ_TARGETS "$simulator" || set_contains REQ_TARGETS "$_sim_base";  }; then
        # Not matched, skip
        _all_skipped_targets=$(set_print REQ_TARGETS)
        print_status_line "SKIP($(toupper "${_all_skipped_targets}")):" "${file} ($run_count of $run_num)"
        num_skiped=$(($num_skiped + 1)); #simply skip the unmatched target
        continue;
      fi
    fi

    # --run-only: skip RUN lines that don't invoke a specified binary
    if [[ -n "$run_only" ]]; then
      # Split comma-separated list; keep if any binary matches as a
      # standalone word (or %placeholder) in the command.
      local _ro_list=() _ro_bin _ro_match
      IFS=',' read -ra _ro_list <<< "$run_only"
      _ro_match=0
      for _ro_bin in "${_ro_list[@]}"; do
        if [[ "$run_command" =~ (^|[[:space:]])(%?${_ro_bin})($|[[:space:]]) ]]; then
          _ro_match=1; break
        fi
      done
      if [[ $_ro_match -eq 0 ]]; then
        print_status_line "SKIP(run-only=${run_only}):" "${file} ($run_count of $run_num)"
        num_skiped=$(($num_skiped + 1));
        continue;
      fi
    fi

    # --target-only: skip RUN lines that don't target a specified -t target
    if [[ -n "$target_only" ]]; then
      # Split comma-separated list; keep if any target matches.
      local _to_list=() _to_tgt _to_match
      IFS=',' read -ra _to_list <<< "$target_only"
      _to_match=0
      for _to_tgt in "${_to_list[@]}"; do
        if [[ "$run_command" =~ [[:space:]]-t[[:space:]]+${_to_tgt}($|[[:space:]]) ]]; then
          _to_match=1; break
        fi
      done
      if [[ $_to_match -eq 0 ]]; then
        print_status_line "SKIP(target-only=${target_only}):" "${file} ($run_count of $run_num)"
        num_skiped=$(($num_skiped + 1));
        continue;
      fi
    fi

    # Arch Matches: Execute the command with replacements
    # Run the command in the background
    if [[ $max_jobs -eq 1 ]] || [[ $sequential -eq 1 ]]; then
      # specialised serial test logic
      execute_command "$file" "$run_command" "$run_count" "$run_num" "${exe_env}" "${unset_env}" "${run_env}"
    else
      while [[ $(jobs | wc -l) -ge $max_jobs ]]; do
        wait -n
      done
      execute_command "$file" "$run_command" "$run_count" "$run_num" "${exe_env}" "${unset_env}" "${run_env}" &
    fi

  done < "$file"

  # Wait for all background processes to finish before moving to the next file
  wait
done

cleantmplocks

# Check for required commands
if [ -n "${save_log}" ] && command -v date >/dev/null 2>&1 && command -v tee >/dev/null 2>&1; then
  # Prepare output directory and file
  LOG_DIR="/tmp/choreo_log_$(whoami)"
  mkdir -p ${LOG_DIR}

  TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
  LOG_FILE=${LOG_DIR}/log_${TIMESTAMP}.txt

  # Run showresult, tee output to log file
  showresult | tee ${LOG_FILE}
  RET_CODE=${PIPESTATUS[0]}  # Get exit code of showresult
  echo "Find the test result: ${LOG_FILE}"
  exit "${RET_CODE}"
else
  # Fallback: run showresult only
  showresult
  exit $?
fi
