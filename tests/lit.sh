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

find_tests_root() {
  # usage: find_tests_root <test_path>
  # returns absolute path to nearest ancestor directory named "tests"
  d="$(abspath_dir_of "$1")"

  while :; do
    base="$(basename "$d")"
    if [ "$base" = "tests" ] || [ "$base" = "benchmark" ]; then
      printf "%s\n" "$d"
      return 0
    fi
    parent="$(dirname "$d")"
    [ "$parent" = "$d" ] && break
    d="$parent"
  done

  echo "lit.sh: could not find ancestor directory named 'tests' for: $1" >&2
  return 2
}

load_cfg_chain_from_tests() {
  # usage: load_cfg_chain_from_tests <test_path> [cfg_name]
  test_path="$1"
  cfg_name="${2:-lit.cfg}"

  tests_root="$(find_tests_root "$test_path")"
  test_dir="$(abspath_dir_of "$test_path")"

  # Build the relative path from tests_root -> test_dir using string stripping
  # Assumes test_dir is under tests_root.
  case "$test_dir" in
    "$tests_root") rel="" ;;
    "$tests_root"/*) rel="${test_dir#"$tests_root"/}" ;;
    *)
      echo "lit.sh: $test_dir is not under $tests_root" >&2
      return 2
      ;;
  esac

  # Source tests_root/lit.cfg first (if present)
  d="$tests_root"
  cfg="$d/$cfg_name"
  if [ -f "$cfg" ]; then
    cfg_abs="$(cd "$(dirname "$cfg")" && pwd)/$(basename "$cfg")"
    if ! already_sourced_cfg "$cfg_abs"; then
      mark_sourced_cfg "$cfg_abs"
       #echo "==> sourcing cfg: $cfg_abs" >&2
      . "$cfg_abs"
    fi
  fi

  # Then walk rel segments: tests_root/seg1, tests_root/seg1/seg2, ...
  # and source cfg at each level if present.
  if [ -n "$rel" ]; then
    oldIFS="$IFS"
    IFS="/"
    set -- $rel
    IFS="$oldIFS"

    for seg in "$@"; do
      d="$d/$seg"
      cfg="$d/$cfg_name"
      if [ -f "$cfg" ]; then
        cfg_abs="$(cd "$(dirname "$cfg")" && pwd)/$(basename "$cfg")"
        if ! already_sourced_cfg "$cfg_abs"; then
          mark_sourced_cfg "$cfg_abs"
           #echo "==> sourcing cfg: $cfg_abs" >&2
          . "$cfg_abs"
        fi
      fi
    done
  fi
}

#=========================================================

# Get the directory where the script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
timestamp=$(date +%Y%m%d%H%M%S)

# Add the script's parent directory to PATH
export PATH="$script_dir:${script_dir}/../:${script_dir}/../extern/bin/:${script_dir}/../extern/:$PATH"

# Check if FileCheck exists in the PATH
FILECHECK=$(which FileCheck \
            FileCheck-18 FileCheck-17 FileCheck-16 \
            FileCheck-15 FileCheck-14 \
            FileCheck-10)

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

if ! which choreo &>/dev/null; then
  echo "Error: choreo is not found in PATH."
  exit 1
fi

if ! which copp &>/dev/null; then
  echo "Error: copp is not found in PATH."
  exit 1
fi

if ! which not.sh &>/dev/null; then
  echo "Error: not.sh is not found in PATH."
  exit 1
fi

if ! which bc &>/dev/null; then
  echo "Error: bc is not found in PATH."
  exit 1
fi

echo "---------------------------------------"
echo "        Choreo SimpleLit - v0.30"
echo "---------------------------------------"
echo ""

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
expect_fail=
expect_skip=

max_jobs=1

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
cuda_arch="none"
mach=
simulator="none"

# some specific features
is_dynshape_supported=0

# Function to fill the target-specific variables
prepare() {
  local file="$1"

  # Validate input
  if [[ -z "$file" ]] || [[ ! -f "$file" ]]; then
    echo "Error: Invalid file specified" >&2
    return 1
  fi

  local ext="${file##*.}"
  local comment_pattern

  # Set comment pattern based on file extension
  if [[ "${ext}" == "co" ]]; then
    comment_pattern="^//"
  elif [[ "${ext}" == "cmt" ]]; then
    comment_pattern="^#"
  else
    echo "Error: Invalid test file: $file" >&2
    return 1
  fi

  # now load the configure
  CFG_SOURCED=""
  HOOKS=""
  load_cfg_chain_from_tests "$file"

  # detect the hardware
  run_hooks "hw_detect";

  # target-specific preparation
  run_hooks "target_prepare"

  # Reset target requirement
  requires_dynamic_shape=0
  need_cute=0
  need_cuda=0
  expect_fail=
  expect_skip=

  # Extract expect_fail and expect_skip with proper comment pattern
  if [[ -n "$comment_pattern" ]]; then
    expect_fail=$(grep "$comment_pattern" "$file" | grep "XFAIL:" | sed 's/.*XFAIL:[[:blank:]]*//')
    expect_skip=$(grep "$comment_pattern" "$file" | grep "SKIP:")
  else
    # Fallback: search entire file if no specific pattern
    expect_fail=$(grep "XFAIL:" "$file" | sed 's/.*XFAIL:[[:blank:]]*//')
    expect_skip=$(grep "SKIP:" "$file")
  fi

  # Extract REQUIRES line from file contents
  local requires=$(grep "REQUIRES:" "$file")

  set_clear REQ_TARGETS

  # If no REQUIRES line found, return early
  if [[ -z "$requires" ]]; then
    return 0
  fi

  # Early returns for comment-style files that only contain comment markers
  if [[ "${ext}" == "co" ]] && [[ "${requires}" != "//"* ]]; then
    return
  fi
  if [[ "${ext}" == "cmt" ]] && [[ "${requires}" != "#"* ]]; then
    return
  fi

  # Extract the actual requirements part
  requires=$(echo "${requires}" | sed 's/.*REQUIRES://')

  # Extract components
  local tgts=$(grep -o "TARGET-[^[:space:]]*" <<< "$requires" | sed 's/TARGET-//')
  local libs=$(grep -o "LIBRARY-[^[:space:]]*" <<< "$requires" | sed 's/LIBRARY-//')
  local cmps=$(grep -o "COMPILER-[^[:space:]]*" <<< "$requires" | sed 's/COMPILER-//')

  # Process targets
  if [ ! -z "${tgts}" ]; then
    run_hooks "set_archs" ${tgts}
  fi

  if set_empty REQ_TARGETS && [ ! -z "${tgts}" ]; then
    echo "invalid target: ${tgts}" >&2
  fi

  # has library requirement
  if [[ "${libs}" == *"CUTE"* ]]; then
    need_cute=1
    need_cuda=1
  fi

  # requires dynamic-shape support (some target only)
  local dynshape=$(grep -q "DYNAMIC-SHAPE" <<< "$requires" && echo "found")
  [ ! -z "${dynshape}" ] && requires_dynamic_shape=1
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

  # Replace %s with the filename
  command=${command//%s/"$file"}

  # Replace 'choreo', 'copp' and 'FileCheck' with their absolute paths
  # Note: It must uses '-n' to remove comments inside host code.
  #       Or else FileCheck will check the line of "// CHECK:"
  command=${command//choreo/"$(which choreo) -n"}
  command=${command//copp/"$(which copp)"}
  command=${command//FileCheck/"${FILECHECK}"}
  command=${command//%cuda_arch/"-arch ${cuda_arch}"}
  local not_command=$(which not.sh | sed 's/[&/\]/\\&/g')
  command=$(echo "$command" | sed "s/\bnot \(.*\)/${not_command} \1/")
  run_hooks "target_cmd" "command"

  # num_tested=$(($num_tested + 1))
  # echo "num_tested before add " $(read_counter "num_tested")
  # echo "num_tested after add " $(read_counter "num_tested")

  # start timing
  local start_time_ns=$(date +%s%N)

  # execute the command
  command="${env_set} ${run_env} $command"
  working_command="$command"

  eval "$command" 2>/dev/null
  local exit_code=$?

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
      elapsed_time="$(bc <<< "scale=3; $elapsed_ns / 1000") µs"
  fi

  local term_width=$(get_terminal_width)
  local max_text_width=$((term_width - 25))

  if [[ $exit_code -eq 0 ]]; then
    if [[ "$expect_fail" == "*"* ]]; then
      increment_counter num_uepass 1
      append_reproduce_command "$command"
      if [[ ${#test_info} -gt $max_text_width ]]; then
        printf "%*s %s\n" $((max_text_width)) "UNEXPECTED PASS: $file ($count of $total)" "| Time: $elapsed_time"
      else
        printf "%-*s %s\n" "$max_text_width" "UNEXPECTED PASS: $file ($count of $total)" "| Time: $elapsed_time"
      fi
    elif [[ ! -z "${expect_fail}" ]] &&
         [[ "$(toupper ${expect_fail})" ==  *"$(toupper ${mach})"* ]]; then
      increment_counter num_uepass 1
      append_reproduce_command "$command"
      if [[ ${#test_info} -gt $max_text_width ]]; then
        printf "%*s %s\n" $((max_text_width)) "UNEXPECTED PASS: $file ($count of $total)" "| Time: $elapsed_time"
      else
        printf "%-*s %s\n" "$max_text_width" "UNEXPECTED PASS: $file ($count of $total)" "| Time: $elapsed_time"
      fi
    else
      increment_counter num_passed 1
      if [[ ${#test_info} -gt $max_text_width ]]; then
        printf "%*s %s\n" $((max_text_width)) "PASS: $file ($count of $total)" "| Time: $elapsed_time"
      else
        printf "%-*s %s\n" "$max_text_width" "PASS: $file ($count of $total)" "| Time: $elapsed_time"
      fi
    fi
  else
    if [[ "${expect_fail}" == "*"* ]]; then
      increment_counter num_xfails 1
      if [[ ${#test_info} -gt $max_text_width ]]; then
        printf "%*s %s\n" $((max_text_width)) "XFAIL: $file ($count of $total)" "| Time: $elapsed_time"
      else
        printf "%-*s %s\n" "$max_text_width" "XFAIL: $file ($count of $total)" "| Time: $elapsed_time"
      fi
    elif [[ ! -z "${expect_fail}" ]] &&
         [[ "$(toupper ${expect_fail})" ==  *"$(toupper ${mach})"* ]]; then
      increment_counter num_xfails 1
      if [[ ${#test_info} -gt $max_text_width ]]; then
        printf "%*s %s\n" $((max_text_width)) "XFAIL: $file ($count of $total)" "| Time: $elapsed_time"
      else
        printf "%-*s %s\n" "$max_text_width" "XFAIL: $file ($count of $total)" "| Time: $elapsed_time"
      fi
    else
      increment_counter num_failed 1
      append_reproduce_command "$command"
      if [[ ${#test_info} -gt $max_text_width ]]; then
        printf "%*s %s\n" $((max_text_width)) "FAIL: $file ($count of $total)" "| Time: $elapsed_time"
      else
        printf "%-*s %s\n" "$max_text_width" "FAIL: $file ($count of $total)" "| Time: $elapsed_time"
      fi
    fi
  fi
}

# ---------------------------------------"
#         Handle arguments
# ---------------------------------------"
if [ $# -lt 1 ]; then
    echo "Usage: $0 [-jN] <file_or_directory>"
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
    -*)
      # Handle invalid option
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      # Handle the first positional argument (file or directory)
      if [ -d "$1" ]; then
          # If it's a directory, find all .co, .cmt(cmake test) files
          files_array=($(find "$1" -type f -name '*.co' -o -name '*.cmt'))
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
  echo "------ Lit Test summary ------"
  echo "Tested:  $(read_counter 'num_tested')"
  echo "Passed:  $(read_counter 'num_passed')"

  [ $num_skiped -ne 0 ] && echo Skipped: $num_skiped
  [ $(read_counter 'num_failed') -ne 0 ] && echo "Failed:  $(read_counter 'num_failed')"
  [ $(read_counter 'num_xfails') -ne 0 ] && echo "Expected Failures: $(read_counter 'num_xfails')"
  [ $(read_counter 'num_uepass') -ne 0 ] && echo "Unexpected Passes: $(read_counter 'num_uepass')"

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
    if [[ "${run_target}" == "$mach" ]]; then
      run_command="${BASH_REMATCH[2]}"
      run_environ="shell"
    elif [[ "${run_target}" == "docker" ]]; then
      run_command="${BASH_REMATCH[2]}"
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

for file in "${files_array[@]}"; do
  # check requirement specified by the file
  prepare $file

  if [ ! -z "$expect_skip" ]; then
    echo "SKIP:  $file"
    num_skiped=$(($num_skiped + 1));
    continue;
  fi

  if [ ${need_cuda} -eq 1 ]; then
    if ! validate_cuda_home; then
      echo "SKIP(CUDA):  $file"
      num_skiped=$(($num_skiped + 1));
      continue;
    elif [ "$device_type" != "gpu" ]; then
      echo "SKIP(GPU):  $file"
      num_skiped=$(($num_skiped + 1));
      continue;
    else
      run_env="CUDA_HOME=${CUDA_HOME}"
    fi
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
      echo "SKIP(GPU-MMA): ${file} "
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
    if [[ ${ext} == "co" ]]; then
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
      echo "SKIP(${run_environ}): ${file} ($run_count of $run_num)"
      num_skiped=$(($num_skiped + 1)); #simply skip the unmatched target
      continue;
    fi

    # There is a RUN-TARGET
    if [[ ! -z "$run_target" ]]; then
      # check if run-target violates the REQUIRES
      if ! set_contains REQ_TARGETS "$run_target"; then
        echo "ERROR($file): run target ($run_target) is not listed as a test targets ($run_target)."
        exit 1
      fi
    fi

    # requires specific device to run
    if ! set_empty REQ_TARGETS; then
      #echo "device: $device_type, reqs: $(set_print REQ_TARGETS), mach: $mach"
      if [[ $device_type == "none"  ]] || ! { set_contains REQ_TARGETS "$mach" || set_contains REQ_TARGETS "$simulator";  }; then
        # Not matched, skip
        _all_skipped_targets=$(set_print REQ_TARGETS)
        echo "SKIP($(toupper "${_all_skipped_targets}")): ${file} ($run_count of $run_num)"
        num_skiped=$(($num_skiped + 1)); #simply skip the unmatched target
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
