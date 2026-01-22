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

#=========================================================

# Get the directory where the script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
timestamp=$(date +%Y%m%d%H%M%S)

# Add the script's parent directory to PATH
export PATH="$script_dir:${script_dir}/../:${script_dir}/../extern/bin:$PATH"

# Check if FileCheck exists in the PATH
if ! which FileCheck &>/dev/null; then
  echo "Error: FileCheck tool not found in PATH."
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

echo "---------------------------------------"
echo "        Choreo SimpleLit - v0.22"
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
tst_targets=()
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

# Function to fill the target-specific variables
check_specific() {
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

  # Reset target requirement
  requires_dynamic_shape=0
  need_cute=0
  need_cuda=0
  expect_fail=
  expect_skip=

  set_clear tst_targets
  set_add tst_targets "gcu210" "gcu300" "gcu400" "sm_86" "sm_90a"

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
  local expect_gcu_sim4=$(grep -q "GCUSIM4" <<< "$requires" && echo "found")
  local expect_gcu_sim5=$(grep -q "GCUSIM5" <<< "$requires" && echo "found")

  # has some targets specified, resolve it
  [ ! -z ${tgts} ] && set_clear tst_targets

  # Process targets
  if [ ! -z "${expect_gcu_sim4}" ]; then set_add tst_targets "gcusim400"; fi
  if [ ! -z "${expect_gcu_sim5}" ]; then set_add tst_targets "gcusim500"; fi
  for tgt in ${tgts}; do
    if [[ "${tgt}" == "GCU400" ]]; then set_add tst_targets "gcu400";
    elif [[ "${tgt}" == "GCU300" ]]; then set_add tst_targets "gcu300";
    elif [[ "${tgt}" == "GCU210" ]]; then set_add tst_targets "gcu210";
    elif [[ "${tgt}" == "SM_90" ]]; then set_add tst_targets "sm_90" "sm_90a";
    elif [[ "${tgt}" == "SM_90A" ]]; then set_add tst_targets "sm_90a";
    elif [[ "${tgt}" == "SM_"* ]]; then set_add tst_targets "$(tolower ${tgt})";
    elif [[ "${tgt}" == "GCUALL" ]]; then
      set_add tst_targets "gcu210" "gcu300" "gcu400"
    elif [[ "${tgt}" == "GPU" ]]; then
      set_add tst_targets "sm_86" "sm_90a"
    fi
  done

  if [ -z "${tgts}" ]; then
    set_add tst_targets "gcu210" "gcu300" "gcu400" "sm_86" "sm_90a"
  fi

  if set_empty tst_targets; then
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

# Check the hardware device availability and type
# Note: consider the machine only installed a single target
device_type="none"
gcu_arch="none"
cuda_arch="none"
mach=
simulator="none"

# some specific features
is_dynshape_supported=0
gcu_sim_lib=
gcu_sim_arch=

gpu_detect() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    # GPU device is available
    while IFS=',' read -r name cap; do
      name=$(echo "$name" | xargs)
      cap=$(echo "$cap" | xargs)

      # Defensive parsing
      if [ -z "$cap" ]; then
        cuda_arch="none"
        continue;
      fi

      major=${cap%%.*}
      minor=${cap##*.}

      # Ignore pre-sm_70
      if [ "$major" -lt 7 ]; then
        cuda_arch="none"
        continue;
      fi

      # Hopper special case
      if [ "$major" -eq 9 ] && [ "$minor" -eq 0 ]; then
        if echo "$name" | grep -qi "\(GH200\|H800\|H20\)"; then
          cuda_arch="sm_90a" # enforce sm_90a now
        else
          cuda_arch="sm_90"
        fi
      else
        cuda_arch="sm_${major}${minor}"
      fi
      break;
    done < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)

    # or else does not find a valid gpu device
    if [ "$cuda_arch" != "none" ]; then
      device_type="gpu"
      mach=${cuda_arch}
    fi
  fi
}

gcu_detect() {
  local _gcu_dstr="$(lspci | grep -E '(Enflame|Tencent)' | head -1)"
  case "${_gcu_dstr}" in
    *S60G*)
      gcu_arch=gcu300
      ;;
    *c035*|*S60*)
      gcu_arch=gcu300
      is_dynshape_supported=1
      ;;
    *I20*)
      gcu_arch=gcu210
      export TOPS_VISIBLE_DEVICES=1
      ;;
    *Tencent*)
      gcu_arch=gcu210
      ;;
    "")
      echo "can not determine the target device type."
      exit 1
      ;;
  esac
  if [ "$gcu_arch" != "none"  ]; then
    device_type="gcu"
    mach=${gcu_arch}
  fi
}

hardware_detect() {
  if [[ "${device_type}" == "none" ]]; then gpu_detect; fi
  if [[ "${device_type}" == "none" ]]; then gcu_detect; fi

  if [[ "{$device_type}" == "none" ]]; then
    echo "can not determine device type."
    exit 1
  fi
}

detect_simulator_features() {
  if [ -f "${script_dir}/../extern/lib/libgcusim.so" ]; then
    # the simulators exist
    gcu_sim_lib=${script_dir}/../extern/lib/
    gcu_sim_arch=gcusim400
    simulator=${gcu_sim_arch}
  elif [ -f "${script_dir}/../extern/lib/libgcusim5.so" ]; then
    gcu_sim_lib=${script_dir}/../extern/lib/
    gcu_sim_arch=gcusim500
    simulator=${gcu_sim_arch}
  fi
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
  command=${command//FileCheck/"$(which FileCheck)"}
  command=${command//%gcu_arch/"-arch=${gcu_arch}"}
  command=${command//%cuda_arch/"-arch ${cuda_arch}"}
  local not_command=$(which not.sh | sed 's/[&/\]/\\&/g')
  command=$(echo "$command" | sed "s/\bnot \(.*\)/${not_command} \1/")

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

# detect the device supported features
hardware_detect
detect_simulator_features
initialize_counters

case $device_type in
  gpu) ;;
  gcu) ;;
  *)
    echo "No supported device was found. abort..."
    exit 1
    ;;
esac

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

trap on_ctrl_c SIGINT

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

for file in "${files_array[@]}"; do
  # check requirement specified by the file
  check_specific $file

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

  if [ "$device_type" == "gcu" ]; then
    if [ $requires_dynamic_shape -eq 1 ]; then
      if [ $is_dynshape_supported -eq 0 ]; then
        echo "SKIP(dyn-shape): ${file} "
        num_skiped=$(($num_skiped + 1));
        continue; #simply skip the unmatched target
      fi
    fi
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

  ext="${file##*.}"
  # Read the file and search for lines starting with "// RUN:"
  run_num=$(grep -E 'RUN(:|-.*:)' $file | wc -l)
  run_count=0
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

    # There is a specified RUN-TARGET
    if [[ ! -z "$run_target" ]]; then # no RUN-TARGET specified
      # check if run-target violates the REQUIRES
      if ! set_contains tst_targets "$run_target"; then
        echo "ERROR($file): run target ($run_target) is not listed as a test targets ($run_target)."
        exit 1
      fi
    fi

    # specific - simulator
    exe_env=
    unset_env=

    # requires simulator
    if set_contains tst_targets "gcusim400" || set_contains tst_targets "gcusim500"; then
      if ! set_contains tst_targets "$simulator"; then
      echo "SKIP(SIM): ${file} ($run_count of $run_num)"
        num_skiped=$(($num_skiped + 1));
        continue;
      else
        exe_env="old_path=${LD_LIBRARY_PATH}; export LD_LIBRARY_PATH=${gcu_sim_lib}:${LD_LIBRARY_PATH};"
        unset_env="export LD_LIBRARY_PATH=${old_path}; unset INTERNAL_GCU_SIM;"
        if [[ "$simulator" == "gcusim400" ]]; then
          exe_env="${exe_env} export INTERNAL_GCU_SIM=LIBRA;"
        elif [[ "$simulator" == "gcusim500" ]]; then
          exe_env="${exe_env} export INTERNAL_GCU_SIM=DRACO;"
        fi
      fi
    fi

    if ! set_contains tst_targets "$mach" && ! set_contains tst_targets "$simulator"; then
      # Not matched, skip
      _all_skipped_targets=$(set_print tst_targets)
      echo "SKIP($(toupper "${_all_skipped_targets}")): ${file} ($run_count of $run_num)"
      num_skiped=$(($num_skiped + 1)); #simply skip the unmatched target
      continue;
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
