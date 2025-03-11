#!/usr/bin/env bash

# Get the directory where the script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
timestamp=$(date +%Y%m%d%H%M%S)

# Add the script's parent directory to PATH
export PATH="$script_dir:${script_dir}/../:${script_dir}/../tools/bin:$PATH"

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
echo "        Choreo SimpleLit - v0.1"
echo "---------------------------------------"
echo ""

reproduce_commands=()
num_tested=0
num_passed=0
num_failed=0
num_uepass=0
num_xfails=0
num_skiped=0

is_in_docker=false
is_in_shell=false
test_target=
requires_dynamic_shape=0
expect_fail=
expect_skip=
expect_docker=
expect_shell=

max_jobs=1

if [ -f /.dockerenv ] || grep -qE "(docker|containerd)" /proc/1/cgroup; then
  is_in_docker=true
  is_in_shell=false
else
  is_in_docker=false
  is_in_shell=true
fi

# Function to fill the target-specific variables
check_requirement() {
  local file=$1
  local requires=$(grep "^\/\/" $file | grep "REQUIRES:" | sed 's/.*REQUIRES://')
  local tgt=$(echo $requires | grep "TARGET-.*\>" |sed 's/TARGET-//g' |sed 's/ .*//')
  # reset target requirement
  requires_dynamic_shape=0
  test_target=
  expect_fail=
  expect_skip=
  expect_docker=
  expect_shell=
  if [ "${tgt}" == "GCU400" ]; then
    [ ! -z "$test_target" ] && echo "Test target has been set to ${test_target}"
    test_target=gcu400
  elif [ "${tgt}" == "GCU300" ]; then
    [ ! -z "$test_target" ] && echo "Test target has been set to ${test_target}"
    test_target=gcu300
  elif [ "${tgt}" == "GCU210" ]; then
    [ ! -z "$test_target" ] && echo "Test target has been set to ${test_target}"
    test_target=gcu210
  elif [ "${tgt}" == "GCUALL" ]; then
    [ ! -z "$test_target" ] && echo "Test target has been set to ${test_target}"
    test_target=gcu-any
  elif [ "${tgt}" == "GPU" ]; then
    [ ! -z "$test_target" ] && echo "Test target has been set to ${test_target}"
    test_target=gpu
  elif [ ! -z "${tgt}" ]; then
    echo "unexpected target $tgt"
  fi

  local dynshape=$(echo $requires | grep "DYNAMIC-SHAPE\>")
  [ ! -z "${dynshape}" ] && requires_dynamic_shape=1;

  expect_fail=$(grep "^\/\/" $file |grep "XFAIL:" | sed 's/.*XFAIL:[[:blank:]]*//')
  expect_skip=$(grep "^\/\/" $file |grep "SKIP:")
  expect_docker=$(grep "^\/\/" $file |grep "DOCKER-ONLY")
  expect_shell=$(grep "^\/\/" $file |grep "SHELL-ONLY")
}

gcu_arch=
is_gpu_available=0
is_gcu_available=0
is_dynshape_supported=0
# check device availability
check_device_features() {
  if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
      echo "GPU is available."
      is_gpu_available=1
      return
    fi
  fi

  GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
  GCU_DEVICE_STR_BACKUP="$(lspci | grep Tencent)"
  if [ "${GCU_DEVICE_STR}" != "" ] || [ "${GCU_DEVICE_STR_BACKUP}" != "" ]; then
    #echo "GCU is available."
    is_gcu_available=1
  fi
  if [[ "${GCU_DEVICE_STR}" == *"S60G"* ]]; then
    gcu_arch=gcu400
  elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]] || [[ "${GCU_DEVICE_STR}" == *"S60"* ]]; then
    gcu_arch=gcu300
    is_dynshape_supported=1
  elif [[ "${GCU_DEVICE_STR}" == *"I20"* ]]; then
    gcu_arch=gcu210
    export TOPS_VISIBLE_DEVICES=1
  elif [[ "${GCU_DEVICE_STR_BACKUP}" != "" ]]; then
    gcu_arch=gcu210
  else
    echo "can not determine the GCU device type."
    exit 1
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

  # Replace %s with the filename
  command=${command//%s/"$file"}

  # Replace 'choreo', 'copp' and 'FileCheck' with their absolute paths
  # Note: It must uses '-n' to remove comments inside host code.
  #       Or else FileCheck will check the line of "// CHECK:"
  command=${command//choreo/"$(which choreo) -n"}
  command=${command//copp/"$(which copp)"}
  command=${command//FileCheck/"$(which FileCheck)"}
  local not_command=$(which not.sh | sed 's/[&/\]/\\&/g')
  command=$(echo "$command" | sed "s/\bnot \(.*\)/${not_command} \1/")

  # num_tested=$(($num_tested + 1))
  # echo "num_tested before add " $(read_counter "num_tested")
  # echo "num_tested after add " $(read_counter "num_tested")

  # start timing
  local start_time_ns=$(date +%s%N)

  # execute the command
  eval "$command" 2>/dev/null
  local exit_code=$?

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

  local term_width=$(tput cols)
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
         [[ "$(toupper ${expect_fail})" ==  *"$(toupper ${gcu_arch})"* ]]; then
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
         [[ "$(toupper ${expect_fail})" ==  *"$(toupper ${gcu_arch})"* ]]; then
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

      max_jobs="$num_jobs"
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
          # If it's a directory, find all .co files
          files_array=($(find "$1" -type f -name '*.co'))
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

# check device supported feature
check_device_features
initialize_counters

if [ $is_gcu_available -eq 0 ] && [ $is_gpu_available -eq 0 ]; then
  echo "No supported device was found. abort..."
  exit 0
fi

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

    return ${failed}
  fi
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
}

trap on_ctrl_c SIGINT

toupper() {
    echo "$1" | tr '[:lower:]' '[:upper:]'
}

for file in "${files_array[@]}"; do
  # check requirement specified by the file
  check_requirement $file

  if [ ! -z "$expect_skip" ]; then
    echo "SKIP:  $file"
    num_skiped=$(($num_skiped + 1));
    continue;
  fi

  if [ ! -z "$expect_docker" ] && [ ! -z "$is_in_docker" ]; then
    echo "SKIP-DOCKER-ONLY:  $file"
    num_skiped=$(($num_skiped + 1));
    continue;
  fi

  if [ ! -z "$expect_shell" ] && [ ! -z "$is_in_shell" ]; then
    echo "SKIP-SHELL-ONLY:  $file"
    num_skiped=$(($num_skiped + 1));
    continue;
  fi

  if [ $is_gcu_available -eq 1 ]; then
    if [ ! -z "$test_target" ] && [ "$test_target" != "gcu-any" ]; then
      if [ "$gcu_arch" != "$test_target" ]; then
        echo "SKIP($test_target): ${file}"
        num_skiped=$(($num_skiped + 1));
        continue; #simply skip the unmatched target
      fi
    fi

    if [ $requires_dynamic_shape -eq 1 ]; then
      if [ $is_dynshape_supported -eq 0 ]; then
        echo "SKIP(dyn-shape): ${file} "
        num_skiped=$(($num_skiped + 1));
        continue; #simply skip the unmatched target
      fi
    fi

  elif [ $is_gpu_available -eq 1 ]; then
    if [ ! -z "$test_target" ] ; then
      if [ "$test_target" != "gpu" ]; then
        echo "SKIP($test_target): ${file}"
        num_skiped=$(($num_skiped + 1));
        continue; #simply skip the unmatched target
      fi
    fi
  fi

  # Check if the file is in the "end2end" folder
  file_name=$(basename "$file")
  folder_name=$(dirname "$file")

  # If it's in the end2end folder, keep it blocking
  if [[ "$folder_name" == *"end2end"* ]]; then
    # Read the file and search for lines starting with "// RUN:"
    run_num=$(grep -E 'RUN(:|-.*:)' $file | wc -l)
    run_count=0
    while IFS= read -r line; do
      if [[ $line =~ ^//[[:blank:]]*RUN:[[:blank:]]*(.+) ]]; then
        run_count=$(($run_count + 1))
        # Extract the command after "RUN:"
        run_command="${BASH_REMATCH[1]}"
        # Execute the command with replacements
        execute_command "$file" "$run_command" "$run_count" "$run_num"
      elif [[ $line =~ ^//[[:blank:]]*RUN-(.+):[[:blank:]]*(.+) ]]; then
        run_count=$(($run_count + 1))
        run_target="${BASH_REMATCH[1]}"
        run_target=$(echo "$run_target" | tr '[:upper:]' '[:lower:]')
        if [[ "${run_target}" == "$gcu_arch" ]]; then
          # Extract the command after "RUN:"
          run_command="${BASH_REMATCH[2]}"
          # Execute the command with replacements
          execute_command "$file" "$run_command" "$run_count" "$run_num"
        else
          echo "SKIP($run_target): ${file} ($run_count of $run_num)"
          num_skiped=$(($num_skiped + 1)); #simply skip the unmatched target
        fi
      fi
    done < "$file"
  else
    # For all other tests, use parallel execution
    run_num=$(grep -E 'RUN(:|-.*:)' $file | wc -l)
    run_count=0
    while IFS= read -r line; do
      if [[ $line =~ ^//[[:blank:]]*RUN:[[:blank:]]*(.+) ]]; then
        run_count=$(($run_count + 1))
        run_command="${BASH_REMATCH[1]}"

        # Run the command in the background
        if [[ $max_jobs -eq 1 ]]; then
          # specialised serial test logic
          execute_command "$file" "$run_command" "$run_count" "$run_num"
        else
          while [[ $(jobs | wc -l) -ge $max_jobs ]]; do
            wait -n
          done
          execute_command "$file" "$run_command" "$run_count" "$run_num" &
        fi
      elif [[ $line =~ ^//[[:blank:]]*RUN-(.+):[[:blank:]]*(.+) ]]; then
        run_count=$(($run_count + 1))
        run_target="${BASH_REMATCH[1]}"
        run_target=$(echo "$run_target" | tr '[:upper:]' '[:lower:]')
        if [[ "${run_target}" == "$gcu_arch" ]]; then
          run_command="${BASH_REMATCH[2]}"
          # Run the command in the background
          if [[ $max_jobs -eq 1 ]]; then
            # specialised serial test logic
            execute_command "$file" "$run_command" "$run_count" "$run_num"
          else
            while [[ $(jobs | wc -l) -ge $max_jobs ]]; do
              wait -n
            done
            execute_command "$file" "$run_command" "$run_count" "$run_num" &
          fi
        else
          echo "SKIP($run_target): ${file} ($run_count of $run_num)"
          num_skiped=$(($num_skiped + 1)); #simply skip the unmatched target
        fi
      fi
    done < "$file"

    # Wait for all background processes to finish before moving to the next file
    wait
  fi
done

cleantmplocks
showresult
