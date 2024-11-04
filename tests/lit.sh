#!/usr/bin/env bash

# Get the directory where the script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

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

if ! which not.sh &>/dev/null; then
    echo "Error: choreo is not found in PATH."
    exit 1
fi


echo "---------------------------------------"
echo "        Choreo SimpleLit - v0.1"
echo "---------------------------------------"
echo ""

failed_commands=()
num_passed=0
num_tested=0
num_xfails=0
num_skiped=0

test_target=
requires_dynamic_shape=0
expect_fail=

# Function to fill the target-specific variables
check_requirement() {
  local file=$1
  local requires=$(grep "^\/\/" $file | grep "REQUIRES:" | sed 's/.*REQUIRES://')
  local tgt=$(echo $requires | grep "TARGET-.*\>" |sed 's/TARGET-//g' |sed 's/ .*//')
  # reset target requirement
  requires_dynamic_shape=0
  test_target=
  expect_fail=
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

  expect_fail=$(grep "^\/\/" $file |grep "XFAIL:" | sed 's/.*XFAIL://')
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
  elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]]; then
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

# Function to replace placeholders and execute command
execute_command() {
    local file=$1
    local command=$2
    local count=$3
    local total=$4

    # Replace %s with the filename
    command=${command//%s/"$file"}

    # Replace 'gcc' and 'FileCheck' with their absolute paths
    command=${command//choreo/"$(which choreo)"}
    command=${command//FileCheck/"$(which FileCheck)"}
    local not_command=$(which not.sh | sed 's/[&/\]/\\&/g')
    command=$(echo "$command" | sed "s/\bnot \(.*\)/${not_command} \1/")

    num_tested=$(($num_tested + 1))

    # Execute the command
    eval "$command" 2>/dev/null

    if [[ $? -eq 0 ]]; then
      num_passed=$(($num_passed + 1));
      echo "PASS: $file ($count of $total)"
    else
      failed_commands+=("$command");
      echo "FAIL: $file ($count of $total)"
    fi
}

# Check if the argument is a valid file or directory
if [ -d "$1" ]; then
    # Directory: Fill the array with .co files from the directory
    files_array=($(find "$1" -type f -name '*.co'))
#    files_array=($(find "$1" -type f -name '*.co' | grep -v 'only'))
#    if [ $is_gpu_available -eq 1 ]; then
#        files_array+=($(find "$1" -type f -name '*.co' | grep 'gpu-only'))
#    fi
#    if [ $is_gcu_available -eq 1 ]; then
#        files_array+=($(find "$1" -type f -name '*.co' | grep 'gcu-only'))
#    fi
    # verbose all tests files
    # for file in "${files_array[@]}"; do
    #     echo "$file"
    # done
elif [ -f "$1" ]; then
    # File: Fill the array with the single file
    files_array=("$1")
else
    echo "Provided argument is not a valid file or directory."
    exit 1
fi

# check device supported feature
check_device_features

if [ $is_gcu_available -eq 0 ] && [ $is_gpu_available -eq 0 ]; then
  echo "No supported device was found. abort..."
  exit 0
fi

showresult() {
  echo ""
  echo "------ Lit Test summary ------"
  echo "Tested:  $num_tested"
  echo "Passed:  $num_passed"

  if [[ $num_passed -ne $num_tested ]]; then
    echo "Failed:  $(($num_tested - $num_passed))"
    echo ""
  fi

  [ ${num_xfails} -ne 0 ] && echo "XFails: $num_xfails"
  [ ${num_skiped} -ne 0 ] && echo "Skipped: $num_skiped"

  if [[ $num_passed -ne $num_tested ]]; then
    echo "Commands to reproduce failures:"
    for com in "${failed_commands[@]}"; do
      echo $com;
    done
  fi
}

on_ctrl_c() {
  showresult
  exit 1
}

trap on_ctrl_c SIGINT

# Iterate over the array
for file in "${files_array[@]}"; do
    # check requirement specified by the file
    check_requirement $file

    if [ "$expect_fail" == "*\**" ]; then
      echo "XFAIL: $file"
      num_xfails=$(($num_xfails + 1));
      continue
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

      if [ ! -z "$expect_fail" ]; then
        if [[ "${expect_fail}" == *"${gcu_arch}"* ]]; then
          echo "XFAIL: $file"
          num_xfails=$(($num_xfails + 1));
          continue;
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

    # Read the file and search for lines starting with "// RUN:"
    run_num=$(grep "RUN:" $file | wc -l)
    run_count=0
    while IFS= read -r line; do
        if [[ $line =~ ^//[[:blank:]]*RUN:[[:blank:]]*(.+) ]]; then
            run_count=$(($run_count + 1))
            # Extract the command after "RUN:"
            run_command="${BASH_REMATCH[1]}"
            # Execute the command with replacements
            execute_command "$file" "$run_command" "$run_count" "$run_num"
        fi
    done < "$file"
done

showresult
