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

# check device availability
is_gpu_available=0
if command -v nvidia-smi &> /dev/null; then
  if nvidia-smi > /dev/null 2>&1; then
    echo "GPU is available."
    is_gpu_available=1
  fi
fi

is_gcu_available=0
GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
GCU_DEVICE_STR_BACKUP="$(lspci | grep Tencent)"
if [ "${GCU_DEVICE_STR}" != "" ] || [ "${GCU_DEVICE_STR_BACKUP}" != "" ]; then
  echo "GCU is available."
  is_gcu_available=1
fi


echo "---------------------------------------"
echo "        Choreo SimpleLit - v0.1"
echo "---------------------------------------"
echo ""

failed_commands=()
num_passed=0
num_tested=0

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
    files_array=($(find "$1" -type f -name '*.co' | grep -v 'only'))
    if [ $is_gpu_available -eq 1 ]; then
        files_array+=($(find "$1" -type f -name '*.co' | grep 'gpu-only'))
    fi
    if [ $is_gcu_available -eq 1 ]; then
        files_array+=($(find "$1" -type f -name '*.co' | grep 'gcu-only'))
    fi
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

# Iterate over the array
for file in "${files_array[@]}"; do
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

echo ""
echo "------ Lit Test summary ------"
echo "Tested: $num_tested"
echo "Passed: $num_passed"

if [[ $num_passed -ne $num_tested ]]; then
  echo "Failed: $(($num_tested - $num_passed))"
  echo ""

  echo "Commands to reproduce failures:"
  for com in "${failed_commands[@]}"; do
    echo $com;
  done
fi
