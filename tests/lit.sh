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

echo "Choreo SimpleLit - v0.1"
echo ""

failed_commands=()
num_passed=0
num_tested=0

# Function to replace placeholders and execute command
execute_command() {
    local file=$1
    local command=$2

    echo "Testing: $file"
    # Replace %s with the filename
    command=${command//%s/"$file"}

    # Replace 'gcc' and 'FileCheck' with their absolute paths
    command=${command//choreo/"$(which choreo)"}
    command=${command//FileCheck/"$(which FileCheck)"}

    num_tested=$(($num_tested + 1))

    # Execute the command
    eval "$command"

    if [[ $? -eq 0 ]]; then num_passed=$(($num_passed + 1));
    else failed_commands+=("$command");
    fi
}

# Check if the argument is a valid file or directory
if [ -d "$1" ]; then
    # Directory: Fill the array with .co files from the directory
    files_array=($(find "$1" -type f -name '*.co'))
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
    while IFS= read -r line; do
        if [[ $line =~ ^//[[:blank:]]*RUN:[[:blank:]]*(.+) ]]; then
            # Extract the command after "RUN:"
            run_command="${BASH_REMATCH[1]}"
            # Execute the command with replacements
            execute_command "$file" "$run_command"
        fi
    done < "$file"
done

echo ""
echo "Tested: $num_tested"
echo "Passed: $num_passed"
if [[ $num_passed -ne $num_tested ]]; then
  echo "Failed: $(($num_tested - $num_passed))"
  for com in "${failed_commands[@]}"; do
    echo $com;
  done
fi
