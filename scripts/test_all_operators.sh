#!/bin/bash

# Test all .co operator files in samples directory
# Exit on first failure

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the project root directory (where this script is located)
PROJECT_ROOT="$SCRIPT_DIR"

# Check if we're in a subdirectory and need to go up
if [ ! -d "$PROJECT_ROOT/samples/topscc" ]; then
    # Try to find samples directory by going up
    while [ "$PROJECT_ROOT" != "/" ] && [ ! -d "$PROJECT_ROOT/samples/topscc" ]; do
        PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
    done
    
    if [ ! -d "$PROJECT_ROOT/samples/topscc" ]; then
        echo "Error: Could not find samples directory from $(pwd)"
        echo "Please run this script from the project root directory or a subdirectory"
        exit 1
    fi
fi

echo "Project root: $PROJECT_ROOT"
echo "Testing all .co operators in samples directory..."
echo "================================================"

# Change to project root directory
cd "$PROJECT_ROOT"

# Find all .co files in samples directory
co_files=$(find samples/topscc -name "*.co" -type f | sort)

if [ -z "$co_files" ]; then
    echo "No .co files found in samples directory"
    exit 1
fi

echo "Found $(echo "$co_files" | wc -l) .co files to test:"
echo "$co_files"
echo ""

# Counter for statistics
total_tests=0
passed_tests=0
failed_tests=0

# Test each .co file
for file in $co_files; do
    echo ""
    echo "Testing: $file"
    echo "----------------------------------------"
    
    # Get the operator name from filename
    operator_name=$(basename "$file" .co)
    
    # Clean up any existing a.out
    rm -f a.out
    
    # Compile the test
    echo "Compiling $operator_name..."
    if choreo "$file" 2>&1; then
        echo "Compilation successful"
    else
        echo "❌ $operator_name: COMPILATION FAILED"
        ((failed_tests++))
        echo "Stopping on first failure..."
        exit 1
    fi
    
    # Check if a.out was created
    if [ ! -f "a.out" ]; then
        echo "❌ $operator_name: NO EXECUTABLE CREATED"
        ((failed_tests++))
        echo "Stopping on first failure..."
        exit 1
    fi
    
    # Run the test
    echo "Running $operator_name..."
    if ./a.out 2>&1; then
        echo "✅ $operator_name: PASSED"
        ((passed_tests++))
    else
        echo "❌ $operator_name: EXECUTION FAILED"
        ((failed_tests++))
        echo "Stopping on first failure..."
        exit 1
    fi
    
    ((total_tests++))
    
    # Clean up
    rm -f a.out
    
    echo "Completed test $total_tests of $(echo "$co_files" | wc -l)"
done

echo ""
echo "================================================"
echo "Test Summary:"
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"
echo ""
echo "🎉 All operators passed!" 
