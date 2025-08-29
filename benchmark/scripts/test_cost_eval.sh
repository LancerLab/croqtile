#!/bin/bash

# Test version of Cost Evaluation Script for Symbolic System
# Processes only a few files for demonstration

set -e

# Configuration
CHOREO_BIN="./choreo"
BENCHMARK_DIR="benchmark"
RESULTS_DIR="benchmark/scripts/results"
TEMP_DIR="benchmark/scripts/temp"

# Statistical Configuration
# SAMPLES: Number of independent benchmark runs per configuration (for statistical confidence)
# MEASURES: Number of timing measurements per sample (for averaging and standard deviation)
SAMPLES=3
MEASURES=3  # Reduced for testing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "$RESULTS_DIR" "$TEMP_DIR"

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to measure compilation time
measure_compilation_time() {
    local source_file=$1
    local output_binary=$2
    local env_vars=$3
    
    local total_time=0
    local times=()
    
    for i in $(seq 1 $MEASURES); do
        local start_time=$(date +%s%N)
        if [ -n "$env_vars" ]; then
            env $env_vars $CHOREO_BIN "$source_file" -o "$output_binary" >/dev/null 2>&1
        else
            $CHOREO_BIN "$source_file" -o "$output_binary" >/dev/null 2>&1
        fi
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        times+=($duration)
        total_time=$((total_time + duration))
    done
    
    # Calculate average and standard deviation
    local avg=$((total_time / MEASURES))
    local variance=0
    for time in "${times[@]}"; do
        local diff=$((time - avg))
        variance=$((variance + diff * diff))
    done
    variance=$((variance / MEASURES))
    local stddev=$(echo "sqrt($variance)" | bc -l 2>/dev/null | cut -d. -f1)
    [ -z "$stddev" ] && stddev=0
    
    echo "$avg $stddev"
}

# Function to measure execution time
measure_execution_time() {
    local binary=$1
    
    local total_time=0
    local times=()
    
    for i in $(seq 1 $MEASURES); do
        local output=$(./"$binary" 2>&1)
        local exec_time=$(echo "$output" | grep "Execution time:" | awk '{print $3}')
        if [ -n "$exec_time" ]; then
            # Convert microseconds to milliseconds
            local time_ms=$(echo "scale=3; $exec_time / 1000" | bc -l)
            local time_int=$(echo "$time_ms * 1000" | bc -l | cut -d. -f1)
            times+=($time_int)
            total_time=$((total_time + time_int))
        fi
    done
    
    if [ ${#times[@]} -eq 0 ]; then
        echo "0 0"
        return
    fi
    
    # Calculate average and standard deviation (in microseconds)
    local avg=$((total_time / MEASURES))
    local variance=0
    for time in "${times[@]}"; do
        local diff=$((time - avg))
        variance=$((variance + diff * diff))
    done
    variance=$((variance / MEASURES))
    local stddev=$(echo "sqrt($variance)" | bc -l 2>/dev/null | cut -d. -f1)
    [ -z "$stddev" ] && stddev=0
    
    echo "$avg $stddev"
}

# Function to process a single benchmark file
process_benchmark_file() {
    local file_path=$1
    local category=$2
    local filename=$(basename "$file_path" .co)
    
    # Check if file contains dynamic shape patterns
    if ! grep -q "#ifdef __STATIC_SHAPE__" "$file_path"; then
        return 0
    fi
    
    print_colored $BLUE "Processing: $filename"
    
    # Compile in static mode
    local static_binary="$TEMP_DIR/${filename}_static"
    local static_comp_result=$(measure_compilation_time "$file_path" "$static_binary" "__STATIC_SHAPE__=1")
    local static_comp_time=$(echo $static_comp_result | cut -d' ' -f1)
    local static_comp_std=$(echo $static_comp_result | cut -d' ' -f2)
    
    # Measure static execution time
    local static_exec_result=$(measure_execution_time "$static_binary")
    local static_exec_time=$(echo $static_exec_result | cut -d' ' -f1)
    local static_exec_std=$(echo $static_exec_result | cut -d' ' -f2)
    
    # Compile in dynamic mode
    local dynamic_binary="$TEMP_DIR/${filename}_dynamic"
    local dynamic_comp_result=$(measure_compilation_time "$file_path" "$dynamic_binary" "")
    local dynamic_comp_time=$(echo $dynamic_comp_result | cut -d' ' -f1)
    local dynamic_comp_std=$(echo $dynamic_comp_result | cut -d' ' -f2)
    
    # Measure dynamic execution time
    local dynamic_exec_result=$(measure_execution_time "$dynamic_binary")
    local dynamic_exec_time=$(echo $dynamic_exec_result | cut -d' ' -f1)
    local dynamic_exec_std=$(echo $dynamic_exec_result | cut -d' ' -f2)
    
    # Calculate deltas
    local comp_delta=$((dynamic_comp_time - static_comp_time))
    local exec_delta=$((dynamic_exec_time - static_exec_time))
    local total_delta=$((comp_delta + exec_delta))

    # Calculate percentage changes for all metrics
    local comp_pct=0
    if [ $static_comp_time -ne 0 ]; then
        comp_pct=$(echo "scale=1; $comp_delta * 100.0 / $static_comp_time" | bc -l)
    fi

    local exec_pct=0
    if [ $static_exec_time -ne 0 ]; then
        exec_pct=$(echo "scale=1; $exec_delta * 100.0 / $static_exec_time" | bc -l)
    fi

    local total_pct=0
    local static_total=$((static_comp_time + static_exec_time))
    if [ $static_total -ne 0 ]; then
        total_pct=$(echo "scale=1; $total_delta * 100.0 / $static_total" | bc -l)
    fi

    # Store results (added comp_pct to CSV)
    echo "$category,$filename,$SAMPLES,$MEASURES,$comp_delta,$comp_pct,$exec_delta,$exec_pct,$static_exec_std,$total_delta,$total_pct" >> "$RESULTS_DIR/raw_results.csv"
    
    # Clean up binaries
    rm -f "$static_binary" "$dynamic_binary"
    
    print_colored $GREEN "✓ Completed: $filename"
    
    # Print immediate results
    printf "  Static:  Comp=%dms, Exec=%dms\n" $static_comp_time $static_exec_time
    printf "  Dynamic: Comp=%dms, Exec=%dms\n" $dynamic_comp_time $dynamic_exec_time
    printf "  Delta:   Comp=%+dms (%.1f%%), Exec=%+dms (%.1f%%), Total=%+dms (%.1f%%)\n" \
           $comp_delta $comp_pct $exec_delta $exec_pct $total_delta $total_pct
    echo
}

# Test with just a few files
main() {
    print_colored $CYAN "🧪 Testing Symbolic System Cost Evaluation"
    print_colored $YELLOW "Configuration: $SAMPLES samples, $MEASURES measures per sample"
    
    # Initialize results file (updated header with comp_pct)
    echo "category,workload,samples,measures,comp_delta,comp_pct,exec_delta,exec_pct,exec_std,total_delta,total_pct" > "$RESULTS_DIR/raw_results.csv"
    
    # Test with a few specific files
    local test_files=(
        "benchmark/reduce_mean/10_dynamic_32xSx768_32x768.co"
        "benchmark/elemwise_add/11_dynamic_32xSx768_32xSx768_32xSx768.co"
        "benchmark/matmul/11_dynamic_32xSx768_768x768_32xSx768.co"
    )
    
    for file in "${test_files[@]}"; do
        if [ -f "$file" ]; then
            local category=$(basename $(dirname "$file"))
            process_benchmark_file "$file" "$category"
        else
            print_colored $RED "⚠️  File not found: $file"
        fi
    done
    
    print_colored $CYAN "✅ Test completed!"
    
    # Show results
    if [ -f "$RESULTS_DIR/raw_results.csv" ]; then
        print_colored $YELLOW "📊 Results Summary:"
        echo "⚙️  Runner: choreo_symbolic_system (TEST)"
        echo "------------------------------------------------------------------------------------------------------------------------"
        printf "%-30s %-8s %-8s %-20s %-20s %-20s %-6s\n" "Workload" "Samples" "Measures" "Compilation Δ" "Execution Δ" "Total Δ" "Rank"
        echo "------------------------------------------------------------------------------------------------------------------------"
        
        local rank=1
        tail -n +2 "$RESULTS_DIR/raw_results.csv" | sort -t',' -k10,10n | while IFS=',' read -r category workload samples measures comp_delta comp_pct exec_delta exec_pct exec_std total_delta total_pct; do
            local rank_symbol="🏆"
            case $rank in
                1) rank_symbol="🏆" ;;
                2) rank_symbol="🥈" ;;
                3) rank_symbol="🥉" ;;
                *) rank_symbol="#$rank" ;;
            esac
            
            # Format deltas with proper signs and units, including percentages
            local comp_delta_str="+${comp_delta}.0ms"
            [ $comp_delta -lt 0 ] && comp_delta_str="${comp_delta}.0ms"

            local exec_delta_str="+${exec_delta}.0ms"
            [ $exec_delta -lt 0 ] && exec_delta_str="${exec_delta}.0ms"

            local total_delta_str="+${total_delta}.0ms"
            [ $total_delta -lt 0 ] && total_delta_str="${total_delta}.0ms"

            # Format percentages with sign for all metrics
            local comp_pct_str="+${comp_pct}%"
            [ $(echo "$comp_pct < 0" | bc -l) -eq 1 ] && comp_pct_str="${comp_pct}%"

            local exec_pct_str="+${exec_pct}%"
            [ $(echo "$exec_pct < 0" | bc -l) -eq 1 ] && exec_pct_str="${exec_pct}%"

            printf "%-30s %-8s %-8s %-20s %-20s %-20s %-6s\n" \
                "${workload:0:29}" "$samples" "$measures" \
                "$comp_delta_str ($comp_pct_str)" \
                "$exec_delta_str ($exec_pct_str) ±${exec_std}" \
                "$total_delta_str" \
                "$rank_symbol"
            
            rank=$((rank + 1))
        done
    fi
    
    print_colored $GREEN "🎉 Test completed! Full script is ready at: benchmark/scripts/cost_eval_symbolic_system.sh"
}

# Run main function
main "$@"
