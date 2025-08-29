#!/bin/bash

# Test version of Shape Resolution Analysis Script
# Tests with a few representative files

set -e

# Configuration
CHOREO_BIN="./choreo"
BENCHMARK_DIR="benchmark"
RESULTS_DIR="benchmark/scripts/results"
TEMP_DIR="benchmark/scripts/temp"

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

# Function to analyze shape inference output
analyze_shape_inference() {
    local source_file=$1
    local category=$2
    local filename=$(basename "$source_file" .co)
    
    print_colored $BLUE "Analyzing: $filename"
    
    # Run shape inference (ensure dynamic mode - no __STATIC_SHAPE__)
    local inference_output="$TEMP_DIR/${filename}_inference.txt"
    if ! $CHOREO_BIN -i "$source_file" > "$inference_output" 2>&1; then
        print_colored $RED "❌ Shape inference failed for: $filename"
        echo "$category,$filename,FAILED,0,0,0,0,0,0,0,0,0,0" >> "$RESULTS_DIR/shape_analysis_raw.csv"
        return 1
    fi
    
    print_colored $YELLOW "📋 Shape inference output preview:"
    head -10 "$inference_output" | while read -r line; do
        echo "    $line"
    done
    echo ""
    
    # Parse the inference output
    local total_nodes=0
    local resolved_nodes=0
    local fully_resolved_nodes=0
    local symbolic_resolved_nodes=0
    local partially_resolved_nodes=0
    
    print_colored $YELLOW "🔍 Analyzing nodes:"
    
    # Count different types of nodes
    while IFS= read -r line; do
        if [[ "$line" =~ ^(Parameter|Symbol|Bounded|Future|Function): ]]; then
            total_nodes=$((total_nodes + 1))
            
            # Extract the type information (everything after "Type: ")
            if [[ "$line" =~ Type:\ (.+)$ ]]; then
                local type_info="${BASH_REMATCH[1]}"
                
                # Check if this node has shape information (contains [...])
                if [[ "$type_info" =~ \[([^\]]+)\] ]]; then
                    local shape_info="${BASH_REMATCH[1]}"
                    resolved_nodes=$((resolved_nodes + 1))
                    
                    # Analyze the shape content
                    if [[ "$shape_info" =~ \? ]]; then
                        # Contains '?' - partially resolved
                        partially_resolved_nodes=$((partially_resolved_nodes + 1))
                        echo "    PARTIAL: $shape_info"
                    elif [[ "$shape_info" =~ [a-zA-Z_] ]]; then
                        # Contains symbolic names - symbolically resolved
                        symbolic_resolved_nodes=$((symbolic_resolved_nodes + 1))
                        echo "    SYMBOLIC: $shape_info"
                    elif [[ "$shape_info" =~ ^[0-9,\ ]+$ ]]; then
                        # Only contains numbers and commas - fully resolved
                        fully_resolved_nodes=$((fully_resolved_nodes + 1))
                        echo "    CONCRETE: $shape_info"
                    else
                        # Has shape but mixed content - count as symbolic
                        symbolic_resolved_nodes=$((symbolic_resolved_nodes + 1))
                        echo "    MIXED: $shape_info"
                    fi
                else
                    echo "    UNRESOLVED: $type_info"
                fi
            fi
        fi
    done < "$inference_output"
    
    # Calculate rates
    local case_success=0
    local node_resolve_rate=0
    local fully_resolve_rate=0
    local symbolic_resolve_rate=0
    local partially_resolve_rate=0
    
    if [ $total_nodes -gt 0 ]; then
        # Case success: all nodes resolved
        if [ $resolved_nodes -eq $total_nodes ]; then
            case_success=1
        fi
        
        # Node resolve rate: resolved nodes / total nodes
        node_resolve_rate=$(echo "scale=1; $resolved_nodes * 100.0 / $total_nodes" | bc -l)
        
        # Fully resolve rate: fully resolved nodes / total nodes
        fully_resolve_rate=$(echo "scale=1; $fully_resolved_nodes * 100.0 / $total_nodes" | bc -l)
        
        # Symbolic resolve rate: symbolic resolved nodes / total nodes
        symbolic_resolve_rate=$(echo "scale=1; $symbolic_resolved_nodes * 100.0 / $total_nodes" | bc -l)
        
        # Partially resolve rate: partially resolved nodes / total nodes
        partially_resolve_rate=$(echo "scale=1; $partially_resolved_nodes * 100.0 / $total_nodes" | bc -l)
    fi
    
    # Store results
    echo "$category,$filename,SUCCESS,$total_nodes,$resolved_nodes,$fully_resolved_nodes,$symbolic_resolved_nodes,$partially_resolved_nodes,$case_success,$node_resolve_rate,$fully_resolve_rate,$symbolic_resolve_rate,$partially_resolve_rate" >> "$RESULTS_DIR/shape_analysis_raw.csv"
    
    print_colored $GREEN "✓ Analysis complete for: $filename"
    printf "  📊 Summary: Total=%d, Resolved=%d (%.1f%%), Concrete=%d (%.1f%%), Symbolic=%d (%.1f%%), Partial=%d (%.1f%%)\n" \
           $total_nodes $resolved_nodes $node_resolve_rate $fully_resolved_nodes $fully_resolve_rate \
           $symbolic_resolved_nodes $symbolic_resolve_rate $partially_resolved_nodes $partially_resolve_rate
    
    local success_status="❌ FAILED"
    [ "$case_success" = "1" ] && success_status="✅ SUCCESS"
    printf "  🎯 Case Status: %s\n" "$success_status"
    echo ""
    
    # Clean up
    rm -f "$inference_output"
}

# Test with specific files
main() {
    print_colored $CYAN "🧪 Testing Shape Resolution Analysis"
    print_colored $YELLOW "Analyzing symbolic system shape inference on dynamic cases"
    
    # Initialize results file
    echo "category,workload,status,total_nodes,resolved_nodes,fully_resolved,symbolic_resolved,partially_resolved,case_success,node_resolve_rate,fully_resolve_rate,symbolic_resolve_rate,partially_resolve_rate" > "$RESULTS_DIR/shape_analysis_raw.csv"
    
    # Test with a few specific files
    local test_files=(
        "benchmark/reduce_mean/10_dynamic_32xSx768_32x768.co"
        "benchmark/elemwise_add/11_dynamic_32xSx768_32xSx768_32xSx768.co"
        "benchmark/matmul/11_dynamic_32xSx768_768x768_32xSx768.co"
    )
    
    for file in "${test_files[@]}"; do
        if [ -f "$file" ]; then
            local category=$(basename $(dirname "$file"))
            analyze_shape_inference "$file" "$category"
        else
            print_colored $RED "⚠️  File not found: $file"
        fi
    done
    
    print_colored $CYAN "✅ Test analysis completed!"
    
    # Show summary results
    if [ -f "$RESULTS_DIR/shape_analysis_raw.csv" ]; then
        print_colored $YELLOW "📊 Test Results Summary:"
        echo ""
        echo "=================================================================================="
        printf "%-15s %-10s %-8s %-8s %-8s %-8s\n" \
            "Runner" "Success" "GRR" "FRSR" "SRSR" "PRSR"
        printf "%-15s %-10s %-8s %-8s %-8s %-8s\n" \
            "" "Rate" "%" "%" "%" "%"
        echo "--------------------------------------------------------------------------------"
        
        # Calculate overall statistics
        local successful_data=$(tail -n +2 "$RESULTS_DIR/shape_analysis_raw.csv" | grep ",SUCCESS,")
        local total_cases=$(echo "$successful_data" | wc -l)
        
        if [ $total_cases -gt 0 ]; then
            local successful_cases=$(echo "$successful_data" | awk -F',' '$9 == 1' | wc -l)
            local overall_success_rate=$(echo "scale=1; $successful_cases * 100.0 / $total_cases" | bc -l)
            
            # Node-level statistics
            local total_nodes=$(echo "$successful_data" | cut -d',' -f4 | awk '{sum+=$1} END {print sum}')
            local total_resolved=$(echo "$successful_data" | cut -d',' -f5 | awk '{sum+=$1} END {print sum}')
            local total_fully_resolved=$(echo "$successful_data" | cut -d',' -f6 | awk '{sum+=$1} END {print sum}')
            local total_symbolic_resolved=$(echo "$successful_data" | cut -d',' -f7 | awk '{sum+=$1} END {print sum}')
            local total_partially_resolved=$(echo "$successful_data" | cut -d',' -f8 | awk '{sum+=$1} END {print sum}')
            
            local node_resolve_rate=0
            local fully_resolve_rate=0
            local symbolic_resolve_rate=0
            local partially_resolve_rate=0
            
            if [ $total_nodes -gt 0 ]; then
                node_resolve_rate=$(echo "scale=1; $total_resolved * 100.0 / $total_nodes" | bc -l)
                fully_resolve_rate=$(echo "scale=1; $total_fully_resolved * 100.0 / $total_nodes" | bc -l)
                symbolic_resolve_rate=$(echo "scale=1; $total_symbolic_resolved * 100.0 / $total_nodes" | bc -l)
                partially_resolve_rate=$(echo "scale=1; $total_partially_resolved * 100.0 / $total_nodes" | bc -l)
            fi
            
            printf "%-15s %-10.1f %-8.1f %-8.1f %-8.1f %-8.1f\n" \
                "choreo_dynamic" "$overall_success_rate" \
                "$node_resolve_rate" "$fully_resolve_rate" \
                "$symbolic_resolve_rate" "$partially_resolve_rate"
        fi
        
        echo "=================================================================================="
        echo ""
        
        print_colored $GREEN "📋 Legend:"
        echo "  Success Rate: Percentage of cases where all nodes were resolved"
        echo "  GRR: General Resolve Rate (resolved nodes / total nodes)"
        echo "  FRSR: Fully Resolved Shape Rate (concrete dimensions only)"
        echo "  SRSR: Symbolic Resolved Shape Rate (contains symbolic dimensions)"
        echo "  PRSR: Partially Resolved Shape Rate (contains '?' dimensions)"
    fi
    
    print_colored $GREEN "🎉 Test completed! Full script is ready at: benchmark/scripts/shape_resolve_analysis.sh"
}

# Run main function
main "$@"
