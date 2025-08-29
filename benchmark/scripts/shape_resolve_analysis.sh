#!/bin/bash

set -e

CHOREO_BIN="./choreo"
BENCHMARK_DIR="benchmark"
RESULTS_DIR="benchmark/scripts/results"
TEMP_DIR="benchmark/scripts/temp"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

mkdir -p "$RESULTS_DIR" "$TEMP_DIR"

print_msg() {
    echo -e "${1}${2}${NC}"
}

analyze_shapes() {
    local file=$1 cat=$2
    local name=$(basename "$file" .co)

    print_msg $BLUE "Analyzing: $name"

    local output="$TEMP_DIR/${name}_inference.txt"
    if ! $CHOREO_BIN -i "$file" > "$output" 2>&1; then
        print_msg $RED "Failed: $name"
        echo "$cat,$name,FAILED,0,0,0,0,0,0,0,0,0,0" >> "$RESULTS_DIR/shape_analysis_raw.csv"
        return 1
    fi
    
    local total=0 resolved=0 concrete=0 symbolic=0 partial=0

    while IFS= read -r line; do
        if [[ "$line" =~ ^(Parameter|Symbol|Bounded|Future|Function): ]]; then
            total=$((total + 1))

            if [[ "$line" =~ Type:\ (.+)$ ]]; then
                local type_info="${BASH_REMATCH[1]}"

                if [[ "$type_info" =~ \[([^\]]+)\] ]]; then
                    local shape="${BASH_REMATCH[1]}"
                    resolved=$((resolved + 1))

                    if [[ "$shape" =~ \? ]]; then
                        partial=$((partial + 1))
                    elif [[ "$shape" =~ [a-zA-Z_] ]]; then
                        symbolic=$((symbolic + 1))
                    elif [[ "$shape" =~ ^[0-9,\ ]+$ ]]; then
                        concrete=$((concrete + 1))
                    else
                        symbolic=$((symbolic + 1))
                    fi
                fi
            fi
        fi
    done < "$output"
    
    local success=0 node_rate=0 concrete_rate=0 symbolic_rate=0 partial_rate=0

    if [ $total -gt 0 ]; then
        if [ $resolved -eq $total ]; then
            success=1
        fi

        node_rate=$(echo "scale=1; $resolved * 100.0 / $total" | bc -l)
        concrete_rate=$(echo "scale=1; $concrete * 100.0 / $total" | bc -l)
        symbolic_rate=$(echo "scale=1; $symbolic * 100.0 / $total" | bc -l)
        partial_rate=$(echo "scale=1; $partial * 100.0 / $total" | bc -l)
    fi

    echo "$cat,$name,SUCCESS,$total,$resolved,$concrete,$symbolic,$partial,$success,$node_rate,$concrete_rate,$symbolic_rate,$partial_rate" >> "$RESULTS_DIR/shape_analysis_raw.csv"

    print_msg $GREEN "✓ $name"
    printf "  Total: %d, Resolved: %d (%.1f%%), Concrete: %d (%.1f%%), Symbolic: %d (%.1f%%), Partial: %d (%.1f%%)\n" \
           $total $resolved $node_rate $concrete $concrete_rate $symbolic $symbolic_rate $partial $partial_rate

    rm -f "$output"
}

# Function to get operator categories in order
get_categories() {
    echo "batch_norm concat conv2d elemwise_add embedding gelu layer_normalization matmul max_pool2d reduce_mean relu reshape sigmoid softmax transpose"
}

# Main execution
main() {
    print_msg $CYAN "Starting shape analysis"
    
    # Initialize results file
    echo "category,workload,status,total_nodes,resolved_nodes,fully_resolved,symbolic_resolved,partially_resolved,case_success,node_resolve_rate,fully_resolve_rate,symbolic_resolve_rate,partially_resolve_rate" > "$RESULTS_DIR/shape_analysis_raw.csv"
    
    # Process each category
    local categories=$(get_categories)
    for cat in $categories; do
        local cat_dir="$BENCHMARK_DIR/$cat"
        if [ ! -d "$cat_dir" ]; then
            print_msg $RED "Missing: $cat_dir"
            continue
        fi

        print_msg $PURPLE "Processing: $cat"

        find "$cat_dir" -name "*.co" -type f | sort | while read -r file; do
            analyze_shapes "$file" "$cat"
        done
    done

    print_msg $CYAN "Generating reports..."
    gen_shape_reports
    print_msg $GREEN "Done. Results in: $RESULTS_DIR"
}

gen_shape_reports() {
    local results="$RESULTS_DIR/shape_analysis_raw.csv"

    if [ ! -f "$results" ]; then
        print_msg $RED "No results file found"
        return 1
    fi

    gen_case_report "$results" "$RESULTS_DIR/shape_analysis_per_case.txt"
    gen_workload_report "$results" "$RESULTS_DIR/shape_analysis_workload.txt"
    gen_summary_report "$results" "$RESULTS_DIR/shape_analysis_global.txt"
}

gen_case_report() {
    local results=$1 output=$2

    {
        echo "Shape Resolution Analysis - Per Case"
        echo "===================================="
        printf "%-30s %-10s %-8s %-8s %-8s %-8s %-8s\n" \
            "Case" "Success" "GRR" "FRSR" "SRSR" "PRSR" "Nodes"
        echo "--------------------------------------------------------------------------------"

        tail -n +2 "$results" | grep ",SUCCESS," | while IFS=',' read -r cat work stat total res concrete symbolic partial success node_rate concrete_rate symbolic_rate partial_rate; do
            local indicator="❌"
            [ "$success" = "1" ] && indicator="✅"

            printf "%-30s %-10s %-8.1f %-8.1f %-8.1f %-8.1f %-8d\n" \
                "${work:0:29}" "$indicator" \
                "$node_rate" "$concrete_rate" \
                "$symbolic_rate" "$partial_rate" \
                "$total"
        done
    } > "$output"
}

gen_workload_report() {
    local results=$1 output=$2

    {
        echo "Workload Summary"
        echo "================"
        printf "%-15s %-10s %-8s %-8s %-8s %-8s %-8s\n" \
            "Workload" "Success" "GRR" "FRSR" "SRSR" "PRSR" "Cases"
        echo "--------------------------------------------------------------------------------"

        tail -n +2 "$results" | grep ",SUCCESS," | cut -d',' -f1 | sort -u | while read -r cat; do
            local data=$(tail -n +2 "$results" | grep "^$cat," | grep ",SUCCESS,")
            local total=$(echo "$data" | wc -l)

            if [ $total -eq 0 ]; then
                continue
            fi

            local success=$(echo "$data" | awk -F',' '$9 == 1' | wc -l)
            local success_rate=$(echo "scale=1; $success * 100.0 / $total" | bc -l)

            local avg_node=$(echo "$data" | cut -d',' -f10 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_concrete=$(echo "$data" | cut -d',' -f11 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_symbolic=$(echo "$data" | cut -d',' -f12 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_partial=$(echo "$data" | cut -d',' -f13 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')

            printf "%-15s %-10.1f %-8.1f %-8.1f %-8.1f %-8.1f %-8d\n" \
                "$cat" "$success_rate" \
                "$avg_node" "$avg_concrete" \
                "$avg_symbolic" "$avg_partial" \
                "$total"
        done
    } > "$output"
}

gen_summary_report() {
    local results=$1 output=$2

    {
        echo "Global Summary"
        echo "=============="

        local data=$(tail -n +2 "$results" | grep ",SUCCESS,")
        local total_cases=$(echo "$data" | wc -l)

        if [ $total_cases -eq 0 ]; then
            echo "No data available."
            return
        fi

        local success_cases=$(echo "$data" | awk -F',' '$9 == 1' | wc -l)
        local success_rate=$(echo "scale=1; $success_cases * 100.0 / $total_cases" | bc -l)

        local total_nodes=$(echo "$data" | cut -d',' -f4 | awk '{sum+=$1} END {print sum}')
        local total_resolved=$(echo "$data" | cut -d',' -f5 | awk '{sum+=$1} END {print sum}')
        local total_concrete=$(echo "$data" | cut -d',' -f6 | awk '{sum+=$1} END {print sum}')
        local total_symbolic=$(echo "$data" | cut -d',' -f7 | awk '{sum+=$1} END {print sum}')
        local total_partial=$(echo "$data" | cut -d',' -f8 | awk '{sum+=$1} END {print sum}')

        local node_rate=0 concrete_rate=0 symbolic_rate=0 partial_rate=0

        if [ $total_nodes -gt 0 ]; then
            node_rate=$(echo "scale=1; $total_resolved * 100.0 / $total_nodes" | bc -l)
            concrete_rate=$(echo "scale=1; $total_concrete * 100.0 / $total_nodes" | bc -l)
            symbolic_rate=$(echo "scale=1; $total_symbolic * 100.0 / $total_nodes" | bc -l)
            partial_rate=$(echo "scale=1; $total_partial * 100.0 / $total_nodes" | bc -l)
        fi

        echo "=================================================================================="
        printf "%-15s %-10s %-8s %-8s %-8s %-8s\n" \
            "Runner" "Success" "GRR" "FRSR" "SRSR" "PRSR"
        printf "%-15s %-10s %-8s %-8s %-8s %-8s\n" \
            "" "Rate" "%" "%" "%" "%"
        echo "--------------------------------------------------------------------------------"
        printf "%-15s %-10.1f %-8.1f %-8.1f %-8.1f %-8.1f\n" \
            "choreo_dynamic" "$success_rate" \
            "$node_rate" "$concrete_rate" \
            "$symbolic_rate" "$partial_rate"
        echo "=================================================================================="

    } > "$output"
}

# Run main function
main "$@"
