#!/bin/bash

set -e

CHOREO_BIN="./choreo"
BENCHMARK_DIR="benchmark"
RESULTS_DIR="benchmark/scripts/results"
TEMP_DIR="benchmark/scripts/temp"
SAMPLES=3
MEASURES=9

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

measure_compile_time() {
    local src=$1 bin=$2 env_vars=$3
    local total=0 times=()

    for i in $(seq 1 $MEASURES); do
        local start=$(date +%s%N)
        if [ -n "$env_vars" ]; then
            env $env_vars $CHOREO_BIN "$src" -o "$bin" >/dev/null 2>&1
        else
            $CHOREO_BIN "$src" -o "$bin" >/dev/null 2>&1
        fi
        local end=$(date +%s%N)
        local dur=$(( (end - start) / 1000000 ))
        times+=($dur)
        total=$((total + dur))
    done

    local avg=$((total / MEASURES))
    local var=0
    for t in "${times[@]}"; do
        local diff=$((t - avg))
        var=$((var + diff * diff))
    done
    var=$((var / MEASURES))
    local std=$(echo "sqrt($var)" | bc -l | cut -d. -f1)

    echo "$avg $std"
}

measure_exec_time() {
    local bin=$1
    local total=0 times=()

    for i in $(seq 1 $MEASURES); do
        local out=$(./"$bin" 2>&1)
        local exec_us=$(echo "$out" | grep "Execution time:" | awk '{print $3}')
        if [ -n "$exec_us" ]; then
            local exec_ms=$(echo "$exec_us * 1000" | bc -l | cut -d. -f1)
            times+=($exec_ms)
            total=$((total + exec_ms))
        fi
    done

    if [ ${#times[@]} -eq 0 ]; then
        echo "0 0"
        return
    fi

    local avg=$((total / MEASURES))
    local var=0
    for t in "${times[@]}"; do
        local diff=$((t - avg))
        var=$((var + diff * diff))
    done
    var=$((var / MEASURES))
    local std=$(echo "sqrt($var)" | bc -l | cut -d. -f1)

    echo "$avg $std"
}

process_file() {
    local file=$1 cat=$2
    local name=$(basename "$file" .co)

    if ! grep -q "#ifdef __STATIC_SHAPE__" "$file"; then
        return 0
    fi

    print_msg $BLUE "Processing: $name"

    local static_bin="$TEMP_DIR/${name}_static"
    local static_comp=$(measure_compile_time "$file" "$static_bin" "__STATIC_SHAPE__=1")
    local static_comp_time=$(echo $static_comp | cut -d' ' -f1)
    local static_comp_std=$(echo $static_comp | cut -d' ' -f2)

    local static_exec=$(measure_exec_time "$static_bin")
    local static_exec_time=$(echo $static_exec | cut -d' ' -f1)
    local static_exec_std=$(echo $static_exec | cut -d' ' -f2)

    local dynamic_bin="$TEMP_DIR/${name}_dynamic"
    local dynamic_comp=$(measure_compile_time "$file" "$dynamic_bin" "")
    local dynamic_comp_time=$(echo $dynamic_comp | cut -d' ' -f1)
    local dynamic_comp_std=$(echo $dynamic_comp | cut -d' ' -f2)

    local dynamic_exec=$(measure_exec_time "$dynamic_bin")
    local dynamic_exec_time=$(echo $dynamic_exec | cut -d' ' -f1)
    local dynamic_exec_std=$(echo $dynamic_exec | cut -d' ' -f2)
    
    local comp_delta=$((dynamic_comp_time - static_comp_time))
    local exec_delta=$((dynamic_exec_time - static_exec_time))
    local total_delta=$((comp_delta + exec_delta))

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

    echo "$cat,$name,$SAMPLES,$MEASURES,$comp_delta,$comp_pct,$exec_delta,$exec_pct,$static_exec_std,$total_delta,$total_pct" >> "$RESULTS_DIR/raw_results.csv"

    rm -f "$static_bin" "$dynamic_bin"
    print_msg $GREEN "✓ $name"
}

get_categories() {
    echo "batch_norm concat conv2d elemwise_add embedding gelu layer_normalization matmul max_pool2d reduce_mean relu reshape sigmoid softmax transpose"
}

main() {
    print_msg $CYAN "Starting benchmark evaluation"

    echo "category,workload,samples,measures,comp_delta,comp_pct,exec_delta,exec_pct,exec_std,total_delta,total_pct" > "$RESULTS_DIR/raw_results.csv"

    for cat in $(get_categories); do
        local cat_dir="$BENCHMARK_DIR/$cat"
        if [ ! -d "$cat_dir" ]; then
            print_msg $RED "Missing: $cat_dir"
            continue
        fi

        print_msg $PURPLE "Processing: $cat"

        find "$cat_dir" -name "*.co" -type f | sort | while read -r file; do
            process_file "$file" "$cat"
        done
    done

    print_msg $CYAN "Generating reports..."
    generate_reports
    print_msg $GREEN "Done. Results in: $RESULTS_DIR"
}

generate_reports() {
    local results="$RESULTS_DIR/raw_results.csv"

    if [ ! -f "$results" ]; then
        print_msg $RED "No results file found"
        return 1
    fi

    gen_individual_report "$results" "$RESULTS_DIR/symbolic_system_report.txt"
    gen_category_report "$results" "$RESULTS_DIR/category_summary.txt"
    gen_global_report "$results" "$RESULTS_DIR/global_summary.txt"
    gen_workload_summary "$results" "$RESULTS_DIR/workload_summary.txt"
    gen_runner_summary "$results" "$RESULTS_DIR/runner_summary.txt"
    gen_cross_tab_matrix "$results" "$RESULTS_DIR/cross_tabulation_matrix.txt"
}

gen_individual_report() {
    local results=$1 output=$2

    {
        echo "Runner: choreo_symbolic_system"
        echo "------------------------------------------------------------------------------------------------------------------------"
        printf "%-30s %-8s %-8s %-20s %-20s %-20s %-6s\n" "Workload" "Samples" "Measures" "Compilation Δ" "Execution Δ" "Total Δ" "Rank"
        echo "------------------------------------------------------------------------------------------------------------------------"

        local rank=1
        tail -n +2 "$results" | sort -t',' -k10,10n | while IFS=',' read -r cat work samp meas comp_d comp_p exec_d exec_p exec_s total_d total_p; do
            local sym="🏆"
            case $rank in
                1) sym="🏆" ;;
                2) sym="🥈" ;;
                3) sym="🥉" ;;
                *) sym="#$rank" ;;
            esac

            local comp_str="+${comp_d}.0ms"
            [ $comp_d -lt 0 ] && comp_str="${comp_d}.0ms"

            local exec_str="+${exec_d}.0ms"
            [ $exec_d -lt 0 ] && exec_str="${exec_d}.0ms"

            local total_str="+${total_d}.0ms"
            [ $total_d -lt 0 ] && total_str="${total_d}.0ms"

            local comp_pct_str="+${comp_p}%"
            [ $(echo "$comp_p < 0" | bc -l) -eq 1 ] && comp_pct_str="${comp_p}%"

            local exec_pct_str="+${exec_p}%"
            [ $(echo "$exec_p < 0" | bc -l) -eq 1 ] && exec_pct_str="${exec_p}%"

            printf "%-30s %-8s %-8s %-20s %-20s %-20s %-6s\n" \
                "${work:0:29}" "$samp" "$meas" \
                "$comp_str ($comp_pct_str)" \
                "$exec_str ($exec_pct_str) ±${exec_s}" \
                "$total_str" \
                "$sym"

            rank=$((rank + 1))
        done
    } > "$output"
}

gen_category_report() {
    local results_file=$1
    local report_file=$2

    {
        echo "📊 Category-wise Performance Summary"
        echo "========================================"
        echo ""

        # Get unique categories and process each
        tail -n +2 "$results_file" | cut -d',' -f1 | sort -u | while read -r category; do
            echo "🔹 Category: $category"
            echo "----------------------------------------"

            # Calculate category statistics
            local category_data=$(tail -n +2 "$results_file" | grep "^$category,")
            local count=$(echo "$category_data" | wc -l)

            if [ $count -eq 0 ]; then
                echo "  No data available"
                echo ""
                continue
            fi

            # Calculate averages (updated for new CSV format)
            local avg_comp_delta=$(echo "$category_data" | cut -d',' -f5 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_comp_pct=$(echo "$category_data" | cut -d',' -f6 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_exec_delta=$(echo "$category_data" | cut -d',' -f7 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_exec_pct=$(echo "$category_data" | cut -d',' -f8 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_total_delta=$(echo "$category_data" | cut -d',' -f10 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_total_pct=$(echo "$category_data" | cut -d',' -f11 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')

            # Find best and worst cases (updated column index)
            local best_case=$(echo "$category_data" | sort -t',' -k10,10n | head -1 | cut -d',' -f2)
            local worst_case=$(echo "$category_data" | sort -t',' -k10,10nr | head -1 | cut -d',' -f2)

            printf "  Cases processed: %d\n" $count
            printf "  Avg Compilation Δ: %+.1fms (%+.1f%%)\n" $avg_comp_delta $avg_comp_pct
            printf "  Avg Execution Δ: %+.1fms (%+.1f%%)\n" $avg_exec_delta $avg_exec_pct
            printf "  Avg Total Δ: %+.1fms (%+.1f%%)\n" $avg_total_delta $avg_total_pct
            printf "  Best case: %s\n" "$best_case"
            printf "  Worst case: %s\n" "$worst_case"
            echo ""
        done

    } > "$report_file"
}

gen_global_report() {
    local results_file=$1
    local report_file=$2

    {
        echo "🌍 Global Performance Summary"
        echo "============================="
        echo ""

        local total_cases=$(tail -n +2 "$results_file" | wc -l)

        if [ $total_cases -eq 0 ]; then
            echo "No benchmark data available."
            return
        fi

        # Overall statistics (updated for new CSV format)
        local overall_avg_comp=$(tail -n +2 "$results_file" | cut -d',' -f5 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local overall_avg_comp_pct=$(tail -n +2 "$results_file" | cut -d',' -f6 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local overall_avg_exec=$(tail -n +2 "$results_file" | cut -d',' -f7 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local overall_avg_exec_pct=$(tail -n +2 "$results_file" | cut -d',' -f8 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local overall_avg_total=$(tail -n +2 "$results_file" | cut -d',' -f10 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local overall_avg_total_pct=$(tail -n +2 "$results_file" | cut -d',' -f11 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')

        # Performance distribution (updated column index)
        local positive_cases=$(tail -n +2 "$results_file" | awk -F',' '$10 > 0' | wc -l)
        local negative_cases=$(tail -n +2 "$results_file" | awk -F',' '$10 < 0' | wc -l)
        local neutral_cases=$(tail -n +2 "$results_file" | awk -F',' '$10 == 0' | wc -l)

        # Best and worst overall (updated column index)
        local global_best=$(tail -n +2 "$results_file" | sort -t',' -k10,10n | head -1)
        local global_worst=$(tail -n +2 "$results_file" | sort -t',' -k10,10nr | head -1)

        local best_workload=$(echo "$global_best" | cut -d',' -f2)
        local best_delta=$(echo "$global_best" | cut -d',' -f10)
        local worst_workload=$(echo "$global_worst" | cut -d',' -f2)
        local worst_delta=$(echo "$global_worst" | cut -d',' -f10)

        printf "📈 Overall Statistics:\n"
        printf "  Total cases analyzed: %d\n" $total_cases
        printf "  Average compilation delta: %+.1fms (%+.1f%%)\n" $overall_avg_comp $overall_avg_comp_pct
        printf "  Average execution delta: %+.1fms (%+.1f%%)\n" $overall_avg_exec $overall_avg_exec_pct
        printf "  Average total delta: %+.1fms (%+.1f%%)\n" $overall_avg_total $overall_avg_total_pct
        echo ""

        printf "📊 Performance Distribution:\n"
        printf "  Improved cases (negative delta): %d (%.1f%%)\n" $negative_cases $(echo "scale=1; $negative_cases * 100.0 / $total_cases" | bc -l)
        printf "  Degraded cases (positive delta): %d (%.1f%%)\n" $positive_cases $(echo "scale=1; $positive_cases * 100.0 / $total_cases" | bc -l)
        printf "  Neutral cases (zero delta): %d (%.1f%%)\n" $neutral_cases $(echo "scale=1; $neutral_cases * 100.0 / $total_cases" | bc -l)
        echo ""

        printf "🏆 Best performing case: %s (%+.1fms)\n" "$best_workload" $best_delta
        printf "⚠️  Worst performing case: %s (%+.1fms)\n" "$worst_workload" $worst_delta
        echo ""

        printf "💡 Symbolic System Impact Summary:\n"
        if [ $(echo "$overall_avg_total < 0" | bc -l) -eq 1 ]; then
            printf "  ✅ Overall POSITIVE impact: %.1fms average improvement\n" $(echo "$overall_avg_total * -1" | bc -l)
        elif [ $(echo "$overall_avg_total > 0" | bc -l) -eq 1 ]; then
            printf "  ❌ Overall NEGATIVE impact: %.1fms average degradation\n" $overall_avg_total
        else
            printf "  ➖ NEUTRAL impact: No significant performance change\n"
        fi

    } > "$report_file"
}

gen_workload_summary() {
    local results_file=$1
    local report_file=$2

    {
        echo "📊 Per-Workload Performance Summary"
        echo "===================================="
        echo ""
        printf "%-15s %-8s %-20s %-20s %-20s %-15s %-15s\n" \
            "Workload" "Cases" "Avg Comp Δ" "Avg Exec Δ" "Avg Total Δ" "Best Case" "Worst Case"
        echo "--------------------------------------------------------------------------------------------------------"

        # Get unique categories and process each
        tail -n +2 "$results_file" | cut -d',' -f1 | sort -u | while read -r category; do
            local category_data=$(tail -n +2 "$results_file" | grep "^$category,")
            local count=$(echo "$category_data" | wc -l)

            if [ $count -eq 0 ]; then
                continue
            fi

            # Calculate averages
            local avg_comp_delta=$(echo "$category_data" | cut -d',' -f5 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_comp_pct=$(echo "$category_data" | cut -d',' -f6 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_exec_delta=$(echo "$category_data" | cut -d',' -f7 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_exec_pct=$(echo "$category_data" | cut -d',' -f8 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_total_delta=$(echo "$category_data" | cut -d',' -f10 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_total_pct=$(echo "$category_data" | cut -d',' -f11 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')

            # Find best and worst cases
            local best_case=$(echo "$category_data" | sort -t',' -k10,10n | head -1 | cut -d',' -f2 | cut -c1-14)
            local worst_case=$(echo "$category_data" | sort -t',' -k10,10nr | head -1 | cut -d',' -f2 | cut -c1-14)

            printf "%-15s %-8d %-20s %-20s %-20s %-15s %-15s\n" \
                "$category" "$count" \
                "${avg_comp_delta}ms (${avg_comp_pct}%)" \
                "${avg_exec_delta}ms (${avg_exec_pct}%)" \
                "${avg_total_delta}ms (${avg_total_pct}%)" \
                "$best_case" \
                "$worst_case"
        done

    } > "$report_file"
}

gen_runner_summary() {
    local results_file=$1
    local report_file=$2

    {
        echo "🏃 Per-Runner Performance Summary"
        echo "================================="
        echo ""
        echo "This analysis compares Static vs Dynamic compilation modes across all workloads."
        echo ""

        local total_cases=$(tail -n +2 "$results_file" | wc -l)

        if [ $total_cases -eq 0 ]; then
            echo "No benchmark data available."
            return
        fi

        # Static mode baseline (all deltas are relative to static)
        echo "📋 Static Mode (Baseline):"
        echo "  - Compilation: 0ms (0%) - Reference baseline"
        echo "  - Execution: 0ms (0%) - Reference baseline"
        echo "  - Total: 0ms (0%) - Reference baseline"
        echo ""

        # Dynamic mode performance
        local avg_comp_delta=$(tail -n +2 "$results_file" | cut -d',' -f5 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local avg_comp_pct=$(tail -n +2 "$results_file" | cut -d',' -f6 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local avg_exec_delta=$(tail -n +2 "$results_file" | cut -d',' -f7 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local avg_exec_pct=$(tail -n +2 "$results_file" | cut -d',' -f8 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local avg_total_delta=$(tail -n +2 "$results_file" | cut -d',' -f10 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        local avg_total_pct=$(tail -n +2 "$results_file" | cut -d',' -f11 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')

        echo "🔄 Dynamic Mode (Symbolic System):"
        printf "  - Compilation: %+.1fms (%+.1f%%) vs Static\n" $avg_comp_delta $avg_comp_pct
        printf "  - Execution: %+.1fms (%+.1f%%) vs Static\n" $avg_exec_delta $avg_exec_pct
        printf "  - Total: %+.1fms (%+.1f%%) vs Static\n" $avg_total_delta $avg_total_pct
        echo ""

        # Performance distribution
        local improved_cases=$(tail -n +2 "$results_file" | awk -F',' '$10 < 0' | wc -l)
        local degraded_cases=$(tail -n +2 "$results_file" | awk -F',' '$10 > 0' | wc -l)
        local neutral_cases=$(tail -n +2 "$results_file" | awk -F',' '$10 == 0' | wc -l)

        echo "📊 Dynamic Mode Impact Distribution:"
        printf "  - Improved cases: %d (%.1f%%) - Dynamic faster than Static\n" \
            $improved_cases $(echo "scale=1; $improved_cases * 100.0 / $total_cases" | bc -l)
        printf "  - Degraded cases: %d (%.1f%%) - Dynamic slower than Static\n" \
            $degraded_cases $(echo "scale=1; $degraded_cases * 100.0 / $total_cases" | bc -l)
        printf "  - Neutral cases: %d (%.1f%%) - No significant difference\n" \
            $neutral_cases $(echo "scale=1; $neutral_cases * 100.0 / $total_cases" | bc -l)

    } > "$report_file"
}

gen_cross_tab_matrix() {
    local results_file=$1
    local report_file=$2

    {
        echo "📊 Cross-Tabulation Matrix: Workload Types vs Compilation Modes"
        echo "================================================================"
        echo ""
        echo "This matrix shows performance deltas (Dynamic - Static) for each workload type."
        echo "Negative values indicate Dynamic mode is faster; Positive values indicate Static mode is faster."
        echo ""

        # Header
        printf "%-15s | %-20s | %-20s | %-20s | %-8s\n" \
            "Workload Type" "Compilation Δ" "Execution Δ" "Total Δ" "Cases"
        echo "----------------+----------------------+----------------------+----------------------+----------"

        # Get unique categories and process each
        tail -n +2 "$results_file" | cut -d',' -f1 | sort -u | while read -r category; do
            local category_data=$(tail -n +2 "$results_file" | grep "^$category,")
            local count=$(echo "$category_data" | wc -l)

            if [ $count -eq 0 ]; then
                continue
            fi

            # Calculate averages for this workload type
            local avg_comp_delta=$(echo "$category_data" | cut -d',' -f5 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_comp_pct=$(echo "$category_data" | cut -d',' -f6 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_exec_delta=$(echo "$category_data" | cut -d',' -f7 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_exec_pct=$(echo "$category_data" | cut -d',' -f8 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_total_delta=$(echo "$category_data" | cut -d',' -f10 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
            local avg_total_pct=$(echo "$category_data" | cut -d',' -f11 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')

            printf "%-15s | %-20s | %-20s | %-20s | %-8d\n" \
                "$category" \
                "${avg_comp_delta}ms (${avg_comp_pct}%)" \
                "${avg_exec_delta}ms (${avg_exec_pct}%)" \
                "${avg_total_delta}ms (${avg_total_pct}%)" \
                "$count"
        done

        echo ""
        echo "📋 Matrix Legend:"
        echo "  - Compilation Δ: Dynamic compilation time - Static compilation time"
        echo "  - Execution Δ: Dynamic execution time - Static execution time"
        echo "  - Total Δ: Sum of compilation and execution deltas"
        echo "  - Cases: Number of benchmark cases in this workload type"
        echo ""
        echo "🎯 Interpretation:"
        echo "  - Negative values (green): Dynamic mode performs better"
        echo "  - Positive values (red): Static mode performs better"
        echo "  - Values near zero: Minimal performance difference"

    } > "$report_file"
}

# Run main function
main "$@"
