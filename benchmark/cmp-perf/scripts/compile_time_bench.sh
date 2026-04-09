#!/bin/bash
set -e

CHOREO_BIN="./choreo"
BENCHMARK_DIR="benchmark/cmp-perf"
RESULTS_DIR="benchmark/cmp-perf/scripts/results"
TEMP_DIR=$(mktemp -d)
MEASURES=5

trap "rm -rf $TEMP_DIR" EXIT

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

mkdir -p "$RESULTS_DIR"

print_msg() { echo -e "${1}${2}${NC}"; }

measure_compile_time() {
    local src=$1 extra_flags=$2
    local times=()
    local total=0

    for i in $(seq 1 $MEASURES); do
        local start=$(date +%s%N)
        $CHOREO_BIN $extra_flags -es "$src" -o "$TEMP_DIR/out_$$" >/dev/null 2>&1
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
    local std=$(echo "sqrt($var)" | bc -l 2>/dev/null | cut -d. -f1)
    [ -z "$std" ] && std=0

    echo "$avg $std"
}

get_phase_timing() {
    local src=$1 extra_flags=$2
    $CHOREO_BIN $extra_flags -tp -es "$src" -o "$TEMP_DIR/out_$$" 2>&1 | \
        grep -E '^\s+[0-9]' | awk '{print $1, $3}'
}

process_file() {
    local file=$1 cat=$2
    local name=$(basename "$file" .co)

    if ! grep -q "#ifdef __STATIC_SHAPE__" "$file"; then
        return 0
    fi

    print_msg $BLUE "  $name"

    local static_result=$(measure_compile_time "$file" "-D__STATIC_SHAPE__=1")
    local static_time=$(echo $static_result | cut -d' ' -f1)
    local static_std=$(echo $static_result | cut -d' ' -f2)

    local dynamic_result=$(measure_compile_time "$file" "")
    local dynamic_time=$(echo $dynamic_result | cut -d' ' -f1)
    local dynamic_std=$(echo $dynamic_result | cut -d' ' -f2)

    local delta=$((dynamic_time - static_time))
    local pct=0
    if [ $static_time -ne 0 ]; then
        pct=$(echo "scale=1; $delta * 100.0 / $static_time" | bc -l)
    fi

    echo "$cat,$name,$static_time,$static_std,$dynamic_time,$dynamic_std,$delta,$pct" >> "$RESULTS_DIR/compile_time_results.csv"

    printf "    Static: %dms (+/-%d)  Dynamic: %dms (+/-%d)  Delta: %+dms (%s%%)\n" \
        $static_time $static_std $dynamic_time $dynamic_std $delta "$pct"
}

process_phase_timing() {
    local file=$1 cat=$2
    local name=$(basename "$file" .co)

    if ! grep -q "#ifdef __STATIC_SHAPE__" "$file"; then
        return 0
    fi

    local static_phases=$(get_phase_timing "$file" "-D__STATIC_SHAPE__=1")
    local dynamic_phases=$(get_phase_timing "$file" "")

    while IFS= read -r line; do
        local time_ms=$(echo "$line" | awk '{print $1}')
        local pass=$(echo "$line" | awk '{print $2}')
        local dyn_time=$(echo "$dynamic_phases" | grep " ${pass}$" | awk '{print $1}' | tail -1)
        [ -z "$dyn_time" ] && dyn_time="0.00"
        echo "$cat,$name,$pass,$time_ms,$dyn_time" >> "$RESULTS_DIR/phase_timing_results.csv"
    done <<< "$static_phases"
}

main() {
    print_msg $CYAN "=============================================="
    print_msg $CYAN " Choreo Compilation Time: Static vs Dynamic"
    print_msg $CYAN "=============================================="
    print_msg $YELLOW "Measures per config: $MEASURES"
    print_msg $YELLOW "Choreo binary: $CHOREO_BIN"
    echo ""

    echo "category,workload,static_ms,static_std,dynamic_ms,dynamic_std,delta_ms,delta_pct" > "$RESULTS_DIR/compile_time_results.csv"
    echo "category,workload,pass,static_ms,dynamic_ms" > "$RESULTS_DIR/phase_timing_results.csv"

    local categories="batch_norm concat conv2d elemwise_add embedding gelu layer_normalization matmul max_pool2d reduce_mean relu reshape sigmoid softmax transpose"

    for cat in $categories; do
        local cat_dir="$BENCHMARK_DIR/$cat"
        if [ ! -d "$cat_dir" ]; then
            print_msg $RED "Missing: $cat_dir"
            continue
        fi

        print_msg $PURPLE "[$cat]"

        find "$cat_dir" -name "*.co" -type f | sort | while read -r file; do
            process_file "$file" "$cat"
        done
    done

    echo ""
    print_msg $CYAN "Phase timing analysis (single run per file)..."

    for cat in $categories; do
        local cat_dir="$BENCHMARK_DIR/$cat"
        [ ! -d "$cat_dir" ] && continue

        find "$cat_dir" -name "*.co" -type f | sort | while read -r file; do
            process_phase_timing "$file" "$cat"
        done
    done

    echo ""
    print_msg $CYAN "=============================================="
    print_msg $CYAN " Results Summary"
    print_msg $CYAN "=============================================="

    local results="$RESULTS_DIR/compile_time_results.csv"
    local total_cases=$(tail -n +2 "$results" | wc -l)
    if [ $total_cases -eq 0 ]; then
        print_msg $RED "No results."
        return
    fi

    local avg_static=$(tail -n +2 "$results" | cut -d',' -f3 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
    local avg_dynamic=$(tail -n +2 "$results" | cut -d',' -f5 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
    local avg_delta=$(tail -n +2 "$results" | cut -d',' -f7 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
    local avg_pct=$(tail -n +2 "$results" | cut -d',' -f8 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')

    printf "\n  Total cases: %d\n" $total_cases
    printf "  Avg static compile:  %.1fms\n" $avg_static
    printf "  Avg dynamic compile: %.1fms\n" $avg_dynamic
    printf "  Avg delta:           %+.1fms (%+.1f%%)\n\n" $avg_delta $avg_pct

    print_msg $YELLOW "Per-category breakdown:"
    printf "  %-20s %8s %8s %8s %8s %6s\n" "Category" "Static" "Dynamic" "Delta" "Pct" "Cases"
    printf "  %-20s %8s %8s %8s %8s %6s\n" "--------" "------" "-------" "-----" "---" "-----"

    tail -n +2 "$results" | cut -d',' -f1 | sort -u | while read -r cat; do
        local cat_data=$(tail -n +2 "$results" | grep "^$cat,")
        local cnt=$(echo "$cat_data" | wc -l)
        local s=$(echo "$cat_data" | cut -d',' -f3 | awk '{sum+=$1} END {printf "%.0f", sum/NR}')
        local d=$(echo "$cat_data" | cut -d',' -f5 | awk '{sum+=$1} END {printf "%.0f", sum/NR}')
        local dt=$(echo "$cat_data" | cut -d',' -f7 | awk '{sum+=$1} END {printf "%.0f", sum/NR}')
        local dp=$(echo "$cat_data" | cut -d',' -f8 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        printf "  %-20s %7sms %7sms %+7sms %+7s%% %6d\n" "$cat" "$s" "$d" "$dt" "$dp" "$cnt"
    done

    echo ""
    print_msg $YELLOW "Top 10 highest delta (dynamic overhead):"
    printf "  %-40s %8s %8s %8s %8s\n" "Workload" "Static" "Dynamic" "Delta" "Pct"
    printf "  %-40s %8s %8s %8s %8s\n" "--------" "------" "-------" "-----" "---"
    tail -n +2 "$results" | sort -t',' -k7,7rn | head -10 | while IFS=',' read -r cat work s ss d ds dt dp; do
        printf "  %-40s %7sms %7sms %+7sms %+7s%%\n" "${cat}/${work}" "$s" "$d" "$dt" "$dp"
    done

    echo ""
    print_msg $YELLOW "Phase timing summary (aggregate by pass):"
    local pt="$RESULTS_DIR/phase_timing_results.csv"
    if [ -f "$pt" ] && [ $(wc -l < "$pt") -gt 1 ]; then
        printf "  %-15s %10s %10s %10s\n" "Pass" "Avg Static" "Avg Dynamic" "Avg Delta"
        printf "  %-15s %10s %10s %10s\n" "----" "----------" "-----------" "---------"
        tail -n +2 "$pt" | awk -F',' '{
            pass=$3; s=$4+0; d=$5+0;
            sum_s[pass]+=s; sum_d[pass]+=d; cnt[pass]++
        } END {
            for (p in sum_s) {
                avg_s=sum_s[p]/cnt[p]; avg_d=sum_d[p]/cnt[p];
                printf "  %-15s %9.2fms %9.2fms %+9.2fms\n", p, avg_s, avg_d, avg_d-avg_s
            }
        }' | sort -k4 -rn
    fi

    print_msg $GREEN "\nDone. CSV results in: $RESULTS_DIR/"
}

main "$@"
