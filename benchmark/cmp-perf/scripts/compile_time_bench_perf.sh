#!/bin/bash
set -e

CHOREO_BIN="./choreo"
PERF_DIR="benchmark/performance"
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

detect_compile_flags() {
    local file=$1
    local flags="-es"

    local run_line=$(grep -m1 "^// RUN:" "$file" 2>/dev/null || true)
    if [ -n "$run_line" ]; then
        if echo "$run_line" | grep -q "\-t cute"; then
            flags="$flags -t cute"
        fi
        if echo "$run_line" | grep -q "\-arch=sm_90a"; then
            flags="$flags -arch=sm_90a"
        elif echo "$run_line" | grep -q "\-arch=sm_86"; then
            flags="$flags -arch=sm_86"
        fi
        if echo "$run_line" | grep -q "\-\-use-prepack-v2"; then
            flags="$flags --use-prepack-v2"
        elif echo "$run_line" | grep -q "\-\-use-prepack"; then
            flags="$flags --use-prepack"
        fi
        if echo "$run_line" | grep -q "\-\-stmatrix"; then
            flags="$flags --stmatrix"
        fi
    fi

    local name=$(basename "$file")
    if echo "$name" | grep -q "sm90\|sm_90\|wgmma\|tma\|warpspec\|tbc"; then
        echo "$flags" | grep -q "\-t cute" || flags="$flags -t cute"
        echo "$flags" | grep -q "\-arch" || flags="$flags -arch=sm_90a"
    elif echo "$name" | grep -q "sm86\|sm_86\|mma"; then
        echo "$flags" | grep -q "\-t cute" || flags="$flags -t cute"
        echo "$flags" | grep -q "\-arch" || flags="$flags -arch=sm_86"
    fi

    echo "$flags"
}

measure_compile_time() {
    local src=$1 flags=$2
    local times=()
    local total=0

    for i in $(seq 1 $MEASURES); do
        local start=$(date +%s%N)
        $CHOREO_BIN $flags "$src" -o "$TEMP_DIR/out_$$" >/dev/null 2>&1
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
    local src=$1 flags=$2
    $CHOREO_BIN $flags -tp "$src" -o "$TEMP_DIR/out_$$" 2>&1 | \
        grep -E '^\s+[0-9]' | awk '{print $1, $3}'
}

main() {
    print_msg $CYAN "================================================="
    print_msg $CYAN " Choreo Compile Time: benchmark/performance files"
    print_msg $CYAN "================================================="
    print_msg $YELLOW "Measures per config: $MEASURES"
    echo ""

    echo "category,workload,compile_ms,compile_std,flags" > "$RESULTS_DIR/perf_compile_time.csv"
    echo "category,workload,pass,time_ms" > "$RESULTS_DIR/perf_phase_timing.csv"

    local skipped=0
    local compiled=0

    for cat_dir in "$PERF_DIR"/*/; do
        local cat=$(basename "$cat_dir")

        local co_files=$(find "$cat_dir" -maxdepth 1 -name "*.co" -type f 2>/dev/null | sort)
        [ -z "$co_files" ] && continue

        print_msg $PURPLE "[$cat]"

        while read -r file; do
            local name=$(basename "$file" .co)
            local flags=$(detect_compile_flags "$file")

            if ! $CHOREO_BIN $flags "$file" -o "$TEMP_DIR/out_$$" >/dev/null 2>&1; then
                printf "  %-60s SKIP\n" "$name"
                skipped=$((skipped + 1))
                continue
            fi

            local result=$(measure_compile_time "$file" "$flags")
            local comp_time=$(echo $result | cut -d' ' -f1)
            local comp_std=$(echo $result | cut -d' ' -f2)

            echo "$cat,$name,$comp_time,$comp_std,$flags" >> "$RESULTS_DIR/perf_compile_time.csv"

            local phases=$(get_phase_timing "$file" "$flags")
            while IFS= read -r line; do
                [ -z "$line" ] && continue
                local time_ms=$(echo "$line" | awk '{print $1}')
                local pass=$(echo "$line" | awk '{print $2}')
                echo "$cat,$name,$pass,$time_ms" >> "$RESULTS_DIR/perf_phase_timing.csv"
            done <<< "$phases"

            printf "  %-60s %6dms (+/-%d)\n" "$name" $comp_time $comp_std
            compiled=$((compiled + 1))
        done <<< "$co_files"
    done

    echo ""
    print_msg $CYAN "================================================="
    print_msg $CYAN " Results Summary"
    print_msg $CYAN "================================================="

    local results="$RESULTS_DIR/perf_compile_time.csv"
    local total_cases=$(tail -n +2 "$results" | wc -l)

    printf "\n  Compiled: %d, Skipped: %d\n" $compiled $skipped

    if [ $total_cases -eq 0 ]; then
        print_msg $RED "No compilable results."
        return
    fi

    local avg_time=$(tail -n +2 "$results" | cut -d',' -f3 | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
    printf "  Avg compile time: %.1fms\n\n" $avg_time

    print_msg $YELLOW "Per-category breakdown:"
    printf "  %-25s %8s %8s %8s %6s\n" "Category" "Avg" "Min" "Max" "Cases"
    printf "  %-25s %8s %8s %8s %6s\n" "--------" "---" "---" "---" "-----"

    tail -n +2 "$results" | cut -d',' -f1 | sort -u | while read -r cat; do
        local cat_data=$(tail -n +2 "$results" | grep "^$cat,")
        local cnt=$(echo "$cat_data" | wc -l)
        local avg=$(echo "$cat_data" | cut -d',' -f3 | awk '{sum+=$1} END {printf "%.0f", sum/NR}')
        local min=$(echo "$cat_data" | cut -d',' -f3 | sort -n | head -1)
        local max=$(echo "$cat_data" | cut -d',' -f3 | sort -rn | head -1)
        printf "  %-25s %7sms %7sms %7sms %6d\n" "$cat" "$avg" "$min" "$max" "$cnt"
    done

    echo ""
    print_msg $YELLOW "Top 10 slowest compiles:"
    printf "  %-60s %8s\n" "Workload" "Time"
    printf "  %-60s %8s\n" "--------" "----"
    tail -n +2 "$results" | sort -t',' -k3,3rn | head -10 | while IFS=',' read -r cat work t s f; do
        printf "  %-60s %7sms\n" "${cat}/${work}" "$t"
    done

    echo ""
    print_msg $YELLOW "Phase timing summary (aggregate by pass):"
    local pt="$RESULTS_DIR/perf_phase_timing.csv"
    if [ -f "$pt" ] && [ $(wc -l < "$pt") -gt 1 ]; then
        printf "  %-15s %10s %10s\n" "Pass" "Avg Time" "Max Time"
        printf "  %-15s %10s %10s\n" "----" "--------" "--------"
        tail -n +2 "$pt" | awk -F',' '{
            pass=$3; t=$4+0;
            sum[pass]+=t; cnt[pass]++;
            if (t > max[pass]) max[pass]=t
        } END {
            for (p in sum) {
                avg=sum[p]/cnt[p];
                printf "  %-15s %9.2fms %9.2fms\n", p, avg, max[p]
            }
        }' | sort -k2 -rn
    fi

    print_msg $GREEN "\nDone. CSV results in: $RESULTS_DIR/"
}

main "$@"
