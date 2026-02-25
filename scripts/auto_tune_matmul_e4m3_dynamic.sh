#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CO_FILE_DEFAULT="benchmark/performance/matmul/matmul_e4m3_dynamic.co"
ARCH="sm_90a"
GPU_INDEX=""
WARMUP="5"
REPEAT="20"
OUT_DIR=""

# fixed problem shape per requirement
M_DEFAULT="2048"
N_DEFAULT="2048"
K_DEFAULT="2048"

show_usage() {
  cat <<'EOF'
Usage:
  scripts/auto_tune_matmul_e4m3_dynamic.sh [options]

Options:
  --co <file>           .co source path (default: benchmark/performance/matmul/matmul_e4m3_dynamic.co)
  --arch <sm_xx>        GPU arch for choreo (default: sm_90a)
  --gpu <index>         fixed GPU index (default: auto select least-used GPU)
  --warmup <n>          timing warmup iterations (default: 5)
  --repeat <n>          timing repeat iterations (default: 20)
  --out-dir <dir>       output directory (default: build/auto_tune/matmul_e4m3_dynamic_tune_<ts>)
  -h, --help            show help
EOF
}

CO_FILE="$CO_FILE_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --co) CO_FILE="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --gpu) GPU_INDEX="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) show_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; show_usage; exit 1 ;;
  esac
done

cd "$ROOT_DIR"

if [[ ! -f "$CO_FILE" ]]; then
  echo "co file not found: $CO_FILE" >&2
  exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="build/auto_tune/matmul_e4m3_dynamic_tune_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_DIR"

if [[ -z "$GPU_INDEX" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found, please pass --gpu" >&2
    exit 1
  fi
  GPU_INDEX="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')"
fi

echo "[auto-tune] ROOT_DIR=$ROOT_DIR"
echo "[auto-tune] CO_FILE=$CO_FILE"
echo "[auto-tune] ARCH=$ARCH GPU=$GPU_INDEX"
echo "[auto-tune] SHAPE=M${M_DEFAULT} N${N_DEFAULT} K${K_DEFAULT}"
echo "[auto-tune] WARMUP=$WARMUP REPEAT=$REPEAT"
echo "[auto-tune] OUT_DIR=$OUT_DIR"

RESULT_CSV="$OUT_DIR/tuning_results.csv"
echo "m,n,k,config,tile_m,tile_n,tile_k,swiz,warp_m,warp_n,warp_k,avg_ms,tflops,status" > "$RESULT_CSV"

run_with_retries() {
  local max_attempts="$1"
  local log_file="$2"
  shift 2

  : > "$log_file"
  local attempt
  for ((attempt = 1; attempt <= max_attempts; ++attempt)); do
    echo "[auto-tune] attempt ${attempt}/${max_attempts}: $*" >> "$log_file"
    set +e
    "$@" >> "$log_file" 2>&1
    local status=$?
    set -e
    if (( status == 0 )); then
      return 0
    fi
    echo "[auto-tune] failed attempt ${attempt}/${max_attempts}, exit=${status}" >> "$log_file"
    if (( attempt < max_attempts )); then
      sleep 1
    fi
  done
  return 1
}

extract_avg_ms() {
  local run_log="$1"
  awk '/Timing avg ms:/ {value = $4} END {if (value != "") print value}' "$run_log"
}

calc_tflops() {
  local avg_ms="$1"
  awk -v m="$M_DEFAULT" -v n="$N_DEFAULT" -v k="$K_DEFAULT" -v ms="$avg_ms" 'BEGIN {
    if (ms <= 0) print "";
    else printf "%.6f", (2.0 * m * n * k) / (ms * 1e9);
  }'
}

# Supported FP8 e4m3 WGMMA-like MMA shapes in this kernel family.
# Constraints:
# 1) MATMUL_SWIZ == MATMUL_TILE_K (must be 32/64/128)
# 2) MATMUL_WARP_M == 64, MATMUL_WARP_K == 32 (fixed by the ISA)
# 3) The tuning space should cover all N sizes exposed by the
#    wgmma ISA (every multiple of 8 from 8 up to 256), not just
#    the coarse 64-grain values used previously.
# SWIZ_CANDIDATES=(32 64 128)
SWIZ_CANDIDATES=(128)
WARP_N_CANDIDATES=()
for n in $(seq 8 8 256); do
  # skip the problematic size that causes hangs
  if [[ "$n" -eq 96 ]]; then
    continue
  fi
  WARP_N_CANDIDATES+=("$n")
done

BEST_TFLOPS=""
BEST_MS=""
BEST_NAME=""
BEST_CO=""
BEST_SCRIPT=""
BEST_TILE_M=""
BEST_TILE_N=""
BEST_TILE_K=""
BEST_SWIZ=""
BEST_WARP_M=""
BEST_WARP_N=""
BEST_WARP_K=""

for SWIZ in "${SWIZ_CANDIDATES[@]}"; do
  TILE_K=$SWIZ
  for WARP_N in "${WARP_N_CANDIDATES[@]}"; do
    NAME="w64n${WARP_N}k32_tk${TILE_K}"
    TILE_M=64
    TILE_N=$WARP_N
    WARP_M=64
    WARP_K=32
  VARIANT_CO="$OUT_DIR/matmul_${NAME}.co"
  VARIANT_SCRIPT="$OUT_DIR/matmul_${NAME}.cute.result"
  RUN_LOG="$OUT_DIR/run_${NAME}.log"

  cp "$CO_FILE" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_TILE_M .*/#define MATMUL_TILE_M ${TILE_M}/" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_TILE_N .*/#define MATMUL_TILE_N ${TILE_N}/" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_TILE_K .*/#define MATMUL_TILE_K ${TILE_K}/" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_WARP_M .*/#define MATMUL_WARP_M ${WARP_M}/" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_WARP_N .*/#define MATMUL_WARP_N ${WARP_N}/" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_WARP_K .*/#define MATMUL_WARP_K ${WARP_K}/" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_SWIZ .*/#define MATMUL_SWIZ ${SWIZ}/" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_DEFAULT_M .*/#define MATMUL_DEFAULT_M ${M_DEFAULT}/" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_DEFAULT_N .*/#define MATMUL_DEFAULT_N ${N_DEFAULT}/" "$VARIANT_CO"
  sed -i "s/^#define MATMUL_DEFAULT_K .*/#define MATMUL_DEFAULT_K ${K_DEFAULT}/" "$VARIANT_CO"

  echo "[auto-tune] testing $NAME (TM=$TILE_M,TN=$TILE_N,TK=$TILE_K,SWIZ=$SWIZ, WM=$WARP_M,WN=$WARP_N,WK=$WARP_K)"

  if ! run_with_retries 2 "$OUT_DIR/compile_${NAME}.log" ./choreo -gs -t cute -arch="$ARCH" "$VARIANT_CO" -o "$VARIANT_SCRIPT"; then
    echo "${M_DEFAULT},${N_DEFAULT},${K_DEFAULT},${NAME},${TILE_M},${TILE_N},${TILE_K},${SWIZ},${WARP_M},${WARP_N},${WARP_K},,,compile_failed" >> "$RESULT_CSV"
    continue
  fi

  if ! run_with_retries 2 "$RUN_LOG" env CUDA_VISIBLE_DEVICES="$GPU_INDEX" CHOREO_SKIP_VERIFY=1 CHOREO_TIMING_WARMUP="$WARMUP" CHOREO_TIMING_REPEAT="$REPEAT" bash "$VARIANT_SCRIPT" --execute; then
    echo "${M_DEFAULT},${N_DEFAULT},${K_DEFAULT},${NAME},${TILE_M},${TILE_N},${TILE_K},${SWIZ},${WARP_M},${WARP_N},${WARP_K},,,run_failed" >> "$RESULT_CSV"
    continue
  fi

  AVG_MS="$(extract_avg_ms "$RUN_LOG")"
  if [[ -z "$AVG_MS" ]]; then
    echo "${M_DEFAULT},${N_DEFAULT},${K_DEFAULT},${NAME},${TILE_M},${TILE_N},${TILE_K},${SWIZ},${WARP_M},${WARP_N},${WARP_K},,,timing_not_found" >> "$RESULT_CSV"
    continue
  fi

  TFLOPS="$(calc_tflops "$AVG_MS")"
  echo "${M_DEFAULT},${N_DEFAULT},${K_DEFAULT},${NAME},${TILE_M},${TILE_N},${TILE_K},${SWIZ},${WARP_M},${WARP_N},${WARP_K},${AVG_MS},${TFLOPS},ok" >> "$RESULT_CSV"

  if [[ -n "$TFLOPS" ]] && { [[ -z "$BEST_TFLOPS" ]] || awk "BEGIN {exit !($TFLOPS > $BEST_TFLOPS)}"; }; then
    BEST_TFLOPS="$TFLOPS"
    BEST_MS="$AVG_MS"
    BEST_NAME="$NAME"
    BEST_CO="$VARIANT_CO"
    BEST_SCRIPT="$VARIANT_SCRIPT"
    BEST_TILE_M="$TILE_M"
    BEST_TILE_N="$TILE_N"
    BEST_TILE_K="$TILE_K"
    BEST_SWIZ="$SWIZ"
    BEST_WARP_M="$WARP_M"
    BEST_WARP_N="$WARP_N"
    BEST_WARP_K="$WARP_K"
  fi
  done
done

if [[ -z "$BEST_NAME" ]]; then
  {
    echo "best_m=${M_DEFAULT}"
    echo "best_n=${N_DEFAULT}"
    echo "best_k=${K_DEFAULT}"
    echo "best_config="
    echo "best_tile_m="
    echo "best_tile_n="
    echo "best_tile_k="
    echo "best_swiz="
    echo "best_warp_m="
    echo "best_warp_n="
    echo "best_warp_k="
    echo "best_avg_ms="
    echo "best_tflops="
    echo "best_co="
    echo "status=no_successful_candidate"
  } > "$OUT_DIR/best_config.txt"

  SUMMARY_MD="$OUT_DIR/bottleneck_summary.md"
  {
    echo "# matmul_e4m3_dynamic Auto-tune Summary"
    echo
    echo "- Shape: M=${M_DEFAULT}, N=${N_DEFAULT}, K=${K_DEFAULT}"
    echo "- Best config: N/A"
    echo "- Status: no successful candidate"
    echo
    echo "## Candidate results"
    echo
    cat "$RESULT_CSV"
  } > "$SUMMARY_MD"

  echo "[auto-tune] no successful candidate found" >&2
  echo "[auto-tune] results: $RESULT_CSV" >&2
  echo "[auto-tune] summary: $SUMMARY_MD" >&2
  exit 1
fi

{
  echo "best_m=${M_DEFAULT}"
  echo "best_n=${N_DEFAULT}"
  echo "best_k=${K_DEFAULT}"
  echo "best_config=${BEST_NAME}"
  echo "best_tile_m=${BEST_TILE_M}"
  echo "best_tile_n=${BEST_TILE_N}"
  echo "best_tile_k=${BEST_TILE_K}"
  echo "best_swiz=${BEST_SWIZ}"
  echo "best_warp_m=${BEST_WARP_M}"
  echo "best_warp_n=${BEST_WARP_N}"
  echo "best_warp_k=${BEST_WARP_K}"
  echo "best_avg_ms=${BEST_MS}"
  echo "best_tflops=${BEST_TFLOPS}"
  echo "best_co=${BEST_CO}"
} > "$OUT_DIR/best_config.txt"

echo "[auto-tune] best: ${BEST_NAME}, avg_ms=${BEST_MS}, tflops=${BEST_TFLOPS}"

SUMMARY_MD="$OUT_DIR/bottleneck_summary.md"
{
  echo "# matmul_e4m3_dynamic Auto-tune Summary"
  echo
  echo "- Shape: M=${M_DEFAULT}, N=${N_DEFAULT}, K=${K_DEFAULT}"
  echo "- Best config: ${BEST_NAME}"
  echo "- Best tile: (${BEST_TILE_M}, ${BEST_TILE_N}, ${BEST_TILE_K})"
  echo "- Best swiz: ${BEST_SWIZ}"
  echo "- Best warp: (${BEST_WARP_M}, ${BEST_WARP_N}, ${BEST_WARP_K})"
  echo "- Best avg ms: ${BEST_MS}"
  echo "- Best TFLOPS: ${BEST_TFLOPS}"
  echo
  echo "## Candidate results"
  echo
  cat "$RESULT_CSV"
} > "$SUMMARY_MD"

echo "[auto-tune] done"
echo "[auto-tune] results: $RESULT_CSV"
echo "[auto-tune] summary: $SUMMARY_MD"
