#!/usr/bin/env bash
# Auto-tune MATMUL_TILE_M / MATMUL_TILE_N / MATMUL_TILE_K for
# matmul_f16_dyn_mma.co  (SM86 mma.sync m16n16k16 kernel)
#
# Usage:
#   scripts/auto_tune_matmul_f16_dyn_mma.sh [options]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CO_FILE_DEFAULT="benchmark/performance/matmul/matmul_f16_dyn_mma.co"
ARCH="sm_86"
GPU_INDEX=""
WARMUP="5"
REPEAT="20"
OUT_DIR=""

M_DEFAULT="2048"
N_DEFAULT="2048"
K_DEFAULT="2048"

# MMA atom is always 16x16x16 on SM86
MMA_M=16
MMA_N=16
MMA_K=16

show_usage() {
  cat <<'EOF'
Usage:
  scripts/auto_tune_matmul_f16_dyn_mma.sh [options]

Options:
  --co <file>         .co source path (default: benchmark/performance/matmul/matmul_f16_dyn_mma.co)
  --arch <sm_xx>      GPU arch (default: sm_86)
  --gpu <index>       fixed GPU index (default: auto select least-used)
  --warmup <n>        timing warmup iterations (default: 5)
  --repeat <n>        timing repeat iterations (default: 20)
  --out-dir <dir>     output directory
  -h, --help          show help
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
  OUT_DIR="build/auto_tune/matmul_f16_dyn_mma_tune_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_DIR"

if [[ -z "$GPU_INDEX" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found, please pass --gpu" >&2
    exit 1
  fi
  GPU_INDEX="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')"
fi

echo "[auto-tune] ROOT_DIR=$ROOT_DIR"
echo "[auto-tune] CO_FILE=$CO_FILE"
echo "[auto-tune] ARCH=$ARCH GPU=$GPU_INDEX"
echo "[auto-tune] SHAPE=M${M_DEFAULT} N${N_DEFAULT} K${K_DEFAULT}"
echo "[auto-tune] WARMUP=$WARMUP REPEAT=$REPEAT"
echo "[auto-tune] OUT_DIR=$OUT_DIR"

RESULT_CSV="$OUT_DIR/tuning_results.csv"
echo "tile_m,tile_n,tile_k,warps_m,warps_n,warps_k,avg_ms,tflops,status" > "$RESULT_CSV"

# ── helpers ──────────────────────────────────────────────────────────────────

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
    if (( status == 0 )); then return 0; fi
    echo "[auto-tune] failed attempt ${attempt}/${max_attempts}, exit=${status}" >> "$log_file"
    if (( attempt < max_attempts )); then sleep 1; fi
  done
  return 1
}

extract_avg_ms() {
  awk '/Timing avg ms:/ {value = $4} END {if (value != "") print value}' "$1"
}

calc_tflops() {
  awk -v m="$M_DEFAULT" -v n="$N_DEFAULT" -v k="$K_DEFAULT" -v ms="$1" \
    'BEGIN { if (ms <= 0) print ""; else printf "%.6f", (2.0*m*n*k) / (ms*1e9); }'
}

# ── search space ─────────────────────────────────────────────────────────────
# TILE_M ∈ {16,32,48,64,80,96,112,128,192,256}  (must be multiple of 16)
# TILE_N ∈ {16,32,48,64,80,96,112,128,192,256}
# TILE_K ∈ {16,32,48,64,96,128}

# SM86 register pressure: 255 regs/thread -> max 256 threads (8 warps)
# (TILE_M/16)*(TILE_N/16) <= 8
# Valid combos (warps_m x warps_n): 1x1,1x2,2x1,1x4,4x1,2x2,1x8,8x1,2x4,4x2
TILE_M_CANDIDATES=(16 32 48 64 80 96 128)
TILE_N_CANDIDATES=(16 32 48 64 80 96 128)
TILE_K_CANDIDATES=(16 32 48 64)

BEST_TFLOPS=""
BEST_MS=""
BEST_NAME=""
BEST_TILE_M=""
BEST_TILE_N=""
BEST_TILE_K=""

TOTAL=0
for TM in "${TILE_M_CANDIDATES[@]}"; do
  for TN in "${TILE_N_CANDIDATES[@]}"; do
    for TK in "${TILE_K_CANDIDATES[@]}"; do
      TOTAL=$((TOTAL + 1))
    done
  done
done
echo "[auto-tune] total candidates: $TOTAL"

IDX=0
for TM in "${TILE_M_CANDIDATES[@]}"; do
  for TN in "${TILE_N_CANDIDATES[@]}"; do
    for TK in "${TILE_K_CANDIDATES[@]}"; do
      IDX=$((IDX + 1))

      WARPS_M=$((TM / MMA_M))
      WARPS_N=$((TN / MMA_N))
      WARPS_K=$((TK / MMA_K))

      # Sanity: shared mem per block (bytes) = (TM*TK + TN*TK) * 2
      SMEM=$(( (TM*TK + TN*TK) * 2 ))
      if (( SMEM > 99000 )); then
        echo "[auto-tune] ($IDX/$TOTAL) SKIP tm=${TM} tn=${TN} tk=${TK}  smem=${SMEM} > 99kB"
        echo "${TM},${TN},${TK},${WARPS_M},${WARPS_N},${WARPS_K},,,smem_too_large" >> "$RESULT_CSV"
        continue
      fi

      # SM86: 255 regs/thread -> max 256 threads (8 warps) per block
      TOTAL_WARPS=$(( WARPS_M * WARPS_N ))
      TOTAL_THREADS=$(( TOTAL_WARPS * 32 ))
      if (( TOTAL_WARPS > 8 )); then
        echo "[auto-tune] ($IDX/$TOTAL) SKIP tm=${TM} tn=${TN} tk=${TK}  warps=${TOTAL_WARPS} > 8 (reg pressure)"
        echo "${TM},${TN},${TK},${WARPS_M},${WARPS_N},${WARPS_K},,,too_many_warps" >> "$RESULT_CSV"
        continue
      fi

      # TILE_M and TILE_N must be multiples of 16 (MMA tile)
      if (( TM % MMA_M != 0 )) || (( TN % MMA_N != 0 )) || (( TK % MMA_K != 0 )); then
        echo "[auto-tune] ($IDX/$TOTAL) SKIP tm=${TM} tn=${TN} tk=${TK}  not aligned to MMA tile"
        echo "${TM},${TN},${TK},${WARPS_M},${WARPS_N},${WARPS_K},,,not_mma_aligned" >> "$RESULT_CSV"
        continue
      fi

      NAME="tm${TM}_tn${TN}_tk${TK}"
      VARIANT_CO="$OUT_DIR/matmul_${NAME}.co"
      VARIANT_SCRIPT="$OUT_DIR/matmul_${NAME}.cute.result"
      RUN_LOG="$OUT_DIR/run_${NAME}.log"

      cp "$CO_FILE" "$VARIANT_CO"
      sed -i "s/^#define MATMUL_TILE_M .*/#define MATMUL_TILE_M ${TM}/" "$VARIANT_CO"
      sed -i "s/^#define MATMUL_TILE_N .*/#define MATMUL_TILE_N ${TN}/" "$VARIANT_CO"
      sed -i "s/^#define MATMUL_TILE_K .*/#define MATMUL_TILE_K ${TK}/" "$VARIANT_CO"
      sed -i "s/^#define MATMUL_DEFAULT_M .*/#define MATMUL_DEFAULT_M ${M_DEFAULT}/" "$VARIANT_CO"
      sed -i "s/^#define MATMUL_DEFAULT_N .*/#define MATMUL_DEFAULT_N ${N_DEFAULT}/" "$VARIANT_CO"
      sed -i "s/^#define MATMUL_DEFAULT_K .*/#define MATMUL_DEFAULT_K ${K_DEFAULT}/" "$VARIANT_CO"

      echo -n "[auto-tune] ($IDX/$TOTAL) tm=${TM} tn=${TN} tk=${TK} smem=${SMEM}B ... "

      # compile
      if ! run_with_retries 2 "$OUT_DIR/compile_${NAME}.log" \
            ./choreo -gs -t cute -arch="$ARCH" "$VARIANT_CO" -o "$VARIANT_SCRIPT"; then
        echo "COMPILE FAILED"
        echo "${TM},${TN},${TK},${WARPS_M},${WARPS_N},${WARPS_K},,,compile_failed" >> "$RESULT_CSV"
        continue
      fi

      # correctness check (with verify, no timing, 1 run)
      VERIFY_LOG="$OUT_DIR/verify_${NAME}.log"
      if ! run_with_retries 1 "$VERIFY_LOG" \
            env CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
            bash "$VARIANT_SCRIPT" --execute --disable-timing; then
        echo "VERIFY FAILED"
        echo "${TM},${TN},${TK},${WARPS_M},${WARPS_N},${WARPS_K},,,verify_failed" >> "$RESULT_CSV"
        continue
      fi

      # check "Test Passed" in verify output
      if ! grep -q 'Test Passed' "$VERIFY_LOG"; then
        echo "CORRECTNESS FAILED"
        echo "${TM},${TN},${TK},${WARPS_M},${WARPS_N},${WARPS_K},,,correctness_failed" >> "$RESULT_CSV"
        continue
      fi

      # timing run (skip verify for speed)
      if ! run_with_retries 2 "$RUN_LOG" \
            env CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
                CHOREO_SKIP_VERIFY=1 \
                CHOREO_TIMING_WARMUP="$WARMUP" \
                CHOREO_TIMING_REPEAT="$REPEAT" \
            bash "$VARIANT_SCRIPT" --execute; then
        echo "TIMING RUN FAILED"
        echo "${TM},${TN},${TK},${WARPS_M},${WARPS_N},${WARPS_K},,,timing_run_failed" >> "$RESULT_CSV"
        continue
      fi

      AVG_MS="$(extract_avg_ms "$RUN_LOG")"
      if [[ -z "$AVG_MS" ]]; then
        echo "NO TIMING"
        echo "${TM},${TN},${TK},${WARPS_M},${WARPS_N},${WARPS_K},,,timing_not_found" >> "$RESULT_CSV"
        continue
      fi

      TFLOPS="$(calc_tflops "$AVG_MS")"
      echo "${AVG_MS} ms  ${TFLOPS} TFLOPS"
      echo "${TM},${TN},${TK},${WARPS_M},${WARPS_N},${WARPS_K},${AVG_MS},${TFLOPS},ok" >> "$RESULT_CSV"

      if [[ -n "$TFLOPS" ]] && { [[ -z "$BEST_TFLOPS" ]] || awk "BEGIN {exit !($TFLOPS > $BEST_TFLOPS)}"; }; then
        BEST_TFLOPS="$TFLOPS"
        BEST_MS="$AVG_MS"
        BEST_NAME="$NAME"
        BEST_TILE_M="$TM"
        BEST_TILE_N="$TN"
        BEST_TILE_K="$TK"
      fi

    done
  done
done

# ── summary ──────────────────────────────────────────────────────────────────

if [[ -z "$BEST_NAME" ]]; then
  {
    echo "status=no_successful_candidate"
  } > "$OUT_DIR/best_config.txt"
  echo "[auto-tune] no successful candidate found" >&2
  exit 1
fi

{
  echo "best_tile_m=${BEST_TILE_M}"
  echo "best_tile_n=${BEST_TILE_N}"
  echo "best_tile_k=${BEST_TILE_K}"
  echo "best_avg_ms=${BEST_MS}"
  echo "best_tflops=${BEST_TFLOPS}"
} > "$OUT_DIR/best_config.txt"

echo ""
echo "============================================================"
echo "[auto-tune] BEST: tile=(${BEST_TILE_M}, ${BEST_TILE_N}, ${BEST_TILE_K})  avg_ms=${BEST_MS}  tflops=${BEST_TFLOPS}"
echo "============================================================"

# sort results by tflops descending
SUMMARY_MD="$OUT_DIR/summary.md"
{
  echo "# matmul_f16_dyn_mma Auto-tune Summary"
  echo ""
  echo "- Shape: M=${M_DEFAULT}, N=${N_DEFAULT}, K=${K_DEFAULT}"
  echo "- Arch: ${ARCH}"
  echo "- Best tile: (${BEST_TILE_M}, ${BEST_TILE_N}, ${BEST_TILE_K})"
  echo "- Best avg ms: ${BEST_MS}"
  echo "- Best TFLOPS: ${BEST_TFLOPS}"
  echo ""
  echo "## All results (sorted by TFLOPS desc)"
  echo ""
  head -1 "$RESULT_CSV"
  tail -n +2 "$RESULT_CSV" | grep ',ok$' | sort -t',' -k8 -rn
  echo ""
  echo "## Failed candidates"
  echo ""
  tail -n +2 "$RESULT_CSV" | grep -v ',ok$' || true
} > "$SUMMARY_MD"

echo "[auto-tune] done"
echo "[auto-tune] results: $RESULT_CSV"
echo "[auto-tune] summary: $SUMMARY_MD"
echo "[auto-tune] best config: $OUT_DIR/best_config.txt"
