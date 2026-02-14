#!/usr/bin/env bash

set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage:
  scripts/run_co_auto_gpu.sh <co_file> [options]

Options:
  --arch <sm_xx>           GPU arch for choreo (default: sm_90a)
  --target <target>        choreo target (default: cute)
  --mode <mode>            generated script action (default: execute)
                           allowed: execute | compile-link | compile-module
  --out <path>             generated script output path
                           (default: /tmp/<basename>.cute.result)
  --gpu <index>            fixed GPU index (skip auto selection)
  --disable-timing         set CHOREO_DISABLE_TIMING=1 when running
  --keep-script            keep generated script file
  -h, --help               show this help

Examples:
  scripts/run_co_auto_gpu.sh tests/gpu/end2end/bench/hgemm_rc_wgmma_tma.co --arch sm_90a --disable-timing
  scripts/run_co_auto_gpu.sh benchmark/performance/hgemm_rc/hgemm_v5_tma_wgmma.co --mode execute --disable-timing
EOF
}

if [[ $# -lt 1 ]]; then
  show_usage
  exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_usage
  exit 0
fi

CO_FILE="$1"
shift

ARCH="sm_90a"
TARGET="cute"
MODE="execute"
OUT_SCRIPT=""
GPU_INDEX=""
DISABLE_TIMING="0"
KEEP_SCRIPT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --out)
      OUT_SCRIPT="$2"
      shift 2
      ;;
    --gpu)
      GPU_INDEX="$2"
      shift 2
      ;;
    --disable-timing)
      DISABLE_TIMING="1"
      shift
      ;;
    --keep-script)
      KEEP_SCRIPT="1"
      shift
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$CO_FILE" ]]; then
  echo "co file not found: $CO_FILE" >&2
  exit 1
fi

if [[ -z "$OUT_SCRIPT" ]]; then
  base_name="$(basename "$CO_FILE" .co)"
  OUT_SCRIPT="/tmp/${base_name}.cute.result"
fi

if [[ "$MODE" != "execute" && "$MODE" != "compile-link" && "$MODE" != "compile-module" ]]; then
  echo "invalid mode: $MODE" >&2
  exit 1
fi

if [[ -z "$GPU_INDEX" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; use --gpu to set device explicitly" >&2
    exit 1
  fi
  GPU_INDEX="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')"
fi

echo "[run_co_auto_gpu] CO_FILE=$CO_FILE"
echo "[run_co_auto_gpu] TARGET=$TARGET ARCH=$ARCH MODE=$MODE"
echo "[run_co_auto_gpu] OUT_SCRIPT=$OUT_SCRIPT"
echo "[run_co_auto_gpu] GPU_INDEX=$GPU_INDEX"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[run_co_auto_gpu] GPU status:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
fi

if [[ "$DISABLE_TIMING" == "1" ]]; then
  CHOREO_DISABLE_TIMING=1 ./choreo -gs -t "$TARGET" -arch="$ARCH" "$CO_FILE" -o "$OUT_SCRIPT"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" CHOREO_DISABLE_TIMING=1 bash "$OUT_SCRIPT" --"$MODE"
else
  ./choreo -gs -t "$TARGET" -arch="$ARCH" "$CO_FILE" -o "$OUT_SCRIPT"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" bash "$OUT_SCRIPT" --"$MODE"
fi

if [[ "$KEEP_SCRIPT" != "1" ]]; then
  rm -f "$OUT_SCRIPT"
fi

echo "[run_co_auto_gpu] done"
