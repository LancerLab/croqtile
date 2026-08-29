#!/bin/bash
# Sweep Triton tile configurations and measure SMEM via ncu
# Each run: ncu measures shared_mem_per_block for the kernel

SCRIPT=/home/wsj/dev/croktile/tools/sala_real_eval/triton_sweep_smem.py
NCU_METRICS="launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block,launch__registers_per_thread,launch__occupancy_limit_shared_mem"

echo "=== Triton SMEM Sweep ==="
echo "Config | WS | SMEM(KB) | DynSMEM(KB) | Regs | OccLimit(SMEM)"
echo "-------|-----|----------|-------------|------|---------------"

# Configs: BM BN BK stages warps
configs=(
  "128 128 64 2 4"
  "128 128 64 3 4"
  "128 128 64 4 4"
  "128 256 64 2 8"
  "128 256 64 3 8"
  "128 256 64 4 8"
  "128 128 128 2 4"
  "128 128 128 3 4"
  "64 128 64 2 4"
  "64 128 64 3 4"
)

for config in "${configs[@]}"; do
  read -r bm bn bk stages warps <<< "$config"
  for ws in 0 1; do
    ws_label="no_WS"
    if [ "$ws" = "1" ]; then ws_label="WS"; fi

    result=$(ncu --metrics $NCU_METRICS \
      --kernel-name regex:'matmul' \
      python3 $SCRIPT $bm $bn $bk $stages $warps $ws 2>&1)

    smem=$(echo "$result" | grep "launch__shared_mem_per_block " | awk '{print $NF}')
    dyn_smem=$(echo "$result" | grep "launch__shared_mem_per_block_dynamic" | awk '{print $NF}')
    regs=$(echo "$result" | grep "launch__registers_per_thread" | awk '{print $NF}')
    occ_smem=$(echo "$result" | grep "launch__occupancy_limit_shared_mem" | awk '{print $NF}')

    echo "${bm}x${bn} BK=${bk} s=${stages} w=${warps} | $ws_label | $smem | $dyn_smem | $regs | $occ_smem"
  done
done
