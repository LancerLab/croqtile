#!/usr/bin/env bash

set -e
set -x

DIR=$(dirname "$(realpath "$0")")

CUDA_SYS_INCLUDES="-I/usr/local/cuda/include"
CUDA_CC="sm_35"
CUDA_ARCH="compute_35"

# TODO, device id
GPU_CC=$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader)

case "${GPU_CC}" in
    3.0)
        CUDA_ARCH="compute_30"
        CUDA_CC="sm_30"
        ;;
    3.5)
        CUDA_ARCH="compute_35"
        CUDA_CC="sm_35"
        ;;
    3.7)
        CUDA_ARCH="compute_37"
        CUDA_CC="sm_37"
        ;;
    5.0)
        CUDA_ARCH="compute_50"
        CUDA_CC="sm_50"
        ;;
    5.2)
        CUDA_ARCH="compute_52"
        CUDA_CC="sm_52"
        ;;
    5.3)
        CUDA_ARCH="compute_53"
        CUDA_CC="sm_53"
        ;;
    6.0)
        CUDA_ARCH="compute_60"
        CUDA_CC="sm_60"
        ;;
    6.1)
        CUDA_ARCH="compute_61"
        CUDA_CC="sm_61"
        ;;
    6.2)
        CUDA_ARCH="compute_62"
        CUDA_CC="sm_62"
        ;;
    7.0)
        CUDA_ARCH="compute_70"
        CUDA_CC="sm_70"
        ;;
    7.2)
        CUDA_ARCH="compute_72"
        CUDA_CC="sm_72"
        ;;
    7.5)
        CUDA_ARCH="compute_75"
        CUDA_CC="sm_75"
        ;;
    8.0)
        CUDA_ARCH="compute_80"
        CUDA_CC="sm_80"
        ;;
    8.6)
        CUDA_ARCH="compute_86"
        CUDA_CC="sm_86"
        ;;
    8.9)
        CUDA_ARCH="compute_89"
        CUDA_CC="sm_89"
        ;;
    9.0)
        CUDA_ARCH="compute_90"
        CUDA_CC="sm_90"
        ;;
    *)
        echo "Unsupported GPU compute capability: ${GPU_CC}"
        exit 1
        ;;
esac

echo "CUDA_ARCH: ${CUDA_ARCH}"
echo "CUDA_CC: ${CUDA_CC}"

HOST_DIR=$1
HOST_SRC=$2
BIN_TARGET=$3
# CUDA_CHOREO_INCLUDES="-I./demos/cuda/sgemm_ref/"
CUDA_INCLUDES="${CUDA_SYS_INCLUDES} -I${HOST_DIR}"

if [ "$#" -ne 3 ]; then
  echo "invalid parameters."
  exit 1
fi

echo "Compile ${BIN_TARGET}"
# LD_LIBRARY_PATH=${TOPS_LIB_PATH} ${TOPS_BIN_PATH}/topsfc ${KERNEL_SRC} -gcu-arch=${GCU_ARCH} -resource=${GCU_RESOURCE} -gen-dir=${FACTOR_DIR} -I${TOPS_INC_PATH} -L${TOPS_LIB_PATH} --host-link-options="-L${TOPS_LIB_PATH}"
# LD_LIBRARY_PATH=${TOPS_LIB_PATH} ${TOPS_BIN_PATH}/topsfc ${HOST_SRC} ${FACTOR_OBJ} -I${FACTOR_DIR} ${TOPS_LINK_ARG} -o ${BIN_TARGET} -I${TOPS_INC_PATH} -L${TOPS_LIB_PATH} --host-link-options="-L${TOPS_LIB_PATH}"
nvcc ${CUDA_INCLUDES} -o ${BIN_TARGET} ${HOST_SRC} -gencode arch=${CUDA_ARCH},code=${CUDA_CC} -rdc=true -lcublas

echo "Run Demo"
./${BIN_TARGET}

