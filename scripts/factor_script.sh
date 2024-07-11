#!/usr/bin/env bash

set -e
set -x

DIR=$(dirname "$(realpath "$0")")

if [ ! -d ${FACTOR_INSTALL} ]; then
  echo "Invalid FACTOR_INSTALL path: ${FACTOR_INSTALL}"
  exit 1
fi

if [ ! -f ${FACTOR_INSTALL}/bin/topsfc ]; then
  echo "Can not find compiler in FACTOR_INSTALL path: ${FACTOR_INSTALL}"
  exit 1
fi

TOPS_BIN_PATH=${FACTOR_INSTALL}/bin
TOPS_INC_PATH=${FACTOR_INSTALL}/include
TOPS_LIB_PATH=${FACTOR_INSTALL}/lib
TOPS_LINK_ARG="-L${TOPS_LIB_PATH} -ltopsrt"

KERNEL_SRC=$1
FACTOR_OBJ=$2
HOST_SRC=$3
BIN_TARGET=$4
GCU_ARCH=$5
GCU_RESOURCE=$6

if [ "$#" -ne 6 ]; then
  echo "invalid parameters."
  exit 1
fi

FACTOR_DIR=$(dirname ${FACTOR_OBJ})

echo "Compile ${BIN_TARGET}"
LD_LIBRARY_PATH=${TOPS_LIB_PATH} ${TOPS_BIN_PATH}/topsfc ${KERNEL_SRC} -gcu-arch=${GCU_ARCH} -resource=${GCU_RESOURCE} -gen-dir=${FACTOR_DIR} -I${TOPS_INC_PATH} -L${TOPS_LIB_PATH} --host-link-options="-L${TOPS_LIB_PATH}"
LD_LIBRARY_PATH=${TOPS_LIB_PATH} ${TOPS_BIN_PATH}/topsfc ${HOST_SRC} ${FACTOR_OBJ} -I${FACTOR_DIR} ${TOPS_LINK_ARG} -o ${BIN_TARGET} -I${TOPS_INC_PATH} -L${TOPS_LIB_PATH} --host-link-options="-L${TOPS_LIB_PATH}"

echo "Run Demo"
LD_LIBRARY_PATH=${TOPS_LIB_PATH} ./${BIN_TARGET}

