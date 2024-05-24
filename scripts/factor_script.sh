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
FATBIN_TARGET=$2
HOST_SRC=$3
BIN_TARGET=$4

echo "Compile ${BIN_TARGET}"
${TOPS_BIN_PATH}/topsfc ${KERNEL_SRC} -gcu-arch=gcu210 -resource=2c24s -o ${FATBIN_TARGET}
${TOPS_BIN_PATH}/topsfc ${HOST_SRC} ${TOPS_INC_PATH} ${TOPS_LINK_ARG} -o ${BIN_TARGET}

echo "Run Demo"
LD_LIBRARY_PATH=${TOPS_LIB_PATH} ./${BIN_TARGET} ${FATBIN_TARGET}

