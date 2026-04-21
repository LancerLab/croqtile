#!/bin/bash

CHOREO_PATH=$(dirname "$0")/../
TOOL_PATH=${CHOREO_PATH}/tools
TOPS_BIN_PATH=${TOOL_PATH}/bin
INC_ARG=${TOOL_PATH}/include
LINK_ARG="-L${TOOL_PATH}/lib -ltopsrt"

KERNEL_PATH=${PWD}

