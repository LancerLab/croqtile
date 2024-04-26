#!/bin/bash

set -e
set -x

source $(dirname "$0")/demo_env_setting.sh

_SRC=$1
_FB=$2
_HOST_SRC=$3
_TARGET=$4

cd $KERNEL_PATH

# $TOPS_BIN_PATH
~/vincent/cmake_tops_build/bin/topsfc $_SRC -gcu-arch=gcu210 -resource=2c24s -o $_FB
~/vincent/cmake_tops_build/bin/topsfc $_HOST_SRC $INC_ARG $LINK_ARG -o $_TARGET

