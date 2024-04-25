#!/bin/bash

set -e
set -x
CHOREO_PATH=~/choreo
TOPS_PATH=~/vincent/tops
TOPS_BIN_PATH=$TOPS_PATH"/cmake_tops_build/bin"

# KERNEL_PATH=$TOPS_PATH"/factor/tools/topsfc/samples/jit"
KERNEL_PATH=$CHOREO_PATH"/demo"
KERNEL_SRC="elementwise_kernel.cc"

cd $KERNEL_PATH

#efdocker run

# efdocker enter

docker exec -it "root_dev" bash -c '
    set -e
    set -x
    echo "Running inside efdocker"
    CHOREO_PATH=~/choreo
    TOPS_PATH=~/vincent/tops
    TOPS_BIN_PATH=$TOPS_PATH"/../cmake_tops_build/bin"
    KERNEL_PATH=$CHOREO_PATH"/demo"
    KERNEL_SRC="elementwise_kernel.cc"
    INC_ARG="-I/opt/tops/include"
    LINK_ARG="-L/opt/tops/lib -ltopsrt"
    FATBIN_TARGET="elementwise_test.fb"
    BIN_TARGET="elementwise_test.bin"
    cd $KERNEL_PATH

    echo "Compile Demo"
    $TOPS_BIN_PATH/"topsfc" $KERNEL_PATH"/"$KERNEL_SRC -gcu-arch=gcu210 -resource=2c24s -o $FATBIN_TARGET
    $TOPS_BIN_PATH/"topsfc" $KERNEL_PATH"/elementwise_test_main.cc" $INC_ARG $LINK_ARG -o $BIN_TARGET

    echo "Run Demo"
    pwd
    $KERNEL_PATH"/"$BIN_TARGET $KERNEL_PATH"/"$FATBIN_TARGET
'

echo "Exiting efdocker"

# exit

cd -

#efdocker rm
