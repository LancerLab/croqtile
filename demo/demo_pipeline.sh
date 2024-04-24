#!/bin/bash

set -e
set -x
TOPS_PATH=~/vincent/tops
TOPS_BIN_PATH=$TOPS_PATH"/cmake_tops_build/bin"

KERNEL_PATH=$TOPS_PATH"/factor/tools/topsfc/samples/jit"
KERNEL_SRC="foo.cc"

cd $KERNEL_PATH

#efdocker run

# efdocker enter

docker exec -it "root_dev" bash -c '
    echo "Running inside efdocker"
    TOPS_PATH=~/vincent/tops
    TOPS_BIN_PATH=$TOPS_PATH"/../cmake_tops_build/bin"
    KERNEL_PATH=$TOPS_PATH"/factor/tools/topsfc/samples/jit"
    KERNEL_SRC="foo.cc"
    INC_ARG="-I/opt/tops/include"
    LINK_ARG="-L/opt/tops/lib -ltopsrt"
    BIN_TARGET="elementwise_test"
    cd $KERNEL_PATH

    echo "Compile Demo"
    $TOPS_BIN_PATH/"topsfc" $KERNEL_PATH"/"$KERNEL_SRC -gcu-arch=gcu210 -resource=2c24s -o $BIN_TARGET".fb"
    $TOPS_BIN_PATH/"topsfc" $KERNEL_PATH"/main.cc" $INC_ARG $LINK_ARG -o $BIN_TARGET

    echo "Run Demo"
    pwd
    $KERNEL_PATH"/"$BIN_TARGET $KERNEL_PATH"/"$BIN_TARGET".fb"
'

echo "Exiting efdocker"

# exit

cd -

#efdocker rm
