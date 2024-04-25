#!/bin/bash

set -e
set -x

source $(dirname "$0")/demo_env_setting.sh

KERNEL_SRC=$KERNEL_PATH"/elementwise_kernel.cc"
FATBIN_TARGET=$KERNEL_PATH"/elementwise_test.fb"
BIN_TARGET=$KERNEL_PATH"/elementwise_test.bin"

docker exec -it "root_dev" bash -c "
    set -e
    set -x
    echo \"Running inside efdocker\"

    echo \"Compile Demo\"
    sudo bash $COMPILE_SHELL $KERNEL_SRC $FATBIN_TARGET $BIN_TARGET

    echo \"Run Demo\"
    $BIN_TARGET $FATBIN_TARGET
"

echo "Exiting efdocker"

