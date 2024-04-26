#!/bin/bash

set -e
set -x

source $(dirname "$0")/demo_env_setting.sh

KERNEL_SRC=$1
FATBIN_TARGET=$2
HOST_SRC=$3
BIN_TARGET=$4

docker exec -it "root_dev" bash -c "
    cd $CHOREO_PATH
    set -e
    set -x
    echo \"Running inside efdocker\"

    echo \"Compile Demo\"
    sudo bash $COMPILE_SHELL $KERNEL_SRC $FATBIN_TARGET $HOST_SRC $BIN_TARGET

    echo \"Run Demo\"
    pwd
    $CHOREO_PATH/demo/$BIN_TARGET $FATBIN_TARGET
"

echo "Exiting efdocker"

