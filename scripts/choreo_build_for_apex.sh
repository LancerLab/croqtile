#!/bin/bash

# Set Python version
PYTHON=python3.10

# Function to check if choreo is installed
check_choreo_installed() {
    if command -v choreo &> /dev/null; then
        echo "✅ Choreo is already installed."
        return 0  # Choreo is installed
    else
        echo "❌ Choreo is not installed."
        return 1  # Choreo is not installed
    fi
}

# Function to download and install choreo
install_choreo() {
    local FILENAME="choreo-1.0.0.alpha-Linux.deb"         # Replace with your actual filename
    local PROJECT_ID="xiaofeng.guan%2Fchoreo"  # Use URL-encoded project path
    local TOKEN=$(<.gitlab-install-token.txt)
    local PKG_PATH="package/"  # Path where packages are stored
    local VERSION="1.0.0.alpha"  # Read version and remove spaces
    local checksum="f1b70b17b30efcb37bbd2a084e4a11ececd73e8a2f80c4aed8f1a88bd025ffe1"
    
    # Construct the download URL
    DOWNLOAD_URL="http://git.enflame.cn/api/v4/projects/$PROJECT_ID/packages/generic/choreo-stable-release-for-apex/$VERSION/$FILENAME"
    
    # Use curl to download the file
    curl --fail --header "PRIVATE-TOKEN: $TOKEN" "$DOWNLOAD_URL" -o "$FILENAME"
    
    # Check for download success
    if [[ $? -eq 0 ]]; then
        echo "✅ Successfully downloaded $FILENAME"
    else
        echo "❌ ERROR downloading $FILENAME!"
    fi

    # Verify the checksum
    echo "🔍 Verifying SHA-256 checksum..."
    local calculated_checksum
    calculated_checksum=$(sha256sum "$FILENAME" | awk '{ print $1 }')

    if [[ "$calculated_checksum" != "$checksum" ]]; then
        echo "❌ Checksum verification failed. Expected $checksum but got $calculated_checksum."
        rm "$FILENAME"
        exit 1
    else
        echo "✅ Checksum verified successfully."
    fi

    # Install the package using dpkg
    echo "🔄 Installing Choreo package..."
    sudo dpkg -i "$FILENAME"
    rm "$FILENAME"

    # Fix any missing dependencies
    echo "🔄 Fixing missing dependencies..."
    sudo apt-get install -f -y
}

# Check if Choreo is installed
if ! check_choreo_installed; then
    install_choreo
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$SOURCE_DIR/apex/dist"
TEST_DIR="$(cd "$SOURCE_DIR/../tests/apex/daily_test" && pwd)"

echo $PKG_DIR

# Function to find the latest .whl file in a directory
find_latest_whl() {
    local dir=$1
    local whl_file
    whl_file=$(ls -t "$dir"/*.whl 2>/dev/null | head -n 1)
    if [[ -z "$whl_file" ]]; then
        echo "⚠️  No .whl file found in $dir. Skipping."
        return 1  # Continue execution instead of exiting
    fi
    echo "$whl_file"
}

# Check for --force flag
FORCE_BUILD=false
if [[ "$1" == "--force" ]]; then
    FORCE_BUILD=true
fi

### ---- APEX MODULE ---- ###
# Check if Apex is installed
APEX_INSTALLED=false
if ${PYTHON} -c "import importlib.util; print(importlib.util.find_spec('apex') is not None)" | grep -q "True"; then
    echo "✅ Apex is already installed."
    APEX_INSTALLED=true
fi

# Build Apex if missing or --force is used
if [[ "$FORCE_BUILD" == true || "$APEX_INSTALLED" == false ]]; then
    echo "⚠️  Building Apex..."
    cd $SOURCE_DIR || { echo "❌ Directory bigmodel/apex not found!"; exit 1; }
    bash $SOURCE_DIR/build_apex.sh clean
    bash $SOURCE_DIR/build_apex.sh all

    # Install Apex
    APEX_WHL=$(find_latest_whl $PKG_DIR) || APEX_WHL=""
    if [[ -n "$APEX_WHL" ]]; then
        echo "🔄 Installing Apex from $APEX_WHL..."
        ${PYTHON} -m pip uninstall apex -y
        ${PYTHON} -m pip install "$APEX_WHL" --no-input
    else
        echo "⚠️  No valid Apex .whl file found. Skipping installation."
    fi
    cd - || exit 1
fi


# ### ---- TRANSFER-TO-GCU MODULE ---- ###
# # Check if Transfer-to-GCU is installed
# TTG_INSTALLED=false
# if ${PYTHON} -c "import importlib.util; print(importlib.util.find_spec('transfer_to_gcu') is not None)" | grep -q "True"; then
#     echo "✅ Transfer-to-GCU is already installed."
#     TTG_INSTALLED=true
# fi
# echo "here"
# 
# # Build Transfer-to-GCU if missing or --force is used
# if [[ "$FORCE_BUILD" == true || "$TTG_INSTALLED" == false ]]; then
#     echo "⚠️  Building Transfer-to-GCU..."
#     cd bigmodel/transfer_to_gcu || { echo "❌ Directory bigmodel/transfer_to_gcu not found!"; exit 1; }
#     ./build_t2g.sh clean
#     ./build_t2g.sh all
#     cd - || exit 1
# fi
# 
# # Install Transfer-to-GCU
# TTT_WHL=$(find_latest_whl "bigmodel/transfer_to_gcu/dist") || TTT_WHL=""
# if [[ -n "$TTT_WHL" ]]; then
#     echo "🔄 Installing Transfer-to-GCU from $TTT_WHL..."
#     ${PYTHON} -m pip uninstall transfer_to_gcu -y
#     ${PYTHON} -m pip install "$TTT_WHL" --no-input
# else
#     echo "⚠️  No valid Transfer-to-GCU .whl file found. Skipping installation."
# fi

echo "✅ All installations complete!"

############### TESTS #################
${PYTHON} -m pytest $TEST_DIR/test_choreo_smoke_test.py
