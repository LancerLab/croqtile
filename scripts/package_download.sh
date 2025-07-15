#!/bin/bash

# Variables for GitLab project and token
VERSION=$(cat VERSION.txt | tr -d '[:space:]')
FILENAME="choreo-${VERSION}-Linux.deb"         # Replace with your actual filename
PROJECT_ID="xiaofeng.guan%2Fchoreo"  # Use URL-encoded project path
TOKEN=$(<.gitlab-install-token.txt)
PKG_PATH="package/"  # Path where packages are stored
VERSION=$(cat VERSION.txt | tr -d '[:space:]')  # Read version and remove spaces

# Construct the download URL
DOWNLOAD_URL="http://git.enflame.cn/api/v4/projects/$PROJECT_ID/packages/generic/choreo-stable-release-for-apex/$VERSION/$FILENAME"

# Use curl to download the file
curl --silent --fail --header "PRIVATE-TOKEN: $TOKEN" "$DOWNLOAD_URL" -o "$FILENAME"

# Check for download success
if [[ $? -eq 0 ]]; then
    echo "✅ Successfully downloaded $FILENAME"
else
    echo "❌ ERROR downloading $FILENAME!"
fi
