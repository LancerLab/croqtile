#!/bin/bash

# Load project information
echo "hello"
PROJECT_ID="xiaofeng.guan%2Fchoreo"  # Use URL-encoded project path
TOKEN=$(<.gitlab-release-token.txt)
PKG_PATH="package/"  # Path where packages are stored
VERSION=$(cat VERSION.txt | tr -d '[:space:]')  # Read version and remove spaces

# Validate VERSION.txt
if [[ -z "$VERSION" ]]; then
    echo "❌ ERROR: VERSION.txt is empty or missing!"
    exit 1
fi

echo "📦 Detected version: $VERSION"

# Define package types
PKG_TYPES=("deb" "zip" "tar.gz")

# Iterate over package types and upload them
for TYPE in "${PKG_TYPES[@]}"; do
    FILE=$(find "$PKG_PATH" -type f -name "*.$TYPE" | head -n 1)  # Find the first file of each type

    if [[ -f "$FILE" ]]; then
        FILENAME=$(basename "$FILE")

        echo "🚀 Uploading $FILENAME to GitLab Package Registry..."

        UPLOAD_URL="http://git.enflame.cn/api/v4/projects/$PROJECT_ID/packages/generic/choreo-stable-release-for-apex/$VERSION/$FILENAME"

        # Upload file to GitLab
        RESPONSE=$(curl --silent --fail --header "PRIVATE-TOKEN: $TOKEN" \
          --upload-file "$FILE" "$UPLOAD_URL")

        # Check for errors
        if [[ $? -eq 0 ]]; then
            echo "✅ Successfully uploaded $FILENAME"
        else
            echo "❌ ERROR uploading $FILENAME!"
            echo "Response: $RESPONSE"
        fi
    else
        echo "⚠️ WARNING: No .$TYPE package found in $PKG_PATH"
    fi
done

echo "🎉 All package uploads completed!"

