#!/bin/bash

PROJECT_ID="xiaofeng.guan%2Fchoreo"  # Replace with your actual project ID
TAG_NAME="v1.0.0"  # Replace with your release tag
PACKAGE_PATH="package/choreo-1.0.0.alpha-Linux.deb"
TOKEN="glpat-xVcdtK_sahpcei86Ey6q"  # Use PAT or CI/CD token

# Upload the .deb package
UPLOAD_RESPONSE=$(curl --header "PRIVATE-TOKEN: $TOKEN" \
  --form "file=@$PACKAGE_PATH" \
  "http://git.enflame.cn/api/v4/projects/$PROJECT_ID/uploads")

# Extract the file URL from the response
echo $UPLOAD_RESPONSE
FILE_URL=$(echo "$UPLOAD_RESPONSE" | jq -r '.url')
echo $FILE_URL

# Create a new GitLab release and attach the .deb package
curl --header "PRIVATE-TOKEN: $TOKEN" \
  --data "name=Release $TAG_NAME" \
  --data "tag_name=$TAG_NAME" \
  --data "description=Automated release for $TAG_NAME" \
  --data "assets[links][0][name]=Debian Package" \
  --data "assets[links][0][url]=http://git.enflame.cn/$PROJECT_ID$FILE_URL" \
  "http://git.enflame.cn/api/v4/projects/$PROJECT_ID/releases"
