#!/bin/bash
#
# Build the Choreo SDK .deb package and upload it to the FTP server.
#
# Usage:
#   ./scripts/sdk/publish-sdk.sh              # build + upload
#   ./scripts/sdk/publish-sdk.sh --build-only # build only, skip upload
#   ./scripts/sdk/publish-sdk.sh --dry-run    # build + show what would be uploaded
#
# After uploading, update depends/Makefile in the CoIR repo with the new MD5.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHOREO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

FTP_SERVER="172.16.11.18"
FTP_USER="ftp_era"
FTP_PASS="Enflame@321"
FTP_DIR="/%2fdev/choreo-sdk"

BUILD_ONLY=false
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --build-only) BUILD_ONLY=true ;;
    --dry-run)    DRY_RUN=true ;;
    -h|--help)
      echo "Usage: $0 [--build-only] [--dry-run]"
      echo "  --build-only  Build the SDK package but don't upload"
      echo "  --dry-run     Build and show upload command without executing"
      exit 0
      ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

echo "=== Building Choreo SDK package ==="
cd "$CHOREO_ROOT"
make sdk-package

VERSION=$(tr -d '[:space:]' < "$CHOREO_ROOT/VERSION.txt")
DEB_NAME="choreo-dev-${VERSION}-Linux.deb"
DEB_PATH="$CHOREO_ROOT/build-release/package/$DEB_NAME"

if [ ! -f "$DEB_PATH" ]; then
  echo "ERROR: Expected package not found: $DEB_PATH"
  exit 1
fi

NEW_MD5=$(md5sum "$DEB_PATH" | cut -d ' ' -f 1)
SIZE=$(du -h "$DEB_PATH" | cut -f1)

echo ""
echo "=== SDK Package Ready ==="
echo "  File: $DEB_PATH"
echo "  Size: $SIZE"
echo "  MD5:  $NEW_MD5"

if $BUILD_ONLY; then
  echo ""
  echo "Build-only mode. To upload manually:"
  echo "  curl -u $FTP_USER:$FTP_PASS -T $DEB_PATH ftp://$FTP_SERVER/$FTP_DIR/$DEB_NAME"
  echo ""
  echo "Then update CoIR depends/Makefile:"
  echo "  CHOREO_SDK_PKG_MD5:=$NEW_MD5"
  exit 0
fi

FTP_URL="ftp://$FTP_SERVER/$FTP_DIR/$DEB_NAME"

if $DRY_RUN; then
  echo ""
  echo "Dry-run mode. Would execute:"
  echo "  curl -u $FTP_USER:$FTP_PASS -T $DEB_PATH $FTP_URL"
  echo ""
  echo "Then update CoIR depends/Makefile:"
  echo "  CHOREO_SDK_PKG_MD5:=$NEW_MD5"
  exit 0
fi

echo ""
echo "=== Uploading to FTP ==="
curl -u "$FTP_USER:$FTP_PASS" -T "$DEB_PATH" "$FTP_URL"

echo ""
echo "=== Upload complete ==="
echo ""
echo "Next step: update CoIR depends/Makefile with the new MD5:"
echo ""
echo "  CHOREO_SDK_PKG_MD5:=$NEW_MD5"
echo ""
