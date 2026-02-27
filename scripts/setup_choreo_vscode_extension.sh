#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXT_ROOT="${REPO_ROOT}/.vscode/choreo-language"
DIST_DIR="${EXT_ROOT}/.dist"

usage() {
  cat <<'EOF'
Usage: scripts/setup_choreo_vscode_extension.sh [--no-install] [--force-reinstall]

What this script does:
  1) Validates extension sources under .vscode/choreo-language
  2) Builds a VSIX package reproducibly (without vsce dependency)
  3) Installs extension via filesystem into remote VS Code extension host

Options:
  --no-install       Build VSIX only, do not install.
  --force-reinstall  Uninstall then reinstall extension.
  -h, --help         Show this help.
EOF
}

log() {
  echo "[choreo-vscode-setup] $*"
}

die() {
  echo "[choreo-vscode-setup][error] $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

NO_INSTALL=0
FORCE_REINSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-install)
      NO_INSTALL=1
      shift
      ;;
    --force-reinstall)
      FORCE_REINSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

require_cmd python3
require_cmd zip
require_cmd unzip

[[ -d "${EXT_ROOT}" ]] || die "extension root not found: ${EXT_ROOT}"
[[ -f "${EXT_ROOT}/package.json" ]] || die "missing ${EXT_ROOT}/package.json"
[[ -f "${EXT_ROOT}/language-configuration.json" ]] || die "missing ${EXT_ROOT}/language-configuration.json"
[[ -f "${EXT_ROOT}/syntaxes/choreo.tmLanguage.json" ]] || die "missing ${EXT_ROOT}/syntaxes/choreo.tmLanguage.json"

readarray -t EXT_META < <(python3 - "${EXT_ROOT}/package.json" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
j = json.loads(p.read_text(encoding="utf-8"))
required = ["name", "publisher", "version", "displayName", "description", "engines"]
for key in required:
    if key not in j:
        raise SystemExit(f"missing key in package.json: {key}")
langs = j.get("contributes", {}).get("languages", [])
if not any(x.get("id") == "choreo" for x in langs):
    raise SystemExit("contributes.languages.id=choreo not found")
print(j["publisher"])
print(j["name"])
print(j["version"])
print(j["displayName"])
print(j["description"])
print(j["engines"]["vscode"])
PY
)

PUBLISHER="${EXT_META[0]}"
NAME="${EXT_META[1]}"
VERSION="${EXT_META[2]}"
DISPLAY_NAME="${EXT_META[3]}"
DESCRIPTION="${EXT_META[4]}"
ENGINE_RANGE="${EXT_META[5]}"
ENGINE_MIN="$(python3 - <<PY
import re
s = "${ENGINE_RANGE}"
m = re.search(r"(\d+\.\d+\.\d+)", s)
print(m.group(1) if m else "1.74.0")
PY
)"
EXT_ID="${PUBLISHER}.${NAME}"
VSIX_NAME="${EXT_ID}-${VERSION}.vsix"
VSIX_PATH="${DIST_DIR}/${VSIX_NAME}"

log "repo: ${REPO_ROOT}"
log "extension id: ${EXT_ID}@${VERSION}"

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/extension"

cp -r "${EXT_ROOT}/package.json" "${DIST_DIR}/extension/"
cp -r "${EXT_ROOT}/language-configuration.json" "${DIST_DIR}/extension/"
cp -r "${EXT_ROOT}/syntaxes" "${DIST_DIR}/extension/"

cat > "${DIST_DIR}/[Content_Types].xml" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="json" ContentType="application/json"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="tmLanguage" ContentType="text/plain"/>
  <Default Extension="tmLanguage.json" ContentType="application/json"/>
</Types>
EOF

cat > "${DIST_DIR}/extension.vsixmanifest" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata>
    <Identity Language="en-US" Id="${NAME}" Version="${VERSION}" Publisher="${PUBLISHER}" />
    <DisplayName>${DISPLAY_NAME}</DisplayName>
    <Description xml:space="preserve">${DESCRIPTION}</Description>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code" Version="[${ENGINE_MIN},)" />
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true" />
  </Assets>
</PackageManifest>
EOF

pushd "${DIST_DIR}" >/dev/null
zip -qr "${VSIX_NAME}" extension extension.vsixmanifest [Content_Types].xml
popd >/dev/null

log "vsix built: ${VSIX_PATH}"

if [[ ${NO_INSTALL} -eq 0 ]]; then
  AGENT_DIR="${VSCODE_AGENT_FOLDER:-${HOME}/.vscode-server}"
  EXTENSIONS_DIR="${AGENT_DIR}/extensions"
  TARGET_DIR="${EXTENSIONS_DIR}/${EXT_ID}-${VERSION}"
  STAGE_DIR="${DIST_DIR}/_unpacked"

  mkdir -p "${EXTENSIONS_DIR}"
  rm -rf "${STAGE_DIR}"
  unzip -q "${VSIX_PATH}" -d "${STAGE_DIR}"

  if [[ ${FORCE_REINSTALL} -eq 1 ]]; then
    rm -rf "${TARGET_DIR}"
    log "removed old extension dir (if existed): ${TARGET_DIR}"
  fi

  rm -rf "${TARGET_DIR}"
  cp -r "${STAGE_DIR}/extension" "${TARGET_DIR}"

  [[ -f "${TARGET_DIR}/package.json" ]] || die "filesystem install failed: ${TARGET_DIR}/package.json missing"

  python3 - <<PY
import json
import time
from pathlib import Path

ext_id = "${EXT_ID}"
version = "${VERSION}"
target_dir = Path("${TARGET_DIR}")
extensions_json = Path("${EXTENSIONS_DIR}") / "extensions.json"
obsolete_json = Path("${EXTENSIONS_DIR}") / ".obsolete"

if obsolete_json.exists():
    try:
        ob = json.loads(obsolete_json.read_text(encoding="utf-8") or "{}")
    except Exception:
        ob = {}
    changed = False
    for key in list(ob.keys()):
        if key == f"{ext_id}-{version}" or key.startswith(f"{ext_id}-"):
            del ob[key]
            changed = True
    if changed:
        obsolete_json.write_text(json.dumps(ob, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

entries = []
if extensions_json.exists():
    try:
        entries = json.loads(extensions_json.read_text(encoding="utf-8") or "[]")
    except Exception:
        entries = []

entries = [e for e in entries if e.get("identifier", {}).get("id") != ext_id]
entries.append({
    "identifier": {"id": ext_id},
    "version": version,
  "location": {"\$mid": 1, "path": str(target_dir), "scheme": "file"},
    "relativeLocation": target_dir.name,
    "metadata": {
        "installedTimestamp": int(time.time() * 1000),
        "pinned": False,
        "source": "external"
    }
})

extensions_json.write_text(json.dumps(entries, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"registered {ext_id}@{version} in {extensions_json}")
PY

  log "installed via filesystem: ${TARGET_DIR}"
  log "reload VS Code window to pick up new extension"
  log "if editor still shows plaintext, run: Developer: Reload Window"
fi

log "done"
