# Reproducible Choreo VS Code Extension Setup

This repository includes the Choreo language extension sources at `.vscode/choreo-language`.

Use the setup script below to run the full workflow from scratch:

1. Validate extension source files
2. Build a VSIX package (without `vsce`)
3. Install into remote VS Code extension host via filesystem

This script does not modify `.vscode/settings.json`.

## One-command install

```bash
bash scripts/setup_choreo_vscode_extension.sh --force-reinstall
```

## Common options

- Build VSIX only (do not install):

```bash
bash scripts/setup_choreo_vscode_extension.sh --no-install
```

- Filesystem install (no `code` CLI required):

```bash
bash scripts/setup_choreo_vscode_extension.sh --force-reinstall --install-mode filesystem
```

## Remote window note

This script installs directly into `~/.vscode-server/extensions` (or `${VSCODE_AGENT_FOLDER}/extensions` if set), so it works even when the remote shell has no `code` command.

It also updates `extensions.json` and clears stale `.obsolete` entries for `local.choreo-language`, so VS Code Server can load the extension after reload.

If files still open as `Plain Text`, run `Developer: Reload Window` once.
