#!/bin/bash
# Re-wrap CPack archive payloads in a <package-name>/ toplevel directory so
# they extract to <package-name>/usr/... . Component-install archives omit the
# toplevel dir even with CPACK_INCLUDE_TOPLEVEL_DIRECTORY, so post-process the
# tar.gz / zip here. Usage: wrap_toplevel.sh <package-dir>
set -e
PKG_DIR="${1:?wrap_toplevel: package dir required}"

for a in "$PKG_DIR"/choreo-*.tar.gz "$PKG_DIR"/choreo-*.zip; do
    [ -f "$a" ] || continue
    base="$(basename "$a")"
    name="${base%.zip}"
    name="${name%.tar.gz}"
    tmp="$(mktemp -d)"
    case "$base" in
        *.tar.gz) tar xzf "$a" -C "$tmp" ;;
        *.zip)    unzip -q "$a" -d "$tmp" ;;
    esac
    mkdir -p "$tmp/$name"
    for item in "$tmp"/* "$tmp"/.[!.]*; do
        [ -e "$item" ] || continue
        bn="$(basename "$item")"
        [ "$bn" = "$name" ] && continue
        mv "$item" "$tmp/$name/"
    done
    case "$base" in
        *.tar.gz) tar czf "$a" -C "$tmp" "$name" ;;
        *.zip)    (cd "$tmp" && zip -qr "$a" "$name") ;;
    esac
    rm -rf "$tmp"
done
