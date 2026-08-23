#!/usr/bin/env bash
#
# fetch_sd_cpp.sh — fetch the PINNED stable-diffusion.cpp release binaries.
#
# Downloads the upstream leejet/stable-diffusion.cpp prebuilt Ubuntu x86_64
# zips (CPU + Vulkan), verifies each against its pinned SHA-256, and extracts
# to vendor/stable-diffusion.cpp/ and vendor/stable-diffusion.cpp-vulkan/.
#
set -euo pipefail

TAG="master-778-c00a9e9"
CPU_ZIP="sd-master-c00a9e9-bin-Linux-Ubuntu-24.04-x86_64.zip"
VULKAN_ZIP="sd-master-c00a9e9-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"
CPU_SHA="da223f809e56d7d24bebfe0ebcfa279394f3549eb557385e3ba8c443a8f8ac5e"
VULKAN_SHA="95328f6e228ba3ada44179cacf415919bd870325275893bad616c65a403a2fa2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$REPO_ROOT/vendor"
BASE_URL="https://github.com/leejet/stable-diffusion.cpp/releases/download/$TAG"

mkdir -p "$VENDOR"

fetch_variant() { # <zip-name> <sha256> <dest-dir>
    local zip="$1" sha="$2" dest="$3"
    local path="$VENDOR/$zip"

    if [ -x "$dest/sd-cli" ]; then echo "already present: $dest"; return 0; fi
    if [ ! -f "$path" ]; then
        echo "downloading $zip ..."
        curl -fsSL -o "$path.part" "$BASE_URL/$zip"
        mv "$path.part" "$path"
    fi
    if [ -n "${BOX_SD_PRINT_HASHES:-}" ]; then sha256sum "$path"; return 0; fi
    echo "$sha  $path" | sha256sum -c - >/dev/null \
        || { echo "SHA-256 MISMATCH for $zip — refusing to extract" >&2; exit 1; }
    local tmp; tmp="$(mktemp -d "$VENDOR/.sdextract.XXXXXX")"
    unzip -oq "$path" -d "$tmp"
    rm -rf "$dest"; mv "$tmp" "$dest"; chmod +x "$dest/sd-cli" 2>/dev/null || true
    echo "ready: $dest"
}

fetch_variant "$CPU_ZIP" "$CPU_SHA" "$VENDOR/stable-diffusion.cpp"
fetch_variant "$VULKAN_ZIP" "$VULKAN_SHA" "$VENDOR/stable-diffusion.cpp-vulkan"
