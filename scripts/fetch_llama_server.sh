#!/usr/bin/env bash
#
# fetch_llama_server.sh — fetch the PINNED llama.cpp release binaries.
#
# Downloads the upstream ggml-org/llama.cpp prebuilt Ubuntu x64 tarballs
# (CPU + Vulkan variants), verifies each against its pinned SHA-256, and
# extracts to vendor/llama.cpp/ and vendor/llama.cpp-vulkan/.
#
set -euo pipefail

TAG="b10034"
CPU_SHA="5a6370a6d27e508e5efa0a348be8f52a4fbbc7f3d2403c1a22cadc41210792ca"
VULKAN_SHA="cec255e083eb5617a08f1923c0b043a391a53b5d09d573675f697a8053008d73"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$REPO_ROOT/vendor"
BASE_URL="https://github.com/ggml-org/llama.cpp/releases/download/$TAG"

mkdir -p "$VENDOR"

fetch_variant() { # <tarball-name> <sha256> <dest-dir>
    local tarball="$1" sha="$2" dest="$3"
    local path="$VENDOR/$tarball"

    if [ -x "$dest/llama-server" ]; then
        echo "already present: $dest"; return 0
    fi
    if [ ! -f "$path" ]; then
        echo "downloading $tarball ..."
        curl -fsSL -o "$path.part" "$BASE_URL/$tarball"
        mv "$path.part" "$path"
    fi
    if [ -n "${BOX_LLAMA_PRINT_HASHES:-}" ]; then sha256sum "$path"; return 0; fi
    echo "$sha  $path" | sha256sum -c - >/dev/null \
        || { echo "SHA-256 MISMATCH for $tarball — refusing to extract" >&2; exit 1; }
    local tmp; tmp="$(mktemp -d "$VENDOR/.extract.XXXXXX")"
    tar xzf "$path" -C "$tmp"
    rm -rf "$dest"; mv "$tmp/llama-$TAG" "$dest"; rmdir "$tmp"
    echo "ready: $dest"
}

fetch_variant "llama-$TAG-bin-ubuntu-x64.tar.gz" "$CPU_SHA" "$VENDOR/llama.cpp"
fetch_variant "llama-$TAG-bin-ubuntu-vulkan-x64.tar.gz" "$VULKAN_SHA" "$VENDOR/llama.cpp-vulkan"
