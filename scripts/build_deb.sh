#!/usr/bin/env bash
#
# build_deb.sh — build the closed-source Box .deb.
#
# Compiles box_chat to a native extension with Nuitka, vendors the pip-only
# deps, and assembles a Debian package. GTK/gi/GStreamer stay as apt Depends.
#
#   Run on a clean Ubuntu (target release) amd64 box with network access.
#   This script only writes under ./build/ — the source tree is untouched.
#
# Build-host requirements (apt):
#   build-essential  dpkg-dev  python3-dev
# Nuitka itself is installed on the fly into ./build/ (never into the .deb).
# The project's ./.venv (its pip) is used as the build pip — the system
# python3 has no pip, and creating a venv would need python3-venv.
#
# Output: build/box_<version>_amd64.deb
#
set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PKG_DEPS=(litert-lm-api ai-edge-litert sentencepiece numpy sounddevice
          pypdf ddgs webrtcvad-wheels matplotlib
          pillow regex argon2-cffi)

ARCH="amd64"
VERSION="$(python3 -c 'import box_chat,sys; sys.stdout.write(box_chat.__version__)')"
PY_MM="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MIN="$PY_MM"
PY_MAX="${PY_MM%.*}.$(( ${PY_MM#*.} + 1 ))"

BUILD="$REPO_ROOT/build"
STAGE="$BUILD/box-pkg"            # dpkg-deb staging root
OPT="$STAGE/opt/box"
SITE="$OPT/site-packages"
NUITKA_OUT="$BUILD/nuitka"
BUILDTOOLS="$BUILD/buildtools"    # Nuitka installed here (build-only, --target)
DEB="$BUILD/box_${VERSION}_${ARCH}.deb"

# The project .venv was created under a now-stale home path, so its bin/pip
# wrapper has a broken shebang — bin/python is a symlink and still works, so
# pip is driven via "-m pip".
BUILDPY="$REPO_ROOT/.venv/bin/python"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mBUILD FAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ── 0. preflight ────────────────────────────────────────────────────────────
say "Preflight (Box $VERSION, python $PY_MM, $ARCH)"
command -v gcc      >/dev/null || die "gcc missing — apt install build-essential"
command -v dpkg-deb >/dev/null || die "dpkg-deb missing — apt install dpkg-dev"
[ -d box_chat ]                || die "run from the repo root (box_chat/ not found)"
[ -x "$BUILDPY" ]              || die ".venv python not found at $BUILDPY"
"$BUILDPY" -m pip --version >/dev/null 2>&1 || die ".venv pip module not usable"
PYINC="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
[ -f "$PYINC/Python.h" ]       || die "Python.h missing ($PYINC) — apt install python3-dev"

# ── 1. clean staging ────────────────────────────────────────────────────────
say "Clean ./build/"
rm -rf "$BUILD"
mkdir -p "$SITE" "$NUITKA_OUT" "$BUILDTOOLS" "$STAGE/DEBIAN" \
         "$STAGE/usr/bin" "$STAGE/usr/share/applications" \
         "$STAGE/usr/share/icons/hicolor/scalable/apps" \
         "$STAGE/usr/share/box/fonts" "$STAGE/usr/share/box/icons" \
         "$OPT/libexec" "$OPT/vision_models" "$OPT/litert_diffusion" \
         "$STAGE/usr/share/doc/box"

# ── 2. install Nuitka (build-only, into ./build/buildtools) ─────────────────
say "Install Nuitka into ./build/buildtools"
"$BUILDPY" -m pip install --quiet --target "$BUILDTOOLS" --upgrade nuitka

# ── 3. vendor the pip-only deps into site-packages ──────────────────────────
# --ignore-installed forces a full, self-contained closure into $SITE so the
# package never silently relies on whatever happens to be on the build host.
say "Vendor pip dependencies into /opt/box/site-packages"
"$BUILDPY" -m pip install --target "$SITE" --ignore-installed "${PKG_DEPS[@]}"
"$BUILDPY" -m pip freeze --path "$SITE" > "$BUILD/requirements-frozen.txt"
echo "  frozen manifest -> build/requirements-frozen.txt"

# ── 4. compile box_chat -> native extension ─────────────────────────────────
# Module mode: only box_chat is compiled; gi/numpy/litert import at runtime.
# No --python-flag=no_docstrings — the SDK needs tool __doc__ for schemas.
# Run with the system python3 (clean: no editable box install on its path),
# whose ABI tag the .deb's Depends pins; Nuitka is found via PYTHONPATH.
say "Compile box_chat with Nuitka (this takes a few minutes)"
PYTHONPATH="$BUILDTOOLS" python3 -m nuitka \
    --mode=module box_chat \
    --include-package=box_chat \
    --output-dir="$NUITKA_OUT" \
    --no-pyi-file \
    --remove-output \
    --assume-yes-for-downloads

shopt -s nullglob
COMPILED=("$NUITKA_OUT"/box_chat*.so)
shopt -u nullglob
[ "${#COMPILED[@]}" -eq 1 ] || die "expected exactly one box_chat*.so, got ${#COMPILED[@]}"
cp "${COMPILED[0]}" "$SITE/"
echo "  compiled -> $(basename "${COMPILED[0]}")"

# guard: no box_chat source must ever ship
if find "$SITE" -path '*box_chat*' -name '*.py' | grep -q .; then
    die "box_chat .py source leaked into site-packages"
fi

# ── 5. build gate — tool schemas must survive compilation ───────────────────
# System python3 + PYTHONPATH=$SITE is exactly the deployed runtime, so the
# gate imports the COMPILED box_chat.so (no editable source on the path).
say "Gate: verify tool schemas survived compilation"
PYTHONPATH="$SITE" python3 packaging/verify_schemas.py \
    || die "tool-schema gate failed (see packaging/verify_schemas.py)"

# ── 5b. bundle engines, models, fonts, icons ────────────────────────────────
# The runtime resolves these from fixed paths: llama_server/sd_backend look in
# /opt/box/libexec, vision_tools/litert_diffusion in /opt/box/{vision_models,
# litert_diffusion}, fonts.py in /usr/share/box/fonts, app.py icons in
# /usr/share/box/icons. Bundle to exactly those.
say "Bundle inference engines"
bundle_engine() {  # <vendor-subdir> <dest-name>
    local src="$REPO_ROOT/vendor/$1" dst="$OPT/libexec/$2"
    [ -d "$src" ] || die "missing vendor engine: vendor/$1 (run scripts/fetch_*.sh)"
    mkdir -p "$dst"
    cp -a "$src"/. "$dst"/
    echo "  vendor/$1 -> /opt/box/libexec/$2 ($(du -sh "$dst" | cut -f1))"
}
bundle_engine llama.cpp                    llama.cpp
bundle_engine llama.cpp-vulkan             llama.cpp-vulkan
bundle_engine stable-diffusion.cpp         stable-diffusion.cpp
bundle_engine stable-diffusion.cpp-vulkan  stable-diffusion.cpp-vulkan

say "Bundle vision + diffusion assets"
cp -a data/vision_models/. "$OPT/vision_models"/
cp -a data/litert_diffusion/. "$OPT/litert_diffusion"/
echo "  vision_models -> /opt/box/vision_models ($(du -sh "$OPT/vision_models" | cut -f1))"
echo "  litert_diffusion -> /opt/box/litert_diffusion ($(du -sh "$OPT/litert_diffusion" | cut -f1))"

say "Bundle fonts + icons"
find data/fonts -maxdepth 1 -name '*.ttf' -exec install -m 0644 {} "$STAGE/usr/share/box/fonts/" \;
cp -a data/icons/. "$STAGE/usr/share/box/icons"/
echo "  fonts -> /usr/share/box/fonts ($(find "$STAGE/usr/share/box/fonts" -name '*.ttf' | wc -l) families)"

# ── 6. stage launcher / desktop / icon / licenses ───────────────────────────
say "Stage launcher, desktop entry, icon, licenses"
install -m 0755 packaging/box.launcher "$STAGE/usr/bin/box"
install -m 0644 data/com.jegly.box.desktop \
        "$STAGE/usr/share/applications/com.jegly.box.desktop"
install -m 0644 data/icons/hicolor/scalable/apps/com.jegly.box.svg \
        "$STAGE/usr/share/icons/hicolor/scalable/apps/com.jegly.box.svg"
PYTHONPATH="$SITE" python3 packaging/gen_licenses.py "$SITE" \
        > "$STAGE/usr/share/doc/box/THIRD-PARTY-LICENSES"

# ── 7. DEBIAN/control ───────────────────────────────────────────────────────
say "Write DEBIAN/control"
sed -e "s/@VERSION@/$VERSION/" \
    -e "s/@ARCH@/$ARCH/" \
    -e "s/@PYMIN@/$PY_MIN/" \
    -e "s/@PYMAX@/$PY_MAX/" \
    packaging/control.in > "$STAGE/DEBIAN/control"

# ── 8. build the .deb ───────────────────────────────────────────────────────
say "Assemble the .deb"
dpkg-deb --build --root-owner-group "$STAGE" "$DEB"

say "DONE"
echo "  package : $DEB"
echo "  size    : $(du -h "$DEB" | cut -f1)"
echo
echo "Next:"
echo "  - install on a FRESH target VM:  sudo apt install $DEB"
echo "  - launch Box from the app menu (or run 'box'); confirm chat,"
echo "    a tool call, voice mode and memory recall all work."
echo "  - if it checks out, commit build/requirements-frozen.txt for the release."
