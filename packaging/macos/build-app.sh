#!/usr/bin/env bash
# SubBuild Studio - builds the macOS app and its disk image, for the architecture of this Mac.
#
#   cd frontend && npm ci && npm run build && cd ..
#   ./packaging/macos/build-app.sh            # -> build/macos/SubBuild-Studio-macOS-<AppleSilicon|Intel>.dmg
#
# The app is self-contained, so users install nothing else: it carries Python 3.12 with all of the
# app's packages already installed, ffmpeg/ffprobe, and the interface already built. Projects,
# settings and API keys live in ~/Library/Application Support/SubBuild Studio (launcher.py), so
# replacing the app with a newer one never touches them.
#
# The app is ad-hoc signed (no Apple Developer ID yet), so on first launch macOS asks the user to
# confirm it (System Settings > Privacy & Security > Open Anyway).
#
# Needs: macOS, Xcode command line tools (clang), uv, git; network access.
# Optional environment: REF (default HEAD), OUT_DIR (default build/macos), SKIP_DMG=1.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF="${REF:-HEAD}"
OUT_DIR="${OUT_DIR:-$REPO/build/macos}"
APP_NAME="SubBuild Studio"
PY_VERSION="3.12"

ARCH="$(uname -m)"   # arm64 | x86_64
case "$ARCH" in
    arm64)
        LABEL="AppleSilicon"; FF_ARCH="arm64"
        # onnxruntime / PyAV wheels for Apple Silicon need macOS 14+
        MIN_MACOS="14.0" ;;
    x86_64)
        LABEL="Intel"; FF_ARCH="amd64"
        # the last onnxruntime with Intel Mac wheels (1.23) needs macOS 13+
        MIN_MACOS="13.0" ;;
    *) echo "[ERROR] Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac
export MACOSX_DEPLOYMENT_TARGET="$MIN_MACOS"

FFMPEG_URLS="https://ffmpeg.martin-riedl.de/redirect/latest/macos/$FF_ARCH/release/ffmpeg.zip"
FFPROBE_URLS="https://ffmpeg.martin-riedl.de/redirect/latest/macos/$FF_ARCH/release/ffprobe.zip"
if [[ "$ARCH" == "x86_64" ]]; then   # second source, Intel only
    FFMPEG_URLS="$FFMPEG_URLS https://evermeet.cx/ffmpeg/getrelease/zip"
    FFPROBE_URLS="$FFPROBE_URLS https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"
fi

step() { echo; echo "==> $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# download <output> <url> [<url>...] - each URL up to 3 times, then the next one
download() {
    local out="$1"; shift
    local url try
    for url in "$@"; do
        for try in 1 2 3; do
            if curl -fL --retry 2 --connect-timeout 30 -o "$out" "$url"; then
                return 0
            fi
            echo "    download failed ($try/3) from $url" >&2
            sleep $((10 * try))
        done
    done
    die "could not download $(basename "$out")"
}

command -v uv >/dev/null 2>&1 || die "uv is required (https://docs.astral.sh/uv/)"
command -v clang >/dev/null 2>&1 || die "clang is required (xcode-select --install)"
[[ -f "$REPO/frontend/dist/index.html" ]] || die "frontend/dist is missing - run 'npm ci' and 'npm run build' in frontend/ first."

WORK="$OUT_DIR/work-$ARCH"
APP="$OUT_DIR/$APP_NAME.app"
CONTENTS="$APP/Contents"
RES="$CONTENTS/Resources"
rm -rf "$WORK" "$APP"
mkdir -p "$WORK" "$CONTENTS/MacOS" "$RES"

# -----------------------------------------------------------------------------------------------
step "1/8 Exporting $REF"
# git archive, as for a GitHub release zip: VERSION is filled in from the newest tag (export-subst)
SRC="$WORK/app"
mkdir -p "$SRC"
git -C "$REPO" archive --format=tar "$REF" | tar -x -C "$SRC" -f -

# Development-only parts are not shipped (same list as packaging/windows/stage.ps1)
for rel in qa-automation-java backend/tests backend/pytest.ini .github docs Jenkinsfile SPEC.md \
           README.md install.bat install.sh install.command run.bat run.sh run.command release-tag.bat \
           frontend packaging/windows packaging/macos .gitattributes .gitignore; do
    rm -rf "${SRC:?}/$rel"
done
mkdir -p "$SRC/frontend"
cp -R "$REPO/frontend/dist" "$SRC/frontend/dist"
# The updater sees this and asks for a new download instead of patching the app (updater.py)
echo "macos" > "$SRC/INSTALLER"

RAW_VERSION="$(tr -d '[:space:]' < "$SRC/VERSION")"
if [[ "$RAW_VERSION" =~ ^v?([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    VERSION="${BASH_REMATCH[1]}"
else
    VERSION="0.0.0"
fi
echo "    version: $VERSION ($RAW_VERSION), $ARCH, macOS $MIN_MACOS+"

# -----------------------------------------------------------------------------------------------
step "2/8 Python $PY_VERSION"
# python-build-standalone (via uv): relocatable, so it can live inside the app bundle
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
export UV_CACHE_DIR="$WORK/uv-cache"
export UV_LINK_MODE=copy
uv python install "$PY_VERSION"
# (found in uv's own install folder: "uv python find" could return a Python of the build machine)
PY_SRC_BIN="$(find "$UV_PYTHON_INSTALL_DIR" -maxdepth 3 -type f -path "*/bin/python$PY_VERSION" | head -n 1)"
[[ -n "$PY_SRC_BIN" ]] || die "uv did not install Python $PY_VERSION"
PY_SRC_HOME="$(cd "$(dirname "$PY_SRC_BIN")/.." && pwd -P)"
ditto "$PY_SRC_HOME" "$RES/python"
PY="$RES/python/bin/python$PY_VERSION"
LIBPY="$RES/python/lib/libpython$PY_VERSION.dylib"
[[ -x "$PY" ]] || die "$PY is missing"
[[ -f "$LIBPY" ]] || die "$LIBPY is missing (the launcher embeds Python and needs the shared library)"
"$PY" -c "import sys, platform; print('   ', sys.version.split()[0], platform.machine())"

# -----------------------------------------------------------------------------------------------
step "3/8 Installing the app's Python packages"
# Straight into the bundled Python: it belongs to the app alone (as on Windows)
uv pip install --python "$PY" --break-system-packages \
    -r "$SRC/backend/requirements.txt" -r "$SRC/packaging/requirements-desktop.txt"
# Not needed at runtime
rm -rf "$RES/python/lib/python$PY_VERSION/test" "$RES/python/lib/python$PY_VERSION/idlelib" \
       "$RES/python/lib/python$PY_VERSION/turtledemo" "$RES/python/lib/python$PY_VERSION/ensurepip" \
       "$RES/python/share"
rm -f "$RES/python/lib/python$PY_VERSION/EXTERNALLY-MANAGED"

# -----------------------------------------------------------------------------------------------
step "4/8 ffmpeg and ffprobe"
FF_DIR="$SRC/packaging/ffmpeg"
mkdir -p "$FF_DIR" "$WORK/ff"
for tool in ffmpeg ffprobe; do
    urls="$FFMPEG_URLS"; [[ "$tool" == "ffprobe" ]] && urls="$FFPROBE_URLS"
    # shellcheck disable=SC2086  # urls is a space-separated list
    download "$WORK/ff/$tool.zip" $urls
    rm -rf "$WORK/ff/$tool"; mkdir -p "$WORK/ff/$tool"
    ditto -x -k "$WORK/ff/$tool.zip" "$WORK/ff/$tool"
    found="$(find "$WORK/ff/$tool" -type f -name "$tool" | head -n 1)"
    [[ -n "$found" ]] || die "$tool not found in its download"
    cp "$found" "$FF_DIR/$tool"
    chmod 755 "$FF_DIR/$tool"
    lipo -archs "$FF_DIR/$tool" | grep -qw "$ARCH" || die "$tool is not built for $ARCH ($(lipo -archs "$FF_DIR/$tool"))"
    # Must not depend on libraries outside the system (e.g. Homebrew's)
    if otool -L "$FF_DIR/$tool" | tail -n +2 | awk '{print $1}' | grep -vE '^(/usr/lib/|/System/)' ; then
        die "$tool links to libraries that are not part of macOS"
    fi
done
"$FF_DIR/ffmpeg" -hide_banner -version | head -n 1
# Burning subtitles into a video needs libass
"$FF_DIR/ffmpeg" -hide_banner -filters 2>/dev/null | grep -qE '^ *[.A-Z|]+ +subtitles ' \
    || die "this ffmpeg build has no 'subtitles' filter (libass)"

# -----------------------------------------------------------------------------------------------
step "5/8 Assembling $APP_NAME.app"
ditto "$SRC" "$RES/app"
cp "$REPO/packaging/icon.icns" "$RES/icon.icns"
sed -e "s/@VERSION@/$VERSION/g" -e "s/@MIN_MACOS@/$MIN_MACOS/g" \
    "$REPO/packaging/macos/Info.plist" > "$CONTENTS/Info.plist"
plutil -lint "$CONTENTS/Info.plist"
printf 'APPL????' > "$CONTENTS/PkgInfo"

# The executable: embeds the bundled Python and runs packaging/launcher.py (launcher.c)
EXE="$CONTENTS/MacOS/$APP_NAME"
clang -O2 -arch "$ARCH" -mmacosx-version-min="$MIN_MACOS" -Wall -Wno-unused-result \
    -I "$RES/python/include/python$PY_VERSION" \
    "$REPO/packaging/macos/launcher.c" \
    -L "$RES/python/lib" -lpython"$PY_VERSION" \
    -o "$EXE"
# Load libpython from inside the bundle, wherever the app is moved
LIBPY_ID="$(otool -D "$LIBPY" | tail -n 1)"
install_name_tool -change "$LIBPY_ID" "@executable_path/../Resources/python/lib/libpython$PY_VERSION.dylib" "$EXE"
otool -L "$EXE" | grep -q "@executable_path/../Resources/python/lib/libpython$PY_VERSION.dylib" \
    || die "the launcher does not load the bundled libpython: $(otool -L "$EXE")"

# Precompiled bytecode that never checks the sources: the app starts faster and Python never
# needs to write into the bundle (which would break its signature)
"$PY" -m compileall -q -j 0 --invalidation-mode unchecked-hash "$RES/python/lib" "$RES/app" >/dev/null || true

# -----------------------------------------------------------------------------------------------
step "6/8 Signing (ad-hoc)"
# Apple Silicon only runs signed code, so every binary needs at least an ad-hoc signature; those
# that already carry a valid one (Python, the wheels, ffmpeg's Developer ID) keep it
signed=0
while IFS= read -r -d '' f; do
    if file -b "$f" | grep -q "Mach-O"; then
        if ! codesign --verify --strict "$f" >/dev/null 2>&1; then
            codesign --force --sign - "$f"
            signed=$((signed + 1))
        fi
    fi
done < <(find "$RES" -type f \( -name "*.so" -o -name "*.dylib" -o -perm -u+x \) -print0)
echo "    signed $signed binaries without a valid signature"
codesign --force --sign - --identifier com.subbuildstudio.app "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

# -----------------------------------------------------------------------------------------------
step "7/8 Smoke test"
# The executable itself (embedded Python + every import the studio needs), then a real start of
# the server without a window
"$EXE" --check
LOG="$WORK/smoke.log"
"$EXE" --no-tray --no-browser >"$LOG" 2>&1 &
PID=$!
ok=0
for _ in $(seq 1 90); do
    code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/system/info || true)"
    if [[ "$code" != "000" ]]; then ok=1; break; fi
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
cat "$LOG"
# (started outside a terminal, the app writes its output to its log file)
cat "$HOME/Library/Application Support/$APP_NAME/launcher.log" 2>/dev/null || true
[[ "$ok" -eq 1 ]] || die "the studio's server did not start"
echo "    the server answered (HTTP $code)"
# The smoke test must not have changed the bundle
codesign --verify --deep --strict "$APP" || die "the app bundle was modified while running"

# -----------------------------------------------------------------------------------------------
DMG="$OUT_DIR/SubBuild-Studio-macOS-$LABEL.dmg"
if [[ "${SKIP_DMG:-0}" == "1" ]]; then
    echo; echo "Built $APP ($VERSION, $ARCH)"; exit 0
fi
step "8/8 Disk image"
DMG_SRC="$WORK/dmg"
rm -rf "$DMG_SRC" "$DMG"
mkdir -p "$DMG_SRC"
ditto "$APP" "$DMG_SRC/$APP_NAME.app"
ln -s /Applications "$DMG_SRC/Applications"
cp "$REPO/packaging/macos/dmg-readme.txt" "$DMG_SRC/Read me first.txt"
# hdiutil now and then fails with "Resource busy" on CI machines: try again
for try in 1 2 3 4 5; do
    if hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_SRC" -fs HFS+ -format UDZO \
        -imagekey zlib-level=9 -ov "$DMG"; then
        break
    fi
    [[ "$try" -lt 5 ]] || die "hdiutil create failed"
    sleep $((5 * try))
done
hdiutil verify "$DMG"
du -sh "$APP" "$DMG"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    { echo "version=$VERSION"; echo "dmg=$DMG"; echo "label=$LABEL"; } >> "$GITHUB_OUTPUT"
fi
echo; echo "Built $DMG ($VERSION, $ARCH)"
