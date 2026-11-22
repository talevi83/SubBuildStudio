#!/usr/bin/env bash
# SubBuild Studio - builds the Linux AppImage (x86_64).
#
#   cd frontend && npm ci && npm run build && cd ..
#   ./packaging/linux/build-appimage.sh       # -> build/linux/SubBuild-Studio-Linux-x86_64.AppImage
#
# The AppImage is self-contained, so users install nothing else: it carries Python 3.12 with all of
# the app's packages already installed (the Qt window included), ffmpeg/ffprobe, and the interface
# already built. Projects, settings and API keys live in ~/.local/share/SubBuild Studio
# (launcher.py), so replacing the AppImage with a newer one never touches them.
#
# The NVIDIA libraries for GPU transcription (~1.5 GB) are not inside: when the app finds an NVIDIA
# card it offers to download them into the data folder (backend/app/core/gpu_runtime.py), with the
# versions pinned here in packaging/gpu-requirements.txt.
#
# Build on the oldest distribution to support: the Qt wheels need glibc 2.34+, so Ubuntu 22.04 gives
# an AppImage for Ubuntu 22.04+, Debian 12+, Fedora 35+ and their relatives.
#
# Needs: Linux x86_64, uv, git, curl, readelf (binutils), xvfb-run when there is no display, and
# the X11 libraries Qt needs (apt: libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1
# libxcb-render-util0 libxcb-shape0 libxcb-util1 libxkbcommon-x11-0), which are bundled; network access.
# Optional environment: REF (default HEAD), OUT_DIR (default build/linux), SKIP_APPIMAGE=1.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF="${REF:-HEAD}"
OUT_DIR="${OUT_DIR:-$REPO/build/linux}"
APP_NAME="SubBuild Studio"
APP_ID="subbuildstudio"
PY_VERSION="3.12"
ARCH="$(uname -m)"
[[ "$ARCH" == "x86_64" ]] || { echo "[ERROR] Only x86_64 is supported (this is $ARCH)" >&2; exit 1; }
APPIMAGE_NAME="SubBuild-Studio-Linux-x86_64.AppImage"
# Lets AppImageUpdate / Gear Lever fetch only the changed blocks of a new release
UPDATE_INFO="gh-releases-zsync|talevi83|SubBuildStudio|latest|$APPIMAGE_NAME.zsync"

# Tried in order, each up to 3 times: both are static builds with libass (the 'subtitles' filter)
FFMPEG_URLS="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-musl.tar.gz"
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
# Libraries Qt's X11 plugin needs that minimal desktops may not have (libxcb-cursor0 above all)
EXTRA_LIBS="libxcb-cursor.so.0 libxcb-icccm.so.4 libxcb-image.so.0 libxcb-keysyms.so.1
libxcb-render-util.so.0 libxcb-shape.so.0 libxcb-util.so.1 libxkbcommon-x11.so.0"

step() { echo; echo "==> $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# download <output> <url> [<url>...] - each URL up to 3 times, then the next one
download() {
    local out="$1"; shift
    local url try
    for url in "$@"; do
        for try in 1 2 3; do
            if curl -fsSL --retry 2 --connect-timeout 30 -o "$out" "$url"; then
                return 0
            fi
            echo "    download failed ($try/3) from $url" >&2
            sleep $((10 * try))
        done
    done
    die "could not download $(basename "$out")"
}

# Runs a command with a display: the real one, or a virtual X server (build machines have none)
with_display() {
    if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
        "$@"
    else
        command -v xvfb-run >/dev/null 2>&1 || die "no display and no xvfb-run (apt install xvfb)"
        xvfb-run -a -s "-screen 0 1440x900x24" "$@"
    fi
}

command -v uv >/dev/null 2>&1 || die "uv is required (https://docs.astral.sh/uv/)"
command -v readelf >/dev/null 2>&1 || die "readelf is required (apt install binutils)"
[[ -f "$REPO/frontend/dist/index.html" ]] || die "frontend/dist is missing - run 'npm ci' and 'npm run build' in frontend/ first."

WORK="$OUT_DIR/work"
APPDIR="$OUT_DIR/AppDir"
rm -rf "$WORK" "$APPDIR"
mkdir -p "$WORK" "$APPDIR/usr/bin" "$APPDIR/usr/lib/extra"

# -----------------------------------------------------------------------------------------------
step "1/9 Exporting $REF"
# git archive, as for a GitHub release zip: VERSION is filled in from the newest tag (export-subst)
SRC="$APPDIR/usr/app"
mkdir -p "$SRC"
git -C "$REPO" archive --format=tar "$REF" | tar -x -C "$SRC" -f -

# Development-only parts are not shipped (same list as packaging/windows/stage.ps1)
for rel in qa-automation-java backend/tests backend/pytest.ini .github docs Jenkinsfile SPEC.md \
           README.md install.bat install.sh install.command run.bat run.sh run.command release-tag.bat \
           frontend packaging/windows packaging/macos packaging/linux .gitattributes .gitignore; do
    rm -rf "${SRC:?}/$rel"
done
mkdir -p "$SRC/frontend"
cp -R "$REPO/frontend/dist" "$SRC/frontend/dist"
# The updater sees this and asks for a new download instead of patching the app (updater.py)
echo "linux" > "$SRC/INSTALLER"

RAW_VERSION="$(tr -d '[:space:]' < "$SRC/VERSION")"
if [[ "$RAW_VERSION" =~ ^v?([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    VERSION="${BASH_REMATCH[1]}"
else
    VERSION="0.0.0"
fi
echo "    version: $VERSION ($RAW_VERSION), $ARCH"

# -----------------------------------------------------------------------------------------------
step "2/9 Python $PY_VERSION"
# python-build-standalone (via uv): relocatable, so it runs from wherever the AppImage is mounted
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
export UV_CACHE_DIR="$WORK/uv-cache"
export UV_LINK_MODE=copy
uv python install "$PY_VERSION"
# (found in uv's own install folder: "uv python find" could return a Python of the build machine)
PY_SRC_BIN="$(find "$UV_PYTHON_INSTALL_DIR" -maxdepth 3 -type f -path "*/bin/python$PY_VERSION" | head -n 1)"
[[ -n "$PY_SRC_BIN" ]] || die "uv did not install Python $PY_VERSION"
PY_SRC_HOME="$(cd "$(dirname "$PY_SRC_BIN")/.." && pwd -P)"
cp -a "$PY_SRC_HOME" "$APPDIR/usr/python"
PY="$APPDIR/usr/python/bin/python$PY_VERSION"
PYLIB="$APPDIR/usr/python/lib/python$PY_VERSION"
[[ -x "$PY" ]] || die "$PY is missing"
"$PY" -c "import sys, platform; print('   ', sys.version.split()[0], platform.machine())"

# -----------------------------------------------------------------------------------------------
step "3/9 Installing the app's Python packages"
# The NVIDIA libraries are left out: they are offered as a download on machines with an NVIDIA
# card. Their exact versions are pinned now, so every user gets the set this build was made with.
grep -vE '^[[:space:]]*nvidia-' "$SRC/backend/requirements.txt" > "$WORK/requirements-no-gpu.txt"
grep -E '^[[:space:]]*nvidia-' "$SRC/backend/requirements.txt" > "$WORK/requirements-gpu.in" \
    || die "backend/requirements.txt lists no nvidia-* packages"
# Straight into the bundled Python: it belongs to the app alone (as on Windows and macOS)
uv pip install --python "$PY" --break-system-packages \
    -r "$WORK/requirements-no-gpu.txt" -r "$SRC/packaging/requirements-desktop.txt" \
    -r "$REPO/packaging/linux/requirements-linux.txt"
uv pip compile --quiet --no-header --no-annotate --python-version "$PY_VERSION" \
    --python-platform x86_64-manylinux_2_28 "$WORK/requirements-gpu.in" -o "$SRC/packaging/gpu-requirements.txt"
echo "    GPU libraries offered as a download:"; sed 's/^/      /' "$SRC/packaging/gpu-requirements.txt"
"$PY" - <<'EOF' || die "an NVIDIA package ended up in the AppImage"
import importlib.util, sys
sys.exit(1 if importlib.util.find_spec("nvidia") else 0)
EOF

# Not needed at runtime
rm -rf "$PYLIB/test" "$PYLIB/idlelib" "$PYLIB/turtledemo" "$PYLIB/ensurepip" "$PYLIB/tkinter" \
       "$APPDIR/usr/python/share" "$APPDIR/usr/python/include"
rm -f "$PYLIB/EXTERNALLY-MANAGED"
# Qt parts the window never uses (QML, Designer, Quick 3D) and translations other than English /
# Hebrew. Nothing that stays may link to a removed library (checked right after).
QT="$PYLIB/site-packages/PyQt6"
# qt_missing: the Qt libraries something in the bundle links to but that are not in it. The wheel
# itself already has a few (plugins for Qt modules it does not ship), so only new ones count.
qt_missing() {
    find "$QT" -type f \( -name '*.so' -o -name '*.so.*' \) -print0 \
        | LD_LIBRARY_PATH="$QT/Qt6/lib" xargs -0 -n 50 ldd 2>/dev/null \
        | awk '/not found/ {print $1}' | grep -E '^lib(Qt6|icu)' | sort -u || true
}
QT_MISSING_BEFORE="$(qt_missing)"
rm -rf "$QT/Qt6/qml" "$QT/Qt6/qsci" "$QT"/QtDesigner* "$QT"/QtQuick3D* "$QT"/QtQml* "$QT"/QtQuick* \
       "$QT"/QtTest* "$QT/Qt6/lib/libQt6Designer"* "$QT/Qt6/lib/libQt6Quick3D"* "$QT/Qt6/plugins/designer" \
       "$QT/Qt6/plugins/assetimporters"
find "$QT/Qt6/translations" -maxdepth 1 -type f ! -name '*_he.qm' ! -name '*_en.qm' -delete
find "$QT/Qt6/translations/qtwebengine_locales" -type f ! -name 'en-US.pak' ! -name 'he.pak' -delete
new_missing="$(comm -13 <(echo "$QT_MISSING_BEFORE") <(qt_missing))"
[[ -z "$new_missing" ]] || die "trimming Qt removed libraries that are still needed:"$'\n'"$new_missing"

# -----------------------------------------------------------------------------------------------
step "4/9 ffmpeg, ffprobe and uv"
FF_DIR="$SRC/packaging/ffmpeg"
mkdir -p "$FF_DIR" "$WORK/ff"
# shellcheck disable=SC2086  # a newline-separated list of URLs
download "$WORK/ff/ffmpeg.tar.xz" $FFMPEG_URLS
tar -xJf "$WORK/ff/ffmpeg.tar.xz" -C "$WORK/ff"
for tool in ffmpeg ffprobe; do
    found="$(find "$WORK/ff" -type f -name "$tool" -perm -u+x | head -n 1)"
    [[ -n "$found" ]] || die "$tool not found in the ffmpeg download"
    cp "$found" "$FF_DIR/$tool"
    chmod 755 "$FF_DIR/$tool"
    # Must not depend on libraries a distribution may not have (only glibc's own and libgcc_s)
    if readelf -d "$FF_DIR/$tool" | grep NEEDED \
        | grep -vE '\[(libc|libm|libmvec|libpthread|libdl|librt|libgcc_s|ld-linux-x86-64)\.so' ; then
        die "$tool links to libraries outside glibc"
    fi
done
license="$(find "$WORK/ff" -type f \( -iname 'GPL*.txt' -o -iname 'LICENSE*' -o -iname 'COPYING*' \) | head -n 1)"
[[ -n "$license" ]] && cp "$license" "$FF_DIR/LICENSE.txt"
"$FF_DIR/ffmpeg" -hide_banner -version | head -n 1
# Burning subtitles into a video needs libass (the list is read first: "grep -q" on a pipe would
# end ffmpeg early, and pipefail would count that as a failure)
filters="$("$FF_DIR/ffmpeg" -hide_banner -filters 2>/dev/null || true)"
grep -qE '^ *[.A-Z|]+ +subtitles ' <<<"$filters" || die "this ffmpeg build has no 'subtitles' filter (libass)"

download "$WORK/uv.tar.gz" "$UV_URL"
tar -xzf "$WORK/uv.tar.gz" -C "$WORK"
cp "$(find "$WORK" -maxdepth 2 -type f -name uv -path '*musl*' | head -n 1)" "$APPDIR/usr/bin/uv"
chmod 755 "$APPDIR/usr/bin/uv"
"$APPDIR/usr/bin/uv" --version

# -----------------------------------------------------------------------------------------------
step "5/9 X11 libraries for Qt"
for lib in $EXTRA_LIBS; do
    path="$(ldconfig -p | awk -v l="$lib" '$1 == l && /x86-64/ {print $NF; exit}')"
    [[ -n "$path" ]] || die "$lib is not installed on this machine (see 'Needs' at the top)"
    cp -L "$path" "$APPDIR/usr/lib/extra/$lib"
done
ls "$APPDIR/usr/lib/extra"
# Qt's X11 plugin must find everything it links to with these (and the system's X11 / OpenGL)
xcb_missing="$(LD_LIBRARY_PATH="$APPDIR/usr/lib/extra:$QT/Qt6/lib" ldd "$QT/Qt6/plugins/platforms/libqxcb.so" \
    | awk '/not found/ {print $1}' | grep -E '^lib(xcb|xkbcommon)' | sort -u || true)"
[[ -z "$xcb_missing" ]] || die "Qt's X11 plugin needs libraries that are not bundled:"$'\n'"$xcb_missing"

# -----------------------------------------------------------------------------------------------
step "6/9 Assembling the AppDir"
install -m 755 "$REPO/packaging/linux/AppRun" "$APPDIR/AppRun"
cp "$REPO/packaging/linux/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
"$PY" - "$REPO/packaging/icon.png" "$APPDIR/$APP_ID.png" <<'EOF'
import sys
from PIL import Image
Image.open(sys.argv[1]).convert("RGBA").resize((256, 256), Image.LANCZOS).save(sys.argv[2])
EOF
ln -s "$APP_ID.png" "$APPDIR/.DirIcon"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp "$APPDIR/$APP_ID.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_ID.png"
# Precompiled bytecode that never checks the sources: the app starts faster from the read-only
# AppImage, where Python could not write it
"$PY" -m compileall -q -j 0 --invalidation-mode unchecked-hash "$APPDIR/usr/python/lib" "$SRC" >/dev/null || true
# Qt WebEngine creates this folder on first start (owner-only); in the read-only AppImage it can't
mkdir -p "$QT/Qt6/resources/locales"
# Readable by everyone: the AppImage keeps these modes, and it runs as users other than the builder
# (some wheels ship folders only their owner can read)
chmod -R a+rX "$APPDIR"

# -----------------------------------------------------------------------------------------------
step "7/9 Smoke test (AppDir)"
SMOKE_HOME="$WORK/smoke-home"   # a throwaway data folder: the build machine's own stays untouched
# The executable itself (Python + every import the studio needs + the Qt window), then a real start
# of the server without a window
with_display env SUBBUILD_HOME="$SMOKE_HOME" CI="${CI:-1}" "$APPDIR/AppRun" --check
smoke_server() {   # smoke_server <command...>
    local log="$WORK/smoke.log" pid ok=0 code=000
    # (in its own process group: the AppImage runtime starts the app as a child, which must stop too)
    SUBBUILD_HOME="$SMOKE_HOME" SUBBUILD_NO_DESKTOP_ENTRY=1 CI="${CI:-1}" setsid "$@" --no-tray --no-browser >"$log" 2>&1 &
    pid=$!
    for _ in $(seq 1 90); do
        code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/system/info || true)"
        if [[ "$code" != "000" ]]; then ok=1; break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do   # the port must be free again for the next start
        [[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/system/info || true)" == "000" ]] && break
        sleep 0.5
    done
    cat "$log"
    [[ "$ok" -eq 1 ]] || die "the studio's server did not start"
    echo "    the server answered (HTTP $code)"
}
smoke_server "$APPDIR/AppRun"

# -----------------------------------------------------------------------------------------------
if [[ "${SKIP_APPIMAGE:-0}" == "1" ]]; then
    echo; echo "Built $APPDIR ($VERSION)"; exit 0
fi
step "8/9 AppImage"
chmod -R a+rX "$APPDIR"   # again: the smoke test ran Qt WebEngine from the AppDir
TOOL="$WORK/appimagetool"
download "$TOOL" "$APPIMAGETOOL_URL"
chmod 755 "$TOOL"
OUT="$OUT_DIR/$APPIMAGE_NAME"
rm -f "$OUT" "$OUT.zsync"
# (extract-and-run: build machines often have no FUSE)
( cd "$OUT_DIR" && APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 VERSION="$VERSION" \
    "$TOOL" --no-appstream --comp zstd -u "$UPDATE_INFO" "$APPDIR" "$OUT" )
[[ -f "$OUT" ]] || die "appimagetool did not write $OUT"
chmod 755 "$OUT"

# -----------------------------------------------------------------------------------------------
step "9/9 Smoke test (AppImage)"
with_display env SUBBUILD_HOME="$SMOKE_HOME" CI="${CI:-1}" APPIMAGE_EXTRACT_AND_RUN=1 "$OUT" --check
APPIMAGE_EXTRACT_AND_RUN=1 smoke_server "$OUT"
du -sh "$APPDIR" "$OUT"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    { echo "version=$VERSION"; echo "appimage=$OUT"; echo "zsync=$OUT.zsync"; } >> "$GITHUB_OUTPUT"
fi
echo; echo "Built $OUT ($VERSION)"
