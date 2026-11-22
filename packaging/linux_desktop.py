"""
SubBuild Studio - what the desktop launcher (packaging/launcher.py) needs only in the Linux AppImage.

The AppImage (packaging/linux/build-appimage.sh) shows the studio in a Qt window (pywebview's Qt
backend with Qt WebEngine, both shipped in it), so this module uses Qt for everything else the
desktop needs too:
  - the tray icon: QSystemTrayIcon speaks StatusNotifierItem, which KDE, Ubuntu's GNOME
    (AppIndicator), Cinnamon and XFCE show; pystray's X11 icon is invisible on most of them and
    blocks the app from quitting when no tray takes it
  - message boxes: a Qt dialog in a child process, so it can be shown from any thread
  - a probe that Qt can open a window here at all (a missing library makes Qt abort the process,
    which no try/except could catch), so the launcher falls back to the browser instead

It also adds the AppImage to the applications menu on first launch, and preloads the few X11
libraries that ship in the AppImage (usr/lib/extra) for desktops that lack them.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional, Sequence

APP_NAME = "SubBuild Studio"
APP_ID = "subbuildstudio"        # the .desktop file, the icon and the window's class
DESKTOP_MARKER = "desktop-entry.json"

IDYES, IDNO, IDCANCEL, IDOK = 6, 7, 2, 1
_EXIT_CODES = {IDYES: 10, IDNO: 11, IDCANCEL: 12, IDOK: 13}
_FROM_EXIT = {v: k for k, v in _EXIT_CODES.items()}


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def python_command(launcher: Path, *args: str) -> List[str]:
    """Run the launcher again on this Python (isolated: the AppImage's own packages only)."""
    return [sys.executable, "-I", "-B", "-X", "utf8", str(launcher), *args]


# ----------------------------------------------------------------------------- libraries

def preload_bundled_libs(appdir: Path) -> None:
    """Load the X11 libraries from usr/lib/extra that the system does not have.

    Qt's X11 plugin links to libxcb-cursor and a few others that minimal desktops lack. Loading
    them by full path before Qt starts satisfies those links, without setting LD_LIBRARY_PATH
    (which would leak into every program the studio opens). The system's own copy wins when there
    is one.
    """
    extra = appdir / "usr" / "lib" / "extra"
    pending = sorted(extra.glob("*.so*")) if extra.is_dir() else []
    mode = getattr(os, "RTLD_GLOBAL", 0)
    remaining = []
    for lib in pending:
        try:
            ctypes.CDLL(lib.name, mode=mode)   # the system's
        except OSError:
            remaining.append(lib)
    # Bundled ones may need each other: load in rounds until nothing more succeeds
    while remaining:
        failed = []
        for lib in remaining:
            try:
                ctypes.CDLL(str(lib), mode=mode)
            except OSError:
                failed.append(lib)
        if len(failed) == len(remaining):
            break
        remaining = failed


# ----------------------------------------------------------------------------- Qt

def qt_arguments(argv0: str) -> List[str]:
    """QApplication's arguments: X11 by default, and Qt WebEngine without Chromium's sandbox.

    - XWayland when there is one: pywebview's Qt backend and the tray icon are better tested
      there than on native Wayland (QT_QPA_PLATFORM still overrides it).
    - Chromium's sandbox needs unprivileged user namespaces, which Ubuntu 24.04+ restricts for
      programs outside /usr. The window shows only the studio's own local pages (other links open
      in the default browser). Passed as an argument, not as QTWEBENGINE_DISABLE_SANDBOX, so it
      does not leak into Qt browsers the studio may open. SUBBUILD_WEBENGINE_SANDBOX=1 keeps it.
    """
    args = [argv0]
    if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        args += ["-platform", "xcb"]
    if os.environ.get("SUBBUILD_WEBENGINE_SANDBOX") != "1":
        args.append("--no-sandbox")
    return args


def qt_application():  # type: ignore[no-untyped-def]
    """The process's QApplication (pywebview reuses it), named so the desktop matches its icon."""
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QIcon
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        if hasattr(Qt, "AA_ShareOpenGLContexts"):  # Qt WebEngine wants it set before the app
            QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
        app = QApplication(qt_arguments(APP_ID))
    app.setApplicationName(APP_ID)           # X11 window class (StartupWMClass)
    app.setApplicationDisplayName(APP_NAME)
    app.setDesktopFileName(APP_ID)           # Wayland app id -> subbuildstudio.desktop
    icon = Path(__file__).resolve().parent / "icon.png"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    app.setQuitOnLastWindowClosed(False)     # the tray icon may keep it running
    return app


PROBE_PAGE = "<!doctype html><title>probe</title><p>ok</p>"


def probe_window(full: bool) -> int:
    """Child-process check that Qt can show a window here (0 = yes).

    full=False: Qt's platform plugin and Qt WebEngine load (fast, done at every start).
    full=True:  Chromium really renders a page (for --check on the build machine).
    """
    try:
        from qtpy.QtWebEngineWidgets import QWebEngineView  # noqa: F401  (loads Qt WebEngine)
        app = qt_application()
        if not full:
            return 0
        from qtpy.QtCore import QTimer

        view = QWebEngineView()
        result = {"ok": False}

        def finished(ok: bool) -> None:
            result["ok"] = bool(ok)
            app.quit()

        view.loadFinished.connect(finished)
        QTimer.singleShot(45000, app.quit)
        view.setHtml(PROBE_PAGE)
        view.resize(320, 200)
        view.show()
        app.exec() if hasattr(app, "exec") else app.exec_()
        print(f"Qt WebEngine rendered a page: {result['ok']}", file=sys.stderr, flush=True)
        return 0 if result["ok"] else 1
    except Exception as e:
        print(f"Qt window probe failed: {e}", file=sys.stderr, flush=True)
        return 1


def window_problem(launcher: Path, full: bool = False) -> Optional[str]:
    """None when the studio can open its Qt window; otherwise the reason it can't."""
    if not has_display():
        return "no graphical display (DISPLAY / WAYLAND_DISPLAY is not set)"
    try:
        run = subprocess.run(python_command(launcher, "--probe-window", "full" if full else "quick"),
                             capture_output=True, text=True, timeout=90 if full else 45)
    except subprocess.TimeoutExpired:
        return "the Qt window did not start in time"
    except OSError as e:
        return f"the Qt window could not be tested ({e})"
    if run.returncode != 0:
        detail = (run.stderr or run.stdout or "").strip().splitlines()
        return "the Qt window cannot start here" + (f" ({detail[-1]})" if detail else "")
    if full and run.stderr.strip():
        print(run.stderr.strip(), file=sys.stderr, flush=True)
    return None


# ----------------------------------------------------------------------------- message boxes

def show_dialog(kind: str, text: str) -> int:
    """The child process's side of message(): a Qt message box; the answer as the exit code."""
    try:
        from qtpy.QtWidgets import QMessageBox

        app = qt_application()  # noqa: F841  (kept alive while the box is shown)
        box = QMessageBox()
        box.setWindowTitle(APP_NAME)
        box.setText(text)
        box.setIcon({"error": QMessageBox.Critical, "info": QMessageBox.Information}.get(kind, QMessageBox.Question))
        if kind == "yesno":
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.Yes)
        elif kind == "yesnocancel":
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Yes)
        else:
            box.setStandardButtons(QMessageBox.Ok)
        box.raise_()
        box.activateWindow()
        box.exec() if hasattr(box, "exec") else box.exec_()
        clicked = box.clickedButton()
        answer = box.standardButton(clicked) if clicked is not None else None
        mapped = IDOK
        for button, code in ((QMessageBox.Yes, IDYES), (QMessageBox.No, IDNO), (QMessageBox.Cancel, IDCANCEL)):
            if answer == button:
                mapped = code
        if kind == "yesno" and mapped == IDOK:   # closed with Esc / the X
            mapped = IDNO
        if kind == "yesnocancel" and mapped == IDOK:
            mapped = IDCANCEL
        return _EXIT_CODES[mapped]
    except Exception as e:
        print(f"Dialog failed: {e}", file=sys.stderr, flush=True)
        return 1


def dialog(launcher: Path, text: str, kind: str) -> Optional[int]:
    """A message box (IDYES / IDNO / IDCANCEL / IDOK); None when none could be shown."""
    if not has_display():
        return None
    try:
        run = subprocess.run(python_command(launcher, "--dialog", kind, text), timeout=3600)
    except Exception:
        return None
    return _FROM_EXIT.get(run.returncode)


# ----------------------------------------------------------------------------- tray icon

class QtTray:
    """The tray icon on Qt, with the small part of pystray's Icon that launcher.py uses.

    Created on the GUI thread once the QApplication exists; title / notify / stop may be called
    from any thread (they are forwarded to the GUI thread).
    """

    def __init__(self, title: str, icon_file: Path, items: Sequence[tuple]) -> None:
        from qtpy.QtCore import QObject, Signal
        from qtpy.QtGui import QIcon
        from qtpy.QtWidgets import QMenu, QSystemTrayIcon

        class _Invoker(QObject):
            call = Signal(object)

        self._invoker = _Invoker()
        self._invoker.call.connect(lambda fn: fn())
        self._tray = QSystemTrayIcon(QIcon(str(icon_file)))
        self._menu = QMenu()
        self._default: Optional[Callable[[], None]] = None
        for item in items:
            if item is None:
                self._menu.addSeparator()
                continue
            label, action, is_default = item
            entry = self._menu.addAction(label)
            entry.triggered.connect(lambda _checked=False, fn=action: fn())
            if is_default:
                self._default = action
                font = entry.font()
                font.setBold(True)
                entry.setFont(font)
        self._tray.setContextMenu(self._menu)
        self._tray.activated.connect(self._activated)
        self._tray.setToolTip(title)

    @staticmethod
    def available() -> bool:
        from qtpy.QtWidgets import QSystemTrayIcon

        return bool(QSystemTrayIcon.isSystemTrayAvailable())

    def _activated(self, reason) -> None:  # type: ignore[no-untyped-def]
        from qtpy.QtWidgets import QSystemTrayIcon

        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick) and self._default:
            self._default()

    def _gui(self, fn: Callable[[], None]) -> None:
        if threading.current_thread() is threading.main_thread():
            fn()
        else:
            self._invoker.call.emit(fn)

    # pystray-like interface
    @property
    def visible(self) -> bool:
        return bool(self._tray.isVisible())

    @visible.setter
    def visible(self, on: bool) -> None:
        self._gui(lambda: self._tray.setVisible(bool(on)))

    @property
    def title(self) -> str:
        return str(self._tray.toolTip())

    @title.setter
    def title(self, text: str) -> None:
        self._gui(lambda: self._tray.setToolTip(text))

    def run_detached(self) -> None:
        self.visible = True

    def notify(self, text: str, title: Optional[str] = None) -> None:
        self._gui(lambda: self._tray.showMessage(title or APP_NAME, text))

    def stop(self) -> None:
        self._gui(lambda: self._tray.hide())


class NoTray:
    """Stands in for the tray icon where the desktop has no tray (launcher.py keeps working)."""

    visible = False
    title = ""

    def run_detached(self) -> None:
        pass

    def notify(self, text: str, title: Optional[str] = None) -> None:
        pass

    def stop(self) -> None:
        pass


# ----------------------------------------------------------------------------- applications menu

def _exec_quote(path: str) -> str:
    """A path as one argument of a .desktop Exec line (Desktop Entry Specification)."""
    escaped = path.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return '"' + escaped.replace("%", "%%") + '"'


def desktop_entry(appimage: str, icon: str) -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "GenericName=Subtitle Editor & Translator\n"
        "Comment=AI subtitles: transcribe, translate and edit\n"
        f"Exec={_exec_quote(appimage)}\n"
        f"TryExec={appimage}\n"
        f"Icon={icon}\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Video;\n"
        "Keywords=subtitles;subtitle;translation;video;transcription;\n"
        f"StartupWMClass={APP_ID}\n"
        "X-AppImage-Integrate=false\n"
    )


def integrate_appimage(appimage: str, home: Path, icon_png: Path,
                       data_home: Optional[Path] = None) -> Optional[Path]:
    """Add the AppImage to the applications menu (first launch) and keep the entry pointing at it.

    The entry follows the AppImage when it is moved or replaced by a newer download. Once added,
    removing it from the menu is respected: it is not added again (delete DESKTOP_MARKER in the
    data folder to get it back). SUBBUILD_NO_DESKTOP_ENTRY=1 skips it altogether.
    Returns the .desktop file when it was written.
    """
    if os.environ.get("SUBBUILD_NO_DESKTOP_ENTRY") == "1" or not appimage:
        return None
    base = data_home or Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    apps = base / "applications"
    desktop_file = apps / f"{APP_ID}.desktop"
    icon_file = base / "icons" / "hicolor" / "256x256" / "apps" / f"{APP_ID}.png"
    marker = home / DESKTOP_MARKER
    if marker.exists() and not desktop_file.exists():
        return None   # the user removed it
    content = desktop_entry(appimage, APP_ID)
    try:
        if desktop_file.exists() and desktop_file.read_text(encoding="utf-8") == content and icon_file.exists():
            return None
        icon_file.parent.mkdir(parents=True, exist_ok=True)
        if icon_png.is_file():
            icon_file.write_bytes(icon_png.read_bytes())
        apps.mkdir(parents=True, exist_ok=True)
        tmp = desktop_file.with_suffix(".desktop.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.chmod(0o755)
        os.replace(tmp, desktop_file)
        marker.write_text(json.dumps({"appimage": appimage, "desktop_file": str(desktop_file)}), encoding="utf-8")
    except OSError as e:
        print(f"Could not add {APP_NAME} to the applications menu: {e}", file=sys.stderr, flush=True)
        return None
    for tool in (["update-desktop-database", str(apps)],
                 ["gtk-update-icon-cache", "-q", "-t", str(base / "icons" / "hicolor")]):
        try:
            subprocess.run(tool, capture_output=True, timeout=20, check=False)
        except Exception:
            pass
    return desktop_file
