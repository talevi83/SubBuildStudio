"""
SubBuild Studio desktop launcher (invoked via run.bat / run.sh --window, the Windows installer's
shortcut, the macOS app's executable - packaging/macos/launcher.c - and the Linux AppImage's
AppRun - packaging/linux/AppRun).

Starts the FastAPI backend inside this process and shows the studio in its own desktop window
(pywebview on Microsoft Edge WebView2, which ships with Windows 10/11, on WebKit on macOS, or on
Qt WebEngine in the Linux AppImage - packaging/linux_desktop.py), with an icon in the system tray /
menu bar. Without a web view - or with --browser - the studio
opens in the default browser instead, and the tray icon is the way back to it.

User data (.env with the API key, storage/ with projects, settings and logs, the window's
size and the studio's own preferences) is kept outside the program directory by default, so
updating or switching branches never touches it:
  1. %SUBBUILD_HOME%, if set (run.bat / run.sh point this to backend/)
  2. a "data" folder in the program root (portable mode - create the folder to opt in)
  3. %LOCALAPPDATA%\\SubBuild Studio (Windows), ~/Library/Application Support/SubBuild Studio (macOS),
     ~/.local/share/SubBuild Studio (Linux)

Also runs directly from source:  python packaging/launcher.py [--browser] [--no-tray] [--no-browser]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable, Optional

APP_NAME = "SubBuild Studio"
WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"
# Set by the macOS app's executable (packaging/macos/launcher.c): the path of SubBuild Studio.app
APP_BUNDLE = os.environ.get("SUBBUILD_APP_BUNDLE") if MACOS else None
LINUX = sys.platform.startswith("linux")
# Set by the AppImage's AppRun (packaging/linux/AppRun): the mounted AppDir. The Qt window, tray
# and dialogs of packaging/linux_desktop.py are used only there.
APPDIR = os.environ.get("SUBBUILD_APPDIR") if LINUX else None
# Set by the AppImage runtime: the .AppImage file itself (what a restart has to run again)
APPIMAGE = os.environ.get("APPIMAGE") if APPDIR else None
STARTUP_TIMEOUT = 180  # seconds; the first start preloads the CUDA libraries, which can be slow
ACTIVE_STATUSES = {"QUEUED", "PROBING", "EXTRACTING_AUDIO", "DUAL_SOURCE_SYNC", "VOCAL_ISOLATION", "TRANSCRIBING",
                   "DIARIZING", "SCANNING_ONSCREEN", "REFINING", "TRANSLATING", "SYNCING"}
WINDOW_SIZE = (1440, 900)
MIN_WINDOW_SIZE = (1024, 640)
IDYES, IDNO = 6, 7

# The running server's session token (app.core.auth): sent by the launcher's own API calls and
# put in the address the window / browser opens. Saved in the data folder so a second launch can
# talk to the copy that is already running.
SESSION_TOKEN: Optional[str] = None
TOKEN_FILE_NAME = ".session-token"

# Window mode: brings the window to the front. A second launch of the exe calls it through the
# local server (POST /api/desktop/show) instead of starting another copy.
SHOW_WINDOW: Optional[Callable[[], None]] = None

LOADING_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#0b0b10;color:#e4e4e7;font-family:Segoe UI,system-ui,sans-serif}
body{display:flex;align-items:center;justify-content:center}
.box{text-align:center}.spin{width:34px;height:34px;margin:0 auto 18px;border:3px solid #27272a;
border-top-color:#6366f1;border-radius:50%;animation:s 0.9s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
h1{font-size:18px;font-weight:600;margin:0 0 6px}p{font-size:12px;color:#a1a1aa;margin:2px 0}
</style></head><body><div class="box"><div class="spin"></div><h1>SubBuild Studio</h1>
<p>Starting the studio&hellip;</p><p dir="rtl">&#1502;&#1508;&#1506;&#1497;&#1500; &#1488;&#1514; &#1492;&#1505;&#1496;&#1493;&#1491;&#1497;&#1493;&hellip;</p>
</div></body></html>"""


# ----------------------------------------------------------------------------- paths

def bundle_dir() -> Path:
    """Root of the repository / application directory."""
    return Path(__file__).resolve().parent.parent


def resource(*parts: str) -> Path:
    return bundle_dir().joinpath(*parts)


def program_dir() -> Path:
    return bundle_dir()


def data_dir() -> Path:
    env = os.environ.get("SUBBUILD_HOME") or os.environ.get("SUBFORGE_HOME")  # old name still honoured
    if env:
        return Path(env).expanduser().resolve()
    portable = program_dir() / "data"
    if portable.is_dir():
        return portable
    if WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".local" / "share" if not os.environ.get("XDG_DATA_HOME") else Path(os.environ["XDG_DATA_HOME"])
    target = base / APP_NAME
    legacy = base / "SubForge Studio"  # the app's name before the rebrand: keep users' data
    if not target.exists() and legacy.is_dir():
        try:
            legacy.rename(target)
        except OSError:
            return legacy  # in use / locked: keep working from the old folder
    return target


# ----------------------------------------------------------------------------- process setup

def redirect_stdio(home: Path) -> None:
    """A windowed launcher has no console: send print/tracebacks to launcher.log.

    (pythonw.exe on Windows has no stdout/stderr at all; the macOS app started from Finder and the
    AppImage started from the applications menu have them, but they lead nowhere.)
    """
    from_finder = bool(APP_BUNDLE or APPDIR) and not (sys.stderr is not None and sys.stderr.isatty())
    if sys.stdout is not None and sys.stderr is not None and not from_finder:
        return
    log = home / "launcher.log"
    try:
        if log.exists() and log.stat().st_size > 5 * 1024 * 1024:
            log.unlink()
    except OSError:
        pass
    stream = open(log, "a", encoding="utf-8", buffering=1, errors="replace")
    if from_finder:
        sys.stdout = sys.stderr = stream
        return
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def hide_child_consoles() -> None:
    """Without this every ffmpeg/ffprobe/nvidia-smi/git call would flash a black console window."""
    if not (WINDOWS and Path(sys.executable).name.lower() == "pythonw.exe"):
        return
    create_no_window = 0x08000000
    original_init = subprocess.Popen.__init__

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["creationflags"] = (kwargs.get("creationflags") or 0) | create_no_window
        original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__  # type: ignore[method-assign]


def prepare_environment(home: Path) -> Path:
    """Create the data folder, a first .env, and point the backend at the bundled files."""
    home.mkdir(parents=True, exist_ok=True)
    env_file = home / ".env"
    if not env_file.exists():
        example = resource("backend", ".env.example")
        if example.exists():
            shutil.copyfile(example, env_file)
    # The backend reads ./.env and keeps ./storage relative to the working directory
    os.chdir(home)
    os.environ.setdefault("SUBBUILD_FRONTEND_DIST", str(resource("frontend", "dist")))
    for ffmpeg_dir in (resource("packaging", "ffmpeg"), resource("ffmpeg")):
        if ffmpeg_dir.is_dir():  # optional local ffmpeg overrides PATH
            os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
            break
    if APPDIR:
        # The optional NVIDIA libraries for GPU transcription are downloaded into the data folder
        # on request (backend/app/core/gpu_runtime.py), with the versions this build pinned
        os.environ.setdefault("SUBBUILD_GPU_DIR", str(home / "gpu-libs"))
        os.environ.setdefault("SUBBUILD_GPU_REQUIREMENTS", str(resource("packaging", "gpu-requirements.txt")))
        os.environ.setdefault("SUBBUILD_UV", str(Path(APPDIR) / "usr" / "bin" / "uv"))
    sys.path.insert(0, str(resource("backend")))
    return env_file


# ----------------------------------------------------------------------------- helpers

# AppleScript dialogs for macOS: the text comes in as an argument, so it needs no quoting
_MAC_DIALOG_BUTTONS = {
    "error": ('{"OK"}', '"OK"', None, "stop"),
    "info": ('{"OK"}', '"OK"', None, "note"),
    "yesno": ('{"No", "Yes"}', '"Yes"', '"No"', "caution"),
    "yesnocancel": ('{"Cancel", "No", "Yes"}', '"Yes"', '"Cancel"', "caution"),
}


def _mac_dialog(text: str, kind: str) -> Optional[int]:
    """A dialog on macOS (IDYES / IDNO / 2 for cancel, 1 for OK); None when osascript fails."""
    buttons, default, cancel, icon = _MAC_DIALOG_BUTTONS[kind]
    script = (f"display dialog (item 1 of argv) with title (item 2 of argv) buttons {buttons} "
              f"default button {default}" + (f" cancel button {cancel}" if cancel else "") +
              f" with icon {icon}")
    try:
        result = subprocess.run(["osascript", "-e", "on run argv", "-e", script, "-e", "end run",
                                 text, APP_NAME], capture_output=True, text=True, timeout=3600)
    except Exception:
        return None
    answer = result.stdout.strip()
    if answer.endswith(":Yes"):
        return IDYES
    if answer.endswith(":No"):
        return IDNO
    if answer.endswith(":OK"):
        return 1
    if "-128" in result.stderr:  # the cancel button (or Esc)
        return IDNO if kind == "yesno" else 2
    return None


def message(text: str, kind: str = "error") -> int:
    """Native message box (returns IDYES / IDNO / 2 for cancel); stderr where there is none."""
    if WINDOWS:
        import ctypes
        flags = {"error": 0x10, "info": 0x40, "yesno": 0x04 | 0x30, "yesnocancel": 0x03 | 0x30}[kind]
        return ctypes.windll.user32.MessageBoxW(None, text, APP_NAME, flags | 0x40000)  # topmost
    print(f"[{kind}] {text}", file=sys.stderr, flush=True)
    if os.environ.get("CI"):  # a dialog on a build machine would wait forever
        return IDYES
    if MACOS:
        answer = _mac_dialog(text, kind)
        if answer is not None:
            return answer
    if APPDIR:
        answer = linux_desktop().dialog(Path(__file__).resolve(), text, kind)
        if answer is not None:
            return answer
    return IDYES


def on_main_thread(fn: Callable[[], None]) -> None:
    """Run fn on the UI thread: AppKit (the macOS menu bar icon) must not be touched from others."""
    if MACOS:
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(fn)
            return
        except Exception:
            pass
    fn()


def open_path(path: Path) -> None:
    if not path.exists():
        message(f"Not found yet:\n{path}", "info")
        return
    if WINDOWS:
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        # -t: files such as .env and server.log have no app of their own; open them in TextEdit
        subprocess.Popen(["open", "-t", str(path)] if path.is_file() else ["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


_no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_json(url: str, timeout: float = 2.0, post: bool = False):
    req = urllib.request.Request(url, data=b"" if post else None, method="POST" if post else "GET")
    if SESSION_TOKEN:
        req.add_header("X-SubBuild-Token", SESSION_TOKEN)
    with _no_proxy.open(req, timeout=timeout) as response:
        return json.loads(response.read())


def is_our_server(url: str) -> bool:
    try:
        info = get_json(url + "/api/system/info")
        return isinstance(info, dict) and "whisper_models" in info
    except Exception:
        return False


def show_running_window(url: str) -> bool:
    """Ask the copy that is already running to bring its window to the front."""
    try:
        return bool(get_json(url + "/api/desktop/show", timeout=3, post=True).get("shown"))
    except Exception:
        return False


def studio_url(url: str) -> str:
    """The address that opens the studio and signs the window / browser in to this session."""
    return f"{url}/?token={SESSION_TOKEN}" if SESSION_TOKEN else url


def read_saved_token(home: Path) -> Optional[str]:
    try:
        return (home / TOKEN_FILE_NAME).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def new_session_token(home: Path) -> str:
    """A fresh token for the server this launch starts (read by app.core.auth at import)."""
    token = secrets.token_urlsafe(32)
    os.environ["SUBBUILD_TOKEN"] = token
    try:
        # Readable by this Windows account only: the portable data folder (or one on another
        # drive) may otherwise be readable by every user of the PC, who could then use the API.
        from app.core.private_files import write_private_text

        write_private_text(home / TOKEN_FILE_NAME, token)
    except OSError:
        pass
    return token


def port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def active_job_count(url: str) -> int:
    try:
        return sum(1 for job in get_json(url + "/api/jobs", timeout=3) if job.get("status") in ACTIVE_STATUSES)
    except Exception:
        return 0


def confirm_stop(url: str, action: str) -> bool:
    busy = active_job_count(url)
    if not busy:
        return True
    return message(f"{busy} project(s) are still running. {action} anyway?\n"
                   "They can be resumed later.", "yesno") == IDYES


def spawn_again(extra_args: list) -> None:
    if APP_BUNDLE:  # macOS app: start the app again (Python on its own would not be "SubBuild Studio")
        subprocess.Popen(["open", "-n", APP_BUNDLE, "--args"] + list(extra_args))
        return
    if APPIMAGE:  # the AppImage: its files go away with this process, so run the .AppImage again
        subprocess.Popen([APPIMAGE] + list(extra_args), start_new_session=True)
        return
    args = [sys.executable, str(Path(__file__).resolve())]
    subprocess.Popen(args + extra_args, cwd=str(program_dir()))


# ----------------------------------------------------------------------------- Linux AppImage

_QT_APP = None  # the QApplication of the Qt window (AppImage, window mode)


def linux_desktop():  # type: ignore[no-untyped-def]
    """packaging/linux_desktop.py (imported only in the AppImage)."""
    import importlib

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module("linux_desktop")


def window_mode_qt() -> bool:
    return _QT_APP is not None


def _call_menu_action(fn: Callable) -> None:
    """Call a tray callback the way pystray does: with the icon when it takes an argument."""
    import inspect

    try:
        takes_icon = len(inspect.signature(fn).parameters) > 0
    except (TypeError, ValueError):
        takes_icon = False
    fn(None) if takes_icon else fn()


def integrate_with_desktop(home: Path) -> None:
    """AppImage: add it to the applications menu on first launch, and keep the entry up to date."""
    if not APPIMAGE:
        return
    icon = Path(APPDIR or "") / "subbuildstudio.png"
    try:
        linux_desktop().integrate_appimage(APPIMAGE, home, icon if icon.is_file() else resource("packaging", "icon.png"))
    except Exception:
        print(f"Could not add the applications menu entry:\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def isolate_child_environment() -> None:
    """AppImage: AppRun's Python settings apply to this interpreter only, not to the programs the
    studio opens (a file manager, an editor, the browser). Our own Python children get them as
    command-line flags instead (linux_desktop.python_command)."""
    if not APPDIR:
        return
    for name in ("PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTHONUTF8"):
        os.environ.pop(name, None)


# ----------------------------------------------------------------------------- server

def _add_desktop_route(app) -> None:  # type: ignore[no-untyped-def]
    """POST /api/desktop/show - lets a second launch bring this window to the front."""
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def show(_request):  # type: ignore[no-untyped-def]
        callback = SHOW_WINDOW
        if callback is not None:
            threading.Thread(target=callback, daemon=True).start()
        return JSONResponse({"shown": callback is not None})

    # Before the static frontend mounted at "/", which would otherwise answer first
    app.router.routes.insert(0, Route("/api/desktop/show", show, methods=["POST"]))


class ServerThread(threading.Thread):
    def __init__(self, host: str, port: int) -> None:
        super().__init__(name="uvicorn", daemon=True)
        self.host, self.port = host, port
        self.server = None
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            import uvicorn
            from app.main import app

            _add_desktop_route(app)
            config = uvicorn.Config(app, host=self.host, port=self.port, log_config=None, access_log=False)
            self.server = uvicorn.Server(config)
            self.server.run()
            if not self.server.started and self.error is None:
                self.error = f"The server did not start (is port {self.port} free?)."
        except BaseException:  # SystemExit from uvicorn included
            self.error = traceback.format_exc()

    def wait_ready(self, timeout: float = STARTUP_TIMEOUT) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.server is not None and self.server.started:
                return True
            if not self.is_alive():
                return False
            time.sleep(0.25)
        return False

    def stop(self, timeout: float = 20.0) -> None:
        if self.server is not None:
            self.server.should_exit = True
        self.join(timeout)


def failure_text(server: ServerThread, home: Path) -> str:
    detail = (server.error or "Startup timed out.").strip().splitlines()
    return ("SubBuild Studio could not start.\n\n" + "\n".join(detail[-6:]) +
            f"\n\nLogs: {home / 'storage' / 'logs' / 'server.log'}\n{home / 'launcher.log'}")


def exit_now(code: int = 0) -> None:
    """Exit even if a transcription worker thread is still busy (jobs resume as interrupted)."""
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)


# ----------------------------------------------------------------------------- tray

def make_tray(url: str, home: Path, env_file: Path, open_label: str, on_open: Callable[[], None],
              on_quit: Callable, on_restart: Callable, with_browser_item: bool):
    """The tray icon and its menu (shared by window and browser mode)."""
    if APPDIR:  # Qt's tray icon (needs the QApplication: window mode only), or none
        ld = linux_desktop()
        if not window_mode_qt():
            return ld.NoTray()
        items = [(open_label, on_open, True)]
        if with_browser_item:
            items.append(("Open in browser", lambda: webbrowser.open(studio_url(url)), False))
        items += [None,
                  ("Open data folder", lambda: open_path(home), False),
                  ("Edit settings (.env)", lambda: open_path(env_file), False),
                  ("Server log", lambda: open_path(home / "storage" / "logs" / "server.log"), False),
                  None,
                  ("Restart server", lambda: _call_menu_action(on_restart), False),
                  ("Quit", lambda: _call_menu_action(on_quit), False)]
        return ld.QtTray(f"{APP_NAME} - starting...", resource("packaging", "icon.png"), items)
    import pystray
    from PIL import Image

    items = [pystray.MenuItem(open_label, lambda: on_open(), default=True)]
    if with_browser_item:
        items.append(pystray.MenuItem("Open in browser", lambda: webbrowser.open(studio_url(url))))
    items += [
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open data folder", lambda: open_path(home)),
        pystray.MenuItem("Edit settings (.env)", lambda: open_path(env_file)),
        pystray.MenuItem("Server log", lambda: open_path(home / "storage" / "logs" / "server.log")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Restart server", on_restart),
        pystray.MenuItem("Quit", on_quit),
    ]
    icon_file = resource("packaging", "icon.ico" if WINDOWS else "icon.png")
    return pystray.Icon("SubBuildStudio", Image.open(icon_file),
                        f"{APP_NAME} - starting...", pystray.Menu(*items))


def run_tray(server: ServerThread, url: str, home: Path, env_file: Path, open_browser: bool) -> None:
    """Browser mode: the tray icon runs on the main thread; the studio opens in the browser."""

    def quit_app(icon) -> None:  # type: ignore[no-untyped-def]
        if not confirm_stop(url, "Quit"):
            return
        icon.visible = False
        server.stop()
        icon.stop()

    def restart(icon) -> None:  # type: ignore[no-untyped-def]
        if not confirm_stop(url, "Restart"):
            return
        icon.visible = False
        server.stop()
        spawn_again(["--browser", "--no-browser"])
        icon.stop()

    icon = make_tray(url, home, env_file, "Open SubBuild Studio", lambda: webbrowser.open(studio_url(url)),
                     quit_app, restart, with_browser_item=False)

    def setup(icon) -> None:  # type: ignore[no-untyped-def]
        icon.visible = True
        if server.wait_ready():
            on_main_thread(lambda: setattr(icon, "title", f"{APP_NAME} - {url}"))
            if open_browser:
                webbrowser.open(studio_url(url))
            else:
                try:
                    icon.notify("Running in the tray. Click the icon to open the studio.", APP_NAME)
                except Exception:
                    pass
        else:
            message(failure_text(server, home))

            def close() -> None:
                icon.visible = False
                icon.stop()

            on_main_thread(close)

    icon.run(setup=setup)


# ----------------------------------------------------------------------------- desktop window

def window_problem() -> Optional[str]:
    """None when the studio can open in its own window; otherwise the reason it can't."""
    try:
        import webview  # noqa: F401
    except Exception as e:
        return f"pywebview is not available ({e})"
    if APPDIR:
        return linux_desktop().window_problem(Path(__file__).resolve())
    if WINDOWS:
        try:
            from webview.platforms import winforms
        except Exception as e:
            return f"the Windows web view could not start ({e})"
        # Without WebView2 pywebview falls back to Internet Explorer, which can't run the studio
        if getattr(winforms, "renderer", "") != "edgechromium":
            return "Microsoft Edge WebView2 Runtime is not installed"
    return None


def read_clipboard_text() -> Optional[str]:
    """Unicode text on the Windows clipboard ('' when it holds none); None where unsupported or busy.

    The page's own navigator.clipboard.readText() needs a permission WebView2 may refuse or
    prompt for, so the window's "Paste" menu asks the app through the bridge instead.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    cf_unicodetext = 13
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]

    for _ in range(10):  # another program may hold the clipboard for a moment
        if user32.OpenClipboard(None):
            break
        time.sleep(0.02)
    else:
        return None
    try:
        handle = user32.GetClipboardData(cf_unicodetext)
        if not handle:
            return ""
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return None
        try:
            return ctypes.wstring_at(pointer)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


MEDIA_FILE_TYPES = (
    "Media files (*.mkv;*.mp4;*.webm;*.avi;*.mov;*.m4v;*.ts;*.mpg;*.mpeg;*.wmv;*.flv;*.mp3;*.wav;*.m4a;*.flac;*.ogg;*.aac)",
    "All files (*.*)",
)


def _enable_webview_dnd() -> None:
    """Tell pywebview's platform backends (WebView2, Cocoa, Qt) to record dropped file paths."""
    try:
        from webview.dom import _dnd_state

        if _dnd_state.get("num_listeners", 0) < 1:
            _dnd_state["num_listeners"] = 1
    except Exception:
        pass


class DesktopApi:
    """Called from the page as window.pywebview.api.* (pywebview's JavaScript bridge)."""

    def __init__(self) -> None:
        self._window = None
        self._fullscreen = False
        _enable_webview_dnd()

    def set_fullscreen(self, on: bool) -> bool:
        """The video player went (or left) fullscreen: make the whole window follow."""
        on = bool(on)
        if self._window is not None and on != self._fullscreen:
            self._fullscreen = on
            self._window.toggle_fullscreen()
        return self._fullscreen

    def read_clipboard(self) -> Optional[str]:
        """Clipboard text for the "Paste" context menu (None: let the page try its own clipboard API)."""
        try:
            return read_clipboard_text()
        except Exception:
            return None

    def resolve_dropped_file(self, filename: str = "") -> Optional[str]:
        """Full disk path for a file just dropped into the window, or None when unavailable."""
        _enable_webview_dnd()
        try:
            from urllib.parse import unquote
            from webview.dom import _dnd_state
        except Exception:
            return None

        target = (filename or "").strip()
        for _ in range(10):
            paths = list(_dnd_state.get("paths") or [])
            if paths:
                matched = None
                for item in paths:
                    if not isinstance(item, (tuple, list)) or len(item) < 2:
                        continue
                    base, full = str(item[0]), str(item[1])
                    if not target or base == target or unquote(base) == target:
                        matched = item
                        break
                if matched is None and len(paths) == 1 and isinstance(paths[0], (tuple, list)) and len(paths[0]) >= 2:
                    matched = paths[0]
                if matched is not None:
                    try:
                        _dnd_state["paths"].remove(matched)
                    except ValueError:
                        pass
                    raw_path = str(matched[1])
                    if Path(raw_path).exists():
                        return raw_path
                    decoded = unquote(raw_path)
                    if Path(decoded).exists():
                        return decoded
                    return raw_path
            time.sleep(0.005)
        return None

    def pick_media_file(self) -> Optional[str]:
        """Open a native file picker for local media; returns the path, '' if cancelled, or None if unavailable."""
        if self._window is None:
            return None
        try:
            import webview

            dialog_type = getattr(getattr(webview, "FileDialog", None), "OPEN", getattr(webview, "OPEN_DIALOG", 10))
            picked = self._window.create_file_dialog(
                dialog_type,
                allow_multiple=False,
                file_types=MEDIA_FILE_TYPES,
            )
            if not picked:
                return ""
            first = picked[0] if isinstance(picked, (list, tuple)) else str(picked)
            return str(first) if first else ""
        except Exception:
            return None

    def pick_media_files(self) -> Optional[list]:
        """Open a native multi-file picker for local media; returns paths list, [] if cancelled, or None if unavailable."""
        if self._window is None:
            return None
        try:
            import webview

            dialog_type = getattr(getattr(webview, "FileDialog", None), "OPEN", getattr(webview, "OPEN_DIALOG", 10))
            picked = self._window.create_file_dialog(
                dialog_type,
                allow_multiple=True,
                file_types=MEDIA_FILE_TYPES,
            )
            if not picked:
                return []
            if isinstance(picked, (list, tuple)):
                return [str(p) for p in picked if p]
            return [str(picked)] if picked else []
        except Exception:
            return None


def _load_window_size(home: Path):
    try:
        data = json.loads((home / "window.json").read_text())
        w, h = int(data["width"]), int(data["height"])
        if w >= MIN_WINDOW_SIZE[0] and h >= MIN_WINDOW_SIZE[1]:
            return w, h
    except Exception:
        pass
    return WINDOW_SIZE


def _save_window_size(home: Path, window) -> None:  # type: ignore[no-untyped-def]
    try:
        w, h = int(window.width), int(window.height)
        if w >= MIN_WINDOW_SIZE[0] and h >= MIN_WINDOW_SIZE[1]:
            (home / "window.json").write_text(json.dumps({"width": w, "height": h}))
    except Exception:
        pass


def run_window(server: ServerThread, url: str, home: Path, env_file: Path) -> None:
    """Window mode: the studio in its own window (main thread); the tray icon runs alongside."""
    global SHOW_WINDOW, _QT_APP
    import webview

    if APPDIR:  # Qt: the application first (named for the desktop), the tray icon needs it
        _QT_APP = linux_desktop().qt_application()

    webview.settings["ALLOW_DOWNLOADS"] = True                 # exports ask where to save
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True  # e.g. the pricing page
    api = DesktopApi()
    width, height = _load_window_size(home)
    window = webview.create_window(
        APP_NAME, html=LOADING_HTML, width=width, height=height, min_size=MIN_WINDOW_SIZE,
        background_color="#0B0B10", js_api=api, text_select=True,
    )
    api._window = window
    state = {"leaving": False, "restart": False}

    def show_window() -> None:
        try:
            window.show()
            window.restore()
            window.on_top = True   # pull it in front of other windows, then let it behave normally
            window.on_top = False
        except Exception:
            pass

    def leave(restart: bool) -> None:
        if not confirm_stop(url, "Restart" if restart else "Quit"):
            return
        state.update(leaving=True, restart=restart)
        _save_window_size(home, window)
        window.destroy()

    icon = make_tray(url, home, env_file, "Open SubBuild Studio", show_window,
                     lambda: leave(False), lambda: leave(True), with_browser_item=True)
    # A Linux desktop may have no tray to come back from: minimize to the taskbar instead
    has_tray = True
    if APPDIR:
        ld = linux_desktop()
        has_tray = isinstance(icon, ld.QtTray) and ld.QtTray.available()

    def on_closing() -> bool:
        if state["leaving"]:
            return True
        busy = active_job_count(url)
        if busy:
            answer = message(
                f"{busy} project(s) are still running.\n\n"
                "Yes - quit now (they can be resumed later)\n"
                + ("No - keep working in the background (tray icon)\n" if has_tray else
                   "No - keep working in the background (minimized)\n") +
                "Cancel - go back", "yesnocancel")
            if answer == IDNO:
                if not has_tray:
                    window.minimize()
                    return False
                window.hide()
                try:
                    icon.notify("Still working in the background. Click the tray icon to open the window.", APP_NAME)
                except Exception:
                    pass
                return False
            if answer != IDYES:
                return False
        _save_window_size(home, window)
        state["leaving"] = True
        return True

    window.events.closing += on_closing

    def boot() -> None:
        if server.wait_ready():
            on_main_thread(lambda: setattr(icon, "title", f"{APP_NAME} - {url}"))
            window.load_url(studio_url(url))
        else:
            message(failure_text(server, home))
            state["leaving"] = True
            window.destroy()

    icon.run_detached()  # Windows: the tray gets its own thread; macOS / Qt: it shares the window's loop
    SHOW_WINDOW = show_window
    start_options = {}
    if APPDIR:
        start_options["gui"] = "qt"
    try:
        webview.start(boot, private_mode=False, storage_path=str(home / "webview"),
                      icon=str(resource("packaging", "icon.png" if APPDIR else "icon.ico")), **start_options)
    finally:
        SHOW_WINDOW = None
        try:
            icon.visible = False
            icon.stop()
        except Exception:
            pass
    server.stop()
    if state["restart"]:
        spawn_again([])


# ----------------------------------------------------------------------------- main

def self_check(home: Path) -> int:
    """--check: import the server, the speech engine and the window / tray packages."""
    import ctranslate2
    import faster_whisper
    import uvicorn  # noqa: F401
    from app.main import app  # noqa: F401

    # (in the AppImage the check also makes Qt WebEngine render a page, in a child process)
    problem = linux_desktop().window_problem(Path(__file__).resolve(), full=True) if APPDIR else window_problem()
    if problem is None and not APPDIR:
        import pystray  # noqa: F401
    print(f"{APP_NAME} check: Python {sys.version.split()[0]}, faster-whisper {faster_whisper.__version__}, "
          f"ctranslate2 {ctranslate2.__version__}, window: {problem or 'ok'}, data: {home}", flush=True)
    return 1 if problem else 0


def run_headless(server: ServerThread, url: str, home: Path, open_browser: bool) -> None:
    """No window and no tray icon: the server runs until Ctrl+C (or the process is ended)."""
    if not server.wait_ready():
        message(failure_text(server, home))
        exit_now(1)
    print(f"{APP_NAME} is running at {studio_url(url)} (data: {home}). Press Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(studio_url(url))
    try:
        while server.is_alive():
            server.join(0.5)
    except KeyboardInterrupt:
        server.stop()
    exit_now(0)


def main() -> int:
    global SESSION_TOKEN
    parser = argparse.ArgumentParser(prog=APP_NAME)
    parser.add_argument("--browser", action="store_true", help="open the studio in the browser, not in its own window")
    parser.add_argument("--no-tray", action="store_true", help="run in the foreground without a tray icon or window")
    parser.add_argument("--no-browser", action="store_true", help="browser mode: do not open the browser at start")
    parser.add_argument("--check", action="store_true", help="import everything the studio needs, then exit")
    # AppImage only, run by the launcher itself in a child process (linux_desktop.py)
    parser.add_argument("--dialog", nargs=2, metavar=("KIND", "TEXT"), help=argparse.SUPPRESS)
    parser.add_argument("--probe-window", choices=("quick", "full"), help=argparse.SUPPRESS)
    args, _ = parser.parse_known_args()
    if APPDIR:
        linux_desktop().preload_bundled_libs(Path(APPDIR))
        if args.dialog:
            return linux_desktop().show_dialog(args.dialog[0], args.dialog[1])
        if args.probe_window:
            return linux_desktop().probe_window(args.probe_window == "full")
        isolate_child_environment()
    use_browser = args.browser or (os.environ.get("SUBBUILD_UI") or os.environ.get("SUBFORGE_UI") or "").lower() == "browser"

    home = data_dir()
    home.mkdir(parents=True, exist_ok=True)
    if not args.check:
        redirect_stdio(home)
    hide_child_consoles()
    env_file = prepare_environment(home)
    if not args.check:
        integrate_with_desktop(home)

    if args.check:
        return self_check(home)

    from app.core.config import settings

    host, port = settings.host, settings.port
    browse_host = "localhost" if host in ("0.0.0.0", "127.0.0.1", "::", "[::]") else host
    url = f"http://{browse_host}:{port}"

    SESSION_TOKEN = read_saved_token(home)  # the token of a copy that may already be running
    if is_our_server(url):  # already running: bring up its window
        if show_running_window(url):
            return 0
        # A copy is running but has no window to show: it runs in the browser, in a console
        # (the old run.bat), or it is an older version. Say so instead of silently opening a tab.
        print(f"{APP_NAME} is already running at {url} without a window.", file=sys.stderr, flush=True)
        if args.no_browser:
            return 0
        if not use_browser and not args.no_tray:
            answer = message(
                f"{APP_NAME} is already running without its own window (in the browser, in a console "
                "window, or as an older version).\n\n"
                "Close that copy first - quit it from its tray icon, or close its black console window - "
                "then start the program again to get the app window.\n\n"
                "Yes - open the running copy in the browser now\n"
                "No - do nothing", "yesno")
            if answer != IDYES:
                return 0
        webbrowser.open(studio_url(url))
        return 0
    if port_busy(port):
        message(f"Port {port} is already used by another program.\n"
                f"Close it, or set PORT=... in:\n{env_file}")
        return 1

    SESSION_TOKEN = new_session_token(home)  # before the server thread imports the app
    server = ServerThread(host, port)
    server.start()

    if args.no_tray:
        run_headless(server, url, home, open_browser=not args.no_browser)

    if not use_browser:
        problem = window_problem()
        if problem is None:
            try:
                run_window(server, url, home, env_file)
                exit_now(0)
            except Exception:
                print(f"Could not open the studio window, using the browser instead:\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)
        else:
            print(f"Opening the studio in the browser: {problem}", file=sys.stderr, flush=True)

    if APPDIR:  # no Qt window here, so no Qt tray either: the studio runs in the browser
        run_headless(server, url, home, open_browser=not args.no_browser)
    run_tray(server, url, home, env_file, open_browser=not args.no_browser)
    exit_now(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
