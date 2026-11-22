"""The Linux AppImage's desktop pieces (packaging/linux_desktop.py) and how the launcher uses them."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[3] / "packaging"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, PACKAGING / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def ld():
    return _load("subbuild_linux_desktop", "linux_desktop.py")


@pytest.fixture()
def icon(tmp_path):
    png = tmp_path / "icon.png"
    png.write_bytes(b"\x89PNG fake")
    return png


def test_first_launch_adds_the_appimage_to_the_applications_menu(ld, tmp_path, icon, monkeypatch):
    monkeypatch.delenv("SUBBUILD_NO_DESKTOP_ENTRY", raising=False)
    home, share = tmp_path / "data", tmp_path / "share"
    home.mkdir()
    appimage = "/home/tal/Apps/SubBuild-Studio-Linux-x86_64.AppImage"
    written = ld.integrate_appimage(appimage, home, icon, data_home=share)
    assert written == share / "applications" / "subbuildstudio.desktop"
    entry = written.read_text()
    assert f'Exec="{appimage}"' in entry and "Icon=subbuildstudio" in entry
    assert "StartupWMClass=subbuildstudio" in entry and "Terminal=false" in entry
    assert (share / "icons" / "hicolor" / "256x256" / "apps" / "subbuildstudio.png").read_bytes() == icon.read_bytes()
    assert json.loads((home / ld.DESKTOP_MARKER).read_text())["appimage"] == appimage
    # unchanged on the next launch
    assert ld.integrate_appimage(appimage, home, icon, data_home=share) is None


def test_the_entry_follows_a_moved_or_replaced_appimage(ld, tmp_path, icon, monkeypatch):
    monkeypatch.delenv("SUBBUILD_NO_DESKTOP_ENTRY", raising=False)
    home, share = tmp_path / "data", tmp_path / "share"
    home.mkdir()
    ld.integrate_appimage("/old/SubBuild.AppImage", home, icon, data_home=share)
    written = ld.integrate_appimage("/new place/SubBuild.AppImage", home, icon, data_home=share)
    assert 'Exec="/new place/SubBuild.AppImage"' in written.read_text()


def test_a_removed_entry_stays_removed(ld, tmp_path, icon, monkeypatch):
    monkeypatch.delenv("SUBBUILD_NO_DESKTOP_ENTRY", raising=False)
    home, share = tmp_path / "data", tmp_path / "share"
    home.mkdir()
    ld.integrate_appimage("/a/SubBuild.AppImage", home, icon, data_home=share).unlink()
    assert ld.integrate_appimage("/a/SubBuild.AppImage", home, icon, data_home=share) is None
    assert not (share / "applications" / "subbuildstudio.desktop").exists()


def test_the_entry_can_be_turned_off(ld, tmp_path, icon, monkeypatch):
    monkeypatch.setenv("SUBBUILD_NO_DESKTOP_ENTRY", "1")
    assert ld.integrate_appimage("/a/x.AppImage", tmp_path, icon, data_home=tmp_path / "share") is None
    assert not (tmp_path / "share").exists()


def test_exec_paths_are_quoted_per_the_desktop_entry_spec(ld):
    assert ld._exec_quote('/home/t/My "Apps"/$x`y`/100%/a\\b') == '"/home/t/My \\"Apps\\"/\\$x\\`y\\`/100%%/a\\\\b"'


def test_qt_arguments(ld, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("SUBBUILD_WEBENGINE_SANDBOX", raising=False)
    assert ld.qt_arguments("x") == ["x", "-platform", "xcb", "--no-sandbox"]
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    monkeypatch.setenv("SUBBUILD_WEBENGINE_SANDBOX", "1")
    assert ld.qt_arguments("x") == ["x"]


def test_no_window_without_a_display(ld, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert "display" in ld.window_problem(PACKAGING / "launcher.py")
    assert ld.dialog(PACKAGING / "launcher.py", "hi", "info") is None


def test_dialog_answers_come_back_through_the_exit_code(ld, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    calls = []

    class Done:
        def __init__(self, code):
            self.returncode = code

    def fake_run(cmd, timeout):
        calls.append(cmd)
        return Done(ld._EXIT_CODES[ld.IDNO])

    monkeypatch.setattr(ld.subprocess, "run", fake_run)
    assert ld.dialog(PACKAGING / "launcher.py", "Quit?", "yesno") == ld.IDNO
    assert calls[0][:2] == [sys.executable, "-I"] and calls[0][-3:] == ["--dialog", "yesno", "Quit?"]
    monkeypatch.setattr(ld.subprocess, "run", lambda cmd, timeout: Done(1))   # the dialog crashed
    assert ld.dialog(PACKAGING / "launcher.py", "Quit?", "yesno") is None


@pytest.fixture()
def appimage_launcher(monkeypatch, tmp_path):
    """The launcher as AppRun starts it inside the AppImage."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("SUBBUILD_APPDIR", str(tmp_path / "AppDir"))
    monkeypatch.setenv("APPIMAGE", "/home/tal/SubBuild.AppImage")
    return _load("subbuild_launcher_appimage", "launcher.py")


def test_appimage_restart_runs_the_appimage_again(appimage_launcher, monkeypatch):
    started = []
    monkeypatch.setattr(appimage_launcher.subprocess, "Popen", lambda args, **kw: started.append(args))
    appimage_launcher.spawn_again(["--browser"])
    assert started == [["/home/tal/SubBuild.AppImage", "--browser"]]


def test_appimage_points_the_backend_at_the_gpu_download(appimage_launcher, monkeypatch, tmp_path):
    for name in ("SUBBUILD_GPU_DIR", "SUBBUILD_GPU_REQUIREMENTS", "SUBBUILD_UV", "SUBBUILD_FRONTEND_DIST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(appimage_launcher.os, "chdir", lambda path: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    home = tmp_path / "home"
    appimage_launcher.prepare_environment(home)
    import os

    assert os.environ["SUBBUILD_GPU_DIR"] == str(home / "gpu-libs")
    assert os.environ["SUBBUILD_GPU_REQUIREMENTS"].endswith("gpu-requirements.txt")
    assert os.environ["SUBBUILD_UV"] == str(tmp_path / "AppDir" / "usr" / "bin" / "uv")


def test_appimage_messages_use_the_qt_dialog(appimage_launcher, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    asked = []
    fake = type("LD", (), {"dialog": staticmethod(lambda launcher, text, kind: asked.append(kind) or 7)})
    monkeypatch.setattr(appimage_launcher, "linux_desktop", lambda: fake)
    assert appimage_launcher.message("Quit?", "yesno") == appimage_launcher.IDNO
    assert asked == ["yesno"]


def test_appimage_tray_menu_calls_window_mode_callbacks_without_arguments(appimage_launcher):
    called = []
    appimage_launcher._call_menu_action(lambda: called.append("window"))
    appimage_launcher._call_menu_action(lambda icon: called.append(("browser", icon)))
    assert called == ["window", ("browser", None)]
