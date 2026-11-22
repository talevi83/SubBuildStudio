"""The Linux AppImage's optional NVIDIA libraries (app/core/gpu_runtime.py, app/core/hardware.py)."""
import stat
import subprocess
import sys
import time

import pytest

from app.core import gpu_runtime, hardware

FAKE_UV_OK = """#!/bin/sh
# uv pip install --python PY --target DIR --no-deps --no-progress -r REQ
while [ $# -gt 0 ]; do
  if [ "$1" = "--target" ]; then target="$2"; fi
  shift
done
mkdir -p "$target/nvidia/cublas/lib" "$target/nvidia/cudnn/lib"
: > "$target/nvidia/cublas/lib/libcublas.so.12"
: > "$target/nvidia/cudnn/lib/libcudnn.so.9"
"""

FAKE_UV_FAIL = """#!/bin/sh
echo "error: Failed to fetch: https://pypi.org/simple/nvidia-cudnn-cu12/" >&2
exit 2
"""


@pytest.fixture()
def appimage_env(tmp_path, monkeypatch):
    """What the launcher sets up in the AppImage, with a stand-in for uv."""
    def make(script: str):
        uv = tmp_path / "uv"
        uv.write_text(script)
        uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
        req = tmp_path / "gpu-requirements.txt"
        req.write_text("nvidia-cublas-cu12==12.9.1.4\nnvidia-cudnn-cu12==9.10.2.21\n")
        target = tmp_path / "data" / "gpu-libs"
        monkeypatch.setenv(gpu_runtime.ENV_DIR, str(target))
        monkeypatch.setenv(gpu_runtime.ENV_REQUIREMENTS, str(req))
        monkeypatch.setenv(gpu_runtime.ENV_UV, str(uv))
        monkeypatch.setitem(gpu_runtime._state, "installing", False)
        monkeypatch.setitem(gpu_runtime._state, "error", None)
        return target
    return make


def _wait_idle(timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not gpu_runtime.status()["installing"]:
            return gpu_runtime.status()
        time.sleep(0.05)
    raise AssertionError("the install did not finish")


def test_off_outside_the_appimage(monkeypatch):
    for name in (gpu_runtime.ENV_DIR, gpu_runtime.ENV_REQUIREMENTS, gpu_runtime.ENV_UV):
        monkeypatch.delenv(name, raising=False)
    assert gpu_runtime.status()["supported"] is False
    with pytest.raises(RuntimeError):
        gpu_runtime.start_install()


@pytest.mark.skipif(sys.platform == "win32", reason="the AppImage is Linux only (shell-script stand-in for uv)")
def test_download_installs_into_the_data_folder_and_loads(appimage_env, monkeypatch):
    target = appimage_env(FAKE_UV_OK)
    loads = []
    monkeypatch.setattr(hardware, "load_gpu_libs", lambda: loads.append(True) or True)
    assert gpu_runtime.status() == {"supported": True, "installed": False, "installing": False,
                                    "error": None, "download_size_mb": gpu_runtime.APPROX_DOWNLOAD_MB}
    gpu_runtime.start_install()
    done = _wait_idle()
    assert done["installed"] is True and done["error"] is None
    assert (target / "nvidia" / "cublas" / "lib" / "libcublas.so.12").exists()
    assert "nvidia-cudnn-cu12==9.10.2.21" in (target / gpu_runtime.COMPLETE_MARKER).read_text()
    assert not target.with_name("gpu-libs.partial").exists() and not target.with_name("gpu-libs.cache").exists()
    assert loads == [True]


@pytest.mark.skipif(sys.platform == "win32", reason="the AppImage is Linux only (shell-script stand-in for uv)")
def test_failed_download_reports_the_error_and_leaves_nothing_half_done(appimage_env, monkeypatch):
    target = appimage_env(FAKE_UV_FAIL)
    monkeypatch.setattr(hardware, "load_gpu_libs", lambda: pytest.fail("nothing to load"))
    gpu_runtime.start_install()
    done = _wait_idle()
    assert done["installed"] is False and "Failed to fetch" in done["error"]
    assert not target.exists() and not target.with_name("gpu-libs.partial").exists()


def test_install_command_is_pinned_and_isolated(tmp_path):
    cmd = gpu_runtime._install_command(tmp_path / "uv", tmp_path / "req.txt", tmp_path / "out")
    assert cmd[1:3] == ["pip", "install"] and "--no-deps" in cmd
    assert cmd[cmd.index("--target") + 1] == str(tmp_path / "out")
    assert cmd[cmd.index("--python") + 1] == sys.executable


def _fake_nvidia_smi(monkeypatch):
    monkeypatch.setattr(hardware, "find_executable", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout="NVIDIA GeForce RTX 2070 SUPER, 8192, 7900\n", stderr=""))


def test_appimage_without_the_download_does_not_offer_cuda(appimage_env, monkeypatch):
    appimage_env(FAKE_UV_OK)
    _fake_nvidia_smi(monkeypatch)
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware, "NVIDIA_LIBS_LOADED", False)
    info = hardware.get_hardware_info()
    assert info["gpu_libs_missing"] is True and info["cuda_available"] is False
    assert info["recommended_device"] == "cpu" and info["gpu_name"] == "NVIDIA GeForce RTX 2070 SUPER"


def test_appimage_with_the_download_uses_cuda(appimage_env, monkeypatch):
    appimage_env(FAKE_UV_OK)
    _fake_nvidia_smi(monkeypatch)
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware, "NVIDIA_LIBS_LOADED", True)
    info = hardware.get_hardware_info()
    assert info["gpu_libs_missing"] is False and info["cuda_available"] is True
    assert info["recommended_device"] == "cuda"


def test_other_installs_are_unchanged(monkeypatch):
    for name in (gpu_runtime.ENV_DIR, gpu_runtime.ENV_REQUIREMENTS, gpu_runtime.ENV_UV):
        monkeypatch.delenv(name, raising=False)
    _fake_nvidia_smi(monkeypatch)
    monkeypatch.setattr(hardware, "NVIDIA_LIBS_LOADED", False)
    info = hardware.get_hardware_info()
    assert info["gpu_libs_missing"] is False and info["cuda_available"] is True


def test_gpu_libs_routes(client, monkeypatch):
    for name in (gpu_runtime.ENV_DIR, gpu_runtime.ENV_REQUIREMENTS, gpu_runtime.ENV_UV):
        monkeypatch.delenv(name, raising=False)
    res = client.get("/api/system/gpu-libs")
    assert res.status_code == 200 and res.json()["supported"] is False
    assert client.post("/api/system/gpu-libs/install").status_code == 400
