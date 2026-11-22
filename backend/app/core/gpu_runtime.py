"""The optional NVIDIA libraries for GPU transcription in the Linux AppImage.

cuBLAS and cuDNN (~1.5 GB) would more than double the AppImage, and most machines have no NVIDIA
card, so the AppImage ships without them (packaging/linux/build-appimage.sh). When the studio finds
an NVIDIA card it offers to download them: `uv pip install --target` into the data folder, with the
exact versions the build pinned (packaging/gpu-requirements.txt). They stay there across updates
of the AppImage; app.core.hardware loads them at start, or right after the download.

Configured by the launcher in the AppImage only (packaging/launcher.py, prepare_environment):
  SUBBUILD_GPU_DIR            where the libraries go (<data folder>/gpu-libs)
  SUBBUILD_GPU_REQUIREMENTS   the pinned package list
  SUBBUILD_UV                 the uv that ships in the AppImage
Everywhere else (Windows, macOS, a Git clone) the feature is off: there the libraries come with
the app's requirements, or are not needed.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV_DIR = "SUBBUILD_GPU_DIR"
ENV_REQUIREMENTS = "SUBBUILD_GPU_REQUIREMENTS"
ENV_UV = "SUBBUILD_UV"
COMPLETE_MARKER = ".complete"
# What the dialog tells the user to expect (cuBLAS + cuDNN for CUDA 12, as wheels)
APPROX_DOWNLOAD_MB = 1500
INSTALL_TIMEOUT_S = 60 * 60

_lock = threading.Lock()
_state: Dict[str, Any] = {"installing": False, "error": None, "started_at": None, "finished_at": None}


def gpu_dir() -> Optional[Path]:
    value = os.environ.get(ENV_DIR)
    return Path(value) if value else None


def _requirements() -> Optional[Path]:
    value = os.environ.get(ENV_REQUIREMENTS)
    return Path(value) if value else None


def _uv() -> Optional[Path]:
    value = os.environ.get(ENV_UV)
    return Path(value) if value else None


def is_supported() -> bool:
    """True in the AppImage, which can download the libraries itself."""
    req, uv = _requirements(), _uv()
    return bool(gpu_dir() and req and req.is_file() and uv and uv.is_file())


def is_installed(target: Optional[Path] = None) -> bool:
    target = target or gpu_dir()
    return bool(target and (target / COMPLETE_MARKER).is_file() and (target / "nvidia").is_dir())


def status() -> Dict[str, Any]:
    with _lock:
        state = dict(_state)
    return {
        "supported": is_supported(),
        "installed": is_installed(),
        "installing": bool(state["installing"]),
        "error": state["error"],
        "download_size_mb": APPROX_DOWNLOAD_MB,
    }


def _tail(text: str, lines: int = 8) -> str:
    return "\n".join([line for line in (text or "").strip().splitlines() if line.strip()][-lines:])


def _install_command(uv: Path, requirements: Path, target: Path) -> List[str]:
    # --no-deps: the list is complete and pinned (uv pip compile at build time)
    return [str(uv), "pip", "install", "--python", sys.executable, "--target", str(target),
            "--no-deps", "--no-progress", "-r", str(requirements)]


def _install() -> None:
    target, requirements, uv = gpu_dir(), _requirements(), _uv()
    assert target and requirements and uv
    partial = target.with_name(target.name + ".partial")
    cache = target.with_name(target.name + ".cache")
    error: Optional[str] = None
    try:
        for leftover in (partial, cache):
            shutil.rmtree(leftover, ignore_errors=True)
        partial.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, UV_CACHE_DIR=str(cache), UV_LINK_MODE="copy", UV_PYTHON_DOWNLOADS="never",
                   UV_NO_CONFIG="1")
        for name in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
            env.pop(name, None)
        logger.info("Downloading the NVIDIA GPU libraries into %s", target)
        run = subprocess.run(_install_command(uv, requirements, partial), capture_output=True, text=True,
                             env=env, timeout=INSTALL_TIMEOUT_S, check=False)
        if run.returncode != 0:
            error = _tail(run.stderr or run.stdout) or f"uv exited with code {run.returncode}"
            raise RuntimeError(error)
        if not (partial / "nvidia").is_dir():
            error = "The download finished but contains no NVIDIA libraries."
            raise RuntimeError(error)
        (partial / COMPLETE_MARKER).write_text(requirements.read_text(encoding="utf-8"), encoding="utf-8")
        # Swap in the complete set only (a half-finished download never looks installed)
        if target.exists():
            shutil.rmtree(target)
        os.replace(partial, target)
        from app.core import hardware

        loaded = hardware.load_gpu_libs()
        logger.info("NVIDIA GPU libraries installed (loaded: %s)", loaded)
    except Exception as exc:  # noqa: BLE001 - reported to the UI
        error = error or str(exc) or exc.__class__.__name__
        logger.warning("Installing the NVIDIA GPU libraries failed: %s", error)
        shutil.rmtree(partial, ignore_errors=True)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
        with _lock:
            _state.update(installing=False, error=error, finished_at=time.time())


def start_install() -> Dict[str, Any]:
    """Start the download in the background (no-op while one runs); returns the new status."""
    if not is_supported():
        raise RuntimeError("GPU libraries can only be downloaded by the Linux AppImage.")
    with _lock:
        if not _state["installing"]:
            _state.update(installing=True, error=None, started_at=time.time(), finished_at=None)
            threading.Thread(target=_install, name="gpu-libs-install", daemon=True).start()
    return status()
