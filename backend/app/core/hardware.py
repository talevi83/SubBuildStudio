import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.executables import find_executable

# True once cuBLAS / cuDNN were found and loaded (Linux). The Linux AppImage downloads them on
# request (app.core.gpu_runtime); until then CUDA must not be offered, or transcription would pick
# the GPU and fail on the missing libraries.
NVIDIA_LIBS_LOADED = False


def _linux_nvidia_dirs() -> list:
    """Where pip-installed NVIDIA libraries can be: this Python's site-packages, and the AppImage's
    download folder (SUBBUILD_GPU_DIR, see app.core.gpu_runtime)."""
    candidates = list((Path(sys.prefix) / "lib").glob("python*/site-packages/nvidia"))
    gpu_dir = os.environ.get("SUBBUILD_GPU_DIR")
    if gpu_dir and (Path(gpu_dir) / ".complete").is_file():
        candidates.append(Path(gpu_dir) / "nvidia")
    return [c for c in candidates if c.is_dir()]


def setup_cuda_dll_paths() -> None:
    """Preload NVIDIA CUDA/cuBLAS/cuDNN shared libraries on Windows and Linux into process space."""
    if sys.platform not in ("win32", "linux"):
        return

    import ctypes
    if sys.platform == "win32":
        candidates = [Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"]
        if getattr(sys, "frozen", False):  # packaged exe: the DLLs sit in <bundle>/nvidia/*/bin
            candidates.insert(0, Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "nvidia")
        nvidia_dir = next((c for c in candidates if c.exists()), None)
        if nvidia_dir is None:
            return

        bin_dirs = list(nvidia_dir.glob("*/bin"))
        for b in bin_dirs:
            try:
                os.add_dll_directory(str(b))
            except Exception:
                pass

        if bin_dirs:
            os.environ["PATH"] = ";".join(str(b) for b in bin_dirs) + ";" + os.environ.get("PATH", "")

        # Preload cuBLAS and cuDNN DLLs into memory
        for b in bin_dirs:
            for dll_file in b.glob("*.dll"):
                try:
                    ctypes.CDLL(str(dll_file))
                except Exception:
                    pass
    else:
        global NVIDIA_LIBS_LOADED
        nvidia_dirs = _linux_nvidia_dirs()
        if not nvidia_dirs:
            return
        lib_dirs = [d for nvidia_dir in nvidia_dirs for d in nvidia_dir.glob("*/lib")]
        if lib_dirs:
            ld_path = ":".join(str(d) for d in lib_dirs)
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{ld_path}:{existing}" if existing else ld_path
        rtld_global = getattr(ctypes, "RTLD_GLOBAL", os.RTLD_GLOBAL if hasattr(os, "RTLD_GLOBAL") else 0)
        loaded = set()
        for d in lib_dirs:
            for so_file in sorted(d.glob("lib*.so*")):
                try:
                    ctypes.CDLL(str(so_file), mode=rtld_global)
                    loaded.add(so_file.name.split(".so")[0])
                except Exception:
                    pass
        if "libcublas" in loaded and any(name.startswith("libcudnn") for name in loaded):
            NVIDIA_LIBS_LOADED = True


def load_gpu_libs() -> bool:
    """Load NVIDIA libraries installed after start (the AppImage's download); True when loaded."""
    setup_cuda_dll_paths()
    return NVIDIA_LIBS_LOADED


# Execute DLL setup immediately
setup_cuda_dll_paths()

try:
    import ctranslate2
except ImportError:
    ctranslate2 = None


def get_hardware_info() -> Dict[str, Any]:
    """
    Detect available hardware acceleration (CUDA/GPU), VRAM, and CPU details.
    Provides recommended Whisper device and compute type.
    """
    cuda_detected = False
    device_count = 0

    if ctranslate2 is not None:
        try:
            device_count = ctranslate2.get_cuda_device_count()
            cuda_detected = device_count > 0
        except Exception:
            cuda_detected = False

    gpu_name: Optional[str] = None
    vram_total_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None

    nvidia_smi = find_executable("nvidia-smi")
    if nvidia_smi:
        try:
            res = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpu_name = parts[0]
                    vram_total_mb = int(parts[1])
                    vram_free_mb = int(parts[2])
                    cuda_detected = True
        except Exception:
            pass

    # The Linux AppImage without its GPU download: an NVIDIA card is there, but transcription
    # can't use it yet (the app offers the download)
    gpu_libs_missing = False
    if cuda_detected and sys.platform.startswith("linux") and not NVIDIA_LIBS_LOADED:
        from app.core import gpu_runtime

        if gpu_runtime.is_supported():
            gpu_libs_missing, cuda_detected, device_count = True, False, 0

    cpu_count = os.cpu_count() or 1
    recommended_device = "cuda" if cuda_detected else "cpu"
    recommended_compute = "float16" if cuda_detected else "int8"

    return {
        "cuda_available": cuda_detected,
        "device_count": device_count,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total_mb,
        "vram_free_mb": vram_free_mb,
        "cpu_count": cpu_count,
        "recommended_device": recommended_device,
        "recommended_compute_type": recommended_compute,
        "gpu_libs_missing": gpu_libs_missing,
    }
