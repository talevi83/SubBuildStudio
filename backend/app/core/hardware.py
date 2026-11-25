import gc
import importlib.util
import logging
import os
import sys
import subprocess
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.executables import find_executable

logger = logging.getLogger(__name__)

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
        # Prefer PyTorch's bundled CUDA/cuDNN DLLs when torch is installed: preloading a
        # standalone nvidia-cudnn-cu12 build first can conflict with torch/lib/cudnn_cnn64_9.dll.
        torch_candidates = [Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"]
        nvidia_candidates = [Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"]
        if getattr(sys, "frozen", False):  # packaged exe: the DLLs sit in <bundle>/...
            meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
            torch_candidates.insert(0, meipass / "torch" / "lib")
            nvidia_candidates.insert(0, meipass / "nvidia")

        torch_lib = next((c for c in torch_candidates if (c / "cudnn64_9.dll").is_file()), None)
        nvidia_dir = next((c for c in nvidia_candidates if c.exists()), None)

        if torch_lib is not None:
            try:
                os.add_dll_directory(str(torch_lib))
            except Exception:
                pass
            torch_dlls = {p.name.lower() for p in torch_lib.glob("*.dll")}
            extra_bins = (
                [b for b in nvidia_dir.glob("*/bin") if b.parent.name.lower() not in ("cudnn", "cublas")]
                if nvidia_dir is not None
                else []
            )
            for b in extra_bins:
                try:
                    os.add_dll_directory(str(b))
                except Exception:
                    pass
            os.environ["PATH"] = ";".join([str(torch_lib), *(str(b) for b in extra_bins)]) + ";" + os.environ.get("PATH", "")
            for core_name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
                dll_path = torch_lib / core_name
                if dll_path.is_file():
                    try:
                        ctypes.CDLL(str(dll_path))
                    except Exception:
                        pass
            for b in extra_bins:
                for dll_file in b.glob("*.dll"):
                    if dll_file.name.lower() in torch_dlls:
                        continue
                    try:
                        ctypes.CDLL(str(dll_file))
                    except Exception:
                        pass
            return

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


def _has_package(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return name in sys.modules


def is_whisperx_installed() -> bool:
    """Return True if the optional whisperx package can be resolved."""
    return _has_package("whisperx")


def is_demucs_installed() -> bool:
    """Return True if the optional demucs package can be resolved."""
    return _has_package("demucs")


def is_audio_separator_installed() -> bool:
    """Return True if the optional audio-separator package can be resolved."""
    return _has_package("audio_separator")


def is_vocal_isolation_installed() -> bool:
    """Return True if either demucs or audio-separator is installed."""
    return is_demucs_installed() or is_audio_separator_installed()


def release_torch_vram(context_label: str = "") -> bool:
    """Run Python GC and release PyTorch CUDA cached VRAM when CUDA is active.

    Never imports ``torch`` at startup; only touches ``torch`` if it is already imported
    in ``sys.modules``. Returns True when ``torch.cuda.empty_cache()`` was called.
    """
    gc.collect()
    label = f" ({context_label})" if context_label else ""
    torch_mod = sys.modules.get("torch")
    cuda_cleared = False
    if torch_mod is not None:
        try:
            cuda_mod = getattr(torch_mod, "cuda", None)
            if cuda_mod is not None and cuda_mod.is_available():
                cuda_mod.empty_cache()
                if hasattr(cuda_mod, "ipc_collect"):
                    cuda_mod.ipc_collect()
                cuda_cleared = True
        except Exception as exc:
            logger.debug(f"VRAM cache release warning{label}: {exc}")
    if cuda_cleared:
        logger.info(f"VRAM cleanup completed{label}: gc.collect() + torch.cuda.empty_cache() executed")
    else:
        logger.info(f"Memory cleanup completed{label}: gc.collect() executed")
    return cuda_cleared


@dataclass
class _CompatAudioMetaData:
    """Compatibility stand-in for removed ``torchaudio.AudioMetaData`` in torchaudio >= 2.9."""

    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def ensure_torchaudio_compat() -> None:
    """Patch removed torchaudio legacy APIs and PyTorch 2.6+ checkpoint loading quirks.

    Torchaudio 2.9–2.11 removed ``AudioMetaData``, ``info``, ``list_audio_backends``, and
    ``set_audio_backend``, and routed ``load``/``save`` through ``torchcodec`` by default,
    which breaks downstream packages such as ``pyannote.audio`` (used by WhisperX VAD/diarization)
    and ``speechbrain``. Meanwhile ``lightning_fabric.utilities.cloud_io._load`` passes
    ``weights_only=None`` for local checkpoints, triggering ``UnpicklingError`` under PyTorch >= 2.6.
    This helper is lazy, idempotent, and safe when ``torchaudio`` is absent or mocked.
    """
    if not _has_package("torchaudio"):
        return

    try:
        import torchaudio  # type: ignore
    except Exception:
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = _CompatAudioMetaData  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "backend"):
        torchaudio.backend = types.SimpleNamespace(  # type: ignore[attr-defined]
            common=types.SimpleNamespace(AudioMetaData=torchaudio.AudioMetaData)
        )
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _backend=None: None  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "info"):
        def _compat_info(uri: Any, format: Optional[str] = None, buffer_size: int = 4096, backend: Optional[str] = None):
            del format, buffer_size, backend
            try:
                import soundfile as sf  # type: ignore

                info = sf.info(uri)
                sub = str(getattr(info, "subtype", "") or "").upper()
                if "24" in sub:
                    bps = 24
                elif "32" in sub or "FLOAT" in sub or "DOUBLE" in sub:
                    bps = 32
                elif "8" in sub:
                    bps = 8
                else:
                    bps = 16
                return torchaudio.AudioMetaData(  # type: ignore[attr-defined]
                    sample_rate=int(info.samplerate),
                    num_frames=int(info.frames),
                    num_channels=int(info.channels),
                    bits_per_sample=bps,
                    encoding=sub or "PCM_S",
                )
            except Exception:
                import wave

                if hasattr(uri, "seek"):
                    uri.seek(0)
                with wave.open(uri, "rb") as wf:
                    return torchaudio.AudioMetaData(  # type: ignore[attr-defined]
                        sample_rate=int(wf.getframerate()),
                        num_frames=int(wf.getnframes()),
                        num_channels=int(wf.getnchannels()),
                        bits_per_sample=int(wf.getsampwidth()) * 8,
                        encoding="PCM_S",
                    )

        torchaudio.info = _compat_info  # type: ignore[attr-defined]

    if not getattr(torchaudio, "_subbuild_io_patched", False):
        orig_load = getattr(torchaudio, "load", None)
        orig_save = getattr(torchaudio, "save", None)
        has_torchcodec = _has_package("torchcodec")

        def _compat_load(
            uri: Any,
            frame_offset: int = 0,
            num_frames: int = -1,
            normalize: bool = True,
            channels_first: bool = True,
            format: Optional[str] = None,
            buffer_size: int = 4096,
            backend: Optional[str] = None,
        ):
            if orig_load is not None and has_torchcodec and backend is None:
                try:
                    return orig_load(
                        uri,
                        frame_offset=frame_offset,
                        num_frames=num_frames,
                        normalize=normalize,
                        channels_first=channels_first,
                        format=format,
                        buffer_size=buffer_size,
                    )
                except (ImportError, RuntimeError, TypeError):
                    pass

            import soundfile as sf  # type: ignore
            import torch  # type: ignore

            frames = -1 if (num_frames is None or int(num_frames) < 0) else int(num_frames)
            start = max(0, int(frame_offset or 0))
            data, sr = sf.read(
                uri,
                frames=frames,
                start=start,
                dtype="float32" if normalize else "int16",
                always_2d=True,
            )
            arr = data.T.copy() if channels_first else data.copy()
            return torch.from_numpy(arr), int(sr)

        def _compat_save(
            uri: Any,
            src: Any,
            sample_rate: int,
            channels_first: bool = True,
            format: Optional[str] = None,
            encoding: Optional[str] = None,
            bits_per_sample: Optional[int] = None,
            buffer_size: int = 4096,
            backend: Optional[str] = None,
            compression: Optional[Any] = None,
        ) -> None:
            if orig_save is not None and has_torchcodec and backend is None:
                try:
                    orig_save(
                        uri,
                        src,
                        sample_rate,
                        channels_first=channels_first,
                        format=format,
                        encoding=encoding,
                        bits_per_sample=bits_per_sample,
                        buffer_size=buffer_size,
                        compression=compression,
                    )
                    return
                except (ImportError, RuntimeError, TypeError):
                    pass

            import soundfile as sf  # type: ignore

            arr = src.detach().cpu().numpy() if hasattr(src, "detach") else src
            if getattr(arr, "ndim", 0) == 2 and channels_first:
                arr = arr.T
            sf.write(uri, arr, int(sample_rate))

        torchaudio.load = _compat_load  # type: ignore[assignment]
        torchaudio.save = _compat_save  # type: ignore[assignment]
        torchaudio._subbuild_io_patched = True  # type: ignore[attr-defined]

    if _has_package("lightning_fabric"):
        try:
            import lightning_fabric.utilities.cloud_io as cio  # type: ignore

            if not getattr(cio, "_subbuild_patched", False):
                orig_pl_load = cio._load

                def _compat_pl_load(path_or_url: Any, map_location: Any = None, weights_only: Optional[bool] = None):
                    resolved_wo = False if weights_only is None else weights_only
                    return orig_pl_load(path_or_url, map_location=map_location, weights_only=resolved_wo)

                cio._load = _compat_pl_load  # type: ignore[attr-defined]
                cio._subbuild_patched = True  # type: ignore[attr-defined]
                if "pyannote.audio.core.model" in sys.modules:
                    setattr(sys.modules["pyannote.audio.core.model"], "pl_load", _compat_pl_load)
        except Exception:
            pass


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
        "whisperx_available": is_whisperx_installed(),
        "vocal_isolation_available": is_vocal_isolation_installed(),
    }
