"""Optional pre-transcription vocal isolation service (Demucs / audio-separator).

Separates dialogue vocals from heavy film scores, orchestral music, and SFX before
transcription and speaker diarization. Heavy ML libraries (`torch`, `demucs`,
`audio_separator`) are strictly imported lazily inside execution functions so the
application can start and run tests even when they are not installed.
"""
import hashlib
import importlib.util
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from app.core.config import settings
from app.core.hardware import (
    ensure_torchaudio_compat,
    is_audio_separator_installed,
    is_demucs_installed,
    is_vocal_isolation_installed,
    release_torch_vram,
)
from app.pipeline.context import PipelineCancelled

logger = logging.getLogger(__name__)

_TARGET_SAMPLE_RATE = 16000
_DEFAULT_CHUNK_SECONDS = 180.0
_DEFAULT_OVERLAP_SECONDS = 1.5
_MAX_DEMUCS_SEGMENT = 7.5  # htdemucs transformer limit is 7.8s


class VocalIsolationUnavailableError(RuntimeError):
    """Raised when neither demucs nor audio-separator is installed in the backend environment."""


def is_demucs_available() -> bool:
    """Return True if demucs can be imported."""
    return is_demucs_installed()


def is_audio_separator_available() -> bool:
    """Return True if audio-separator can be imported."""
    return is_audio_separator_installed()


def is_vocal_isolation_available() -> bool:
    """Return True if any vocal separation backend is installed."""
    return is_vocal_isolation_installed()


def vocal_cache_path(job_id: str) -> Path:
    """Return the per-job cached isolated vocals WAV path."""
    cache_dir = settings.storage_dir / "vocals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{job_id}.vocals.wav"


def source_vocal_cache_path(
    source_media_path: Optional[str],
    audio_stream_index: int = 0,
    model_name: str = "htdemucs",
) -> Optional[Path]:
    """Compute a deterministic cache path keyed by source file identity, stream, and model."""
    if not source_media_path:
        return None
    try:
        resolved = Path(source_media_path).resolve()
        if not resolved.is_file():
            return None
        st = resolved.stat()
        digest = hashlib.sha256(
            f"{resolved}|{st.st_size}|{int(st.st_mtime)}|{audio_stream_index}|{model_name}".encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = settings.storage_dir / "vocals"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"src_{digest}.vocals.wav"
    except OSError:
        return None


def _is_valid_wav(path: Path) -> bool:
    """Return True when path is an existing, non-empty, readable WAV file."""
    try:
        if not path.is_file() or path.stat().st_size <= 1024:
            return False
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() > 0 and wf.getframerate() > 0
    except Exception:
        return False


def unload_vocal_model(model: Any = None) -> None:
    """Drop model references and reclaim PyTorch CUDA VRAM and Python memory."""
    del model
    release_torch_vram("vocal isolation")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.resolve() == src.resolve():
        return
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        try:
            if tmp.exists():
                tmp.unlink()
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _read_pcm16_window(wf: wave.Wave_read, start_frame: int, num_frames: int, channels: int) -> np.ndarray:
    """Read `num_frames` from an open PCM16 WAV stream as float32 array shape `(channels, frames)`."""
    wf.setpos(start_frame)
    raw = wf.readframes(num_frames)
    if not raw:
        return np.zeros((channels, 0), dtype=np.float32)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        frame_count = len(samples) // channels
        samples = samples[: frame_count * channels].reshape(frame_count, channels).T
    else:
        samples = samples.reshape(1, -1)
    return samples


def _resample_tensor(torch_mod: Any, audio: Any, orig_sr: int, target_sr: int) -> Any:
    """Resample a `(channels, time)` tensor cleanly from `orig_sr` to `target_sr`."""
    if orig_sr == target_sr or audio.shape[-1] == 0:
        return audio
    if importlib.util.find_spec("torchaudio") is not None:
        try:
            import torchaudio  # type: ignore

            return torchaudio.functional.resample(audio, orig_sr, target_sr)
        except Exception:
            pass
    new_len = max(1, int(round(audio.shape[-1] * (float(target_sr) / float(orig_sr)))))
    return torch_mod.nn.functional.interpolate(
        audio.unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0)


def _match_channels(torch_mod: Any, audio: Any, target_channels: int) -> Any:
    """Convert `(channels, time)` tensor to `target_channels`."""
    cur = audio.shape[0]
    if cur == target_channels:
        return audio
    if cur == 1 and target_channels > 1:
        return audio.repeat(target_channels, 1)
    mono = audio.mean(dim=0, keepdim=True)
    if target_channels == 1:
        return mono
    return mono.repeat(target_channels, 1)


def _run_demucs(
    input_wav_path: Path,
    output_wav_path: Path,
    *,
    model_name: str,
    device: str,
    chunk_seconds: float,
    overlap_seconds: float,
    cancel_flag: Optional[threading.Event],
    on_progress: Optional[Callable[[float, str], None]],
    preloaded_model: Any = None,
    keep_model_loaded: bool = False,
) -> Dict[str, Any]:
    ensure_torchaudio_compat()
    import torch  # type: ignore
    from demucs.apply import apply_model  # type: ignore
    from demucs.pretrained import get_model  # type: ignore

    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    model = preloaded_model
    tmp_path = output_wav_path.with_name(f"{output_wav_path.stem}.tmp{output_wav_path.suffix}")
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    try:
        if model is None:
            if on_progress:
                on_progress(2.0, f"Loading Demucs model '{model_name}' on {target_device}...")
            logger.info(f"Loading vocal isolation model '{model_name}' (demucs) on {target_device}")
            model = get_model(model_name)
            model.eval()
            try:
                model.to(target_device)
            except Exception as exc:
                if target_device == "cuda":
                    logger.warning(f"Demucs CUDA initialization failed ({exc}); falling back to CPU")
                    target_device = "cpu"
                    model.to("cpu")
                else:
                    raise
        else:
            try:
                first_param = next(model.parameters(), None)
                if first_param is not None:
                    target_device = str(first_param.device).split(":")[0]
            except Exception:
                pass

        sources = list(getattr(model, "sources", ["drums", "bass", "other", "vocals"]))
        vocals_idx = sources.index("vocals") if "vocals" in sources else len(sources) - 1
        model_sr = int(getattr(model, "samplerate", 44100) or 44100)
        model_channels = int(getattr(model, "audio_channels", 2) or 2)
        seg_limit = min(_MAX_DEMUCS_SEGMENT, float(getattr(model, "segment", _MAX_DEMUCS_SEGMENT) or _MAX_DEMUCS_SEGMENT))

        with wave.open(str(input_wav_path), "rb") as wf_in:
            in_sr = wf_in.getframerate()
            in_channels = wf_in.getnchannels()
            total_frames = wf_in.getnframes()
            if wf_in.getsampwidth() != 2:
                raise RuntimeError(f"Expected 16-bit PCM WAV input, got sample width {wf_in.getsampwidth()}")

            chunk_frames = max(in_sr * 10, int(round(chunk_seconds * in_sr)))
            overlap_frames = max(0, int(round(overlap_seconds * in_sr)))

            with wave.open(str(tmp_path), "wb") as wf_out:
                wf_out.setnchannels(1)
                wf_out.setsampwidth(2)
                wf_out.setframerate(_TARGET_SAMPLE_RATE)

                prev_tail: Optional[np.ndarray] = None
                core_start = 0

                while core_start < total_frames:
                    if cancel_flag is not None and cancel_flag.is_set():
                        raise PipelineCancelled()

                    core_end = min(total_frames, core_start + chunk_frames)
                    win_start = max(0, core_start - overlap_frames)
                    win_end = min(total_frames, core_end + overlap_frames)

                    pcm_win = _read_pcm16_window(wf_in, win_start, win_end - win_start, in_channels)
                    wav_tensor = torch.from_numpy(pcm_win)
                    wav_model = _resample_tensor(torch, wav_tensor, in_sr, model_sr)
                    wav_model = _match_channels(torch, wav_model, model_channels)

                    mean = wav_model.mean()
                    std = wav_model.std()
                    if float(std) > 1e-6:
                        norm_tensor = (wav_model - mean) / std
                    else:
                        norm_tensor = wav_model - mean

                    def _infer_chunk(dev: str) -> Any:
                        batch = norm_tensor.unsqueeze(0).to(dev)
                        with torch.no_grad():
                            if dev == "cuda":
                                with torch.amp.autocast("cuda", dtype=torch.float16):
                                    stems = apply_model(
                                        model,
                                        batch,
                                        shifts=0,
                                        split=True,
                                        overlap=0.25,
                                        progress=False,
                                        device=dev,
                                        segment=seg_limit,
                                    )
                            else:
                                stems = apply_model(
                                    model,
                                    batch,
                                    shifts=0,
                                    split=True,
                                    overlap=0.25,
                                    progress=False,
                                    device=dev,
                                    segment=seg_limit,
                                )
                        return stems[0, vocals_idx].detach().float().cpu()

                    try:
                        vocals_model = _infer_chunk(target_device)
                    except Exception as exc:
                        if target_device == "cuda" and ("out of memory" in str(exc).lower() or "cuda" in str(exc).lower()):
                            logger.warning(f"Demucs CUDA error/OOM on chunk ({exc}); falling back to CPU")
                            target_device = "cpu"
                            model.to("cpu")
                            release_torch_vram("demucs OOM fallback")
                            vocals_model = _infer_chunk("cpu")
                        else:
                            raise

                    if float(std) > 1e-6:
                        vocals_model = vocals_model * std + mean
                    else:
                        vocals_model = vocals_model + mean

                    vocals_mono = vocals_model.mean(dim=0, keepdim=True)
                    vocals_out = _resample_tensor(torch, vocals_mono, model_sr, _TARGET_SAMPLE_RATE).squeeze(0).numpy()

                    # Compute boundaries inside vocals_out matching [core_start, core_end]
                    scale = float(_TARGET_SAMPLE_RATE) / float(in_sr)
                    out_core_start = int(round((core_start - win_start) * scale))
                    expected_core_len = max(1, int(round((core_end - core_start) * scale)))
                    out_core_end = min(len(vocals_out), out_core_start + expected_core_len)
                    out_overlap_len = int(round(overlap_frames * scale))

                    core_samples = vocals_out[out_core_start:out_core_end].copy()
                    if prev_tail is not None and len(prev_tail) > 0 and len(core_samples) > 0:
                        xfade = min(len(prev_tail), len(core_samples))
                        fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
                        core_samples[:xfade] = prev_tail[:xfade] * (1.0 - fade_in) + core_samples[:xfade] * fade_in

                    if core_end < total_frames and out_overlap_len > 0 and out_core_end < len(vocals_out):
                        next_tail_end = min(len(vocals_out), out_core_end + out_overlap_len)
                        prev_tail = vocals_out[out_core_end:next_tail_end].copy()
                    else:
                        prev_tail = None

                    pcm_out = np.clip(core_samples, -1.0, 1.0)
                    pcm_int16 = (pcm_out * 32767.0).astype(np.int16)
                    wf_out.writeframes(pcm_int16.tobytes())

                    core_start = core_end
                    if on_progress and total_frames > 0:
                        pct = min(98.0, 5.0 + (core_end / float(total_frames)) * 93.0)
                        on_progress(pct, f"Isolating dialogue vocals... {pct:.0f}%")

        os.replace(tmp_path, output_wav_path)
        elapsed = round(time.monotonic() - started, 2)
        if on_progress:
            on_progress(100.0, f"Vocal isolation completed in {elapsed:.1f}s")
        return {
            "path": str(output_wav_path),
            "cached": False,
            "backend": "demucs",
            "model": model_name,
            "device": target_device,
            "elapsed_seconds": elapsed,
            **({"loaded_model": model} if keep_model_loaded else {}),
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if not keep_model_loaded:
            unload_vocal_model(model)


def _convert_to_16k_mono(src_path: Path, dst_path: Path) -> None:
    """Normalize any audio file to 16kHz mono PCM16 WAV via FFmpeg (or direct copy if matching)."""
    try:
        with wave.open(str(src_path), "rb") as wf:
            if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == _TARGET_SAMPLE_RATE:
                _link_or_copy(src_path, dst_path)
                return
    except Exception:
        pass

    from app.core.ffmpeg_safety import safe_input
    from app.pipeline.stages.audio_stage import get_ffmpeg_bin

    tmp_path = dst_path.with_name(f"{dst_path.stem}.tmp{dst_path.suffix}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            get_ffmpeg_bin(),
            "-y",
            "-nostats",
            *safe_input(str(src_path)),
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(_TARGET_SAMPLE_RATE),
            "-ac",
            "1",
            str(tmp_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 or not tmp_path.is_file():
            raise RuntimeError(f"FFmpeg conversion of isolated vocals failed: {res.stderr[-500:]}")
        os.replace(tmp_path, dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _run_audio_separator(
    input_wav_path: Path,
    output_wav_path: Path,
    *,
    cancel_flag: Optional[threading.Event],
    on_progress: Optional[Callable[[float, str], None]],
) -> Dict[str, Any]:
    ensure_torchaudio_compat()
    from audio_separator.separator import Separator  # type: ignore

    started = time.monotonic()
    separator = None
    try:
        if cancel_flag is not None and cancel_flag.is_set():
            raise PipelineCancelled()
        if on_progress:
            on_progress(10.0, "Loading MDX-Net vocal separator...")
        with tempfile.TemporaryDirectory(prefix="subbuild_sep_") as tmp_dir:
            separator = Separator(
                output_dir=tmp_dir,
                output_format="WAV",
                output_single_stem="Vocals",
            )
            separator.load_model()
            if cancel_flag is not None and cancel_flag.is_set():
                raise PipelineCancelled()
            if on_progress:
                on_progress(35.0, "Separating dialogue vocals with MDX-Net...")
            out_files = separator.separate(str(input_wav_path)) or []
            candidates = [Path(tmp_dir) / f if not Path(f).is_absolute() else Path(f) for f in out_files]
            vocals_file = next((p for p in candidates if p.is_file() and "vocal" in p.name.lower()), None)
            if vocals_file is None:
                vocals_file = next((p for p in candidates if p.is_file()), None)
            if vocals_file is None:
                raise RuntimeError("audio-separator produced no vocal output file")
            _convert_to_16k_mono(vocals_file, output_wav_path)
        elapsed = round(time.monotonic() - started, 2)
        if on_progress:
            on_progress(100.0, f"Vocal isolation completed in {elapsed:.1f}s")
        return {
            "path": str(output_wav_path),
            "cached": False,
            "backend": "audio-separator",
            "model": "mdx-net",
            "elapsed_seconds": elapsed,
        }
    finally:
        unload_vocal_model(separator)


def isolate_vocals(
    input_wav_path: str,
    output_wav_path: str,
    *,
    model_name: Optional[str] = None,
    device: str = "auto",
    chunk_seconds: float = _DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = _DEFAULT_OVERLAP_SECONDS,
    source_media_path: Optional[str] = None,
    audio_stream_index: int = 0,
    cancel_flag: Optional[threading.Event] = None,
    on_progress: Optional[Callable[[float, str], None]] = None,
    preloaded_model: Any = None,
    keep_model_loaded: bool = False,
) -> Dict[str, Any]:
    """Extract an isolated dialogue vocals track (`16kHz` mono PCM16 WAV) with disk caching.

    Checks both `output_wav_path` and a content-addressed source cache before running separation,
    processes long tracks in memory-safe windows (`chunk_seconds=180s` by default), and always
    frees separation models and PyTorch CUDA cache before returning.
    """
    chosen_model = model_name or settings.vocal_isolation_model or "htdemucs"
    in_path = Path(input_wav_path)
    out_path = Path(output_wav_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input audio WAV does not exist: {in_path}")

    if _is_valid_wav(out_path):
        logger.info(f"Using cached isolated vocals: {out_path}")
        if on_progress:
            on_progress(100.0, "Using cached isolated vocals")
        return {
            "path": str(out_path),
            "cached": True,
            "backend": "cache",
            "model": chosen_model,
            "elapsed_seconds": 0.0,
        }

    src_cache = source_vocal_cache_path(source_media_path, audio_stream_index, chosen_model)
    if src_cache is not None and _is_valid_wav(src_cache):
        _link_or_copy(src_cache, out_path)
        logger.info(f"Reusing source-cached isolated vocals: {src_cache} -> {out_path}")
        if on_progress:
            on_progress(100.0, "Using cached isolated vocals")
        return {
            "path": str(out_path),
            "cached": True,
            "backend": "cache",
            "model": chosen_model,
            "elapsed_seconds": 0.0,
        }

    with tempfile.TemporaryDirectory(prefix="subbuild_demucs_src_") as clean_src_dir:
        separation_input = in_path
        # When the original container exists on disk, feed Demucs clean 44.1 kHz stereo directly
        # from the selected audio stream rather than a 16 kHz / comms-filtered mono extraction.
        if source_media_path:
            src_media = Path(source_media_path)
            if src_media.is_file():
                clean_stereo = Path(clean_src_dir) / "clean_44k_stereo.wav"
                try:
                    from app.core.executables import executable
                    from app.core.ffmpeg_safety import safe_input

                    cmd = [
                        executable("ffmpeg"),
                        "-y",
                        "-nostats",
                        "-loglevel",
                        "error",
                        *safe_input(str(src_media.resolve())),
                        "-map",
                        f"0:a:{audio_stream_index}",
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        str(clean_stereo.resolve()),
                    ]
                    res = subprocess.run(cmd, capture_output=True, timeout=600)
                    if res.returncode == 0 and _is_valid_wav(clean_stereo):
                        separation_input = clean_stereo
                except Exception as exc:
                    logger.debug(f"Clean stereo extraction for vocal isolation skipped ({exc}); using {in_path}")

        if is_demucs_available():
            result = _run_demucs(
                separation_input,
                out_path,
                model_name=chosen_model,
                device=device,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                cancel_flag=cancel_flag,
                on_progress=on_progress,
                preloaded_model=preloaded_model,
                keep_model_loaded=keep_model_loaded,
            )
        elif is_audio_separator_available():
            result = _run_audio_separator(
                separation_input,
                out_path,
                cancel_flag=cancel_flag,
                on_progress=on_progress,
            )
        else:
            raise VocalIsolationUnavailableError(
                "Neither 'demucs' nor 'audio-separator' is installed. "
                "Install optional vocal isolation dependencies via: pip install -r backend/requirements-optional.txt"
            )

    if src_cache is not None and _is_valid_wav(out_path):
        try:
            _link_or_copy(out_path, src_cache)
        except OSError as exc:
            logger.debug(f"Could not populate source vocal cache {src_cache}: {exc}")

    return result
