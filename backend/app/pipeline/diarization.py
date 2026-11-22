"""Local zero-token acoustic speaker diarization powered by WeSpeaker CAM++ ONNX + NumPy."""
import functools
import logging
import os
import threading
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np

from app.core.config import settings
from app.pipeline.context import PipelineCancelled, RawSegment, TranscribedWord

logger = logging.getLogger(__name__)

MODEL_FILENAME = "wespeaker_en_voxceleb_CAM++.onnx"
# Note: upstream release tag in k2-fsa/sherpa-onnx uses 'speaker-recongition-models' spelling
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx"
)
SPEAKER_PALETTE = (
    "#38BDF8",
    "#F472B6",
    "#4ADE80",
    "#FBBF24",
    "#C084FC",
    "#FB923C",
    "#2DD4BF",
    "#F87171",
)


def model_path() -> Path:
    """Return local cache path for the ONNX speaker embedding model."""
    return settings.storage_dir / "models" / MODEL_FILENAME


def ensure_diarization_model(
    cancel_flag: Optional[threading.Event] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Path:
    """Ensure the WeSpeaker ONNX model is cached on disk, streaming from GitHub if missing."""
    path = model_path()
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    if os.environ.get("SUBBUILD_OFFLINE") == "1":
        raise RuntimeError("Speaker diarization model is not cached locally (SUBBUILD_OFFLINE=1).")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".onnx.tmp")
    logger.info(f"Downloading speaker diarization model to {path}...")
    try:
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            with client.stream("GET", MODEL_URL) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or 0)
                downloaded = 0
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=131_072):
                        if cancel_flag is not None and cancel_flag.is_set():
                            raise PipelineCancelled()
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if on_progress and total > 0:
                            on_progress(min(100.0, (downloaded / total) * 100.0))
        if tmp.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded diarization model file is unexpectedly small or incomplete.")
        os.replace(tmp, path)
        logger.info(f"Speaker diarization model ready ({path.stat().st_size // (1024 * 1024)} MB)")
        return path
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.exp(np.asarray(mel, dtype=np.float64) / 1127.0) - 1.0)


@functools.lru_cache(maxsize=8)
def _mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    low_freq: float = 20.0,
    high_freq: float = 7600.0,
) -> np.ndarray:
    """Compute a Kaldi-compatible triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    high = min(high_freq, sample_rate * 0.5)
    mel_pts = np.linspace(_hz_to_mel(low_freq), _hz_to_mel(high), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        if center > left:
            up = (fft_freqs - left) / (center - left)
            mask = (fft_freqs >= left) & (fft_freqs <= center)
            fb[m, mask] = up[mask]
        if right > center:
            down = (right - fft_freqs) / (right - center)
            mask = (fft_freqs > center) & (fft_freqs <= right)
            fb[m, mask] = down[mask]
    return fb


def extract_fbank_80(samples_16k: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract Kaldi-compatible 80-dim log-Mel filterbanks with per-utterance CMN for WeSpeaker CAM++.

    `samples_16k` is mono float32 audio normalized in [-1.0, 1.0].
    Returns float32 features of shape (T, 80) with zero-mean per frequency bin.
    """
    arr = np.asarray(samples_16k, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((1, 80), dtype=np.float32)

    # Scale to 16-bit PCM range as expected by Kaldi / sherpa-onnx fbank extractors
    wav = arr * 32768.0
    frame_len = max(16, int(round(0.025 * sample_rate)))  # 400 at 16kHz
    frame_shift = max(8, int(round(0.010 * sample_rate)))  # 160 at 16kHz
    n_fft = 1 << (frame_len - 1).bit_length()  # 512

    if wav.size < frame_len:
        wav = np.pad(wav, (0, frame_len - wav.size), mode="constant")

    num_frames = 1 + (wav.size - frame_len) // frame_shift
    idx = (
        np.arange(frame_len, dtype=np.int64)[None, :]
        + np.arange(num_frames, dtype=np.int64)[:, None] * frame_shift
    )
    frames = wav[idx].astype(np.float32, copy=True)

    # Per-frame DC offset removal, then 0.97 pre-emphasis, then Povey window
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    frames[:, 0] *= 0.03

    n = np.arange(frame_len, dtype=np.float64)
    povey = np.power(0.5 - 0.5 * np.cos((2.0 * np.pi * n) / max(1, frame_len - 1)), 0.85).astype(np.float32)
    frames *= povey[None, :]

    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)

    fb = _mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=80, low_freq=20.0, high_freq=7600.0)
    mel_energy = np.maximum(power @ fb.T, 1e-10)
    feats = np.log(mel_energy).astype(np.float32)

    # Per-utterance Cepstral Mean Normalization (CMN only, no variance scaling)
    feats -= feats.mean(axis=0, keepdims=True)
    return feats


class OnnxSpeakerEmbedder:
    """Thread-safe wrapper caching an onnxruntime session for WeSpeaker CAM++ speaker embeddings."""

    _lock = threading.Lock()
    _session: Optional[Any] = None
    _session_path: Optional[Path] = None

    def __init__(self, onnx_path: Optional[Path] = None):
        self.onnx_path = onnx_path or model_path()

    def _get_session(self) -> Any:
        import onnxruntime as ort

        with self._lock:
            if OnnxSpeakerEmbedder._session is not None and OnnxSpeakerEmbedder._session_path == self.onnx_path:
                return OnnxSpeakerEmbedder._session
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
            session = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])
            OnnxSpeakerEmbedder._session = session
            OnnxSpeakerEmbedder._session_path = self.onnx_path
            return session

    def embed_fbank(self, feats: np.ndarray) -> np.ndarray:
        """Compute an L2-normalized 1D speaker embedding from (T, 80) fbank features."""
        session = self._get_session()
        inp_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        x = np.asarray(feats, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, :, :]
        emb = np.asarray(session.run([out_name], {inp_name: x})[0], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        return emb / max(norm, 1e-12)

    def embed_samples(self, samples_16k: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        return self.embed_fbank(extract_fbank_80(samples_16k, sample_rate=sample_rate))

    @classmethod
    def unload(cls) -> None:
        with cls._lock:
            cls._session = None
            cls._session_path = None


def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = 0.45,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[str]:
    """
    Pure-NumPy agglomerative average-linkage cosine clustering with singleton noise absorption.

    Returns chronological labels ('SPEAKER_01', 'SPEAKER_02', ...).
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    n = int(arr.shape[0]) if arr.ndim == 2 else 0
    if n == 0:
        return []
    if n == 1:
        return ["SPEAKER_01"]

    norms = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    normed = arr / norms
    dist = np.clip(1.0 - (normed @ normed.T), 0.0, 2.0)

    clusters: List[List[int]] = [[i] for i in range(n)]
    floor_k = max(1, int(min_speakers)) if min_speakers is not None else 1
    ceil_k = max(floor_k, int(max_speakers)) if max_speakers is not None else None

    def _pair_dist(ca: List[int], cb: List[int]) -> float:
        return float(dist[np.ix_(ca, cb)].mean())

    while len(clusters) > floor_k:
        best_i, best_j, best_d = -1, -1, float("inf")
        k = len(clusters)
        for i in range(k):
            for j in range(i + 1, k):
                d = _pair_dist(clusters[i], clusters[j])
                if d < best_d:
                    best_i, best_j, best_d = i, j, d
        need_forced_merge = ceil_k is not None and len(clusters) > ceil_k
        if best_d < threshold or need_forced_merge:
            clusters[best_i] = clusters[best_i] + clusters[best_j]
            clusters.pop(best_j)
        else:
            break

    # Absorb isolated single-window outliers when multi-window clusters exist
    absorb_limit = min(threshold + 0.15, 0.72)
    changed = True
    while changed and len(clusters) > floor_k:
        changed = False
        multi = [idx for idx, c in enumerate(clusters) if len(c) >= 2]
        if not multi:
            break
        for idx, c in enumerate(list(clusters)):
            if len(c) != 1:
                continue
            best_target, best_d = -1, float("inf")
            for target_idx in range(len(clusters)):
                if target_idx == idx or len(clusters[target_idx]) < 2:
                    continue
                d = _pair_dist(c, clusters[target_idx])
                if d < best_d:
                    best_target, best_d = target_idx, d
            if best_target >= 0 and best_d < absorb_limit:
                clusters[best_target] = clusters[best_target] + c
                clusters.pop(idx)
                changed = True
                break

    # Chronological relabeling by earliest embedding index in each cluster
    clusters.sort(key=min)
    labels = [""] * n
    for sp_num, members in enumerate(clusters, start=1):
        label = f"SPEAKER_{sp_num:02d}"
        for m in members:
            labels[m] = label
    return labels


def _read_wav_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {sampwidth * 8}-bit")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, sr


def diarize_audio_spans(
    wav_path: Path,
    spans: Sequence[Tuple[float, float]],
    *,
    embedder: Optional[OnnxSpeakerEmbedder] = None,
    threshold: float = 0.45,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    cancel_flag: Optional[threading.Event] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Extract voice windows over `spans`, cluster embeddings, and return (timeline, speakers)."""
    samples, sr = _read_wav_float32(Path(wav_path))
    if samples.size == 0 or not spans:
        return [], {}

    total_dur = samples.size / sr
    # Merge adjacent speech spans separated by < 0.15s
    clean_spans: List[List[float]] = []
    for s, e in sorted((max(0.0, float(a)), min(total_dur, float(b))) for a, b in spans):
        if e - s < 0.15:
            continue
        if clean_spans and s - clean_spans[-1][1] <= 0.15:
            clean_spans[-1][1] = max(clean_spans[-1][1], e)
        else:
            clean_spans.append([s, e])
    if not clean_spans:
        return [], {}

    win_sec, hop_sec, min_win = 1.6, 0.8, 0.35
    windows: List[Tuple[float, float]] = []
    for s, e in clean_spans:
        dur = e - s
        if dur < min_win:
            continue
        if dur <= win_sec:
            windows.append((s, e))
        else:
            t = s
            while t + min_win < e:
                w_end = min(e, t + win_sec)
                if w_end - t >= min_win:
                    windows.append((t, w_end))
                if w_end >= e:
                    break
                t += hop_sec

    if not windows:
        # Fallback: single longest span >= 0.15s
        longest = max(clean_spans, key=lambda pair: pair[1] - pair[0])
        windows = [(longest[0], longest[1])]

    active_embedder = embedder or OnnxSpeakerEmbedder()
    embeddings: List[np.ndarray] = []
    total_w = len(windows)
    for idx, (ws, we) in enumerate(windows):
        if cancel_flag is not None and cancel_flag.is_set():
            raise PipelineCancelled()
        i0 = max(0, int(round(ws * sr)))
        i1 = min(samples.size, int(round(we * sr)))
        chunk = samples[i0:i1]
        embeddings.append(active_embedder.embed_samples(chunk, sample_rate=sr))
        if on_progress:
            on_progress(((idx + 1) / total_w) * 100.0)

    emb_mat = np.vstack(embeddings)
    win_labels = cluster_embeddings(
        emb_mat,
        threshold=threshold,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    timeline: List[Dict[str, Any]] = []
    for (ws, we), sp in zip(windows, win_labels):
        if timeline and timeline[-1]["speaker"] == sp and ws - timeline[-1]["end"] <= 1.2:
            timeline[-1]["end"] = max(timeline[-1]["end"], round(we, 3))
        else:
            # Resolve boundary overlap cleanly against prior turn of a different speaker
            eff_start = round(ws, 3)
            if timeline and eff_start < timeline[-1]["end"]:
                mid = round((timeline[-1]["end"] + eff_start) * 0.5, 3)
                timeline[-1]["end"] = mid
                eff_start = mid
            timeline.append({"start": eff_start, "end": round(we, 3), "speaker": sp})

    ordered_speakers: List[str] = []
    for item in timeline:
        sp = item["speaker"]
        if sp not in ordered_speakers:
            ordered_speakers.append(sp)

    speakers: Dict[str, Dict[str, Any]] = {}
    for idx, sp_id in enumerate(ordered_speakers):
        speakers[sp_id] = {
            "label": f"Speaker {idx + 1}",
            "gender": "unknown",
            "color": SPEAKER_PALETTE[idx % len(SPEAKER_PALETTE)],
        }
    return timeline, speakers


def _dominant_speaker(start: float, end: float, timeline: Sequence[Dict[str, Any]]) -> Optional[str]:
    if not timeline:
        return None
    overlap_by_speaker: Dict[str, float] = {}
    for entry in timeline:
        ov = min(end, float(entry["end"])) - max(start, float(entry["start"]))
        if ov > 0:
            sp = str(entry["speaker"])
            overlap_by_speaker[sp] = overlap_by_speaker.get(sp, 0.0) + ov
    if overlap_by_speaker:
        return max(overlap_by_speaker.items(), key=lambda kv: kv[1])[0]

    # Fallback: nearest timeline segment within 1.0s
    nearest_sp: Optional[str] = None
    best_dist = 1.01
    for entry in timeline:
        es, ee = float(entry["start"]), float(entry["end"])
        gap = max(0.0, es - end, start - ee)
        if gap < best_dist:
            best_dist = gap
            nearest_sp = str(entry["speaker"])
    return nearest_sp


def split_raw_segments_by_speaker(
    raw_segments: List[RawSegment],
    timeline: Sequence[Dict[str, Any]],
) -> List[RawSegment]:
    """Assign speakers to words/segments and split segments on sustained (>=2 word) speaker switches."""
    if not raw_segments or not timeline:
        return raw_segments

    out: List[RawSegment] = []
    for seg in raw_segments:
        words = list(seg.words or [])
        if not words:
            out.append(
                RawSegment(
                    index=len(out) + 1,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=seg.text,
                    confidence=seg.confidence,
                    words=[],
                    speaker=_dominant_speaker(seg.start_time, seg.end_time, timeline),
                )
            )
            continue

        word_speakers: List[Optional[str]] = [
            _dominant_speaker(float(w.start), float(w.end), timeline) for w in words
        ]
        # Suppress isolated 1-word flickers surrounded by the same speaker [A, B, A] -> [A, A, A]
        if len(word_speakers) >= 3:
            smoothed = list(word_speakers)
            for idx in range(1, len(word_speakers) - 1):
                if (
                    word_speakers[idx - 1]
                    and word_speakers[idx - 1] == word_speakers[idx + 1]
                    and word_speakers[idx] != word_speakers[idx - 1]
                ):
                    smoothed[idx] = word_speakers[idx - 1]
            word_speakers = smoothed

        for w, sp in zip(words, word_speakers):
            w.speaker = sp

        # Partition runs only on sustained (>= 2 consecutive words) speaker switches
        runs: List[Tuple[Optional[str], List[TranscribedWord]]] = []
        for w, sp in zip(words, word_speakers):
            if not runs:
                runs.append((sp, [w]))
                continue
            prev_sp, prev_words = runs[-1]
            if sp is None or prev_sp is None or sp == prev_sp:
                runs[-1] = (prev_sp or sp, prev_words + [w])
            else:
                runs.append((sp, [w]))

        # Merge any 1-word switch run back into adjacent run so fleeting flickers never fragment cues
        merged_runs: List[Tuple[Optional[str], List[TranscribedWord]]] = []
        for sp, r_words in runs:
            if merged_runs and len(r_words) < 2:
                prev_sp, prev_words = merged_runs[-1]
                merged_runs[-1] = (prev_sp, prev_words + r_words)
            else:
                merged_runs.append((sp, r_words))

        if len(merged_runs) <= 1:
            sp = merged_runs[0][0] if merged_runs else _dominant_speaker(seg.start_time, seg.end_time, timeline)
            out.append(
                RawSegment(
                    index=len(out) + 1,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=seg.text,
                    confidence=seg.confidence,
                    words=words,
                    speaker=sp,
                )
            )
            continue

        for sp, r_words in merged_runs:
            for w in r_words:
                w.speaker = sp
            chunk_text = "".join(w.word for w in r_words).strip()
            if not chunk_text:
                continue
            out.append(
                RawSegment(
                    index=len(out) + 1,
                    start_time=round(float(r_words[0].start), 3),
                    end_time=round(max(float(r_words[0].start) + 0.05, float(r_words[-1].end)), 3),
                    text=chunk_text,
                    confidence=seg.confidence,
                    words=r_words,
                    speaker=sp,
                )
            )
    return out


def assign_speakers_to_segments(segments: Sequence[Any], timeline: Sequence[Dict[str, Any]]) -> None:
    """Assign `speaker` onto segment dicts or ORM rows by temporal overlap (skips ON_SCREEN)."""
    if not segments:
        return
    for seg in segments:
        is_dict = isinstance(seg, dict)
        seg_type = (seg.get("segment_type") if is_dict else getattr(seg, "segment_type", "DIALOGUE")) or "DIALOGUE"
        if seg_type == "ON_SCREEN":
            if is_dict:
                seg["speaker"] = None
            else:
                seg.speaker = None
            continue
        start = float(seg.get("start_time", 0.0) if is_dict else getattr(seg, "start_time", 0.0))
        end = float(seg.get("end_time", start) if is_dict else getattr(seg, "end_time", start))
        sp = _dominant_speaker(start, end, timeline)
        if is_dict:
            seg["speaker"] = sp
        else:
            seg.speaker = sp
