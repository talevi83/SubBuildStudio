"""WhisperX transcription provider: batched VAD ASR + wav2vec2 phoneme alignment + optional diarization.

Heavy packages (`whisperx`, `torch`) are resolved via `importlib.util.find_spec` and imported
lazily inside execution methods so SubBuild Studio runs cleanly without WhisperX installed.
"""
import asyncio
import importlib.util
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.hardware import (
    ensure_torchaudio_compat,
    get_hardware_info,
    is_whisperx_installed,
    release_torch_vram,
)
from app.pipeline.context import PipelineCancelled, RawSegment, TranscribedWord
from app.pipeline.diarization import SPEAKER_PALETTE
from app.providers.transcription.base import BaseTranscriber, SegmentProgressCallback
from app.providers.transcription.faster_whisper import (
    COMMS_LOG_PROB_THRESHOLD,
    COMMS_NO_SPEECH_THRESHOLD,
    TACTICAL_HOTWORDS,
    build_whisper_initial_prompt,
)

logger = logging.getLogger(__name__)


class WhisperXUnavailableError(RuntimeError):
    """Raised when the optional whisperx package is not installed."""


def is_whisperx_available() -> bool:
    """Return True if whisperx is importable in the current environment."""
    return is_whisperx_installed()


def _register_pytorch_safe_globals() -> None:
    """Allow PyTorch 2.6+ weights_only=True loader to unpickle WhisperX / pyannote VAD checkpoints."""
    try:
        import torch  # type: ignore

        add_fn = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
        if not callable(add_fn):
            return

        import collections
        import typing

        safe_classes: List[Any] = [list, dict, tuple, set, typing.Any, collections.defaultdict]
        for mod_name, attr_names in (
            ("omegaconf", ("ListConfig", "DictConfig")),
            ("omegaconf.base", ("ContainerMetadata", "Metadata")),
            ("omegaconf.nodes", ("AnyNode", "ValueNode")),
            ("pyannote.audio.core.model", ("Introspection",)),
            ("pyannote.audio.core.task", ("Specifications", "Problem", "Resolution")),
        ):
            try:
                mod = __import__(mod_name, fromlist=list(attr_names))
                for attr in attr_names:
                    obj = getattr(mod, attr, None)
                    if obj is not None:
                        safe_classes.append(obj)
            except Exception:
                continue
        add_fn(safe_classes)
    except Exception:
        pass


def _interpolate_word_timings(
    raw_words: List[Dict[str, Any]],
    seg_start: float,
    seg_end: float,
    speaker_map: Dict[str, str],
) -> List[TranscribedWord]:
    """Convert WhisperX aligned word dicts into valid TranscribedWord entries.

    WhisperX leaves digits/symbols without character-level acoustic alignment (`start`/`end`
    missing or None). Fill missing bounds smoothly from surrounding aligned words or segment edges.
    """
    clean_items = [w for w in raw_words if isinstance(w, dict) and str(w.get("word") or "").strip()]
    if not clean_items:
        return []

    n = len(clean_items)
    starts: List[Optional[float]] = []
    ends: List[Optional[float]] = []
    for item in clean_items:
        s_val = item.get("start")
        e_val = item.get("end")
        starts.append(float(s_val) if s_val is not None else None)
        ends.append(float(e_val) if e_val is not None else None)

    for idx in range(n):
        if starts[idx] is None or ends[idx] is None:
            left = ends[idx - 1] if idx > 0 and ends[idx - 1] is not None else seg_start
            right: Optional[float] = None
            for look in range(idx + 1, n):
                if starts[look] is not None:
                    right = starts[look]
                    break
            if right is None:
                right = max(left + 0.05, seg_end)
            missing_run = 1
            for look in range(idx + 1, n):
                if starts[look] is None or ends[look] is None:
                    missing_run += 1
                else:
                    break
            step = max(0.02, (right - left) / missing_run)
            starts[idx] = starts[idx] if starts[idx] is not None else round(left, 3)
            ends[idx] = ends[idx] if ends[idx] is not None else round(max(starts[idx] + 0.02, left + step), 3)

    words_out: List[TranscribedWord] = []
    for idx, item in enumerate(clean_items):
        ws = max(0.0, float(starts[idx] if starts[idx] is not None else seg_start))
        we = max(ws + 0.01, float(ends[idx] if ends[idx] is not None else max(ws + 0.05, seg_end)))
        prob_raw = item.get("score", item.get("probability", 0.92))
        try:
            prob = max(0.0, min(1.0, float(prob_raw)))
        except (TypeError, ValueError):
            prob = 0.92
        raw_sp = item.get("speaker")
        mapped_sp = speaker_map.get(str(raw_sp), str(raw_sp)) if raw_sp else None
        words_out.append(
            TranscribedWord(
                word=f" {str(item.get('word')).strip()}",
                start=round(ws, 3),
                end=round(we, 3),
                probability=round(prob, 3),
                speaker=mapped_sp,
            )
        )
    return words_out


# Stock phrases Whisper invents on breaths, clicks and room tone (normalized, lower case)
PHANTOM_PHRASES = frozenset({
    "thank you", "thanks", "thank you very much", "oh", "ah", "uh", "um", "hmm", "mm", "huh",
    "whoa", "yeah", "okay", "ok", "all right", "alright", "bye", "you",
})
# WhisperX alignment scores below this mean wav2vec2 found almost no matching phonemes
PHANTOM_MAX_SCORE = 0.30
PHANTOM_MAX_WORDS = 3
PHANTOM_MIN_ISOLATION = 1.0  # seconds of silence required on both sides


def filter_phantom_interjections(segments: List[RawSegment]) -> Tuple[List[RawSegment], int]:
    """Drop isolated stock interjections ("Thank you.", "Yeah.") the aligner could not hear.

    Deliberately conservative: a line is dropped only when ALL hold — its whole text is a known
    Whisper breath/noise phrase, it has at most three words, its mean alignment score is below
    `PHANTOM_MAX_SCORE`, and no other speech lies within `PHANTOM_MIN_ISOLATION` seconds of it.
    Real one-word replies sit inside a conversation (small gaps) or align well, so they stay.
    """
    import re as _re

    kept: List[RawSegment] = []
    dropped = 0
    ordered = sorted(segments, key=lambda s: s.start_time)
    for i, seg in enumerate(ordered):
        norm = _re.sub(r"\s+", " ", _re.sub(r"[^\w\s']", " ", (seg.text or "").lower())).strip()
        prev_end = ordered[i - 1].end_time if i > 0 else float("-inf")
        next_start = ordered[i + 1].start_time if i + 1 < len(ordered) else float("inf")
        isolated = (seg.start_time - prev_end) >= PHANTOM_MIN_ISOLATION and (next_start - seg.end_time) >= PHANTOM_MIN_ISOLATION
        if (
            norm in PHANTOM_PHRASES
            and len(norm.split()) <= PHANTOM_MAX_WORDS
            and bool(seg.words)
            and float(seg.confidence) < PHANTOM_MAX_SCORE
            and isolated
        ):
            dropped += 1
            continue
        kept.append(seg)
    for idx, seg in enumerate(kept, start=1):
        seg.index = idx
    return kept, dropped


class WhisperXProvider(BaseTranscriber):
    """Speech-to-Text provider powered by WhisperX.

    Executes:
      - Step A: Batched Whisper ASR with VAD (`batch_size=8` by default, `float16` on CUDA)
      - Step B: Phoneme forced alignment via `whisperx.load_align_model` + `whisperx.align`
      - Step C: Optional integrated pyannote diarization when `HF_TOKEN` is configured

    Each model is unloaded immediately after its step (`gc.collect()` + `torch.cuda.empty_cache()`)
    so peak VRAM stays within 8GB-class GPUs (e.g. RTX 2070 Super). Does NOT run destructive
    un-VAD'd gap recovery.
    """

    supports_gap_recovery: bool = False
    supports_word_snap: bool = False

    def __init__(
        self,
        model_size_or_path: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        vad_filter: bool = True,
        batch_size: Optional[int] = None,
        comms_boost: bool = False,
        context_notes: Optional[str] = None,
        high_speech_sensitivity: bool = False,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: Optional[bool] = None,
        file_name: Optional[str] = None,
        enable_diarization: bool = False,
        hf_token: Optional[str] = None,
        isolated_vocals: bool = False,
    ):
        self.model_name = model_size_or_path or settings.whisper_default_model

        target_device = device or settings.whisper_device
        if target_device == "auto":
            target_device = "cuda" if get_hardware_info()["cuda_available"] else "cpu"
        self.device = target_device

        target_compute = compute_type or settings.whisper_compute_type
        if target_compute == "auto":
            target_compute = "float16" if self.device == "cuda" else "int8"
        self.compute_type = target_compute

        self.vad_filter = bool(vad_filter or comms_boost or high_speech_sensitivity)
        self.batch_size = max(
            1,
            int(settings.whisperx_batch_size if batch_size is None else batch_size),
        )
        self.comms_boost = bool(comms_boost)
        self.high_speech_sensitivity = bool(high_speech_sensitivity)
        self.context_notes = context_notes
        self.initial_prompt = initial_prompt
        self.file_name = file_name
        self.condition_on_previous_text = (
            settings.whisper_condition_on_previous_text
            if condition_on_previous_text is None
            else bool(condition_on_previous_text)
        )
        self.enable_diarization = bool(enable_diarization)
        self.hf_token = hf_token
        self.isolated_vocals = bool(isolated_vocals)

        self.detected_language: Optional[str] = None
        self.diarization_timeline: List[Dict[str, Any]] = []
        self.speakers: Dict[str, Dict[str, Any]] = {}
        self._active_model: Any = None

    def unload_model(self) -> None:
        """Unload any held WhisperX models and clear PyTorch CUDA VRAM."""
        self._active_model = None
        release_torch_vram("WhisperXProvider")

    def _build_asr_options(self, lang_arg: Optional[str], audio_path: str) -> Dict[str, Any]:
        # NOTE: whisperx 3.x batched decoding does not apply the three thresholds below (it calls
        # CTranslate2 `generate` directly); they are kept for API parity only. Speech gating in
        # WhisperX happens in VAD (`_build_vad_options`) and in the post-filters of `_run_sync`.
        asr_options: Dict[str, Any] = {
            "condition_on_previous_text": self.condition_on_previous_text,
            "compression_ratio_threshold": settings.whisper_compression_ratio_threshold,
            "log_prob_threshold": settings.whisper_log_prob_threshold,
            "no_speech_threshold": settings.whisper_no_speech_threshold,
        }
        if self.high_speech_sensitivity:
            asr_options["log_prob_threshold"] = -1.2
            asr_options["no_speech_threshold"] = 0.75
        if self.comms_boost:
            asr_options["log_prob_threshold"] = COMMS_LOG_PROB_THRESHOLD
            asr_options["no_speech_threshold"] = COMMS_NO_SPEECH_THRESHOLD
            if not lang_arg or lang_arg.lower() in ("en", "english"):
                asr_options["hotwords"] = TACTICAL_HOTWORDS

        prompt = build_whisper_initial_prompt(
            initial_prompt=self.initial_prompt,
            context_notes=self.context_notes,
            language=lang_arg,
            comms_boost=self.comms_boost,
            file_name=self.file_name or audio_path,
        )
        if prompt:
            asr_options["initial_prompt"] = prompt
        return asr_options

    def _build_vad_options(self) -> Dict[str, float]:
        default = {"vad_onset": 0.42, "vad_offset": 0.28}
        if self.isolated_vocals:
            # Whole-file isolated vocals have no music left to dig speech out of: keep stricter onset
            # so residual breath clicks are not transcribed as "." / "Oh."
            return {"vad_onset": 0.50, "vad_offset": 0.363}
        if self.comms_boost:
            return {"vad_onset": 0.28, "vad_offset": 0.20}
        if self.high_speech_sensitivity:
            return {"vad_onset": 0.33, "vad_offset": 0.23}
        return default

    def _run_sync(
        self,
        audio_path: str,
        lang_arg: Optional[str],
        cancel_flag: threading.Event,
    ) -> Tuple[List[RawSegment], str]:
        ensure_torchaudio_compat()
        _register_pytorch_safe_globals()
        import whisperx  # type: ignore

        asr_options = self._build_asr_options(lang_arg, audio_path)
        vad_options = self._build_vad_options()
        asr_model = None
        align_model = None
        align_metadata = None
        diarize_model = None

        self.diarization_timeline = []
        self.speakers = {}
        started = time.monotonic()

        try:
            # --------------------------------------------------------------
            # Step A: Batched Whisper ASR with VAD
            # --------------------------------------------------------------
            def _load_asr(dev: str, compute: str) -> Any:
                try:
                    return whisperx.load_model(
                        self.model_name,
                        device=dev,
                        compute_type=compute,
                        language=lang_arg,
                        asr_options=asr_options,
                        vad_options=vad_options,
                    )
                except TypeError:
                    return whisperx.load_model(
                        self.model_name,
                        device=dev,
                        compute_type=compute,
                        language=lang_arg,
                        asr_options=asr_options,
                    )

            logger.info(
                f"WhisperX Step A: loading '{self.model_name}' on {self.device} ({self.compute_type}), "
                f"batch_size={self.batch_size}"
            )
            try:
                asr_model = _load_asr(self.device, self.compute_type)
            except (ImportError, AttributeError):
                raise
            except Exception as exc:
                if self.device == "cuda":
                    logger.warning(f"WhisperX CUDA init failed ({exc}); falling back to CPU INT8")
                    self.device, self.compute_type = "cpu", "int8"
                    asr_model = _load_asr("cpu", "int8")
                else:
                    raise

            self._active_model = asr_model
            if cancel_flag.is_set():
                raise PipelineCancelled()

            audio = whisperx.load_audio(audio_path)
            asr_result = asr_model.transcribe(audio, batch_size=self.batch_size, language=lang_arg)
            detected_lang = str(asr_result.get("language") or lang_arg or "en")
            self.detected_language = detected_lang
            logger.info(
                f"WhisperX Step A complete: {len(asr_result.get('segments') or [])} raw segment(s), "
                f"language='{detected_lang}'"
            )
        finally:
            self._active_model = None
            del asr_model
            release_torch_vram("WhisperX Step A ASR")

        if cancel_flag.is_set():
            raise PipelineCancelled()

        aligned_result: Dict[str, Any] = dict(asr_result or {})
        raw_segs = list(aligned_result.get("segments") or [])

        # ------------------------------------------------------------------
        # Step B: Phoneme Forced Alignment
        # ------------------------------------------------------------------
        if raw_segs:
            try:
                logger.info(f"WhisperX Step B: loading phoneme aligner for language '{detected_lang}' on {self.device}")
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=self.device,
                )
                self._active_model = align_model
                if cancel_flag.is_set():
                    raise PipelineCancelled()
                aligned_result = whisperx.align(
                    raw_segs,
                    align_model,
                    align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
                logger.info(
                    f"WhisperX Step B complete: aligned {len(aligned_result.get('segments') or [])} segment(s)"
                )
            except PipelineCancelled:
                raise
            except Exception as exc:
                logger.warning(f"WhisperX phoneme alignment skipped for '{detected_lang}': {exc}")
                aligned_result = dict(asr_result or {})
            finally:
                self._active_model = None
                del align_model, align_metadata
                release_torch_vram("WhisperX Step B alignment")

        if cancel_flag.is_set():
            raise PipelineCancelled()

        # ------------------------------------------------------------------
        # Step C: Optional Integrated Diarization (when HF_TOKEN is configured)
        # ------------------------------------------------------------------
        token = (self.hf_token or settings.hf_token or os.environ.get("HF_TOKEN") or "").strip() or None
        if self.enable_diarization and token and aligned_result.get("segments"):
            try:
                diarize_cls = getattr(getattr(whisperx, "diarize", None), "DiarizationPipeline", None)
                if diarize_cls is None:
                    diarize_cls = getattr(whisperx, "DiarizationPipeline", None)
                if diarize_cls is None:
                    logger.info("WhisperX DiarizationPipeline unavailable; deferring to local ONNX DiarizeStage")
                else:
                    logger.info("WhisperX Step C: running integrated speaker diarization")
                    try:
                        diarize_model = diarize_cls(use_auth_token=token, device=self.device)
                    except TypeError:
                        diarize_model = diarize_cls(token=token, device=self.device)
                    self._active_model = diarize_model
                    diarize_segments = diarize_model(audio)
                    aligned_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            except PipelineCancelled:
                raise
            except Exception as exc:
                logger.warning(f"WhisperX integrated diarization failed non-fatally: {exc}")
            finally:
                self._active_model = None
                del diarize_model
                release_torch_vram("WhisperX Step C diarization")
        elif self.enable_diarization:
            logger.info("WhisperX diarization deferred to local ONNX DiarizeStage (no HF_TOKEN configured)")

        # ------------------------------------------------------------------
        # Convert output to SubBuild Studio RawSegment / TranscribedWord format
        # ------------------------------------------------------------------
        from app.pipeline.gap_recovery import is_watermark_hallucination
        from app.utils.text_rules import has_spoken_content

        try:
            media_duration = float(len(audio)) / 16000.0  # whisperx.load_audio always resamples to 16 kHz
        except Exception:
            media_duration = None
        dropped_punct = 0

        speaker_map: Dict[str, str] = {}
        speakers_dict: Dict[str, Dict[str, Any]] = {}
        timeline: List[Dict[str, Any]] = []

        def _map_speaker(raw_sp: Optional[str]) -> Optional[str]:
            if not raw_sp:
                return None
            key = str(raw_sp).strip()
            if not key:
                return None
            if key not in speaker_map:
                idx = len(speaker_map) + 1
                norm = f"SPEAKER_{idx:02d}"
                speaker_map[key] = norm
                speakers_dict[norm] = {
                    "label": f"Speaker {idx}",
                    "gender": "unknown",
                    "color": SPEAKER_PALETTE[(idx - 1) % len(SPEAKER_PALETTE)],
                }
            return speaker_map[key]

        output: List[RawSegment] = []
        for seg in aligned_result.get("segments") or []:
            if cancel_flag.is_set():
                raise PipelineCancelled()
            seg_text = str(seg.get("text") or "").strip()
            if not seg_text:
                continue
            if not has_spoken_content(seg_text):
                # ".", ". .", "..." decoded from breaths/clicks: nothing to read or translate
                dropped_punct += 1
                continue
            if is_watermark_hallucination(seg_text, float(seg.get("start") or 0.0), media_duration):
                logger.info(
                    f"Filtered watermark/outro hallucination at {float(seg.get('start') or 0):.1f}-"
                    f"{float(seg.get('end') or 0):.1f}s: {seg_text[:70]!r}"
                )
                continue

            seg_start = float(seg.get("start") or 0.0)
            seg_end = max(seg_start + 0.05, float(seg.get("end") or (seg_start + 0.5)))

            raw_sp = seg.get("speaker")
            if raw_sp:
                _map_speaker(str(raw_sp))
            for w in seg.get("words") or []:
                if isinstance(w, dict) and w.get("speaker"):
                    _map_speaker(str(w.get("speaker")))

            words = _interpolate_word_timings(list(seg.get("words") or []), seg_start, seg_end, speaker_map)
            # Punctuation-only "words" (a stray "." aligned to a breath) never become subtitle text
            words = [w for w in words if has_spoken_content(w.word)]
            mapped_seg_sp = _map_speaker(str(raw_sp)) if raw_sp else None
            if mapped_seg_sp is None and words:
                first_w_sp = next((w.speaker for w in words if w.speaker), None)
                mapped_seg_sp = first_w_sp

            if mapped_seg_sp:
                timeline.append({"start": round(seg_start, 3), "end": round(seg_end, 3), "speaker": mapped_seg_sp})

            probs = [w.probability for w in words]
            conf = round(sum(probs) / len(probs), 2) if probs else 0.92
            output.append(
                RawSegment(
                    index=len(output) + 1,
                    start_time=round(seg_start, 3),
                    end_time=round(seg_end, 3),
                    text=seg_text,
                    confidence=conf,
                    words=words,
                    speaker=mapped_seg_sp,
                )
            )

        output, dropped_phantom = filter_phantom_interjections(output)
        if dropped_punct or dropped_phantom:
            logger.info(
                f"Dropped {dropped_punct} punctuation-only and {dropped_phantom} unaligned stock-interjection "
                f"WhisperX segment(s) (breath/noise artifacts)"
            )
        self.speakers = speakers_dict
        self.diarization_timeline = timeline
        elapsed = round(time.monotonic() - started, 2)
        logger.info(
            f"WhisperX pipeline complete in {elapsed:.1f}s: {len(output)} segment(s), "
            f"{len(speakers_dict)} integrated speaker(s)"
        )
        return output, detected_lang

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        on_segment_callback: Optional[SegmentProgressCallback] = None,
        cancel_flag: Optional[threading.Event] = None,
    ) -> List[RawSegment]:
        if not is_whisperx_available():
            raise WhisperXUnavailableError(
                "WhisperX is not installed. Install optional dependencies via: pip install -r backend/requirements-optional.txt"
            )

        cancel_ev = cancel_flag or threading.Event()
        lang_arg = None if (not language or language.lower() in ("auto", "none")) else language

        try:
            segments, _ = await asyncio.to_thread(self._run_sync, audio_path, lang_arg, cancel_ev)
            total_end = max([s.end_time for s in segments] or [1.0])
            if on_segment_callback:
                for seg in segments:
                    if cancel_ev.is_set():
                        raise PipelineCancelled()
                    pct = min(99.0, round((seg.end_time / max(1.0, total_end)) * 100.0, 2))
                    await on_segment_callback(seg, pct)
            return segments
        finally:
            self.unload_model()
