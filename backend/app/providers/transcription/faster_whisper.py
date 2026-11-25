import asyncio
import gc
import math
import logging
import threading
from typing import Dict, List, Optional, Tuple

from app.core import compat as _compat  # noqa: F401  (must run before faster_whisper import)

from faster_whisper import WhisperModel

try:  # faster-whisper >= 1.1
    from faster_whisper import BatchedInferencePipeline
except ImportError:  # pragma: no cover
    BatchedInferencePipeline = None  # type: ignore[assignment]

from app.core.config import settings
from app.core.hardware import get_hardware_info
from app.pipeline.context import RawSegment, TranscribedWord, PipelineCancelled
from app.pipeline.foreign_speech import SAMPLE_RATE, build_foreign_spans, group_speech, load_wav
from app.pipeline.word_snap import SnapStats, snap_word_starts
from app.providers.transcription.base import BaseTranscriber, SegmentProgressCallback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# "Tactical Comms Boost" decoding profile
# ---------------------------------------------------------------------------
TACTICAL_PRIMER = (
    "Clear transcript including military radio transmissions, walkie-talkie comms, callsigns, "
    "over, copy, roger, sitrep, klicks, and low-voice tactical dialogue. "
    "All right, do you copy? Yes, loud and clear, over. Let's take a look at what we have here."
)
# Short vocabulary list passed as `hotwords`: unlike `initial_prompt` (which faster-whisper only
# uses for the first window when condition_on_previous_text=False) hotwords bias every window.
TACTICAL_HOTWORDS = "over, copy, roger, wilco, sitrep, klicks, callsign, actual, say again, radio check"

# A permissive VAD so band-limited / static-laden transmissions are not discarded as noise
COMMS_VAD_PARAMETERS = dict(
    threshold=0.28,
    min_speech_duration_ms=120,
    min_silence_duration_ms=400,
    speech_pad_ms=450,
)
COMMS_NO_SPEECH_THRESHOLD = 0.88
COMMS_LOG_PROB_THRESHOLD = -1.25
DEFAULT_VAD_PARAMETERS = dict(
    min_silence_duration_ms=500,
    speech_pad_ms=200,
)
_MAX_NOTES_IN_PROMPT = 500
_GENERIC_MEDIA_STEMS = {
    "audio", "video", "movie", "clip", "sample", "test", "input", "output", "stream", "track",
}
_RELEASE_NOISE_RE = re.compile(
    r"\b(?:2160p|1080p|720p|480p|4k|uhd|hdr|hdr10|dv|dovi|bluray|blu[\-\s]?ray|bdrip|brrip|"
    r"web[\-\s]?dl|webrip|web|hdrip|dvdrip|remux|x264|x265|h\.?264|h\.?265|hevc|av1|aac\d*|"
    r"ddp\d*|eac3|ac3|dts(?:\-hd)?|truehd|atmos|10bit|8bit|proper|repack|amzn|nf|dsnp|atvp|hmax)\b.*$",
    re.IGNORECASE,
) if "re" in globals() else None


def derive_title_prompt(file_name: Optional[str]) -> Optional[str]:
    """Extract a clean movie/episode title hint from media filename metadata when meaningful."""
    import os
    import re

    raw = (file_name or "").strip()
    if not raw:
        return None
    stem = os.path.splitext(os.path.basename(raw))[0].strip()
    if not stem or len(stem) < 4:
        return None
    low_stem = stem.lower()
    if low_stem in _GENERIC_MEDIA_STEMS or re.match(r"^(?:job|tmp|temp|v|f|audio|vid)[_\-][a-z0-9]+$", low_stem):
        return None
    if re.fullmatch(r"[0-9a-f_\-]{6,}", low_stem):
        return None

    cleaned = re.sub(r"[._]+", " ", stem)
    cleaned = re.sub(r"\[[^\]]*\]|\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(
        r"\b(?:2160p|1080p|720p|480p|4k|uhd|hdr|hdr10|dv|dovi|bluray|blu[\-\s]?ray|bdrip|brrip|"
        r"web[\-\s]?dl|webrip|hdrip|dvdrip|remux|x264|x265|h\.?264|h\.?265|hevc|av1|aac\d*|"
        r"ddp\d*|eac3|ac3|dts|truehd|atmos|10bit|8bit|proper|repack)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_.,;:")
    # Format trailing release year nicely: "The Uprising 2026" -> "The Uprising (2026)"
    m_year = re.match(r"^(.+?)\s+\(?((?:19|20)\d{2})\)?$", cleaned)
    if m_year:
        title_part, year_part = m_year.group(1).strip(" -_("), m_year.group(2)
        if len(title_part) >= 3:
            cleaned = f"{title_part} ({year_part})"
    words = [w for w in re.findall(r"[A-Za-z\u00C0-\u024F0-9']+", cleaned) if any(c.isalpha() for c in w)]
    if not words or (len(words) == 1 and words[0].lower() in _GENERIC_MEDIA_STEMS):
        return None
    if len(cleaned) < 4:
        return None
    return cleaned[:120]


def _sanitize_prompt_for_audio(raw_text: Optional[str], language: Optional[str]) -> str:
    """Prepare user context/glossary text for Whisper's initial_prompt without cross-script bias.

    When the spoken audio is not an RTL language (e.g. English), target-language RTL annotations
    such as ``John Ball = ג'ון בול`` have their RTL spans stripped so the source-script proper
    nouns (``John Ball``) survive cleanly. Pure-RTL notes (e.g. translator gender instructions in
    Hebrew) return an empty string so Whisper is not nudged into another language.
    """
    import re
    from app.utils.languages import is_rtl
    from app.utils.rtl_fixer import contains_rtl

    text = (raw_text or "").strip()
    if not text:
        return ""
    lang = (language or "auto").strip().lower()
    if lang not in ("auto", "none", "") and is_rtl(lang):
        return re.sub(r"\s+", " ", text)[:_MAX_NOTES_IN_PROMPT].strip()

    if not contains_rtl(text):
        return re.sub(r"\s+", " ", text)[:_MAX_NOTES_IN_PROMPT].strip()

    # Strip RTL character runs and adjacent mapping markers (=, ->, →) while preserving Latin/source terms
    stripped = re.sub(
        r"\s*(?:=|->|→|:)?\s*[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufeff]+"
        r"(?:[\s'\"׳״\-–—()[\],./]*[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufeff]+)*\s*(?:=|->|→)?",
        " ",
        text,
    )
    # Remove empty parentheses/brackets left behind after stripping RTL explanations
    stripped = re.sub(r"\(\s*\)|\[\s*\]", " ", stripped)
    stripped = re.sub(r"\s*,\s*,+", ", ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ,;:-=>→")
    if not re.search(r"[A-Za-z0-9\u00C0-\u024F\u0400-\u04FF]", stripped):
        return ""
    return stripped[:_MAX_NOTES_IN_PROMPT].strip()


def build_whisper_initial_prompt(
    initial_prompt: Optional[str],
    context_notes: Optional[str],
    language: Optional[str],
    comms_boost: bool = False,
    file_name: Optional[str] = None,
) -> Optional[str]:
    """Build Whisper's ``initial_prompt`` from explicit ``initial_prompt``, ``context_notes``, title metadata, and mode primer."""
    parts: List[str] = []
    seen_cf: set[str] = set()
    for candidate in (initial_prompt, context_notes):
        cleaned = _sanitize_prompt_for_audio(candidate, language)
        if cleaned and cleaned.casefold() not in seen_cf:
            parts.append(cleaned)
            seen_cf.add(cleaned.casefold())

    if not parts and file_name:
        title_hint = _sanitize_prompt_for_audio(derive_title_prompt(file_name), language)
        if title_hint:
            parts.append(title_hint)
            seen_cf.add(title_hint.casefold())

    if comms_boost:
        lang = (language or "auto").strip().lower()
        if lang in ("auto", "none", "", "en", "english"):
            parts.append(TACTICAL_PRIMER)
    return " ".join(parts) if parts else None


def build_comms_prompt(context_notes: Optional[str], language: Optional[str]) -> Optional[str]:
    """
    Combine the user's show notes with the tactical primer for Whisper's initial_prompt.

    - Notes written purely in a right-to-left script (e.g. Hebrew notes for the translator) are left out:
      a prompt in another script pushes Whisper toward that language.
    - The English primer is only added when the audio is (or may be) English.
    - Whisper keeps the *last* ~224 prompt tokens, so the primer goes last and survives truncation.
    """
    return build_whisper_initial_prompt(
        initial_prompt=None,
        context_notes=context_notes,
        language=language,
        comms_boost=True,
    )

# ---------------------------------------------------------------------------
# Process-wide model cache. Loading large-v3 takes many seconds and several GB of
# VRAM, so the most recently used model is kept and reused by the next job. Only one
# model is kept at a time so switching sizes doesn't pile models up on the GPU.
# ---------------------------------------------------------------------------
_model_cache: Dict[Tuple[str, str, str], WhisperModel] = {}
_model_lock = threading.Lock()


def _load_model(name: str, device: str, compute_type: str) -> WhisperModel:
    key = (name, device, compute_type)
    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]
        if _model_cache:
            logger.info("Unloading previously cached Whisper model(s)")
            _model_cache.clear()
            gc.collect()
        logger.info(f"Loading Faster-Whisper model '{name}' on {device} ({compute_type})...")
        model = WhisperModel(name, device=device, compute_type=compute_type)
        _model_cache[key] = model
        return model


class FasterWhisperTranscriber(BaseTranscriber):
    """
    Local Speech-to-Text provider powered by faster-whisper (CTranslate2).
    Supports CUDA FP16 / CPU INT8, Silero VAD, word-level timestamps, batched GPU
    inference and cooperative cancellation.
    """

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

        self.comms_boost = comms_boost
        self.high_speech_sensitivity = high_speech_sensitivity
        self.context_notes = context_notes
        self.initial_prompt = initial_prompt
        self.file_name = file_name
        self.condition_on_previous_text = (
            settings.whisper_condition_on_previous_text
            if condition_on_previous_text is None
            else bool(condition_on_previous_text)
        )
        # Comms mode & high speech sensitivity always run VAD with their respective parameters
        self.vad_filter = vad_filter or comms_boost or high_speech_sensitivity
        self.batch_size = settings.whisper_batch_size if batch_size is None else batch_size
        self.detected_language: Optional[str] = None
        self._model: Optional[WhisperModel] = None

    def _get_model(self) -> WhisperModel:
        """Load (or reuse) the model, falling back to CPU INT8 if CUDA init fails."""
        try:
            self._model = _load_model(self.model_name, self.device, self.compute_type)
            return self._model
        except Exception as e:
            if self.device != "cuda":
                raise
            logger.warning(f"Failed to initialize Faster-Whisper on CUDA ({e}). Falling back to CPU INT8...")
            self.device, self.compute_type = "cpu", "int8"
            self._model = _load_model(self.model_name, "cpu", "int8")
            return self._model

    def unload_model(self) -> None:
        """Free the loaded Whisper model from GPU/CPU memory."""
        with _model_lock:
            self._model = None
            _model_cache.clear()
        gc.collect()
        logger.info("Unloaded Faster-Whisper model to free GPU/CPU memory")

    def _use_batched(self) -> bool:
        # Batched inference needs VAD to cut the audio into chunks and only pays off on GPU.
        return (
            BatchedInferencePipeline is not None
            and self.batch_size > 0
            and self.device == "cuda"
            and self.vad_filter
        )

    def _start_transcription(self, model: WhisperModel, audio_path: str, lang_arg: Optional[str], batched: bool):
        common: Dict[str, object] = dict(
            language=lang_arg,
            word_timestamps=True,
            # Prevents cascading hallucinations across segments during music or crowd noise:
            # each decoding window starts clean without conditioning on previous output.
            condition_on_previous_text=self.condition_on_previous_text,
            compression_ratio_threshold=settings.whisper_compression_ratio_threshold,
            log_prob_threshold=settings.whisper_log_prob_threshold,
            no_speech_threshold=settings.whisper_no_speech_threshold,
        )
        if self.vad_filter:
            common["vad_parameters"] = dict(
                min_silence_duration_ms=settings.whisper_min_silence_duration_ms,
                speech_pad_ms=settings.whisper_speech_pad_ms,
            )
        if self.high_speech_sensitivity:
            common.update(
                vad_parameters=dict(
                    threshold=0.35,
                    min_speech_duration_ms=150,
                    min_silence_duration_ms=400,
                    speech_pad_ms=400,
                ),
                no_speech_threshold=0.75,
                log_prob_threshold=-1.2,
            )
        if self.comms_boost:
            common.update(
                vad_parameters=dict(COMMS_VAD_PARAMETERS),
                no_speech_threshold=COMMS_NO_SPEECH_THRESHOLD,
                log_prob_threshold=COMMS_LOG_PROB_THRESHOLD,
            )
            if not lang_arg or lang_arg.lower() in ("en", "english"):
                common["hotwords"] = TACTICAL_HOTWORDS
        prompt = build_whisper_initial_prompt(
            initial_prompt=self.initial_prompt,
            context_notes=self.context_notes,
            language=lang_arg,
            comms_boost=self.comms_boost,
            file_name=self.file_name or audio_path,
        )
        if prompt:
            common["initial_prompt"] = prompt
        if batched:
            logger.info(f"Transcribing with batched inference (batch_size={self.batch_size})")
            pipeline = BatchedInferencePipeline(model=model)
            chunk_length = 25 if self.high_speech_sensitivity else 15
            return pipeline.transcribe(
                audio_path, batch_size=self.batch_size, vad_filter=True, chunk_length=chunk_length, **common
            )
        return model.transcribe(audio_path, vad_filter=self.vad_filter, **common)

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        on_segment_callback: Optional[SegmentProgressCallback] = None,
        cancel_flag: Optional[threading.Event] = None,
    ) -> List[RawSegment]:
        """
        Transcribe audio without blocking the event loop, streaming each segment
        through on_segment_callback as soon as it is recognized.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        lang_arg = None if (not language or language.lower() in ("auto", "none")) else language
        cancel_flag = cancel_flag or threading.Event()
        # Local stop signal: set when we stop listening early (error in a callback etc.)
        # without marking the whole job as cancelled.
        stop = threading.Event()

        def _should_stop() -> bool:
            return cancel_flag.is_set() or stop.is_set()

        def _put(item) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def _run(model: WhisperModel, batched: bool) -> int:
            """Decode the whole file; returns the number of segments emitted."""
            segments_iter, info = self._start_transcription(model, audio_path, lang_arg, batched)
            _put(("INFO", (float(info.duration or 1.0), getattr(info, "language", None))))
            emitted = 0
            # segments_iter is lazy: decoding happens while we iterate, so breaking
            # out of the loop really stops the GPU work.
            from app.pipeline.gap_recovery import is_watermark_hallucination
            from app.utils.text_rules import has_spoken_content

            try:
                for s in segments_iter:
                    if _should_stop():
                        _put(("CANCELLED", None))
                        return -1
                    seg_text = (s.text or "").strip()
                    if not seg_text or not has_spoken_content(seg_text):
                        continue  # empty or punctuation-only ("...", ". .") breath/noise artifact
                    if is_watermark_hallucination(seg_text, float(s.start), float(info.duration or 0.0) or None):
                        logger.info(f"Filtered watermark/outro hallucination at {s.start:.1f}-{s.end:.1f}s: {seg_text[:70]!r}")
                        continue
                    words = [
                        TranscribedWord(
                            word=w.word,
                            start=float(w.start),
                            end=float(w.end),
                            probability=float(getattr(w, "probability", 1.0)),
                        )
                        for w in (getattr(s, "words", None) or [])
                    ]
                    avg_logprob = getattr(s, "avg_logprob", None)
                    conf = max(0.0, min(1.0, math.exp(avg_logprob))) if avg_logprob is not None else 0.95
                    emitted += 1
                    _put(("SEGMENT", RawSegment(
                        index=emitted,
                        start_time=float(s.start),
                        end_time=float(s.end),
                        text=seg_text,
                        confidence=round(conf, 2),
                        words=words,
                    )))
            except Exception as e:
                e.emitted = emitted  # type: ignore[attr-defined]
                raise
            return emitted

        def _worker():
            try:
                model = self._get_model()
                if _should_stop():
                    _put(("CANCELLED", None))
                    return
                batched = self._use_batched()
                try:
                    result = _run(model, batched)
                except Exception as e:
                    # Batched decoding can run out of VRAM. If nothing was emitted yet we
                    # can transparently retry with classic sequential decoding.
                    if batched and getattr(e, "emitted", 0) == 0:
                        logger.warning(f"Batched transcription failed ({e}); retrying sequentially")
                        gc.collect()
                        result = _run(model, False)
                    elif batched:
                        raise RuntimeError(
                            f"{e} (batched GPU inference; try lowering WHISPER_BATCH_SIZE in backend/.env)"
                        ) from e
                    else:
                        raise
                if result >= 0:
                    _put(("DONE", None))
            except Exception as e:
                _put(("ERROR", e))

        worker_task = asyncio.ensure_future(asyncio.to_thread(_worker))

        results: List[RawSegment] = []
        audio_duration = 1.0
        try:
            while True:
                msg_type, payload = await queue.get()
                if msg_type == "INFO":
                    audio_duration = max(1.0, payload[0])
                    self.detected_language = payload[1]
                elif msg_type == "SEGMENT":
                    results.append(payload)
                    progress_pct = min(99.0, round((payload.end_time / audio_duration) * 100, 2))
                    if on_segment_callback:
                        await on_segment_callback(payload, progress_pct)
                elif msg_type == "DONE":
                    break
                elif msg_type == "CANCELLED":
                    raise PipelineCancelled()
                elif msg_type == "ERROR":
                    raise RuntimeError(f"Whisper transcription failed: {payload}") from payload
        finally:
            # If we leave early (cancel / callback error), tell the worker to stop and
            # wait for it so the GPU is actually free before the next job starts.
            stop.set()
            await asyncio.gather(worker_task, return_exceptions=True)
        return results


    # ------------------------------------------------------------------
    # Foreign-language passages
    # ------------------------------------------------------------------

    def _analyze_foreign_sync(
        self,
        audio_path: str,
        main_language: Optional[str],
        spoken: List[RawSegment],
        cancel_flag: threading.Event,
    ):
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        model = self._get_model()
        audio = load_wav(audio_path)
        timestamps = get_speech_timestamps(
            audio, VadOptions(min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=100)
        )
        chunks = []
        for start, end in group_speech(timestamps):
            if cancel_flag.is_set():
                raise PipelineCancelled()
            piece = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            language, probability, _ = model.detect_language(audio=piece)
            chunks.append({"start": start, "end": end, "language": language, "probability": float(probability)})

        spans = build_foreign_spans(chunks, main_language, [(s.start_time, s.end_time) for s in spoken])

        # Fallback text: re-run Whisper on each foreign span in that span's own language.
        # (Transcribing rather than Whisper's "translate" task: large-v3-turbo can't translate,
        # and Gemini translates the original language better than a double translation.)
        fallback: List[RawSegment] = []
        for span in spans:
            if not span["language"]:
                continue
            for core_start, core_end in span["cores"]:
                if cancel_flag.is_set():
                    raise PipelineCancelled()
                piece = audio[int(core_start * SAMPLE_RATE):int(core_end * SAMPLE_RATE)]
                segments, _ = model.transcribe(
                    piece,
                    language=span["language"],
                    word_timestamps=True,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                for s in segments:
                    text = s.text.strip()
                    if not text:
                        continue
                    words = [
                        TranscribedWord(word=w.word, start=float(w.start) + core_start, end=float(w.end) + core_start,
                                        probability=float(getattr(w, "probability", 1.0)))
                        for w in (getattr(s, "words", None) or [])
                    ]
                    avg = getattr(s, "avg_logprob", None)
                    fallback.append(RawSegment(
                        index=len(fallback) + 1,
                        start_time=float(s.start) + core_start,
                        end_time=float(s.end) + core_start,
                        text=text,
                        confidence=round(max(0.0, min(1.0, math.exp(avg))), 2) if avg is not None else 0.9,
                        words=words,
                    ))
        logger.info(f"Foreign speech analysis: {len(chunks)} chunks, {len(spans)} span(s)")
        return spans, fallback

    def _recover_gaps_sync(
        self,
        audio_path: str,
        language: Optional[str],
        spoken: List[RawSegment],
        exclude: List[Tuple[float, float]],
        cancel_flag: threading.Event,
    ) -> List[RawSegment]:
        from app.pipeline import gap_recovery as gr

        audio = load_wav(audio_path)
        duration = len(audio) / SAMPLE_RATE
        gaps = gr.find_gaps(gr.covered_intervals(spoken), duration, exclude)
        min_level_db = -54.0 if getattr(self, "high_speech_sensitivity", False) else gr.MIN_LEVEL_DB
        gaps = [g for g in gr.split_long(gaps, max_len=14.0) if gr.level_db(audio, *g) > min_level_db]
        if not gaps:
            logger.info("Gap recovery: no uncovered audio worth re-listening to")
            return []
        total = sum(e - s for s, e in gaps)
        logger.info(f"Gap recovery: re-listening to {len(gaps)} uncovered stretch(es), {total:.0f}s, without VAD")

        model = self._get_model()
        gap_prompt = build_whisper_initial_prompt(
            initial_prompt=self.initial_prompt,
            context_notes=self.context_notes,
            language=language,
            comms_boost=False,
            file_name=self.file_name or audio_path,
        )
        options: Dict[str, object] = dict(
            language=language,
            clip_timestamps=[t for g in gaps for t in g],
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
        )
        if gap_prompt:
            options["initial_prompt"] = gap_prompt
        segments, _ = model.transcribe(audio, **options)

        context = [{"start": s.start_time, "end": s.end_time, "text": s.text} for s in spoken]
        recovered: List[RawSegment] = []
        rejected = 0
        for seg in segments:
            if cancel_flag.is_set():
                raise PipelineCancelled()
            start, end, text = float(seg.start), float(seg.end), (seg.text or "").strip()
            mid = (start + end) / 2
            if not any(a - 0.3 <= mid <= b + 0.3 for a, b in gaps):
                continue
            words = [w for w in (getattr(seg, "words", None) or []) if w.end > w.start]
            # Judge the span of the words, not the segment: Whisper pads segments with silence
            spoken_span = (float(words[-1].end) - float(words[0].start)) if words else end - start
            reason = gr.rejection_reason(
                text,
                getattr(seg, "avg_logprob", None),
                getattr(seg, "no_speech_prob", None),
                getattr(seg, "compression_ratio", None),
                [float(getattr(w, "probability", 1.0)) for w in words],
                spoken_span,
                gr.neighbours_of(start, end, context + [{"start": r.start_time, "end": r.end_time, "text": r.text} for r in recovered]),
                prompt_text=gap_prompt,
            )
            stats = (f"logprob {getattr(seg, 'avg_logprob', 0):.2f}, no_speech {getattr(seg, 'no_speech_prob', 0):.2f}")
            if reason:
                rejected += 1
                logger.info(f"Gap recovery: - {start:.1f}-{end:.1f}s {text[:70]!r} rejected: {reason} ({stats})")
                continue
            logger.info(f"Gap recovery: + {start:.1f}-{end:.1f}s {text[:70]!r} ({stats})")
            avg = getattr(seg, "avg_logprob", None)
            recovered.append(RawSegment(
                index=0,
                start_time=start,
                end_time=end,
                text=text,
                confidence=round(max(0.0, min(1.0, math.exp(avg))), 2) if avg is not None else 0.8,
                words=[TranscribedWord(word=w.word, start=float(w.start), end=float(w.end),
                                       probability=float(getattr(w, "probability", 1.0))) for w in words],
            ))
        logger.info(f"Gap recovery: kept {len(recovered)} line(s), rejected {rejected}")
        return recovered

    def _snap_starts_sync(self, audio_path: str, spoken: List[RawSegment]) -> SnapStats:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        audio = load_wav(audio_path)
        # Tight settings on purpose: the question is where speech *begins*, so no padding and
        # short pauses count as pauses
        timestamps = get_speech_timestamps(
            audio, VadOptions(threshold=0.5, min_speech_duration_ms=100, min_silence_duration_ms=150, speech_pad_ms=0)
        )
        regions = [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE) for t in timestamps]
        stats = snap_word_starts(spoken, regions)
        logger.info(
            f"Word timing: {stats.moved} of {stats.words} word start(s) moved to the speech onset "
            f"(mean {stats.mean_shift:.2f}s, max {stats.max_shift:.2f}s)"
        )
        return stats

    async def snap_starts_to_speech(self, audio_path: str, spoken: List[RawSegment]) -> SnapStats:
        """Correct word starts that Whisper placed inside a pause (see app.pipeline.word_snap)."""
        return await asyncio.to_thread(self._snap_starts_sync, audio_path, spoken)

    async def recover_missed_speech(
        self,
        audio_path: str,
        language: Optional[str],
        spoken: List[RawSegment],
        exclude: Optional[List[Tuple[float, float]]] = None,
        cancel_flag: Optional[threading.Event] = None,
    ) -> List[RawSegment]:
        """Re-listen (without VAD) to audio no subtitle covers; returns the lines worth keeping."""
        return await asyncio.to_thread(
            self._recover_gaps_sync, audio_path, language, spoken, list(exclude or []), cancel_flag or threading.Event()
        )

    async def analyze_foreign_speech(
        self,
        audio_path: str,
        main_language: Optional[str],
        spoken: List[RawSegment],
        cancel_flag: Optional[threading.Event] = None,
    ):
        if not main_language:
            return [], []
        return await asyncio.to_thread(
            self._analyze_foreign_sync, audio_path, main_language, spoken, cancel_flag or threading.Event()
        )
