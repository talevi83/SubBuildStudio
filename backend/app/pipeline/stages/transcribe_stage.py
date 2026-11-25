import logging
import time
from typing import Optional
from app.core.config import settings
from app.pipeline.context import PipelineCancelled, PipelineContext, RawSegment
from app.pipeline.stages.base import PipelineStage, ProgressCallback
from app.providers.transcription.base import BaseTranscriber

logger = logging.getLogger(__name__)


class TranscribeStage(PipelineStage):
    """Pipeline stage executing Speech-to-Text transcription via Faster-Whisper."""

    def __init__(self, transcriber: Optional[BaseTranscriber] = None):
        self._transcriber = transcriber

    @property
    def name(self) -> str:
        return "TRANSCRIBING"

    async def execute(
        self,
        ctx: PipelineContext,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        if ctx.embedded_dialogue_segments and self._transcriber is None and not ctx.supplement_with_whisper:
            ctx.raw_segments = [
                RawSegment(
                    index=seg["index"],
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    text=seg["original_text"],
                    confidence=seg.get("confidence", 1.0),
                    words=[],
                )
                for seg in ctx.embedded_dialogue_segments
            ]
            if on_progress:
                stream_name = (
                    f"'{ctx.embedded_subtitle_stream.title}'"
                    if ctx.embedded_subtitle_stream and ctx.embedded_subtitle_stream.title
                    else "embedded subtitle track"
                )
                await on_progress(
                    100.0,
                    None,
                    0.0,
                    f"Using {len(ctx.raw_segments)} segments from {stream_name} — skipping Whisper transcription",
                    None,
                )
            return

        if not ctx.audio_wav_path:
            raise ValueError("Audio wav path missing from pipeline context")

        transcriber = self._transcriber
        active_provider = getattr(ctx, "transcription_provider", "faster_whisper") or "faster_whisper"
        if transcriber is None:
            # Imported lazily so the API can start (and tests run) without loading CUDA libs
            from app.providers.transcription import create_transcriber
            from app.services.smart_audio_map import resolve_audio_modes

            _, c_mode = resolve_audio_modes(
                vocal_isolation=bool(getattr(ctx, "vocal_isolation", False)),
                vocal_isolation_mode=getattr(ctx, "vocal_isolation_mode", None),
                comms_boost=bool(getattr(ctx, "comms_boost", False)),
                comms_boost_mode=getattr(ctx, "comms_boost_mode", None),
            )
            # Only inject military TACTICAL_HOTWORDS when comms_boost is explicitly forced "on";
            # in "auto" mode Smart Audio Map already conditioned comms/whisper windows cleanly.
            force_comms_hotwords = (c_mode == "on")
            effective_high_sens = bool(ctx.high_speech_sensitivity or c_mode == "auto")

            transcriber, active_provider, fallback_warning = create_transcriber(
                active_provider,
                model_size_or_path=ctx.whisper_model,
                vad_filter=ctx.vad_filter,
                comms_boost=force_comms_hotwords,
                context_notes=ctx.context_notes,
                initial_prompt=ctx.initial_prompt,
                condition_on_previous_text=ctx.condition_on_previous_text,
                high_speech_sensitivity=effective_high_sens,
                file_name=ctx.file_name,
                enable_diarization=getattr(ctx, "enable_diarization", False),
                isolated_vocals=bool(
                    getattr(ctx, "isolated_vocals_path", None)
                    and resolve_audio_modes(
                        vocal_isolation=bool(getattr(ctx, "vocal_isolation", False)),
                        vocal_isolation_mode=getattr(ctx, "vocal_isolation_mode", None),
                    )[0] == "on"
                ),
                allow_fallback=True,
            )
            ctx.effective_transcription_provider = active_provider
            if fallback_warning:
                ctx.warnings.append(fallback_warning)
        else:
            if not getattr(transcriber, "supports_gap_recovery", True):
                active_provider = "whisperx"
            ctx.effective_transcription_provider = active_provider

        start_time = time.time()
        ctx.raw_segments = []

        async def _segment_callback(raw_seg: RawSegment, progress_pct: float) -> None:
            ctx.raw_segments.append(raw_seg)
            elapsed = time.time() - start_time
            speed_factor = round(raw_seg.end_time / elapsed, 1) if elapsed > 0.5 else None
            remaining_audio = max(0.0, ctx.duration - raw_seg.end_time) if ctx.duration > 0 else 0.0
            eta_seconds = round(remaining_audio / speed_factor, 1) if speed_factor else None

            if on_progress:
                await on_progress(
                    progress_pct,
                    speed_factor,
                    eta_seconds,
                    f"Transcribing: segment #{raw_seg.index} recognized",
                    raw_seg,
                )

        if on_progress:
            engine_label = "WhisperX" if active_provider == "whisperx" else "Whisper"
            await on_progress(0.0, None, None, f"Loading {engine_label} model...", None)

        await transcriber.transcribe(
            audio_path=ctx.audio_wav_path,
            language=ctx.source_language,
            on_segment_callback=_segment_callback,
            cancel_flag=ctx.cancel_flag,
        )
        ctx.detected_language = getattr(transcriber, "detected_language", None) or ctx.detected_language

        # Preserve optional integrated speaker diarization produced by WhisperX
        provider_speakers = getattr(transcriber, "speakers", None)
        provider_timeline = getattr(transcriber, "diarization_timeline", None)
        if isinstance(provider_speakers, dict) and provider_speakers:
            ctx.speakers = provider_speakers
        if isinstance(provider_timeline, list) and provider_timeline:
            ctx.diarization_timeline = provider_timeline

        # Foreign-language passages (e.g. Russian inside an English episode) feed the burned-in
        # subtitle reader in the on-screen text stage. Never fatal.
        if ctx.detect_onscreen_text and getattr(transcriber, "analyze_foreign_speech", None) is not None:
            if on_progress:
                await on_progress(99.0, None, None, "Looking for speech in other languages...", None)
            try:
                main = ctx.effective_source_language
                ctx.foreign_spans, ctx.foreign_audio_segments = await transcriber.analyze_foreign_speech(
                    ctx.audio_wav_path, main, ctx.raw_segments, ctx.cancel_flag
                )
            except PipelineCancelled:
                raise
            except Exception as e:
                logger.warning(f"Foreign speech analysis failed: {e}")
                ctx.warnings.append(f"Foreign-language detection failed: {e}")
            if ctx.foreign_spans and on_progress:
                from app.pipeline.foreign_speech import summarize_spans

                await on_progress(99.5, None, None, f"Speech in other languages: {summarize_spans(ctx.foreign_spans)}", None)

        # Speech the voice detector skipped (shouting over gunfire, radio voices under music).
        # Never run un-VAD'd gap recovery or Silero start-snapping for WhisperX (phoneme forced alignment).
        supports_gap_recovery = getattr(transcriber, "supports_gap_recovery", True) and active_provider != "whisperx"
        recover = getattr(transcriber, "recover_missed_speech", None)
        if supports_gap_recovery and settings.whisper_gap_recovery and recover is not None:
            if on_progress:
                await on_progress(99.6, None, None, "Re-listening to stretches without subtitles...", None)
            try:
                from app.pipeline.foreign_speech import READ_MARGIN

                # The whole window the subtitle reader will scan belongs to the foreign passage:
                # an English guess at Russian speech must not compete with the show's own subtitles
                exclude = [(span["start"] - READ_MARGIN, span["end"] + READ_MARGIN)
                           for span in (ctx.foreign_spans or [])]
                recovered = await recover(
                    ctx.audio_wav_path, ctx.effective_source_language, ctx.raw_segments, exclude, ctx.cancel_flag
                )
            except PipelineCancelled:
                raise
            except Exception as e:
                logger.warning(f"Gap recovery failed: {e}", exc_info=True)
                recovered = []
            if recovered:
                ctx.raw_segments = sorted(ctx.raw_segments + recovered, key=lambda s: s.start_time)
                for i, seg in enumerate(ctx.raw_segments, start=1):
                    seg.index = i
                if on_progress:
                    await on_progress(99.8, None, None, f"Recovered {len(recovered)} line(s) the voice detector missed", None)
        elif active_provider == "whisperx" or not getattr(transcriber, "supports_gap_recovery", True):
            logger.info("Skipping un-VAD'd gap recovery for WhisperX provider (phoneme alignment active)")

        # Whisper puts a word that follows a pause right after the previous word, so subtitles can
        # appear a second or two before anyone speaks; the voice detector knows where speech begins
        supports_word_snap = getattr(transcriber, "supports_word_snap", True) and active_provider != "whisperx"
        snap = getattr(transcriber, "snap_starts_to_speech", None)
        if supports_word_snap and settings.whisper_snap_starts and snap is not None and ctx.raw_segments:
            if on_progress:
                await on_progress(99.9, None, None, "Aligning subtitle starts to the speech...", None)
            try:
                await snap(ctx.audio_wav_path, ctx.raw_segments)
            except PipelineCancelled:
                raise
            except Exception as e:
                logger.warning(f"Aligning word starts failed: {e}", exc_info=True)

        self._last_transcriber = transcriber
        try:
            from app.core.user_settings import load_settings

            if load_settings().cost_saver_mode:
                unload = getattr(transcriber, "unload_model", None)
                if callable(unload):
                    unload()
        except Exception as e:
            logger.debug(f"Could not check cost_saver_mode after transcription: {e}")

        if on_progress:
            lang = f" (language: {ctx.detected_language})" if ctx.detected_language else ""
            await on_progress(100.0, None, None, f"Transcribed {len(ctx.raw_segments)} raw segments{lang}", None)
