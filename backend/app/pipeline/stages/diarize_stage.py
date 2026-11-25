"""Acoustic speaker diarization stage using local WeSpeaker CAM++ ONNX embeddings."""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.pipeline import diarization
from app.pipeline.context import PipelineCancelled, PipelineContext
from app.pipeline.diarization import (
    OnnxSpeakerEmbedder,
    diarize_audio_spans,
    ensure_diarization_model,
    split_raw_segments_by_speaker,
)
from app.pipeline.stages.base import PipelineStage, ProgressCallback

logger = logging.getLogger(__name__)


class DiarizeStage(PipelineStage):
    """Clusters acoustic speaker voices and stamps SPEAKER_XX onto transcribed words and segments."""

    @property
    def name(self) -> str:
        return "DIARIZING"

    async def execute(
        self,
        ctx: PipelineContext,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        if not getattr(ctx, "enable_diarization", False):
            return

        # Always prefer the clean, un-separated extracted WAV (`raw_audio_wav_path`) over Demucs /
        # conditioned audio so separation phase artifacts never corrupt WeSpeaker voiceprints or F0 pitch.
        raw_candidate = Path(ctx.raw_audio_wav_path) if getattr(ctx, "raw_audio_wav_path", None) else None
        if raw_candidate is not None and raw_candidate.is_file():
            wav_path = raw_candidate
        else:
            wav_path = Path(ctx.audio_wav_path) if ctx.audio_wav_path else (settings.storage_dir / "audio" / f"{ctx.job_id}.wav")

        # If integrated WhisperX diarization already clustered speakers, enrich with voice pitch
        # (non-fatal) and ensure segments carry speaker labels instead of re-running ONNX clustering.
        has_integrated_speakers = bool(ctx.speakers) and (
            bool(ctx.diarization_timeline)
            or any(getattr(s, "speaker", None) for s in (ctx.raw_segments or []))
        )
        if has_integrated_speakers:
            if ctx.diarization_timeline and wav_path.is_file():
                try:
                    from app.pipeline.voice_gender import label_speakers_by_voice

                    await asyncio.to_thread(label_speakers_by_voice, wav_path, ctx.diarization_timeline, ctx.speakers)
                except Exception as exc:
                    logger.info(f"Voice gender estimate skipped for {ctx.job_id}: {exc}")
            if (
                ctx.raw_segments
                and ctx.diarization_timeline
                and not any(getattr(s, "speaker", None) for s in ctx.raw_segments)
            ):
                ctx.raw_segments = split_raw_segments_by_speaker(ctx.raw_segments, ctx.diarization_timeline)
            if on_progress:
                await on_progress(
                    100.0,
                    None,
                    None,
                    f"Detected {len(ctx.speakers)} speaker(s) via integrated diarization",
                    None,
                )
            return

        if not wav_path.is_file():
            msg = "Speaker diarization skipped: extracted audio file is not available."
            ctx.warnings.append(msg)
            logger.warning(msg)
            if on_progress:
                await on_progress(100.0, None, None, msg, None)
            return

        if ctx.raw_segments:
            spans = [(float(s.start_time), float(s.end_time)) for s in ctx.raw_segments]
        elif ctx.embedded_dialogue_segments:
            spans = [(float(s["start_time"]), float(s["end_time"])) for s in ctx.embedded_dialogue_segments]
        else:
            spans = []

        if not spans:
            if on_progress:
                await on_progress(100.0, None, None, "No speech spans to diarize", None)
            return

        loop = asyncio.get_running_loop()
        last_emitted: list[object] = [-10.0, ""]

        def _emit(pct: float, message: str) -> None:
            if on_progress and not ctx.cancel_flag.is_set():
                clamped = round(max(0.0, min(100.0, float(pct))), 1)
                if (
                    message != last_emitted[1]
                    or abs(clamped - float(last_emitted[0])) >= 1.0
                    or clamped >= 100.0
                ):
                    last_emitted[0] = clamped
                    last_emitted[1] = message
                    asyncio.run_coroutine_threadsafe(
                        on_progress(clamped, None, None, message, None),
                        loop,
                    )

        try:
            cached = diarization.model_path()
            needs_download = not (cached.is_file() and cached.stat().st_size > 1_000_000)
            if on_progress:
                init_pct = 0.0 if needs_download else 2.0
                init_msg = (
                    "Downloading local speaker model..."
                    if needs_download
                    else "Preparing local speaker diarization model..."
                )
                last_emitted[0] = init_pct
                last_emitted[1] = init_msg
                await on_progress(init_pct, None, None, init_msg, None)

            onnx_path = await asyncio.to_thread(
                ensure_diarization_model,
                ctx.cancel_flag,
                (lambda p: _emit(p * 0.20, "Downloading local speaker model...")) if needs_download else None,
            )
            ctx.raise_if_cancelled()

            embedder = OnnxSpeakerEmbedder(onnx_path=onnx_path)
            base_pct, span_pct = (20.0, 0.78) if needs_download else (2.0, 0.96)
            timeline, speakers = await asyncio.to_thread(
                diarize_audio_spans,
                wav_path,
                spans,
                embedder=embedder,
                cancel_flag=ctx.cancel_flag,
                on_progress=lambda p: _emit(
                    base_pct + p * span_pct,
                    "Clustering speaker voices..." if p >= 90.0 else "Identifying voices across timeline...",
                ),
            )
            ctx.raise_if_cancelled()

            ctx.diarization_timeline = timeline
            if speakers:
                if timeline:
                    # Gendered target languages need each speaker's gender; the voice is often the
                    # only clue (a line with no name or pronoun). A guess, never a reason to fail.
                    try:
                        from app.pipeline.voice_gender import label_speakers_by_voice

                        await asyncio.to_thread(label_speakers_by_voice, wav_path, timeline, speakers)
                    except Exception as exc:
                        logger.info(f"Voice gender estimate skipped for {ctx.job_id}: {exc}")
                ctx.speakers = speakers
                from app.pipeline.diarization import assess_diarization

                quality = assess_diarization(speakers, timeline)
                ctx.diarization_quality = quality
                logger.info(
                    f"Diarization for {ctx.job_id}: {quality['speakers']} speaker(s) with >=2% of the speech"
                    + (f"; {'; '.join(quality['reasons'])}" if quality["reasons"] else "")
                )
            if ctx.raw_segments and timeline:
                ctx.raw_segments = split_raw_segments_by_speaker(ctx.raw_segments, timeline)

            try:
                from app.core.user_settings import load_settings

                if load_settings().cost_saver_mode:
                    OnnxSpeakerEmbedder.unload()
            except Exception:
                pass

            if on_progress:
                await on_progress(
                    100.0,
                    None,
                    None,
                    f"Detected {len(ctx.speakers)} speaker(s)",
                    None,
                )
        except PipelineCancelled:
            raise
        except Exception as exc:
            msg = f"Speaker diarization skipped: {exc}"
            logger.warning(f"DiarizeStage failed non-fatally for {ctx.job_id}: {exc}")
            ctx.warnings.append(msg)
            if on_progress:
                await on_progress(100.0, None, None, msg, None)
