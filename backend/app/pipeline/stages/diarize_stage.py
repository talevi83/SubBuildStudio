"""Acoustic speaker diarization stage using local WeSpeaker CAM++ ONNX embeddings."""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings
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

        wav_path = Path(ctx.audio_wav_path) if ctx.audio_wav_path else (settings.storage_dir / "audio" / f"{ctx.job_id}.wav")
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

        def _emit(pct: float, message: str) -> None:
            if on_progress and not ctx.cancel_flag.is_set():
                asyncio.run_coroutine_threadsafe(
                    on_progress(round(pct, 1), None, None, message, None),
                    loop,
                )

        try:
            if on_progress:
                await on_progress(5.0, None, None, "Preparing local speaker diarization model...", None)

            onnx_path = await asyncio.to_thread(
                ensure_diarization_model,
                ctx.cancel_flag,
                lambda p: _emit(5.0 + p * 0.2, "Downloading local speaker model..."),
            )
            ctx.raise_if_cancelled()

            embedder = OnnxSpeakerEmbedder(onnx_path=onnx_path)
            timeline, speakers = await asyncio.to_thread(
                diarize_audio_spans,
                wav_path,
                spans,
                embedder=embedder,
                cancel_flag=ctx.cancel_flag,
                on_progress=lambda p: _emit(25.0 + p * 0.7, "Identifying voices across timeline..."),
            )
            ctx.raise_if_cancelled()

            ctx.diarization_timeline = timeline
            if speakers:
                ctx.speakers = speakers
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
