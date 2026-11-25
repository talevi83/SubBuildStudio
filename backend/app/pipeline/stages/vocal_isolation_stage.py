"""Optional pipeline stage for vocal isolation and Smart Audio Map window conditioning."""
import asyncio
import logging
from typing import Optional

from app.core.config import settings
from app.pipeline.context import PipelineCancelled, PipelineContext
from app.pipeline.stages.base import PipelineStage, ProgressCallback
from app.services.audio_isolation import (
    VocalIsolationUnavailableError,
    isolate_vocals,
    vocal_cache_path,
)
from app.services.smart_audio_map import (
    analyze_audio_timeline,
    condition_audio_by_map,
    resolve_audio_modes,
)

logger = logging.getLogger(__name__)


class VocalIsolationStage(PipelineStage):
    """Run full-file vocal isolation or localized Smart Audio Map conditioning for ASR.

    Always preserves `ctx.raw_audio_wav_path` pointing at the clean extracted audio so speaker
    diarization (`WeSpeaker`) and voice-pitch gender estimation (`F0`) never see separation or
    compression artifacts.
    """

    @property
    def name(self) -> str:
        return "VOCAL_ISOLATION"

    async def execute(
        self,
        ctx: PipelineContext,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        v_mode, c_mode = resolve_audio_modes(
            vocal_isolation=bool(getattr(ctx, "vocal_isolation", False)),
            vocal_isolation_mode=getattr(ctx, "vocal_isolation_mode", None),
            comms_boost=bool(getattr(ctx, "comms_boost", False)),
            comms_boost_mode=getattr(ctx, "comms_boost_mode", None),
        )
        if v_mode == "off" and c_mode != "auto":
            return

        if (
            ctx.embedded_dialogue_segments
            and not ctx.supplement_with_whisper
            and not getattr(ctx, "enable_diarization", False)
        ):
            if on_progress:
                await on_progress(100.0, None, None, "Using embedded subtitles — audio conditioning skipped", None)
            return

        if not ctx.audio_wav_path:
            raise ValueError("Audio wav path missing from pipeline context")

        if not getattr(ctx, "raw_audio_wav_path", None):
            ctx.raw_audio_wav_path = ctx.audio_wav_path

        loop = asyncio.get_running_loop()
        last_pct = [-10.0]

        def _thread_progress(pct: float, message: str) -> None:
            if on_progress is None or ctx.cancel_flag.is_set():
                return
            clamped = round(max(0.0, min(100.0, float(pct))), 1)
            if clamped - last_pct[0] >= 1.0 or clamped >= 100.0:
                last_pct[0] = clamped
                asyncio.run_coroutine_threadsafe(
                    on_progress(clamped, None, None, message, None),
                    loop,
                )

        # Smart Audio Map mode: scan soundtrack first, apply localized conditioning only where needed,
        # and immediately return to OFF everywhere else.
        if v_mode != "on":
            if on_progress:
                await on_progress(2.0, None, None, "Scanning soundtrack for whispers, comms & music masking...", None)
            try:
                channels = 2
                for stream_info in getattr(ctx, "audio_streams", None) or []:
                    if getattr(stream_info, "index", None) == ctx.selected_audio_stream:
                        channels = getattr(stream_info, "channels", 2) or 2
                        break
                # In `auto` mode, apply gentle linear whisper gain without narrowband `comms`
                # frequency cuts (reserving full radio bandpass filtering for explicit `on` mode).
                windows = await asyncio.to_thread(
                    analyze_audio_timeline,
                    ctx.raw_audio_wav_path,
                    detect_whispers=(c_mode == "auto" or bool(getattr(ctx, "high_speech_sensitivity", False))),
                    detect_comms=False,
                    detect_music_masked=(v_mode == "auto"),
                    surround_center_extracted=(channels >= 6),
                )
                ctx.raise_if_cancelled()
                conditioned_target = settings.storage_dir / "vocals" / f"{ctx.job_id}.conditioned.wav"
                report = await asyncio.to_thread(
                    condition_audio_by_map,
                    ctx.raw_audio_wav_path,
                    conditioned_target,
                    windows,
                    source_media_path=ctx.video_path,
                    audio_stream_index=ctx.selected_audio_stream,
                    model_name=settings.vocal_isolation_model,
                    cancel_flag=ctx.cancel_flag,
                    on_progress=_thread_progress,
                )
                ctx.raise_if_cancelled()
                summary = report.summary_line()
                ctx.smart_audio_summary = summary
                logger.info(f"Job {ctx.job_id}: {summary}")
                if report.conditioned_wav_path and report.conditioned_wav_path != ctx.raw_audio_wav_path:
                    ctx.audio_wav_path = report.conditioned_wav_path
                    if report.used_demucs:
                        ctx.isolated_vocals_path = report.conditioned_wav_path
                if on_progress:
                    await on_progress(100.0, None, None, summary, None)
            except PipelineCancelled:
                raise
            except Exception as exc:
                msg = f"Smart Audio Map conditioning failed ({exc}); continuing with clean extracted audio."
                logger.warning(f"Job {ctx.job_id}: {msg}", exc_info=True)
                ctx.warnings.append(msg)
                if on_progress:
                    await on_progress(100.0, None, None, "Smart Audio Map skipped; using standard audio", None)
            return

        target_path = vocal_cache_path(ctx.job_id)
        if on_progress:
            await on_progress(0.0, None, None, "Preparing vocal isolation...", None)

        try:
            result = await asyncio.to_thread(
                isolate_vocals,
                ctx.raw_audio_wav_path,
                str(target_path),
                model_name=settings.vocal_isolation_model,
                source_media_path=ctx.video_path,
                audio_stream_index=ctx.selected_audio_stream,
                cancel_flag=ctx.cancel_flag,
                on_progress=_thread_progress,
            )
            ctx.raise_if_cancelled()

            isolated_path = result["path"]
            ctx.isolated_vocals_path = isolated_path
            ctx.audio_wav_path = isolated_path

            status_note = (
                "Loaded cached isolated vocals"
                if result.get("cached")
                else f"Isolated dialogue vocals ({result.get('backend', 'demucs')} in {result.get('elapsed_seconds', 0):.1f}s)"
            )
            logger.info(f"Job {ctx.job_id}: {status_note} -> {isolated_path}")
            if on_progress:
                await on_progress(100.0, None, None, status_note, None)
        except PipelineCancelled:
            raise
        except VocalIsolationUnavailableError as exc:
            msg = f"Vocal isolation skipped: {exc}"
            logger.warning(f"Job {ctx.job_id}: {msg}")
            ctx.warnings.append(
                "Vocal isolation skipped: demucs/audio-separator is not installed; used standard extracted audio."
            )
            if on_progress:
                await on_progress(100.0, None, None, "Vocal isolation skipped (optional dependency missing)", None)
        except Exception as exc:
            msg = f"Vocal isolation failed ({exc}); continuing with standard extracted audio."
            logger.warning(f"Job {ctx.job_id}: {msg}", exc_info=True)
            ctx.warnings.append(msg)
            if on_progress:
                await on_progress(100.0, None, None, "Vocal isolation failed; using standard audio", None)
