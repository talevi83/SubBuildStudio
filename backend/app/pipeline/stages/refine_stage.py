"""
Refine stage: raw Whisper output -> final, editable subtitles.

The work is done by two modules next to this stage:

* `pipeline.segmentation` — re-segments Whisper's words into broadcast-style blocks
  (sentence grouping, line limits, reading speed, timing).
* `pipeline.timeline_merge` — tags COMMS lines and merges ON_SCREEN and FOREIGN cues into the
  spoken timeline.

The stage then applies speaker-turn splitting, the Whisper supplement for embedded/synced
subtitles and end-credits filtering.
"""
import logging
from typing import Optional

from app.core.config import settings
from app.models.db_models import SegmentType
from app.pipeline.context import PipelineContext
from app.pipeline.segmentation import (
    enforce_min_duration,
    merge_micro_segments,
    merge_subsecond_cues,
    refine_raw_segments,
)
from app.pipeline.stages.base import PipelineStage, ProgressCallback
from app.pipeline.timeline_merge import classify_comms, merge_foreign, merge_timeline
from app.utils.text_rules import has_spoken_content

logger = logging.getLogger(__name__)


class RefineStage(PipelineStage):
    """Pipeline stage for refining raw transcribed subtitle blocks to broadcast standards."""

    @property
    def name(self) -> str:
        return "REFINING"

    async def execute(
        self,
        ctx: PipelineContext,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        if on_progress:
            await on_progress(10.0, None, None, "Refining subtitle segments and timing...", None)

        from app.utils.speaker_split import split_dual_speaker_segments

        if ctx.embedded_dialogue_segments:
            spoken = [
                {
                    "index": i + 1,
                    "start_time": float(seg["start_time"]),
                    "end_time": float(seg["end_time"]),
                    "original_text": seg["original_text"],
                    "translated_text": None,
                    "translation_failed": False,
                    "segment_type": seg.get("segment_type", SegmentType.DIALOGUE.value),
                    "confidence": float(seg.get("confidence", 1.0)),
                }
                for i, seg in enumerate(ctx.embedded_dialogue_segments)
            ]
            if ctx.supplement_with_whisper and ctx.raw_segments:
                from app.pipeline.missing_speech import supplement_missing_speech
                spoken, recovered_count = supplement_missing_speech(
                    spoken,
                    ctx.raw_segments,
                    same_language=True,
                    min_gap=settings.min_gap_seconds,
                    max_chars_per_line=settings.max_chars_per_line,
                )
                if recovered_count > 0:
                    logger.info(f"RefineStage: recovered {recovered_count} missing speech segments from Whisper.")
        else:
            spoken = refine_raw_segments(
                ctx.raw_segments,
                max_chars_per_line=settings.max_chars_per_line,
            )
            foreign_audio = refine_raw_segments(ctx.foreign_audio_segments, max_chars_per_line=settings.max_chars_per_line)
            spoken = merge_foreign(spoken, ctx.foreign_spans, ctx.foreign_ocr_segments, foreign_audio)
            spoken = merge_micro_segments(
                spoken,
                max_chars_per_line=settings.max_chars_per_line,
                max_duration=settings.max_subtitle_duration,
            )

        if not ctx.embedded_dialogue_segments:
            before = len(spoken)
            spoken = [s for s in spoken if has_spoken_content(s.get("original_text") or "")]
            if len(spoken) != before:
                logger.info(f"RefineStage: dropped {before - len(spoken)} punctuation-only cue(s)")

        comms = classify_comms(spoken) if ctx.comms_boost else 0
        spoken = split_dual_speaker_segments(
            spoken,
            max_chars_per_line=settings.max_chars_per_line,
            min_gap=settings.min_gap_seconds,
        )
        if not ctx.embedded_dialogue_segments:
            spoken = merge_subsecond_cues(
                spoken,
                min_duration=settings.min_subtitle_duration,
                max_gap=0.35,
                max_chars_per_line=settings.max_chars_per_line,
                max_duration=settings.max_subtitle_duration,
            )
            spoken = enforce_min_duration(
                spoken,
                min_duration=settings.min_subtitle_duration,
                buffer_sec=0.05,
                max_chars_per_line=settings.max_chars_per_line,
            )
        foreign = sum(1 for s in spoken if s.get("segment_type") == SegmentType.FOREIGN.value)
        allow_overlap = getattr(ctx, "allow_onscreen_overlap", True)
        ctx.refined_segments = merge_timeline(
            spoken,
            ctx.onscreen_segments,
            settings.max_chars_per_line,
            allow_overlap=allow_overlap,
        )

        if getattr(ctx, "diarization_timeline", None):
            from app.pipeline.diarization import assign_speakers_to_segments

            missing = [
                s for s in ctx.refined_segments
                if (s.get("segment_type") or "DIALOGUE") != SegmentType.ON_SCREEN.value and not s.get("speaker")
            ]
            if missing:
                assign_speakers_to_segments(missing, ctx.diarization_timeline)
            for s in ctx.refined_segments:
                if (s.get("segment_type") or "DIALOGUE") == SegmentType.ON_SCREEN.value:
                    s["speaker"] = None

        credits_removed = 0
        if getattr(ctx, "skip_end_credits", True):
            from app.pipeline.credits_detect import filter_credits_segments

            chapters = (ctx.video_metadata or {}).get("chapters") or ()
            visual_spans = getattr(ctx, "visual_credit_spans", ()) or ()
            ctx.refined_segments, credits_removed = filter_credits_segments(
                ctx.refined_segments,
                ctx.duration or 0.0,
                chapters=chapters,
                visual_credit_spans=visual_spans,
            )

        if getattr(ctx, "speakers", None):
            from app.pipeline.diarization import prune_unused_speakers

            ctx.speakers = prune_unused_speakers(ctx.speakers, ctx.refined_segments)

        if on_progress:
            extras = []
            if comms:
                extras.append(f"{comms} radio/comms")
            if ctx.onscreen_segments:
                extras.append(f"{len(ctx.onscreen_segments)} on-screen")
            if foreign:
                extras.append(f"{foreign} foreign-language")
            if credits_removed:
                extras.append(f"{credits_removed} credits skipped")
            suffix = f" ({', '.join(extras)})" if extras else ""
            await on_progress(100.0, None, None, f"Refined {len(ctx.refined_segments)} subtitle segments{suffix}", None)
