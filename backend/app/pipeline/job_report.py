"""
Job log report: what a run was asked to do (settings, models) and what it produced
(subtitle counts, warnings, Gemini usage and cost). Written to the per-job log by the
orchestrator; failures here never break the pipeline.
"""
import logging
from typing import Dict, Optional, Sequence

from sqlmodel import Session

from app.core.config import settings
from app.core.database import engine
from app.models.db_models import Job
from app.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def log_job_start(
    ctx: PipelineContext,
    mode: str,
    only_failed: bool = False,
    rebuild_glossary: bool = False,
    only_indices: Optional[Sequence[int]] = None,
    hint: Optional[str] = None,
) -> None:
    """Write the job's settings to the job log at the start of a run."""
    try:
        from app.core.user_settings import load_settings

        s = load_settings()

        def call(c) -> str:
            return f"{c.model} (thinking {c.thinking}{', Flex' if c.flex else ''})"

        run = {"translate": "retranslate", "sync": "sync existing subtitles"}.get(mode, "full run")
        logger.info(f"===== Job {ctx.job_id}: {run} =====")
        if mode == "sync":
            logger.info(f"Subtitle file: {ctx.sync_file_name} | {len(ctx.sync_cues or [])} lines | "
                        f"language {ctx.sync_language} | Gemini matching {ctx.sync_use_gemini} | "
                        f"burned-in detection {settings.sync_detect_burned_in}")
        logger.info(f"File: {ctx.file_name} | duration {ctx.duration:.1f}s | path {ctx.video_path}")
        engine_name = getattr(ctx, "transcription_provider", "faster_whisper") or "faster_whisper"
        v_mode = getattr(ctx, "vocal_isolation_mode", None) or ("on" if getattr(ctx, "vocal_isolation", False) else "off")
        c_mode = getattr(ctx, "comms_boost_mode", None) or ("on" if getattr(ctx, "comms_boost", False) else "off")
        logger.info(
            f"Transcription: engine {engine_name}, model {ctx.whisper_model}, vocal isolation {v_mode}, "
            f"source language {ctx.source_language}, VAD {ctx.vad_filter}, "
            f"comms boost {c_mode}, on-screen text {ctx.detect_onscreen_text}"
        )
        if getattr(ctx, "enable_dual_source_pivot", False):
            logger.info("Dual-Source Pivot (EN + AR): enabled for this job")
        translation_model = ctx.gemini_model or s.translation.model
        logger.info(
            f"AI: translation {'on' if ctx.enable_translation else 'off'} -> {ctx.target_language}, "
            f"model {call(s.translation) if translation_model == s.translation.model else translation_model}; "
            f"glossary {call(s.glossary)}; vision {call(s.vision)}, frame detail {s.vision_media_resolution}"
        )
        if mode == "translate":
            logger.info(f"Retranslate: only failed={only_failed}, rebuild glossary={rebuild_glossary}, "
                        f"segments={len(ctx.refined_segments)}"
                        + (f", selected rows={len(only_indices)}" if only_indices else "")
                        + (f", hint: {hint}" if hint else ""))
    except Exception as e:  # logging must never break the pipeline
        logger.warning(f"Could not log the job settings: {e}")

def log_job_summary(job_id: str, ctx: Optional[PipelineContext]) -> None:
    """Write the result (subtitle counts, warnings, Gemini usage and cost) to the job log."""
    try:
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            counts: Dict[str, int] = {}
            for seg in job.segments:
                kind = seg.segment_type or "DIALOGUE"
                counts[kind] = counts.get(kind, 0) + 1
            untranslated = sum(1 for seg in job.segments if not seg.translated_text)
            failed = sum(1 for seg in job.segments if seg.translation_failed)
            logger.info(
                f"Result: status {job.status.value}, {len(job.segments)} subtitles {counts}, "
                f"untranslated {untranslated}, flagged failed {failed}"
            )
            if ctx is not None:
                eff_engine = getattr(ctx, "effective_transcription_provider", None) or getattr(
                    ctx, "transcription_provider", "faster_whisper"
                )
                smart_summary = getattr(ctx, "smart_audio_summary", None)
                logger.info(
                    f"ASR Summary: effective engine {eff_engine}, "
                    f"vocal isolation used {bool(getattr(ctx, 'isolated_vocals_path', None))}"
                    + (f" | {smart_summary}" if smart_summary else "")
                )
                ds_report = getattr(ctx, "dual_source_report", None)
                if isinstance(ds_report, dict):
                    if ds_report.get("active"):
                        align = ds_report.get("alignment") or {}
                        en_sync = ds_report.get("primary_sync") or ds_report.get("en_sync") or {}
                        ar_sync = ds_report.get("anchor_sync") or ds_report.get("ar_sync") or {}
                        p_lang = (ds_report.get("primary_lang") or "en").upper()
                        a_lang = (ds_report.get("anchor_lang") or "").upper()
                        pair_mode = ds_report.get("pair_mode") or "en_ar"
                        gap_sup = ds_report.get("gap_asr_supplement")
                        logger.info(
                            f"Dual-Source Pivot ({ds_report.get('source', 'api')}, mode={pair_mode}, "
                            f"primary={p_lang}, anchor={a_lang or 'none'}): "
                            f"{p_lang} sync {en_sync.get('engine', 'none')} ({float(en_sync.get('offset_seconds', 0.0)):+.3f}s)"
                            + (
                                f", {a_lang} sync {ar_sync.get('engine', 'none')} ({float(ar_sync.get('offset_seconds', 0.0)):+.3f}s)"
                                if a_lang
                                else ""
                            )
                            + f" | paired={align.get('paired_count', 0)}, unmatched={align.get('en_only_count', 0)}, "
                            f"AR-only ON_SCREEN={align.get('ar_only_count', 0)} | "
                            f"{'gap ASR supplement active; ' if gap_sup else ''}"
                            f"skipped {', '.join(ds_report.get('skipped_stages') or [])}"
                        )
                    else:
                        logger.info(f"Dual-Source Pivot fallback: {ds_report.get('fallback_reason', 'unavailable')}")
            if job.warning_message:
                logger.warning(f"Job warning: {job.warning_message}")
            if job.error_message:
                logger.error(f"Job error: {job.error_message}")
        if ctx is not None and ctx.usage is not None:
            for name, st in ctx.usage.stages.items():
                cost = "unknown" if st.cost_usd is None else f"${st.cost_usd:.4f}"
                std_count = getattr(st, "standard_requests", None)
                if std_count is None:
                    std_count = max(0, st.requests - st.flex_requests - st.flex_fallbacks)
                logger.info(
                    f"AI {name}: {st.model}, {st.requests} request(s), in {st.input_tokens}, "
                    f"out {st.output_tokens}, thinking {st.thinking_tokens}, cost {cost}, "
                    f"Flex {st.flex_requests}, Flex->Standard {st.flex_fallbacks}, Standard {std_count}"
                )
            loc = getattr(ctx.usage, "local", None)
            if loc is not None and (loc.requests > 0 or loc.saved_tokens > 0 or loc.frames_filtered > 0):
                logger.info(
                    f"Local AI & Cost-Saver: model {loc.model or 'local'}, {loc.requests} local query(s), "
                    f"local tokens {loc.input_tokens + loc.output_tokens}, frames filtered {loc.frames_filtered}, "
                    f"saved {loc.saved_tokens} cloud tokens (~${loc.saved_cost_usd:.4f})"
                )
        logger.info(f"===== Job {job_id}: end =====")
    except Exception as e:
        logger.warning(f"Could not log the job summary: {e}")
