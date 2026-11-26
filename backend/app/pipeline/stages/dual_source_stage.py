"""Feature-flagged Smart Subtitles-First -> Fallback ASR stage (EN + AR + fallback languages)."""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.models.db_models import SegmentType
from app.pipeline.context import PipelineContext
from app.pipeline.embedded_tracks import extract_track_as_srt, pick_dialogue_track, pick_track_by_language
from app.pipeline.stages.base import PipelineStage, ProgressCallback
from app.services.dual_source_aligner import (
    DEFAULT_GENDER_FALLBACK_FOR_EN,
    DEFAULT_SECONDARY_FOR_AR,
    align_dual_cues,
    evaluate_alignment_quality,
    normalize_lang_code,
    parse_srt_cues,
    select_subtitle_pair,
)
from app.services.dual_source_fetcher import fetch_dual_subtitles
from app.services.dual_source_sync import sync_srt_to_audio
from app.utils.languages import language_code, language_name

logger = logging.getLogger(__name__)


def _build_ordered_chain(defaults: Sequence[str], custom_priority: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    if custom_priority:
        for item in custom_priority:
            norm = normalize_lang_code(item)
            if norm in defaults and norm not in out:
                out.append(norm)
    for d in defaults:
        if d not in out:
            out.append(d)
    return out


class DualSourceSyncStage(PipelineStage):
    """Smart Subtitles-First stage: resolves best subtitle pair, syncs to audio, and triggers ASR fallback if needed."""

    @property
    def name(self) -> str:
        return "DUAL_SOURCE_SYNC"

    async def execute(
        self,
        ctx: PipelineContext,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        if not settings.enable_dual_source_pivot or not getattr(ctx, "enable_dual_source_pivot", False):
            return

        ctx.dual_source_active = False
        if on_progress:
            await on_progress(5.0, None, None, "Checking embedded and online EN / AR subtitle sources...", None)

        raw_by_lang: Dict[str, str] = {}
        meta_by_lang: Dict[str, Optional[Dict[str, Any]]] = {}
        streams = getattr(ctx, "subtitle_streams", None) or []
        custom_priority = getattr(ctx, "dual_source_fallback_languages", None) or getattr(
            settings, "dual_source_fallback_languages", None
        )

        manual_map = {
            "en": (getattr(ctx, "dual_source_en_srt", None) or "").strip(),
            "ar": (getattr(ctx, "dual_source_ar_srt", None) or "").strip(),
        }
        movie_hash: Optional[str] = None
        parsed_media: Optional[Dict[str, Any]] = None

        async def _try_embedded(lang: str) -> bool:
            if not streams:
                return False
            stream = None
            selected_idx = getattr(ctx, "selected_subtitle_stream", None)
            if selected_idx is not None:
                explicit = next((s for s in streams if s.index == selected_idx and s.text_based), None)
                if explicit is not None and pick_track_by_language([explicit], lang) is not None:
                    stream = explicit
            if stream is None:
                stream = pick_track_by_language(streams, lang)
            if stream is None:
                return False
            try:
                extracted = await extract_track_as_srt(ctx.video_path, stream)
            except Exception as exc:
                logger.warning(
                    f"Job {ctx.job_id}: failed to extract embedded {lang.upper()} track #{stream.index}: {exc}"
                )
                return False
            if extracted:
                raw_by_lang[lang] = extracted
                meta_by_lang[lang] = {
                    "provider": "embedded",
                    "language": lang,
                    "stream_index": stream.index,
                    "title": stream.title,
                }
                logger.info(
                    f"Job {ctx.job_id}: extracted embedded {lang.upper()} track #{stream.index} "
                    f"({stream.title or stream.language or lang})"
                )
                return True
            return False

        async def _fetch_langs(langs: Tuple[str, ...]) -> None:
            nonlocal movie_hash, parsed_media
            if not langs:
                return
            fetched = await asyncio.to_thread(
                fetch_dual_subtitles,
                Path(ctx.video_path),
                ctx.file_name,
                opensubtitles_api_key=settings.opensubtitles_api_key,
                subdl_api_key=settings.subdl_api_key,
                languages=langs,
            )
            ctx.raise_if_cancelled()
            movie_hash = movie_hash or fetched.get("movie_hash")
            parsed_media = parsed_media or fetched.get("parsed")
            for lang in langs:
                txt = (fetched.get(f"{lang}_srt") or "").strip()
                if txt:
                    raw_by_lang[lang] = txt
                    meta_by_lang[lang] = fetched.get(f"{lang}_meta")

        # 1. Resolve EN and AR first (manual override -> embedded track -> online API)
        for lang in ("en", "ar"):
            if manual_map[lang]:
                raw_by_lang[lang] = manual_map[lang]
                meta_by_lang[lang] = {"provider": "manual", "language": lang}
            else:
                await _try_embedded(lang)

        missing_core = tuple(l for l in ("en", "ar") if not raw_by_lang.get(l))
        if missing_core:
            if on_progress:
                await on_progress(
                    15.0,
                    None,
                    None,
                    f"Searching online subtitles for {' & '.join(l.upper() for l in missing_core)}...",
                    None,
                )
            await _fetch_langs(missing_core)

        # 2. If only one of EN/AR was found (or neither yet), try ordered gender/secondary fallback languages
        if "en" in raw_by_lang and "ar" not in raw_by_lang:
            for fb in _build_ordered_chain(DEFAULT_GENDER_FALLBACK_FOR_EN, custom_priority):
                if await _try_embedded(fb):
                    break
                await _fetch_langs((fb,))
                if fb in raw_by_lang:
                    break
        elif "ar" in raw_by_lang and "en" not in raw_by_lang:
            for sec in _build_ordered_chain(DEFAULT_SECONDARY_FOR_AR, custom_priority):
                if await _try_embedded(sec):
                    break
                await _fetch_langs((sec,))
                if sec in raw_by_lang:
                    break

        pair = select_subtitle_pair(raw_by_lang, fallback_priority=custom_priority)
        pair_mode = pair["pair_mode"]
        primary_lang = pair["primary_lang"]
        anchor_lang = pair["anchor_lang"]

        # Priority #1 safety net: if no EN/AR/fallback pair matched, still prefer any embedded
        # dialogue track present in the container before falling back to Whisper ASR.
        if (pair_mode == "none" or not primary_lang) and streams:
            fallback_track = pick_dialogue_track(
                streams,
                source_language=ctx.source_language,
                audio_streams=ctx.audio_streams,
                selected_audio_stream=ctx.selected_audio_stream,
                selected_subtitle_stream=ctx.selected_subtitle_stream,
            )
            if fallback_track is not None:
                try:
                    extracted_any = await extract_track_as_srt(ctx.video_path, fallback_track)
                except Exception as exc:
                    logger.warning(
                        f"Job {ctx.job_id}: failed to extract embedded dialogue track #{fallback_track.index}: {exc}"
                    )
                    extracted_any = ""
                if extracted_any:
                    track_lang = (
                        language_code(fallback_track.language)
                        or (fallback_track.language or "").strip().lower()
                    )
                    if not track_lang or track_lang in ("und", "unknown", "none", "auto"):
                        track_lang = "en"
                    raw_by_lang[track_lang] = extracted_any
                    meta_by_lang[track_lang] = {
                        "provider": "embedded",
                        "language": track_lang,
                        "stream_index": fallback_track.index,
                        "title": fallback_track.title,
                    }
                    pair_mode = "single_source"
                    primary_lang = track_lang
                    anchor_lang = None
                    logger.info(
                        f"Job {ctx.job_id}: using embedded dialogue track #{fallback_track.index} "
                        f"({track_lang.upper()}) as single_source priority before ASR fallback."
                    )

        if pair_mode == "none" or not primary_lang:
            reason = "No matching EN, AR, or fallback subtitles found; running standard WhisperX / Vision pipeline."
            logger.warning(f"Job {ctx.job_id}: {reason}")
            ctx.warnings.append(reason)
            ctx.dual_source_report = {
                "active": False,
                "pair_mode": "none",
                "source": "api",
                "fallback_reason": reason,
                "movie_hash": movie_hash,
                "parsed": parsed_media,
            }
            if on_progress:
                await on_progress(100.0, None, None, reason, None)
            return

        # For single-source EN/AR without a second language, fall back cleanly only if strict dual required,
        # OR continue as verified single_source subtitle pipeline (unless alignment fails).
        used_langs = [l for l in (primary_lang, anchor_lang) if l]
        providers = {
            str((meta_by_lang.get(l) or {}).get("provider") or "")
            for l in used_langs
            if meta_by_lang.get(l)
        }
        if providers == {"manual"}:
            source_kind = "manual"
        elif providers == {"embedded"}:
            source_kind = "embedded"
        elif len(providers) == 1 and next(iter(providers)) in ("opensubtitles", "subdl"):
            source_kind = "api"
        elif len(providers) > 1:
            source_kind = "mixed"
        else:
            source_kind = "api"

        art_dir = settings.storage_dir / "dual_source" / ctx.job_id
        art_dir.mkdir(parents=True, exist_ok=True)
        reference_audio = Path(ctx.audio_wav_path or ctx.video_path)

        async def _sync_track(
            lang: str,
            pct: float,
            reference_srt_path: Optional[Path] = None,
        ) -> Tuple[str, Dict[str, Any], Path]:
            raw_p = art_dir / f"{lang}.raw.srt"
            out_p = art_dir / f"{lang}.synced.srt"
            raw_p.write_text(raw_by_lang[lang], encoding="utf-8")
            if (meta_by_lang.get(lang) or {}).get("provider") == "embedded":
                out_p.write_text(raw_by_lang[lang], encoding="utf-8")
                return raw_by_lang[lang], {
                    "synced": True,
                    "engine": "embedded",
                    "offset_seconds": 0.0,
                    "scale_factor": None,
                    "warning": None,
                }, out_p
            if on_progress:
                await on_progress(pct, None, None, f"Synchronizing {lang.upper()} subtitles to audio...", None)
            if reference_srt_path is not None:
                res = await asyncio.to_thread(
                    sync_srt_to_audio,
                    reference_audio,
                    raw_p,
                    out_p,
                    reference_srt_path=reference_srt_path,
                )
            else:
                res = await asyncio.to_thread(sync_srt_to_audio, reference_audio, raw_p, out_p)
            return out_p.read_text(encoding="utf-8", errors="replace"), res, out_p

        primary_synced_text, primary_sync, primary_synced_path = await _sync_track(primary_lang, 35.0)
        ctx.raise_if_cancelled()
        anchor_sync: Optional[Dict[str, Any]] = None
        anchor_synced_text = ""
        if anchor_lang:
            ref_srt = primary_synced_path if primary_sync.get("synced") else None
            anchor_synced_text, anchor_sync, _ = await _sync_track(
                anchor_lang,
                65.0,
                reference_srt_path=ref_srt,
            )
            ctx.raise_if_cancelled()

        for sync_res in (primary_sync, anchor_sync):
            if sync_res and sync_res.get("warning"):
                ctx.warnings.append(sync_res["warning"])

        strip_speakers = bool(getattr(ctx, "remove_sdh_speakers", False))
        primary_cues = parse_srt_cues(
            primary_synced_text,
            clean_sdh=True,
            strip_speaker_labels=strip_speakers,
        )
        anchor_cues = (
            parse_srt_cues(
                anchor_synced_text,
                clean_sdh=(pair_mode != "en_ar"),
                strip_speaker_labels=False,
            )
            if anchor_lang
            else []
        )
        if not primary_cues:
            reason = f"Primary {primary_lang.upper()} subtitle contained no valid timed cues; falling back to WhisperX."
            logger.warning(f"Job {ctx.job_id}: {reason}")
            ctx.warnings.append(reason)
            ctx.dual_source_report = {
                "active": False,
                "pair_mode": pair_mode,
                "primary_lang": primary_lang,
                "anchor_lang": anchor_lang,
                "source": source_kind,
                "fallback_reason": reason,
            }
            if on_progress:
                await on_progress(100.0, None, None, reason, None)
            return

        if on_progress:
            await on_progress(85.0, None, None, "Aligning subtitles and verifying coverage...", None)

        aligned, stats = align_dual_cues(primary_cues, anchor_cues, pair_mode=pair_mode, window_sec=1.5)
        quality = evaluate_alignment_quality(
            aligned,
            float(ctx.duration or 0.0),
            stats,
            pair_mode=pair_mode,
            primary_synced=bool(primary_sync.get("synced")),
        )

        # Degrade misaligned anchor references cleanly to single_source instead of poisoning gender
        if quality.get("anchor_unreliable") and anchor_lang:
            warn_msg = (
                f"Anchor {anchor_lang.upper()} alignment ratio was low ({quality['pair_ratio']:.0%}); "
                f"continuing with {primary_lang.upper()} single-source subtitles."
            )
            logger.warning(f"Job {ctx.job_id}: {warn_msg}")
            ctx.warnings.append(warn_msg)
            pair_mode = "single_source"
            anchor_lang = None
            aligned, stats = align_dual_cues(primary_cues, [], pair_mode="single_source")
            quality = evaluate_alignment_quality(
                aligned,
                float(ctx.duration or 0.0),
                stats,
                pair_mode=pair_mode,
                primary_synced=bool(primary_sync.get("synced")),
            )

        smart_fallback = bool(getattr(ctx, "dual_source_smart_asr_fallback", True))
        if quality["needs_full_asr_fallback"] and smart_fallback:
            reason = f"Smart ASR Fallback triggered: {quality['reason']}; running WhisperX transcription."
            logger.warning(f"Job {ctx.job_id}: {reason}")
            ctx.warnings.append(reason)
            ctx.dual_source_active = False
            ctx.dual_source_report = {
                "active": False,
                "source": source_kind,
                "pair_mode": pair_mode,
                "primary_lang": primary_lang,
                "anchor_lang": anchor_lang,
                "quality": quality,
                "alignment": stats,
                "fallback_reason": reason,
            }
            if on_progress:
                await on_progress(100.0, None, None, reason, None)
            return

        dialogue_segs: List[Dict[str, Any]] = []
        onscreen_segs: List[Dict[str, Any]] = []
        for item in aligned:
            row = {
                "start_time": item.start_time,
                "end_time": item.end_time,
                "original_text": item.original_text,
                "segment_type": item.segment_type,
                "confidence": 1.0,
                "reference_text": item.reference_text,
                "dual_source_kind": item.alignment_kind,
            }
            if item.alignment_kind == "AR_ONLY" or item.segment_type == SegmentType.ON_SCREEN.value:
                row["segment_type"] = SegmentType.ON_SCREEN.value
                onscreen_segs.append(row)
            else:
                dialogue_segs.append(row)

        ctx.embedded_dialogue_segments = dialogue_segs
        ctx.onscreen_segments = onscreen_segs
        ctx.detected_language = primary_lang
        ctx.dual_source_primary_lang = primary_lang
        ctx.dual_source_anchor_lang = anchor_lang
        ctx.dual_source_pair_mode = pair_mode
        ctx.dual_source_active = True

        # Preserve legacy report shape (`en_sync` / `ar_sync` aliases + explicit primary/anchor keys)
        en_sync_alias = primary_sync if primary_lang == "en" else (anchor_sync if anchor_lang == "en" else primary_sync)
        ar_sync_alias = anchor_sync if anchor_lang == "ar" else (primary_sync if primary_lang == "ar" else (anchor_sync or primary_sync))

        gap_supplement = bool(quality["needs_gap_asr_supplement"] and smart_fallback)
        skip_onscreen = bool(onscreen_segs) or not bool(getattr(ctx, "detect_onscreen_text", False))
        if gap_supplement:
            ctx.supplement_with_whisper = True
            skipped_stages = ["DIARIZING"]
            if skip_onscreen:
                skipped_stages.append("SCANNING_ONSCREEN")
            gap_msg = (
                f"Smart ASR Gap Supplement enabled: {quality['reason']} — "
                "running Whisper on uncovered spans only."
            )
            logger.info(f"Job {ctx.job_id}: {gap_msg}")
            ctx.warnings.append(gap_msg)
        else:
            # Ensure a stale UI supplement_with_whisper flag cannot re-enable full Whisper / Demucs
            ctx.supplement_with_whisper = False
            skipped_stages = ["VOCAL_ISOLATION", "TRANSCRIBING", "DIARIZING"]
            if skip_onscreen:
                skipped_stages.append("SCANNING_ONSCREEN")

        ctx.dual_source_report = {
            "active": True,
            "source": source_kind,
            "pair_mode": pair_mode,
            "primary_lang": primary_lang,
            "primary_lang_name": language_name(primary_lang, primary_lang),
            "anchor_lang": anchor_lang,
            "anchor_lang_name": language_name(anchor_lang, anchor_lang) if anchor_lang else None,
            "movie_hash": movie_hash,
            "parsed": parsed_media,
            "en_meta": meta_by_lang.get("en"),
            "ar_meta": meta_by_lang.get("ar"),
            "primary_meta": meta_by_lang.get(primary_lang),
            "anchor_meta": meta_by_lang.get(anchor_lang) if anchor_lang else None,
            "en_sync": en_sync_alias,
            "ar_sync": ar_sync_alias,
            "primary_sync": primary_sync,
            "anchor_sync": anchor_sync,
            "alignment": stats,
            "quality": quality,
            "gap_asr_supplement": gap_supplement,
            "artifacts_dir": str(art_dir),
            "skipped_stages": skipped_stages,
        }

        logger.info(
            f"Dual-Source Sync ready for {ctx.job_id} ({source_kind}, mode={pair_mode}, "
            f"primary={primary_lang}, anchor={anchor_lang or 'none'}): "
            f"primary={stats['primary_total']}, paired={stats['paired_count']}, "
            f"AR-only ON_SCREEN={stats['ar_only_count']}, gap_supplement={gap_supplement}"
        )
        if on_progress:
            await on_progress(
                100.0,
                None,
                None,
                f"Aligned {stats['paired_count']} {primary_lang.upper()}"
                + (f"+{anchor_lang.upper()}" if anchor_lang else "")
                + f" cues ({pair_mode})",
                None,
            )
