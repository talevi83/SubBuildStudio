import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional
from app.core.config import settings
from app.core.glossary_profiles import load_profile, merge_glossary, save_profile
from app.pipeline.context import PipelineContext, PipelineCancelled
from app.pipeline.stages.base import PipelineStage, ProgressCallback
from app.providers.translation.base import BaseTranslator, TranslatableSegment
from app.utils.languages import language_name
from app.utils.srt_formatter import calculate_cps

logger = logging.getLogger(__name__)

# Called after each translated batch so the orchestrator can persist it immediately
BatchPersistCallback = Callable[[List[TranslatableSegment]], Awaitable[None]]


def _notes_with_hint(notes: Optional[str], hint: Optional[str], rows: List[TranslatableSegment]) -> str:
    """The user's show notes, plus a note that applies only to the rows being re-translated now."""
    base = (notes or "").strip()
    if not hint or not hint.strip() or not rows:
        return base
    first, last = min(r.index for r in rows), max(r.index for r in rows)
    scoped = (
        f"NOTE FOR THE SEGMENTS YOU ARE TRANSLATING NOW (#{first}-#{last}); it overrides everything else, "
        f"including the glossary: {hint.strip()}"
    )
    return f"{base}\n\n{scoped}" if base else scoped


_DEFAULT_SPEAKER_LABEL_RE = re.compile(r"^Speaker \d+$", re.IGNORECASE)

_LISTENER_FIX_NOTE = (
    "GENDER CHECK - the rows in `target_batch` were already translated, but their 2nd-person forms did not "
    "match the listener. Their `addressee` is verified: translate each row again so that every 2nd-person "
    "pronoun, verb, imperative and adjective agrees with `addressee` (masculine singular, feminine singular "
    "or plural). Keep the meaning, register and length otherwise."
)


def _format_speaker_info(speaker_id: Optional[str], speakers: Optional[Dict[str, Any]]) -> Optional[str]:
    from app.pipeline.speaker_match import format_speaker_identity

    return format_speaker_identity(speaker_id, speakers)


def _sync_glossary_to_speakers(glossary: Optional[Dict[str, Any]], speakers: Optional[Dict[str, Any]]) -> None:
    if not glossary or not isinstance(speakers, dict) or not speakers:
        return
    for ch in glossary.get("characters") or []:
        if not isinstance(ch, dict):
            continue
        sid = str(ch.get("speaker_id") or "").strip()
        if not sid or sid not in speakers or not isinstance(speakers[sid], dict):
            continue
        cur_label = str(speakers[sid].get("label") or "").strip()
        new_label = str(ch.get("target_name") or ch.get("name") or "").strip()
        if new_label and (not cur_label or _DEFAULT_SPEAKER_LABEL_RE.match(cur_label)):
            speakers[sid]["label"] = new_label
        ch_gender = str(ch.get("gender") or "").strip().lower()
        cur_gender = str(speakers[sid].get("gender") or "unknown").strip().lower()
        from_voice = speakers[sid].get("gender_source") == "voice"
        if ch_gender in ("male", "female") and (cur_gender in ("", "unknown") or from_voice):
            speakers[sid]["gender"] = ch_gender
            speakers[sid].pop("gender_source", None)  # the transcript names the character now


class TranslateStage(PipelineStage):
    """Pipeline stage executing contextual batch subtitle translation via Gemini."""

    def __init__(
        self,
        translator: Optional[BaseTranslator] = None,
        on_batch_persist: Optional[BatchPersistCallback] = None,
    ):
        self._translator = translator
        self._on_batch_persist = on_batch_persist

    @property
    def name(self) -> str:
        return "TRANSLATING"

    def _make_translator(self, ctx: PipelineContext) -> BaseTranslator:
        from app.core.provider_catalog import infer_provider_from_model
        from app.core.user_settings import load_settings
        from app.providers.translation.gemini_provider import GeminiTranslator

        app_settings = load_settings()
        # Thinking and Flex are configured for the Settings model; a per-job override uses defaults
        target_model = ctx.gemini_model or app_settings.translation.model
        matches_settings_model = (target_model == app_settings.translation.model)
        inferred = None if matches_settings_model else (infer_provider_from_model(target_model) if ctx.gemini_model else None)
        target_provider = inferred or app_settings.translation.provider
        same_model = (target_provider == app_settings.translation.provider and matches_settings_model)
        return GeminiTranslator(
            api_key=ctx.gemini_api_key_override,
            model_name=target_model,
            thinking=ctx.gemini_thinking or (app_settings.translation.thinking if same_model else "default"),
            glossary_model=app_settings.glossary.model,
            glossary_thinking=app_settings.glossary.thinking,
            usage=ctx.usage,
            flex=app_settings.translation.flex if same_model else False,
            glossary_flex=app_settings.glossary.flex,
            flex_state=ctx.flex_state,
            convert_to_metric=getattr(ctx, "convert_to_metric", False) or app_settings.convert_to_metric,
            remove_sdh_speakers=getattr(ctx, "remove_sdh_speakers", False) or getattr(app_settings, "remove_sdh_speakers", False),
            provider=target_provider,
            key_id=app_settings.translation.key_id if same_model else None,
            glossary_provider=app_settings.glossary.provider,
            glossary_key_id=app_settings.glossary.key_id,
        )

    async def _fix_listener_mismatches(
        self,
        ctx: PipelineContext,
        translator: BaseTranslator,
        translated: List[TranslatableSegment],
        all_segments: List[TranslatableSegment],
        source_lang: str,
        target_lang: str,
        notes: str,
        store_batch: Callable[[List[TranslatableSegment]], Awaitable[None]],
        on_progress: Optional[ProgressCallback],
    ) -> int:
        """Translate once more the lines whose 2nd-person forms contradict a confident listener.

        A retry only replaces the first translation when it agrees with the listener; otherwise the
        first one stays and the quality check reports it. Returns how many lines were corrected.
        """
        from dataclasses import replace

        from app.pipeline.addressee import HIGH, AddresseeRecord
        from app.pipeline.qa import listener_form_mismatch

        def _expected_form(seg: TranslatableSegment) -> Optional[str]:
            record = AddresseeRecord.from_dict(seg.addressee)
            if record is None or record.confidence < HIGH or (seg.segment_type or "DIALOGUE") == "ON_SCREEN":
                return None
            return record.form

        wrong = [
            seg for seg in translated
            if seg.translated_text and not seg.translation_failed
            and listener_form_mismatch(_expected_form(seg), seg.translated_text, seg.addressee_form)
        ]
        if not wrong:
            return 0
        limit = max(0, int(settings.addressee_autofix_max_rows))
        if len(wrong) > limit:
            logger.info(f"{len(wrong)} line(s) disagree with their listener; re-checking the first {limit}")
            wrong = wrong[:limit]
        if not wrong:
            return 0
        if on_progress:
            await on_progress(99.0, None, None, f"Re-checking 2nd-person gender on {len(wrong)} line(s)...", None)

        # Work on copies: a failed retry must never cost a line the translation it already has
        retries = [replace(seg, translated_text=None, translation_failed=False, addressee_form=None) for seg in wrong]
        try:
            await ctx.run_cancellable(translator.translate_segments(
                segments=retries,
                source_lang=source_lang,
                target_lang=target_lang,
                context_notes=f"{notes}\n\n{_LISTENER_FIX_NOTE}".strip(),
                apply_rtl=False,
                on_batch_complete=None,
                glossary=ctx.glossary,
                context_segments=all_segments,
            ))
        except PipelineCancelled:
            raise
        except Exception as exc:
            logger.warning(f"2nd-person gender re-check skipped: {exc}")
            return 0

        fixed: List[TranslatableSegment] = []
        for seg, retry in zip(wrong, retries):
            if not retry.translated_text or retry.translation_failed:
                continue
            if listener_form_mismatch(_expected_form(seg), retry.translated_text, retry.addressee_form):
                continue
            seg.translated_text = retry.translated_text
            seg.addressee_form = retry.addressee_form
            fixed.append(seg)
        if fixed:
            await store_batch(fixed)
            logger.info(f"Corrected the 2nd-person gender of {len(fixed)} line(s) to match their listener")
        return len(fixed)

    async def execute(
        self,
        ctx: PipelineContext,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        ctx.translated_segments = ctx.refined_segments
        if not ctx.enable_translation:
            if on_progress:
                await on_progress(100.0, None, None, "Translation disabled; skipping", None)
            return

        if not ctx.refined_segments:
            if on_progress:
                await on_progress(100.0, None, None, "No segments to translate", None)
            return

        translator = self._translator
        if translator is None:
            try:
                translator = self._make_translator(ctx)
            except Exception as e:
                # Keep the transcription; tell the user instead of silently skipping.
                ctx.translation_warning = (
                    f"Translation skipped: {e} Add the key and use 'Retranslate' — "
                    f"the transcription is saved."
                )
                logger.warning(ctx.translation_warning)
                if on_progress:
                    await on_progress(100.0, None, None, ctx.translation_warning, None)
                return

        speakers_map = getattr(ctx, "speakers", None) or {}
        by_index: Dict[int, dict] = {s["index"]: s for s in ctx.refined_segments}
        all_segments: List[TranslatableSegment] = [
            TranslatableSegment(
                index=s["index"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                original_text=s["original_text"],
                translated_text=s.get("translated_text"),
                translation_failed=bool(s.get("translation_failed")),
                segment_type=s.get("segment_type") or "DIALOGUE",
                speaker=s.get("speaker"),
                speaker_info=_format_speaker_info(s.get("speaker"), speakers_map),
            )
            for s in ctx.refined_segments
        ]
        if ctx.only_indices:
            wanted = set(ctx.only_indices)
            to_translate = [s for s in all_segments if s.index in wanted]
            for s in to_translate:
                s.translated_text = None
                s.translation_failed = False
        elif ctx.only_failed:
            to_translate = [s for s in all_segments if s.translation_failed or not s.translated_text]
        else:
            to_translate = all_segments
            for s in to_translate:
                s.translated_text = None
                s.translation_failed = False

        if not to_translate:
            if on_progress:
                await on_progress(100.0, None, None, "Nothing left to translate", None)
            return

        source_lang = language_name(ctx.effective_source_language, default="the original language")
        target_lang = ctx.target_language

        total_to_translate = max(1, len(to_translate))
        completed_so_far = 0

        async def _store_batch(batch: List[TranslatableSegment]) -> None:
            for b in batch:
                seg = by_index.get(b.index)
                if seg is None:
                    continue
                seg["translated_text"] = b.translated_text
                seg["translation_failed"] = b.translation_failed
                seg["cps"] = calculate_cps(
                    b.translated_text or seg["original_text"],
                    max(0.1, seg["end_time"] - seg["start_time"]),
                )
                if b.addressee is not None:
                    # Which 2nd-person form the translator says it wrote, kept for the quality check
                    if b.addressee_form:
                        b.addressee["used"] = b.addressee_form
                    else:
                        b.addressee.pop("used", None)
                seg["addressee"] = b.addressee
            if self._on_batch_persist:
                await self._on_batch_persist(batch)

        async def _on_batch_complete(batch: List[TranslatableSegment], _stream_pct: float) -> None:
            nonlocal completed_so_far
            completed_so_far += len(batch)
            overall_pct = min(100.0, round((completed_so_far / total_to_translate) * 100.0, 2))
            await _store_batch(batch)
            if on_progress:
                await on_progress(
                    10.0 + overall_pct * 0.9,
                    None,
                    None,
                    f"Translated batch up to #{batch[-1].index}",
                    batch,
                )
            # Raising here stops the translator between batches
            if ctx.is_cancelled:
                raise PipelineCancelled()

        from app.providers.translation.gemini_provider import TranslationAborted

        try:
            known = load_profile(ctx.glossary_profile)
            if settings.gemini_glossary_pass and (ctx.glossary is None or ctx.rebuild_glossary):
                if on_progress:
                    await on_progress(2.0, None, None, "Analysing characters and terminology...", None)
                extra = {"known_glossary": known} if known else {}
                ctx.glossary = merge_glossary(
                    known,
                    await ctx.run_cancellable(translator.build_glossary(
                        all_segments, source_lang, target_lang, ctx.context_notes or "", **extra
                    )),
                )
                if ctx.glossary_profile and ctx.glossary:
                    # The series profile grows with every episode: new characters are added, the rest stays
                    try:
                        save_profile(ctx.glossary_profile, ctx.glossary)
                    except (OSError, ValueError) as e:
                        logger.warning(f"Could not update glossary profile '{ctx.glossary_profile}': {e}")
                glossary_error = getattr(translator, "last_glossary_error", None)
                if ctx.glossary is None and glossary_error:
                    ctx.warnings.append(f"Glossary pass skipped — {glossary_error[:300]}")
                if on_progress:
                    await on_progress(6.0, None, None, "Glossary analysis complete", None)
                ctx.raise_if_cancelled()
            elif ctx.glossary is None and known:
                ctx.glossary = known

            _sync_glossary_to_speakers(ctx.glossary, getattr(ctx, "speakers", None))
            if getattr(ctx, "speakers", None) and ctx.refined_segments:
                try:
                    from app.pipeline.diarization import prune_unused_speakers
                    from app.pipeline.speaker_match import match_speakers_to_characters

                    await match_speakers_to_characters(
                        ctx.refined_segments,
                        ctx.speakers,
                        ctx.glossary,
                        target_language=target_lang,
                        use_ai=(self._translator is None),
                        overwrite_manual=False,
                        auto_merge=True,
                        api_key_override=ctx.gemini_api_key_override,
                        usage=ctx.usage,
                        flex_state=ctx.flex_state,
                    )
                    ctx.speakers = prune_unused_speakers(ctx.speakers, ctx.refined_segments)
                    by_seg_idx = {int(s["index"]): s.get("speaker") for s in ctx.refined_segments if "index" in s}
                    for seg_obj in all_segments:
                        if seg_obj.index in by_seg_idx:
                            seg_obj.speaker = by_seg_idx[seg_obj.index]
                except Exception as exc:
                    logger.debug(f"Speaker-character matching skipped: {exc}")

            from app.pipeline.addressee import (
                AddresseeRecord,
                merge_local_guess,
                needs_listener_gender,
                resolve_addressees,
                unresolved_record,
            )
            from app.pipeline.gender_pretag import is_gendered_target_language

            gendered_target = is_gendered_target_language(target_lang)
            speakers_now = getattr(ctx, "speakers", None)
            records: Dict[int, AddresseeRecord] = {}
            try:
                records = resolve_addressees(
                    ctx.refined_segments,
                    speakers_now,
                    (ctx.glossary or {}).get("characters"),
                )
            except Exception as exc:
                logger.debug(f"Addressee inference skipped: {exc}")

            for seg_obj in all_segments:
                seg_obj.speaker_info = _format_speaker_info(seg_obj.speaker, speakers_now)

            effective_notes = _notes_with_hint(ctx.context_notes, ctx.translation_hint, to_translate)
            try:
                from app.core.user_settings import load_settings

                if gendered_target and load_settings().cost_saver_mode:
                    from app.pipeline.gender_pretag import parse_gender_tag, pretag_gender_hints

                    async def _on_local_batch() -> None:
                        if on_progress:
                            await on_progress(8.0, None, None, "Pre-tagging speaker gender with local AI...", None)

                    # A second opinion on the lines the rules above left open, read in their scene
                    local_hints = await pretag_gender_hints(
                        to_translate,
                        target_lang=target_lang,
                        glossary=ctx.glossary,
                        usage=ctx.usage,
                        cloud_model=getattr(translator, "model_name", None),
                        on_batch=_on_local_batch,
                        speakers=speakers_now,
                        addressees=records,
                        context_segments=all_segments,
                    )
                    by_seg_obj = {seg_obj.index: seg_obj for seg_obj in all_segments}
                    for seg_idx, tag in local_hints.items():
                        seg_obj = by_seg_obj.get(seg_idx)
                        if seg_obj is None:
                            continue
                        sp_g, to_g = parse_gender_tag(tag)
                        if to_g:
                            merge_local_guess(records, seg_idx, to_g)
                        if sp_g and not seg_obj.speaker_info:
                            seg_obj.speaker_info = sp_g
            except Exception as e:
                logger.debug(f"Local gender pre-tagging skipped: {e}")

            # Confident listeners go to the translator as facts, likely ones as guesses it may overrule;
            # lines that need a listener nobody could name are marked so the editor can show them.
            translating = {s.index for s in to_translate}
            for seg_obj in all_segments:
                record = records.get(seg_obj.index)
                if (
                    record is None
                    and gendered_target
                    and seg_obj.index in translating
                    and (seg_obj.segment_type or "DIALOGUE") != "ON_SCREEN"
                    and needs_listener_gender(seg_obj.original_text)
                ):
                    record = unresolved_record()
                shown = record.display(speakers_now) if record is not None else None
                seg_obj.addressee_info = shown if record is not None and record.tier == "high" else None
                seg_obj.addressee_guess = shown if record is not None and record.tier == "medium" else None
                stored = by_index.get(seg_obj.index)
                if seg_obj.index in translating:
                    seg_obj.addressee = record.to_dict() if record is not None else None
                elif stored is not None:
                    seg_obj.addressee = stored.get("addressee")  # not re-translated now: its record stands

            if on_progress:
                await on_progress(10.0, None, None, f"Translating {len(to_translate)} segments to {target_lang}...", None)

            spoken_to_translate = [s for s in to_translate if (s.segment_type or "DIALOGUE") != "ON_SCREEN"]
            onscreen_to_translate = [s for s in to_translate if (s.segment_type or "DIALOGUE") == "ON_SCREEN"]
            if spoken_to_translate and onscreen_to_translate:
                # Translate continuous spoken dialogue first without interleaved ON_SCREEN titles
                # interrupting multi-segment sentences, then translate ON_SCREEN titles with full context.
                spoken_context = [s for s in all_segments if (s.segment_type or "DIALOGUE") != "ON_SCREEN"]
                await ctx.run_cancellable(translator.translate_segments(
                    segments=spoken_to_translate,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context_notes=effective_notes,
                    apply_rtl=False,
                    on_batch_complete=_on_batch_complete,
                    glossary=ctx.glossary,
                    context_segments=spoken_context,
                ))
                await ctx.run_cancellable(translator.translate_segments(
                    segments=onscreen_to_translate,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context_notes=effective_notes,
                    apply_rtl=False,
                    on_batch_complete=_on_batch_complete,
                    glossary=ctx.glossary,
                    context_segments=all_segments,
                ))
            else:
                await ctx.run_cancellable(translator.translate_segments(
                    segments=to_translate,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context_notes=effective_notes,
                    apply_rtl=False,  # stored clean; RTL marks are added at export time
                    on_batch_complete=_on_batch_complete,
                    glossary=ctx.glossary,
                    context_segments=all_segments,
                ))

            if gendered_target and settings.addressee_autofix:
                await self._fix_listener_mismatches(
                    ctx, translator, to_translate, all_segments, source_lang, target_lang,
                    effective_notes, _store_batch, on_progress,
                )
        except TranslationAborted as e:
            for b in to_translate:
                if not b.translated_text:
                    b.translation_failed = True
                    by_index[b.index]["translation_failed"] = True
            if self._on_batch_persist:
                await self._on_batch_persist([b for b in to_translate if not b.translated_text])
            ctx.translation_warning = f"Translation stopped: {e}"
            logger.error(ctx.translation_warning)
            return

        from app.utils.speaker_split import split_dual_speaker_segments

        ctx.refined_segments = split_dual_speaker_segments(
            ctx.refined_segments,
            max_chars_per_line=settings.max_chars_per_line,
            min_gap=settings.min_gap_seconds,
        )
        ctx.translated_segments = ctx.refined_segments

        failed = sum(1 for s in ctx.refined_segments if s.get("translation_failed"))
        if failed:
            ctx.translation_warning = (
                f"{failed} segment(s) could not be translated. Use 'Retranslate failed' to retry them."
            )

        if on_progress:
            await on_progress(100.0, None, None, f"Translation finished ({len(to_translate) - failed} ok, {failed} failed)", None)
