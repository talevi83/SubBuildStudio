import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional
from app.core.config import settings
from app.core.glossary_profiles import (
    combine_context_notes,
    load_profile,
    merge_glossary,
    parse_user_glossary,
    save_profile,
)
from app.pipeline.context import PipelineContext, PipelineCancelled
from app.pipeline.stages.base import PipelineStage, ProgressCallback
from app.providers.translation.base import BaseTranslator, TranslatableSegment
from app.utils.languages import language_name
from app.utils.srt_formatter import calculate_cps

logger = logging.getLogger(__name__)

# Called after each translated batch so the orchestrator can persist it immediately
BatchPersistCallback = Callable[[List[TranslatableSegment]], Awaitable[None]]


def _scoped_hint(hint: Optional[str], rows: List[TranslatableSegment]) -> Optional[str]:
    """Build a per-batch user-message override note without mutating the cached system prompt."""
    if not hint or not hint.strip() or not rows:
        return None
    first, last = min(r.index for r in rows), max(r.index for r in rows)
    return (
        f"NOTE FOR THE SEGMENTS YOU ARE TRANSLATING NOW (#{first}-#{last}); it overrides everything else, "
        f"including the glossary: {hint.strip()}"
    )


def _notes_with_hint(notes: Optional[str], hint: Optional[str], rows: List[TranslatableSegment]) -> str:
    """The user's show notes, plus a note that applies only to the rows being re-translated now."""
    base = (notes or "").strip()
    scoped = _scoped_hint(hint, rows)
    if not scoped:
        return base
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
        profile = speakers[sid]
        # Do not let a coarse glossary speaker_id overwrite a high/medium match to a different character
        matched_char = str(profile.get("character_name") or "").strip()
        matched_conf = str(profile.get("confidence") or "").strip().lower()
        ch_name = str(ch.get("name") or "").strip()
        if matched_char and matched_conf in ("high", "medium") and ch_name and matched_char.lower() != ch_name.lower():
            continue
        cur_label = str(profile.get("label") or "").strip()
        new_label = str(ch.get("target_name") or ch.get("name") or "").strip()
        if new_label and (not cur_label or _DEFAULT_SPEAKER_LABEL_RE.match(cur_label)):
            profile["label"] = new_label
        ch_gender = str(ch.get("gender") or "").strip().lower()
        cur_gender = str(profile.get("gender") or "unknown").strip().lower()
        from_voice = profile.get("gender_source") == "voice"
        if ch_gender in ("male", "female") and (cur_gender in ("", "unknown") or from_voice):
            profile["gender"] = ch_gender
            profile.pop("gender_source", None)  # the transcript names the character now


def _apply_diarization_guard(ctx: Any) -> set:
    """Assess the speaker clustering and quarantine speaker ids that cannot be trusted.

    Untrusted ids (a cluster pooling several voices, or every id when the clustering collapsed)
    lose any character/gender link, so they can no longer push one character's gender onto
    another's lines. Returns the set of untrusted speaker ids.
    """
    from app.pipeline.diarization import assess_diarization

    speakers = getattr(ctx, "speakers", None)
    if not isinstance(speakers, dict) or not speakers:
        return set()
    quality = assess_diarization(speakers, getattr(ctx, "diarization_timeline", None) or (), (ctx.glossary or {}).get("characters"))
    ctx.diarization_quality = quality
    untrusted = set(quality["untrusted"])
    if not untrusted:
        return untrusted
    _forget_speaker_identities(ctx, untrusted)
    if not quality["reliable"]:
        msg = (
            "Speaker detection looks unreliable (" + "; ".join(quality["reasons"]) + "): speaker and listener "
            "genders come only from each line's own wording (names, titles, 'everyone')."
        )
        logger.warning(f"{ctx.job_id}: {msg}")
        ctx.warnings.append(msg)
    else:
        logger.info(f"{ctx.job_id}: {'; '.join(quality['reasons'])} — their ids are not used for gender inference")
    return untrusted


def _forget_speaker_identities(ctx: Any, untrusted: set) -> None:
    """Drop character names and inferred genders from untrusted speakers (manual edits stay)."""
    for ch in (ctx.glossary or {}).get("characters") or []:
        if isinstance(ch, dict) and str(ch.get("speaker_id") or "") in untrusted:
            ch.pop("speaker_id", None)
    for sid in untrusted:
        prof = (getattr(ctx, "speakers", None) or {}).get(sid)
        if not isinstance(prof, dict) or prof.get("locked") or prof.get("manual"):
            continue
        prof.pop("character_name", None)
        prof.pop("gender_source", None)
        prof["gender"] = "unknown"
        prof["confidence"] = "low"


_SELF_MASC_RE = re.compile(
    r"\b(?:i\s+am|i['’]m|as\s+(?:a|an|the|your))\s+"
    r"(?:(?:good|old|young|poor|free|honest|loyal|simple|humble)\s+)?"
    r"(?:man|father|husband|king|priest|son|brother|boy|lord|gentleman|knight|monk|parson)\b",
    re.IGNORECASE,
)
_SELF_FEM_RE = re.compile(
    r"\b(?:i\s+am|i['’]m|as\s+(?:a|an|the|your))\s+"
    r"(?:(?:good|old|young|poor|free|honest|loyal|simple|humble)\s+)?"
    r"(?:woman|mother|wife|queen|lady|daughter|sister|girl|widow|maiden|nun)\b",
    re.IGNORECASE,
)


def _infer_contextual_speaker_genders(
    segments: List[Dict[str, Any]],
    speakers: Optional[Dict[str, Any]],
    glossary: Optional[Dict[str, Any]] = None,
) -> int:
    """Infer speaker genders from dialogue context (1st-person nouns, vocatives, and honorific replies)
    and prioritize contextual evidence over acoustic pitch (`gender_source == 'voice'`)."""
    if not segments or not isinstance(speakers, dict) or not speakers:
        return 0
    from collections import defaultdict
    from app.pipeline.addressee import _FEM_STRONG_RE, _MASC_STRONG_RE, _sentences
    from app.pipeline.speaker_match import _clean_text, _extract_character_aliases, _match_vocative_character

    dialogue = sorted(
        (s for s in segments if (s.get("segment_type") or "DIALOGUE") != "ON_SCREEN"),
        key=lambda s: (float(s.get("start_time", 0.0)), int(s.get("index", 0))),
    )
    if not dialogue:
        return 0

    chars = (glossary or {}).get("characters") or []
    alias_to_char = []
    char_by_idx: Dict[int, Dict[str, Any]] = {}
    for idx, ch, aliases in _extract_character_aliases(chars):
        char_by_idx[idx] = ch
        alias_to_char.extend((al, idx) for al in aliases)
    alias_to_char.sort(key=lambda item: len(item[0]), reverse=True)

    evidence: Dict[str, Dict[str, float]] = defaultdict(lambda: {"male": 0.0, "female": 0.0})
    for i, seg in enumerate(dialogue):
        sp = str(seg.get("speaker") or "").strip()
        text = _clean_text(seg.get("original_text"))
        if not text:
            continue
        if sp and sp in speakers:
            if _SELF_MASC_RE.search(text):
                evidence[sp]["male"] += 2.5
            if _SELF_FEM_RE.search(text):
                evidence[sp]["female"] += 2.5

        sents = _sentences(text) or [text]
        masc_h = any(_MASC_STRONG_RE.search(s) for s in sents)
        fem_h = any(_FEM_STRONG_RE.search(s) for s in sents)
        addr_gender: Optional[str] = "male" if (masc_h and not fem_h) else ("female" if (fem_h and not masc_h) else None)
        if addr_gender is None and alias_to_char:
            voc = _match_vocative_character(text, alias_to_char, char_by_idx)
            if voc is not None:
                vg = str(voc.get("gender") or "").strip().lower()
                if vg in ("male", "female"):
                    addr_gender = vg

        if addr_gender in ("male", "female"):
            # Immediate reply after a titled/named address ("My lord, wait!" -> SPEAKER_01 replies)
            if i + 1 < len(dialogue):
                nxt = dialogue[i + 1]
                nxt_sp = str(nxt.get("speaker") or "").strip()
                gap = float(nxt.get("start_time", 0.0)) - float(seg.get("end_time", 0.0))
                if nxt_sp and nxt_sp != sp and nxt_sp in speakers and 0.0 <= gap <= 3.5:
                    evidence[nxt_sp][addr_gender] += 2.0
            # Speaker right before an acknowledging honorific ("Yes, my lord." / "Thank you, father.")
            if i - 1 >= 0:
                prv = dialogue[i - 1]
                prv_sp = str(prv.get("speaker") or "").strip()
                gap = float(seg.get("start_time", 0.0)) - float(prv.get("end_time", 0.0))
                if prv_sp and prv_sp != sp and prv_sp in speakers and 0.0 <= gap <= 2.5:
                    evidence[prv_sp][addr_gender] += 1.5

    updated = 0
    for sid, counts in evidence.items():
        profile = speakers.get(sid)
        if not isinstance(profile, dict):
            continue
        cur_gender = str(profile.get("gender") or "unknown").strip().lower()
        from_voice = profile.get("gender_source") == "voice"
        if cur_gender in ("male", "female") and not from_voice:
            continue
        m_score, f_score = counts["male"], counts["female"]
        if m_score >= 1.5 and m_score >= 2.0 * max(0.1, f_score):
            profile["gender"] = "male"
            profile.pop("gender_source", None)
            profile["context_gender_score"] = round(m_score, 2)
            updated += 1
        elif f_score >= 1.5 and f_score >= 2.0 * max(0.1, m_score):
            profile["gender"] = "female"
            profile.pop("gender_source", None)
            profile["context_gender_score"] = round(f_score, 2)
            updated += 1
    if updated:
        logger.info(f"Contextual dialogue gender resolved for {updated} speaker(s), overriding acoustic pitch")
    return updated


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
        from app.core.user_settings import load_settings, preferred_thinking
        from app.providers.translation.gemini_provider import GeminiTranslator

        app_settings = load_settings()
        # Thinking and Flex are configured per stage in Settings; a per-job model override uses
        # task-appropriate low/minimal thinking for batch translation rather than expensive medium/default.
        target_model = ctx.gemini_model or app_settings.translation.model
        matches_settings_model = (target_model == app_settings.translation.model)
        inferred = None if matches_settings_model else (infer_provider_from_model(target_model) if ctx.gemini_model else None)
        target_provider = inferred or app_settings.translation.provider
        same_model = (target_provider == app_settings.translation.provider and matches_settings_model)
        fallback_thinking = preferred_thinking(target_provider, target_model, "translation")
        return GeminiTranslator(
            api_key=ctx.gemini_api_key_override,
            model_name=target_model,
            thinking=ctx.gemini_thinking or (app_settings.translation.thinking if same_model else fallback_thinking),
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
            # Keep system_prompt + glossary 100% static across cloud batches for prefix caching;
            # only prune per-batch on small local Ollama models with tight context windows.
            prune_glossary=bool(app_settings.cost_saver_mode and target_provider == "ollama"),
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

        speakers_lookup = getattr(ctx, "speakers", None) or {}

        def _expected_form(seg: TranslatableSegment) -> Optional[str]:
            record = AddresseeRecord.from_dict(seg.addressee)
            if record is None or record.confidence < HIGH or (seg.segment_type or "DIALOGUE") == "ON_SCREEN":
                return None
            # Never force-rewrite a translation when the listener's gender came only from a medium/uncertain voice guess
            if (
                record.target
                and record.source not in ("manual", "vocative", "honorific", "plural")
                and isinstance(speakers_lookup.get(record.target), dict)
                and speakers_lookup[record.target].get("gender_source") == "voice"
                and speakers_lookup[record.target].get("voice_confidence") != "high"
            ):
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
        import inspect

        supports_batch_instr = "batch_instruction" in inspect.signature(translator.translate_segments).parameters
        extra_fix_kw = (
            {"batch_instruction": _LISTENER_FIX_NOTE}
            if supports_batch_instr
            else {}
        )
        fix_notes = f"{notes}\n\n{_LISTENER_FIX_NOTE}".strip()
        try:
            await ctx.run_cancellable(translator.translate_segments(
                segments=retries,
                source_lang=source_lang,
                target_lang=target_lang,
                context_notes=fix_notes,
                apply_rtl=False,
                on_batch_complete=None,
                glossary=ctx.glossary,
                context_segments=all_segments,
                **extra_fix_kw,
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

        combined_context_notes = combine_context_notes(ctx.context_notes, getattr(ctx, "initial_prompt", None))
        user_glossary = parse_user_glossary(combined_context_notes)

        try:
            known = merge_glossary(user_glossary, load_profile(ctx.glossary_profile))
            if settings.gemini_glossary_pass and (ctx.glossary is None or ctx.rebuild_glossary):
                if on_progress:
                    await on_progress(2.0, None, None, "Analysing characters and terminology...", None)
                extra = {"known_glossary": known} if known else {}
                ctx.glossary = merge_glossary(
                    known,
                    await ctx.run_cancellable(translator.build_glossary(
                        all_segments, source_lang, target_lang, combined_context_notes, **extra
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
            elif ctx.glossary is not None and user_glossary:
                ctx.glossary = merge_glossary(user_glossary, ctx.glossary)

            untrusted_speakers = _apply_diarization_guard(ctx)
            diarization_reliable = bool((ctx.diarization_quality or {}).get("reliable", True))
            _sync_glossary_to_speakers(ctx.glossary, getattr(ctx, "speakers", None))
            if getattr(ctx, "speakers", None) and ctx.refined_segments and diarization_reliable:
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
                _sync_glossary_to_speakers(ctx.glossary, getattr(ctx, "speakers", None))
                _infer_contextual_speaker_genders(ctx.refined_segments, getattr(ctx, "speakers", None), ctx.glossary)
                if untrusted_speakers:
                    # Matching and self-reference inference must not re-attach an identity to them
                    _forget_speaker_identities(ctx, untrusted_speakers)

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
                # An untrusted id is treated as "unknown speaker": turn-taking, two-person talks and
                # carried listeners all assume the id names one person
                addressee_segments = [
                    dict(s, speaker=None) if isinstance(s, dict) and s.get("speaker") in untrusted_speakers else s
                    for s in ctx.refined_segments
                ]
                records = resolve_addressees(
                    addressee_segments,
                    speakers_now,
                    (ctx.glossary or {}).get("characters"),
                )
            except Exception as exc:
                logger.debug(f"Addressee inference skipped: {exc}")

            for seg_obj in all_segments:
                trusted_sp = None if seg_obj.speaker in untrusted_speakers else seg_obj.speaker
                seg_obj.speaker_info = _format_speaker_info(trusted_sp, speakers_now)

            import inspect

            supports_batch_instr = "batch_instruction" in inspect.signature(translator.translate_segments).parameters
            scoped_hint = _scoped_hint(ctx.translation_hint, to_translate)
            static_notes = (combined_context_notes or "").strip() if supports_batch_instr else _notes_with_hint(combined_context_notes, ctx.translation_hint, to_translate)
            hint_kw = {"batch_instruction": scoped_hint} if (supports_batch_instr and scoped_hint) else {}
            effective_notes = static_notes
            try:
                from app.core.user_settings import load_settings

                if gendered_target and load_settings().cost_saver_mode and diarization_reliable:
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
                        speakers={k: v for k, v in (speakers_now or {}).items() if k not in untrusted_speakers},
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
                    **hint_kw,
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
                    **hint_kw,
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
                    **hint_kw,
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

        from app.pipeline.segmentation import enforce_min_duration, merge_subsecond_cues
        from app.utils.speaker_split import split_dual_speaker_segments

        ctx.refined_segments = split_dual_speaker_segments(
            ctx.refined_segments,
            max_chars_per_line=settings.max_chars_per_line,
            min_gap=settings.min_gap_seconds,
        )
        if not ctx.only_indices and not ctx.only_failed:
            ctx.refined_segments = merge_subsecond_cues(
                ctx.refined_segments,
                min_duration=settings.min_subtitle_duration,
                max_gap=0.35,
                max_chars_per_line=settings.max_chars_per_line,
                max_duration=settings.max_subtitle_duration,
            )
            ctx.refined_segments = enforce_min_duration(
                ctx.refined_segments,
                min_duration=settings.min_subtitle_duration,
                buffer_sec=0.05,
                max_chars_per_line=settings.max_chars_per_line,
            )
        ctx.translated_segments = ctx.refined_segments

        failed = sum(1 for s in ctx.refined_segments if s.get("translation_failed"))
        if failed:
            ctx.translation_warning = (
                f"{failed} segment(s) could not be translated. Use 'Retranslate failed' to retry them."
            )

        if on_progress:
            await on_progress(100.0, None, None, f"Translation finished ({len(to_translate) - failed} ok, {failed} failed)", None)
