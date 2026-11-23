import asyncio
import json
import re
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.model_catalog import UsageTracker, default_temperature, thinking_config
from app.providers import gemini_calls
from app.providers.translation.base import (
    BaseTranslator,
    TranslatableSegment,
    BatchProgressCallback,
)
from app.utils.languages import is_rtl, language_code
from app.utils.rtl_fixer import apply_rtl_fix, strip_directional_marks

logger = logging.getLogger(__name__)


def is_same_language(source_lang: str, target_lang: str) -> bool:
    """True when source and target resolve to the same language code (same-language proofreading mode)."""
    src = language_code(source_lang)
    tgt = language_code(target_lang)
    return bool(src and tgt and src == tgt)

# Keep the glossary request comfortably inside the model's context window
_GLOSSARY_MAX_CHARS = 150_000


class TranslationAborted(Exception):
    """A non-retryable error (e.g. invalid API key) — stop translating remaining batches."""


# =====================================================================
# Pydantic Schemas for Gemini Structured Output
# =====================================================================

class TranslatedItemSchema(BaseModel):
    index: int = Field(description="The exact index of the subtitle segment from the input.")
    translated_text: str = Field(
        description="The translated subtitle text. Use '\\n' if split into two lines (max 2 lines)."
    )


class TranslationBatchResponseSchema(BaseModel):
    translations: List[TranslatedItemSchema] = Field(
        description="List of translated subtitle segments matching the exact input indices."
    )


class GlossaryCharacter(BaseModel):
    name: str = Field(description="Character name as spoken in the source")
    target_name: str = Field(description="How the name should be written in the target language")
    gender: str = Field(description="male, female, mixed group, or unknown")
    notes: str = Field(default="", description="Role, relationships, how they speak (formal, slang...)")
    speaker_id: str = Field(default="", description="Optional audio SPEAKER_XX id matching this character, or empty")


class GlossaryTerm(BaseModel):
    source: str
    translation: str
    notes: str = ""


class GlossarySchema(BaseModel):
    summary: str = Field(description="Two or three sentences: genre, setting, tone")
    characters: List[GlossaryCharacter] = []
    terms: List[GlossaryTerm] = Field(default=[], description="Recurring jargon, ranks, places, call signs")
    register_notes: str = Field(default="", description="Guidance on tone, profanity, formality")


# =====================================================================
# Prompt builders
# =====================================================================

def format_glossary(glossary: Optional[Dict[str, Any]]) -> str:
    if not glossary:
        return ""
    lines = ["### SHOW GLOSSARY (derived from the full transcript — stay consistent with it)"]
    if glossary.get("summary"):
        lines.append(f"Summary: {glossary['summary']}")
    chars = glossary.get("characters") or []
    if chars:
        lines.append("Characters:")
        for c in chars:
            note = f" — {c.get('notes')}" if c.get("notes") else ""
            lines.append(f"- {c.get('name')} → {c.get('target_name') or c.get('name')} ({c.get('gender', 'unknown')}){note}")
    terms = glossary.get("terms") or []
    if terms:
        lines.append("Terminology:")
        for t in terms:
            note = f" ({t.get('notes')})" if t.get("notes") else ""
            lines.append(f"- {t.get('source')} → {t.get('translation')}{note}")
    if glossary.get("register_notes"):
        lines.append(f"Register: {glossary['register_notes']}")
    return "\n".join(lines) + "\n"


def _known_glossary_note(known: Optional[Dict[str, Any]]) -> str:
    """Tell the glossary pass what the series already established, so it only adds what is new."""
    if not known:
        return ""
    return (
        "\n\nThis show has a saved glossary from earlier episodes. Treat it as authoritative and do NOT "
        "repeat or change anything in it; list only characters and terms that are NEW in this transcript.\n"
        + format_glossary(known)
    )


def build_subtitle_system_prompt(
    source_lang: str,
    target_lang: str,
    context_notes: str = "",
    max_chars_per_line: int = 42,
    glossary: Optional[Dict[str, Any]] = None,
    convert_to_metric: bool = False,
    remove_sdh_speakers: bool = False,
) -> str:
    """Construct the system prompt engineered for cinema/broadcast subtitle translation or same-language polish."""
    extra_context = (
        f"\n### SHOW / MOVIE SPECIFIC CONTEXT (from the user — highest priority):\n{context_notes.strip()}\n"
        if context_notes and context_notes.strip()
        else ""
    )
    glossary_block = format_glossary(glossary)
    same_lang = is_same_language(source_lang, target_lang)

    if same_lang:
        hebrew_polish = ""
        if target_lang.strip().lower() in ("hebrew", "he", "עברית"):
            hebrew_polish = """
6. **Hebrew Proofreading & Orthography:**
   - Fix common Whisper/ASR Hebrew homophone and suffix mishearings from context (e.g., אם/עם, אל/על, לו/לא, ש/כש, gender endings matching the speaker/listener).
   - Use natural modern Israeli spelling (כתיב מלא תקני) without changing spoken slang or colloquial phrasing (keep "יאללה", "אחי", "וואלה", "ממש", etc. intact).
   - Use proper Hebrew gershayim ״ and geresh ׳ for abbreviations and foreign letters (ד״ר, צה״ל, אב״כ, ג׳ון).
"""
        metric_polish = ""
        if convert_to_metric:
            metric_polish = """
7. **Metric Conversion (if any imperial units appear):**
   - Convert literal non-metric measurements into natural metric equivalents while leaving figurative idioms untouched.
"""
        sdh_polish = ""
        if remove_sdh_speakers:
            sdh_polish = """
8. **SDH Speaker Labels Stripping:**
   - Strip `[Name]` or `Name:` closed-caption speaker tags from dialogue while preserving genuine on-screen `[...]` titles.
"""
        return f"""You are an elite broadcast subtitle proofreader and copy-editor for {target_lang} speech transcripts.

### CORE OBJECTIVE
Polish the `target_batch` {target_lang} ASR transcript into clean, broadcast-ready {target_lang} subtitles while **preserving the speaker's exact wording, phrasing, register, and slang** and strictly maintaining a 1-to-1 mapping with each segment `index`.
- **DO NOT rewrite, paraphrase, or over-formalize** what the speaker actually said.
- **ONLY fix:** automatic speech recognition (ASR) typos/misheard words (guided by context and the glossary), punctuation, capitalization/spelling, proper nouns/brands from the glossary, speaker-turn hyphens (`- `), and 2-line formatting.
{extra_context}
{glossary_block}
### STRICT TECHNICAL RULES
1. **1-to-1 Index Mapping:**
   - Return the exact same number of items as provided in `target_batch`, with exact matching `index`. Never merge, skip, or invent indices, and do NOT output items from `previous_context` or `lookahead_context`.

2. **Line Length & Formatting Constraints:**
   - Maximum **{max_chars_per_line} characters per line** and maximum **2 lines per subtitle segment** (separated by `\\n`).
   - Respect `duration_sec` (aim for {settings.target_cps:.0f} characters per second or fewer) without omitting words unless trimming clear ASR filler stutters.

3. **Split Sentences & Continuity (No Orphan Words):**
   - Inspect `previous_context`, adjacent items in `target_batch`, and `lookahead_context` so sentences spanning boundaries are punctuated coherently and 1–2 orphan words are not left on the wrong side of a boundary.

4. **Dialogue, Dual Speakers & Speaker vs. Addressee:**
   - If a segment contains two different speakers, place each turn on its own line starting with `- ` (`- ...\\n- ...`).
   - **`speaker` vs. `addressee` (CRITICAL DISTINCTION):** `speaker` is who is speaking — use `speaker`'s gender **ONLY for 1st-person self-reference** ("I", "I'm", self-referential verbs/adjectives). **NEVER** conjugate 2nd-person forms ("you") from `speaker`'s gender! Use `addressee` (or the conversational listener in `previous_context`/`target_batch`/`lookahead_context`) for **all 2nd-person forms** (masculine `אתה`, feminine `את`, or plural `אתם/אתן` when `addressee` is `group (plural)`).

5. **Segment Types (`segment_type`):**
   - Preserve `ON_SCREEN` square brackets `[...]`; keep `COMMS` terse; polish `DIALOGUE` and `FOREIGN` naturally.
{hebrew_polish}
{metric_polish}
{sdh_polish}
9. **Output hygiene:** Plain text only. Keep punctuation at the logical end of sentences; do NOT insert Unicode direction marks or manually reverse punctuation. Never put the ASCII double-quote character (") inside `translated_text` — use “ ” or Hebrew ״ / ׳.
"""

    hebrew_rules = ""
    if target_lang.strip().lower() in ("hebrew", "he", "עברית"):
        chyron_sdh_note = " (Note: Do not confuse on-screen chyrons with SDH speaker labels in dialogue like `[John] Hello` when SDH stripping is enabled)" if remove_sdh_speakers else ""
        hebrew_rules = f"""
6. **Natural Spoken Hebrew & Linguistic Guidelines:**
   - **Avoid "Translationese" (תרגומית):** Do not translate word-for-word. Translate the intent, subtext, and natural Hebrew conversational flow (e.g., translate "Are you kidding me?" as "אתה צוחק עליי?" or "את עובדת עליי?", never literally).
   - **Gender & Plurality Resolution (`speaker` vs. `addressee`):** English "you" or gender-neutral verbs must match the **addressee (listener)**, NOT the speaker! Use `speaker`'s gender **ONLY** for 1st-person self-reference ("אני חושב/חושבת", "אני מוכן/מוכנה"). Use `addressee`'s gender and number (or infer the listener from vocatives and adjacent lines in `previous_context`/`target_batch`/`lookahead_context`) for **all 2nd-person forms** ("אתה / את / אתם", imperatives, adjectives — e.g. a female speaker talking to a man uses feminine 1st-person "אני הולכת" and masculine 2nd-person "אתה בא?"; addressing multiple people or `group (plural)` uses "אתם/אתן").
   - **Slang, Humor & Profanity:** Preserve the exact register, tone, and intensity of the original dialogue. Do not censor profanity; use authentic Hebrew equivalents that match the character's voice.
   - **Proper Nouns & Technical Terms:** Transliterate character names consistently across batches (use the glossary spellings).
   - **On-screen titles & speaker chyrons (`segment_type` = "ON_SCREEN", or any line wrapped in `[...]`):** translate concisely as an official Hebrew broadcast screen title and keep the square brackets, e.g. `[One week later...]` → `[שבוע לאחר מכן...]`, `[Damascus, Syria]` → `[דמשק, סוריה]`, `[DR. JAN FRANCKE — INTERNATIONAL GPR EXPERT]` → `[ד״ר יאן פרנקה — מומחה בינלאומי ל-GPR]`. Write dates/times the way Israeli broadcasters do.{chyron_sdh_note}
   - **Radio & military comms (`segment_type` = "COMMS", or military/radio terminology in any line):** use authentic IDF/tactical Hebrew radio phrasing, short and clipped, e.g. "Copy that" → "קיבלתי" / "רות", "Over" → "עבור", "Over and out" → "סוף", "Roger" → "רות", "Say again" → "חזור", "Sitrep" → "דו"ח מצב", "Do you copy?" → "שומע?". Keep callsigns as transliterated names.
"""

    metric_rules = ""
    if convert_to_metric:
        metric_rules = """
7. **Metric System Conversion (Imperial to Metric):**
   - Automatically convert non-metric measurements (miles, feet, inches, yards, pounds/lbs, ounces, Fahrenheit, gallons, acres, etc.) into their natural, idiomatic Metric system equivalents (km, km/h, meters, cm, kg, grams, Celsius, liters, dunams).
   - Round converted numbers naturally and comfortably for subtitle readability, matching common human speech (e.g., "50 feet" → "15 מטר", "6 feet tall" / "6-foot" → "בגובה 1.80 מ׳", "100 degrees" → "38 מעלות", "90 miles an hour" → "145 קמ״ש", "200 pounds" → "90 ק״ג", "half a mile" → "כ-800 מטר", "10 miles" → "16 ק״מ", "5 acres" → "20 דונם").
   - Do NOT convert figurative idioms or symbolic expressions where the unit is metaphorical rather than a literal measurement (e.g. "give him an inch and he'll take a mile", "miles away" meaning distracted, "walk a mile in my shoes", "million-mile stare").
"""

    sdh_rules = ""
    if remove_sdh_speakers:
        sdh_rules = """
8. **SDH Speaker Labels Stripping:**
   - Input subtitles (SDH / Closed Captions) often include hearing-impaired speaker labels (e.g. `[JOHN] Hello`, `[Sarah]: I'm fine`, `(MAN 1) Stop!`, `NARRATOR: Long ago...`).
   - Use the character's name to determine the speaker's gender (masculine/feminine) for accurate grammatical conjugation in the target language (e.g. `[Sarah] I'm ready.` → feminine `אני מוכנה.`).
   - **Never include the speaker's name or bracketed speaker label in your translation output.** Output ONLY the translated dialogue text (e.g. `[John] Hello!` → `שלום!`, NOT `[ג'ון] שלום!`).
   - If a segment contains two speakers with labels (e.g. `[John] Yes.\\n[Sarah] No.`), format with standard dialogue hyphens without names:
     `- כן.\\n- לא.`
   - Keep square brackets ONLY for genuine on-screen text, dates, and locations (e.g. `[One week later...]` → `[שבוע לאחר מכן...]`, `[Damascus, Syria]` → `[דמשק, סוריה]`).
"""

    return f"""You are an elite, broadcast-grade audiovisual subtitle translator specializing in translating movies and TV series from {source_lang} to {target_lang}.

### CORE OBJECTIVE
Translate the `target_batch` subtitle segments into natural, idiomatic, spoken {target_lang} suitable for cinema and television (Netflix/HBO broadcast standards), while strictly maintaining a 1-to-1 mapping with the original segment `index`.
{extra_context}
{glossary_block}
### STRICT TECHNICAL RULES
1. **1-to-1 Index Mapping:**
   - You MUST return the exact same number of items as provided in `target_batch`.
   - Every returned item MUST have the exact `index` from `target_batch`.
   - NEVER merge two segments into one, and NEVER skip or invent an index.
   - Do NOT translate items from `previous_context` or `lookahead_context`—those are provided strictly for narrative continuity.

2. **Line Length & Formatting Constraints:**
   - Maximum **{max_chars_per_line} characters per line**.
   - Maximum **2 lines per subtitle segment** (separated by a single `\\n`).
   - If a translation exceeds {max_chars_per_line} characters, split it into two visually balanced lines at a natural linguistic boundary (e.g., after a comma, conjunction, or preposition).
   - Respect `duration_sec`: keep phrasing concise so the viewer can read it comfortably (aim for {settings.target_cps:.0f} characters per second or fewer). Condense rather than overflow.

3. **Split Sentences & Continuity (No Orphan Words):**
   - Sentences may span across two or more consecutive subtitle segments when a single long thought continues across cues.
   - Inspect `previous_context`, adjacent items in `target_batch`, and `lookahead_context` to understand the full sentence before translating.
   - **CRITICAL — Never strand 1–2 orphan words across a boundary:** Automatic speech recognition (ASR) sometimes cuts a segment 1–2 words too early or too late (especially in unpunctuated segments). NEVER end a segment with a complete sentence followed by 1–2 dangling words that actually start the next segment's sentence, and NEVER start a segment with the final 1–2 words of the previous segment's sentence followed by a period and a new sentence (e.g. avoid ending segment N with "...אני לא בטוח מה" and starting segment N+1 with "זה. מה לעזאזל זה?"). Keep each complete sentence whole inside the segment where it belongs so the previous sentence ends with its final word and period in segment N, and segment N+1 begins cleanly with its own sentence.

4. **Dialogue, Dual Speakers & Speaker vs. Addressee:**
   - If a single segment contains two different speakers (whether indicated by leading hyphens, e.g., `- Hello.\\n- Hi.`, or an unpunctuated ASR run-on containing a question/statement and another person's reply), format each speaker's turn on its own line starting with `- ` (`- ...\\n- ...`) so downstream processing can split them into separate single-speaker subtitles.
   - **`speaker` vs. `addressee` (CRITICAL DISTINCTION):**
     - `speaker` is the person talking. Use `speaker`'s gender **ONLY for 1st-person self-reference** ("I", "I'm", present-tense self verbs/adjectives like "אני חושב/חושבת", "אני מוכן/מוכנה"). **NEVER** conjugate 2nd-person verbs/pronouns ("you", "your", imperatives) according to `speaker`'s gender!
     - `addressee` (when present) is the person or group being spoken TO. Use the **addressee's** gender and number for **all 2nd-person forms** ("you", "your", imperatives, 2nd-person verbs/adjectives: masculine singular, feminine singular, or plural when `addressee` is `group (plural)` or multiple people).
     - When `addressee` is not explicitly tagged on a line, infer who the speaker is talking to from vocative names/titles in the line, who spoke immediately before (`previous_context` / earlier in `target_batch`), and who replies immediately after (`target_batch` / `lookahead_context`).

5. **Segment Types (`segment_type`):**
   - "DIALOGUE": normal speech.
   - "COMMS": a radio / walkie-talkie transmission — keep it terse, in radio procedure style.
   - "ON_SCREEN": text shown on screen (speaker/expert identification chyron, time jump, location, date, phone text message). Translate it as a short screen title (transliterating person names and translating their title/role) and keep any surrounding square brackets `[...]`. (Note: an inline prefix like `[Travis] Hello` inside a DIALOGUE segment is a speaker tag, NOT an on-screen title.)
   - "FOREIGN": a line spoken in a different language than the rest of the show (e.g. Russian in an English series). Its text is either the show's own subtitle for that line or a transcription in the original language — either way, translate it into the target language as ordinary dialogue (no brackets).
{hebrew_rules}
{metric_rules}
{sdh_rules}
9. **Output hygiene:** Plain text only. Keep punctuation at the logical end of sentences and do NOT insert Unicode direction marks or manually reverse punctuation — a downstream processor handles right-to-left display. Never put the ASCII double-quote character (") inside a translation — it ends the JSON string and cuts the subtitle off. For Hebrew abbreviations use gershayim ״ and geresh ׳ (ד״ר, אב״כ, נ״צ, כנ״ל, צה״ל, ג׳יימס); for quotations use “ ”.
"""


def _is_fatal(exc: Exception) -> bool:
    """Errors retrying won't fix: bad/missing key, permission, unknown model."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if code in (401, 403):
        return True
    if any(k in msg for k in ("http 401", "http 403", "invalid_api_key", "incorrect api key", "authentication_error", "api key not valid")):
        return True
    if (code in (400, 404) or "http 404" in msg) and (
        "api key" in msg or "api_key" in msg or "not found" in msg or "permission" in msg or "does not exist" in msg
    ):
        return True
    return False


def _looks_truncated(source: str, translation: str) -> bool:
    """A translation far shorter than its source is almost always cut off mid-answer
    (seen: 'Stand by for exact coordinates. Nishioka, eyes on.' -> 'המתן לנ'), or corrupted
    into bare punctuation/brackets ('}]}')."""
    src_norm = re.sub(r"\s+", " ", source).strip()
    tr_norm = translation.strip()
    if re.search(r"\w", src_norm) and not re.search(r"\w", tr_norm):
        return True
    src = len(src_norm)
    return src >= 25 and len(tr_norm) < 0.35 * src


def _is_rate_limit(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 429 or "resource_exhausted" in str(exc).lower()


_SDH_BRACKET_PREFIX = re.compile(r"^(\s*[-–—]?\s*)\[[^\]\n]{1,35}\]\s*:?\s*")
_SDH_COLON_PREFIX = re.compile(r"^(\s*[-–—]?\s*)[A-Za-z\u0590-\u05fe]{2,20}\s*:\s*")


def _strip_sdh_speaker_prefix(text: str) -> str:
    """Strip accidental leftover speaker label in dialogue lines when SDH stripping is enabled."""
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            lines.append(line)
            continue
        cleaned = _SDH_BRACKET_PREFIX.sub(r"\1", line)
        cleaned = _SDH_COLON_PREFIX.sub(r"\1", cleaned)
        lines.append(cleaned)
    return "\n".join(lines)


# =====================================================================
# Gemini Translator Implementation
# =====================================================================

class GeminiTranslator(BaseTranslator):
    """
    Context-aware batch subtitle translator powered by the Google GenAI SDK.
    Supports a whole-transcript glossary pass, sliding context windows, structured JSON
    output, partial retries of missing indices, and explicit failure flags.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        prev_context_size: int = 5,
        lookahead_size: int = 3,
        max_chars_per_line: Optional[int] = None,
        max_retries: int = 3,
        thinking: str = "default",
        glossary_model: Optional[str] = None,
        glossary_thinking: str = "default",
        usage: Optional[UsageTracker] = None,
        flex: bool = False,
        glossary_flex: bool = False,
        flex_state: Optional[gemini_calls.FlexState] = None,
        convert_to_metric: bool = False,
        remove_sdh_speakers: bool = False,
        provider: str = "google",
        key_id: Optional[str] = None,
        glossary_provider: Optional[str] = None,
        glossary_key_id: Optional[str] = None,
    ):
        from app.core.provider_keys import resolve_secret_key

        self.provider = (provider or "google").strip().lower()
        self.key_id = key_id
        self.glossary_provider = (glossary_provider or self.provider).strip().lower()
        self.glossary_key_id = glossary_key_id if glossary_provider else (glossary_key_id or key_id)
        self.api_key_override = (api_key or "").strip() or None

        self.client: Any = None
        if self.provider == "google":
            resolved_key = self.api_key_override
            if not resolved_key:
                vault_key = resolve_secret_key("google", key_id=self.key_id)
                resolved_key = vault_key.api_key if vault_key else settings.gemini_api_key
            if not resolved_key or resolved_key == "your_gemini_api_key_here":
                raise ValueError(
                    "GEMINI_API_KEY is missing. Set it in backend/.env or provide it in the UI."
                )
            self.client = genai.Client(api_key=resolved_key)
        else:
            # Verify credential exists early for non-Google providers (unless Ollama local fallback)
            resolved = resolve_secret_key(self.provider, key_id=self.key_id)
            if resolved is None:
                raise ValueError(f"No API key configured for provider '{self.provider}'.")

        self.model_name = model_name or settings.gemini_model
        if batch_size is not None:
            self.batch_size = batch_size
        elif self.provider == "ollama":
            self.batch_size = min(settings.gemini_batch_size, 30)
        elif self.provider in ("openai", "anthropic"):
            self.batch_size = min(settings.gemini_batch_size, 50)
        else:
            self.batch_size = settings.gemini_batch_size
        self.prev_context_size = prev_context_size
        self.lookahead_size = lookahead_size
        self.max_chars_per_line = max_chars_per_line or settings.max_chars_per_line
        self.max_retries = max_retries
        self.thinking = thinking
        self.glossary_model = glossary_model or self.model_name
        self.glossary_thinking = glossary_thinking
        self.usage = usage
        # Flex service tier: half price, slower, may fall back to Standard when Google is busy
        self.flex = flex if self.provider == "google" else False
        self.glossary_flex = glossary_flex if self.glossary_provider == "google" else False
        self.flex_state = flex_state if flex_state is not None else gemini_calls.FlexState()
        self.convert_to_metric = convert_to_metric
        self.remove_sdh_speakers = remove_sdh_speakers
        self.last_glossary_error: Optional[str] = None

    async def _generate(
        self,
        contents: str,
        system_prompt: str,
        schema,
        temperature: float,
        stage: str = "translation",
        model: Optional[str] = None,
        thinking: Optional[str] = None,
        flex: Optional[bool] = None,
        provider: Optional[str] = None,
        key_id: Optional[str] = None,
    ):
        target_provider = (provider or (self.glossary_provider if stage == "glossary" else self.provider) or "google").lower()
        target_key_id = key_id if key_id is not None else (self.glossary_key_id if stage == "glossary" else self.key_id)
        model = model or self.model_name
        thinking = self.thinking if thinking is None else thinking
        use_flex = self.flex if flex is None else flex

        if target_provider == "google" and target_key_id is None and self.client is not None:
            return await gemini_calls.generate_content(
                self.client,
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=default_temperature(model, temperature),
                    thinking_config=thinking_config(model, thinking),
                ),
                flex=use_flex,
                usage=self.usage,
                stage=stage,
                flex_state=self.flex_state,
            )

        from app.providers.llm import call_text_json, resolve_provider_spec

        spec = resolve_provider_spec(
            target_provider,  # type: ignore[arg-type]
            model,
            key_id=target_key_id,
            api_key_override=self.api_key_override if target_provider == "google" else None,
            thinking=thinking,
            flex=use_flex,
        )
        return await call_text_json(
            spec,
            system_instruction=system_prompt,
            prompt=contents,
            schema_cls=schema,
            temperature=temperature,
            usage=self.usage,
            stage=stage,
            flex_state=self.flex_state,
        )

    async def build_glossary(
        self,
        segments: List[TranslatableSegment],
        source_lang: str,
        target_lang: str,
        context_notes: str = "",
        known_glossary: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        has_speaker_tags = any(getattr(s, "speaker", None) for s in segments)
        transcript = "\n".join(
            f"{s.index}: "
            + (f"[{s.speaker}] " if getattr(s, "speaker", None) else "")
            + s.original_text.replace(chr(10), " ")
            for s in segments
        )
        if len(transcript) > _GLOSSARY_MAX_CHARS:
            transcript = transcript[:_GLOSSARY_MAX_CHARS]
        notes = f"\nUser notes about the show:\n{context_notes.strip()}\n" if context_notes.strip() else ""
        speaker_note = (
            " When transcript lines start with [SPEAKER_XX] audio tags that clearly correspond to a character, set `speaker_id` to that tag."
            if has_speaker_tags
            else ""
        )
        if is_same_language(source_lang, target_lang):
            system_prompt = (
                f"You prepare subtitle proofreaders. Read the full {target_lang} transcript of a video and produce "
                f"a concise glossary for copy-editing it in {target_lang}: the main speakers/people mentioned "
                f"(exact {target_lang} spelling, gender), recurring technical terminology, brands, places and jargon "
                f"with their authoritative {target_lang} spelling, and notes on tone/register. "
                f"Only include what the transcript supports; mark gender 'unknown' when unsure.{speaker_note}{notes}"
                f"{_known_glossary_note(known_glossary)}"
            )
        else:
            system_prompt = (
                f"You prepare subtitle translators. Read the full {source_lang} transcript of a movie/episode and "
                f"produce a concise glossary for translating it into {target_lang}: the main characters (with their "
                f"gender, inferred from pronouns, titles and how others address them), recurring terminology, ranks, "
                f"places and call signs with a recommended {target_lang} rendering, and notes on tone/register. "
                f"Only include what the transcript supports; mark gender 'unknown' when unsure.{speaker_note}{notes}"
                f"{_known_glossary_note(known_glossary)}"
            )
        try:
            response = await self._generate(
                transcript, system_prompt, GlossarySchema, temperature=0.1,
                stage="glossary", model=self.glossary_model, thinking=self.glossary_thinking,
                flex=self.glossary_flex,
            )
            glossary = GlossarySchema.model_validate_json(response.text).model_dump()
            logger.info(
                f"Glossary built: {len(glossary.get('characters', []))} characters, "
                f"{len(glossary.get('terms', []))} terms"
            )
            return glossary
        except Exception as exc:
            msg = str(exc).lower()
            # Only a bad key stops everything; a problem with the glossary *model* must not
            # prevent the translation (which may use a different model).
            if getattr(exc, "code", None) == 401 or "api key not valid" in msg or "api_key_invalid" in msg:
                raise TranslationAborted(str(exc)) from exc
            self.last_glossary_error = f"{self.glossary_model}: {exc}"
            logger.warning(f"Glossary pass failed (continuing without it): {exc}")
            return None

    async def translate_segments(
        self,
        segments: List[TranslatableSegment],
        source_lang: str = "English",
        target_lang: str = "Hebrew",
        context_notes: str = "",
        apply_rtl: bool = False,
        on_batch_complete: Optional[BatchProgressCallback] = None,
        glossary: Optional[Dict[str, Any]] = None,
        context_segments: Optional[List[TranslatableSegment]] = None,
    ) -> List[TranslatableSegment]:
        """Translate segments in batches with sliding context."""
        if not segments:
            return []

        context = context_segments or segments
        position = {seg.index: pos for pos, seg in enumerate(context)}
        total_segments = len(segments)
        from app.providers.translation.glossary_pruner import prune_glossary_for_batch

        do_rtl = apply_rtl and is_rtl(target_lang)

        for start_idx in range(0, total_segments, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_segments)
            current_batch = segments[start_idx:end_idx]

            first_pos = position.get(current_batch[0].index, 0)
            last_pos = position.get(current_batch[-1].index, len(context) - 1)
            prev_context = context[max(0, first_pos - self.prev_context_size):first_pos]
            lookahead_context = context[last_pos + 1:last_pos + 1 + self.lookahead_size]

            batch_glossary = prune_glossary_for_batch(
                glossary=glossary,
                batch_texts=current_batch,
                previous_context=prev_context,
                lookahead_context=lookahead_context,
            )
            if glossary and batch_glossary and self.usage is not None and hasattr(self.usage, "record_local_savings"):
                from app.core.model_catalog import request_cost

                saved_chars = max(0, len(format_glossary(glossary)) - len(format_glossary(batch_glossary)))
                saved_toks = saved_chars // 4
                if saved_toks > 0:
                    saved_cost = request_cost(self.model_name, saved_toks, 0, flex=self.flex, provider=self.provider) or 0.0
                    self.usage.record_local_savings(saved_tokens=saved_toks, saved_cost_usd=saved_cost)

            system_prompt = build_subtitle_system_prompt(
                source_lang=source_lang,
                target_lang=target_lang,
                context_notes=context_notes,
                max_chars_per_line=self.max_chars_per_line,
                glossary=batch_glossary,
                convert_to_metric=self.convert_to_metric,
                remove_sdh_speakers=self.remove_sdh_speakers,
            )

            try:
                await self._translate_single_batch(
                    target_batch=current_batch,
                    prev_context=prev_context,
                    lookahead_context=lookahead_context,
                    system_prompt=system_prompt,
                    apply_rtl=do_rtl,
                )
            except TranslationAborted:
                for seg in segments[start_idx:]:
                    if not seg.translated_text:
                        seg.translation_failed = True
                raise

            progress_pct = round((end_idx / total_segments) * 100, 2)
            logger.info(f"Translated batch {start_idx + 1}-{end_idx}/{total_segments} ({progress_pct}%)")

            if on_batch_complete:
                await on_batch_complete(current_batch, progress_pct)

        return segments

    async def _translate_single_batch(
        self,
        target_batch: List[TranslatableSegment],
        prev_context: List[TranslatableSegment],
        lookahead_context: List[TranslatableSegment],
        system_prompt: str,
        apply_rtl: bool,
    ) -> None:
        """Translate one batch. Missing indices are retried; leftovers are flagged as failed."""
        pending = list(target_batch)
        for seg in pending:
            seg.translation_failed = False

        def _ctx_item(
            seg: TranslatableSegment,
            *,
            with_translated: bool = False,
            with_meta: bool = False,
        ) -> Dict[str, Any]:
            item: Dict[str, Any] = {"index": seg.index}
            if with_meta:
                item["segment_type"] = seg.segment_type
                item["duration_sec"] = seg.duration
            item["original_text"] = seg.original_text
            if with_translated:
                item["translated_text"] = strip_directional_marks(seg.translated_text or "")
            sp = getattr(seg, "speaker_info", None) or getattr(seg, "speaker", None)
            if sp:
                item["speaker"] = sp
            addr = getattr(seg, "addressee_info", None)
            if addr:
                item["addressee"] = addr
            return item

        for attempt in range(1, self.max_retries + 1):
            payload = {
                "previous_context": [_ctx_item(seg, with_translated=True) for seg in prev_context],
                "target_batch": [_ctx_item(seg, with_meta=True) for seg in pending],
                "lookahead_context": [_ctx_item(seg) for seg in lookahead_context],
            }
            try:
                response = await self._generate(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    system_prompt,
                    TranslationBatchResponseSchema,
                    temperature=0.2,
                )
                parsed = TranslationBatchResponseSchema.model_validate_json(response.text)
                translation_map = {item.index: item.translated_text for item in parsed.translations}

                still_missing = []
                for seg in pending:
                    raw = strip_directional_marks((translation_map.get(seg.index) or "").strip())
                    if raw and _looks_truncated(seg.original_text, raw):
                        logger.warning(f"Translation of #{seg.index} looks cut off ({raw!r} for "
                                       f"{seg.original_text!r}); asking again")
                        raw = ""
                    if raw:
                        if self.remove_sdh_speakers and (seg.segment_type or "DIALOGUE") == "DIALOGUE":
                            raw = _strip_sdh_speaker_prefix(raw)
                        elif (seg.segment_type or "DIALOGUE") == "ON_SCREEN":
                            orig_s = (seg.original_text or "").strip()
                            if orig_s.startswith("[") and orig_s.endswith("]") and not (raw.startswith("[") and raw.endswith("]")):
                                raw = f"[{raw.strip('[]').strip()}]"
                        seg.translated_text = apply_rtl_fix(raw) if apply_rtl else raw
                    else:
                        still_missing.append(seg)
                pending = still_missing
                if not pending:
                    return
                raise ValueError(f"Gemini response omitted segment indices: {[s.index for s in pending]}")

            except Exception as exc:
                if _is_fatal(exc):
                    raise TranslationAborted(str(exc)) from exc
                logger.warning(f"Batch translation attempt {attempt}/{self.max_retries} failed: {exc}")
                err_low = str(exc).lower()
                is_size_or_timeout = any(
                    tok in err_low
                    for tok in (
                        "eof",
                        "json_invalid",
                        "validation error",
                        "field required",
                        "max_completion_tokens",
                        "timeout",
                        "omitted segment indices",
                    )
                )
                if len(pending) > 15 and (is_size_or_timeout or attempt >= self.max_retries):
                    mid = len(pending) // 2
                    left, right = pending[:mid], pending[mid:]
                    logger.info(
                        f"Splitting failed batch of {len(pending)} segments into {len(left)} + {len(right)}"
                    )
                    await self._translate_single_batch(
                        left, prev_context, right[:self.lookahead_size], system_prompt, apply_rtl
                    )
                    right_prev = (prev_context + left)[-self.prev_context_size:] if self.prev_context_size > 0 else []
                    await self._translate_single_batch(
                        right, right_prev, lookahead_context, system_prompt, apply_rtl
                    )
                    return
                if attempt < self.max_retries:
                    delay = 1.5 * (2 ** (attempt - 1))
                    if _is_rate_limit(exc):
                        delay *= 8
                    await asyncio.sleep(delay)

        logger.error(f"Giving up on {len(pending)} segment(s) in this batch; flagged as failed")
        for seg in pending:
            seg.translated_text = None
            seg.translation_failed = True
