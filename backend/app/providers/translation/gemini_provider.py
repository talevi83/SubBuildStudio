import asyncio
import json
import re
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator
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

# One logger for every provider this translator drives (Gemini, Anthropic, OpenAI, Ollama), so
# Anthropic calls are no longer logged under a "gemini_provider" name.
logger = logging.getLogger("app.providers.translation.translator")


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
# Pydantic Schemas for Structured Output
# =====================================================================

class TranslatedItemSchema(BaseModel):
    id: int = Field(description="The exact segment id from target_batch (strict 1-to-1 correspondence required).")
    text: str = Field(
        description="The translated subtitle text for this id. Use '\\n' if split into two lines (max 2 lines)."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            if "id" not in out and "index" in out:
                out["id"] = out["index"]
            if "text" not in out and "translated_text" in out:
                out["text"] = out["translated_text"]
            return out
        return data

    @property
    def index(self) -> int:
        return self.id

    @property
    def translated_text(self) -> str:
        return self.text


class TranslationBatchResponseSchema(BaseModel):
    translations: List[TranslatedItemSchema] = Field(
        description="List of translated subtitle items matching the exact input ids in 1-to-1 order."
    )


class TranslatedItemWithFormSchema(TranslatedItemSchema):
    addressee_form: str = Field(
        default="",
        description=(
            "The 2nd-person form actually written in text: 'm' (masculine singular), "
            "'f' (feminine singular), 'pl' (plural), or '' when it has no gender/number-marked 2nd-person form."
        ),
    )


class TranslationBatchWithFormResponseSchema(BaseModel):
    """Used for target languages whose 2nd person is gendered, so the result can be checked."""
    translations: List[TranslatedItemWithFormSchema] = Field(
        description="List of translated subtitle items matching the exact input ids in 1-to-1 order."
    )


_ADDRESSEE_FORMS = ("m", "f", "pl")


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
    lines = [
        "### LOCKED TERMINOLOGY & CHARACTER GLOSSARY (MANDATORY — 100% CONSISTENCY REQUIRED)",
        "Use these exact target-language spellings, titles, and translations in EVERY scene and batch from start to finish. "
        "Never invent alternate transliterations or switch renderings across scenes. If speech recognition phonetically "
        "distorted a listed character name or domain term in `original_text`, render the locked glossary entry rather than "
        "transliterating the ASR mishearing literally.",
    ]
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


# Heading that starts the part of the system prompt that changes from batch to batch (show notes,
# the glossary pruned to the batch). Everything before it is identical for every batch of a job, so
# providers with prefix caching (Anthropic `cache_control`, Gemini implicit caching) reuse it.
PER_REQUEST_CONTEXT_HEADING = "### PER-REQUEST CONTEXT (show notes and locked glossary for this batch)"


def split_cacheable_system_prompt(system_prompt: str) -> tuple:
    """Split a system prompt into `(static_prefix, per_request_suffix)` at `PER_REQUEST_CONTEXT_HEADING`.

    Returns `(system_prompt, "")` when the heading is absent.
    """
    text = system_prompt or ""
    pos = text.find(PER_REQUEST_CONTEXT_HEADING)
    if pos <= 0:
        return text, ""
    return text[:pos].rstrip() + "\n", text[pos:]


def _per_request_block(extra_context: str, glossary_block: str) -> str:
    body = f"{extra_context.strip()}\n\n{glossary_block.strip()}".strip()
    return f"\n{PER_REQUEST_CONTEXT_HEADING}\n{body}\n" if body else ""


def build_subtitle_system_prompt(
    source_lang: str,
    target_lang: str,
    context_notes: str = "",
    max_chars_per_line: int = 42,
    glossary: Optional[Dict[str, Any]] = None,
    convert_to_metric: bool = False,
    remove_sdh_speakers: bool = False,
    report_addressee_form: bool = False,
    static_glossary: bool = False,
    dual_source_mode: bool = False,
    dual_source_primary_lang: str = "English",
    dual_source_anchor_lang: Optional[str] = "Arabic",
    dual_source_pair_mode: str = "en_ar",
) -> str:
    """Construct the system prompt engineered for cinema/broadcast subtitle translation or same-language polish."""
    extra_context = (
        f"\n### SHOW / MOVIE SPECIFIC CONTEXT (from the user — highest priority):\n{context_notes.strip()}\n"
        if context_notes and context_notes.strip()
        else ""
    )
    glossary_block = format_glossary(glossary)
    if static_glossary:
        body = f"{extra_context.strip()}\n\n{glossary_block.strip()}".strip()
        context_tail = f"\n\n{body}\n" if body else ""
    else:
        context_tail = _per_request_block(extra_context, glossary_block)
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
Polish the `target_batch` {target_lang} ASR transcript into clean, broadcast-ready {target_lang} subtitles while **preserving the speaker's exact wording, phrasing, register, and slang** and strictly maintaining a 1-to-1 mapping with each segment `id`.
- **DO NOT rewrite, paraphrase, or over-formalize** what the speaker actually said.
- **ONLY fix:** automatic speech recognition (ASR) typos/misheard words (guided by context and the glossary), punctuation, capitalization/spelling, proper nouns/brands from the glossary, speaker-turn hyphens (`- `), and 2-line formatting.

### STRICT TECHNICAL RULES
1. **1-to-1 ID Mapping:**
   - Return the exact same number of items as provided in `target_batch`, with exact matching `id` and `text`. Never merge, skip, shift, or invent IDs, and do NOT output items from `previous_context` or `lookahead_context`.

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
9. **Output hygiene:** Plain text only. Keep punctuation at the logical end of sentences; do NOT insert Unicode direction marks or manually reverse punctuation. Never put the ASCII double-quote character (") inside `text` — use “ ” or Hebrew ״ / ׳.
{context_tail}"""

    hebrew_rules = ""
    if target_lang.strip().lower() in ("hebrew", "he", "עברית"):
        chyron_sdh_note = " (Note: Do not confuse on-screen chyrons with SDH speaker labels in dialogue like `[John] Hello` when SDH stripping is enabled)" if remove_sdh_speakers else ""
        hebrew_rules = f"""
7. **Natural Spoken Hebrew, Anti-Calque Rules & Linguistic Guidelines:**
   - **Strictly Forbid Word-for-Word Calques ("Translationese" / תרגומית):** NEVER translate English idioms, exclamations, or directives word-for-word. Always translate pragmatic intent into natural Hebrew:
     - *"Easy" / "Easy, easy" / "Take it easy"* (calming someone, steadying a crowd, or slowing down) → **"לאט, לאט"** or **"הירגע / תירגע / תירגעי / הירגעו"** — **NEVER** literal **"קל, קל"**!
     - *"Are you kidding me?"* → **"אתה צוחק עליי? / את עובדת עליי?"**
     - *"Hold on" / "Hang on"* → **"חכה רגע / תחזיק מעמד"** (never literal *"החזק על"*).
     - *"Watch out!" / "Look out!"* → **"תיזהר! / תיזהרי! / תיזהרו!"** (never literal *"תסתכל החוצה!"*).
     - *"Make way!"* → **"פנו דרך!"**; *"Stand back!"* → **"זוזו אחורה! / התרחקו!"**; *"Work fast"* → **"עבוד מהר / עבדו מהר"**.
     - *"No way"* → **"אין סיכוי / מה פתאום"**; *"I got you"* → **"תפסתי אותך / אני איתך"** (never *"קיבלתי אותך"*).
   - **Genre, Era & Register Fidelity:** Match phrasing to the genre and historical period established in the glossary/context. In historical or period dramas, use authentic, natural elevated/period Hebrew and avoid modern Israeli street slang anachronisms ("אחי", "סבבה", "יאללה"); in contemporary dialogue, preserve authentic spoken register, humor, and uncensored equivalents matching each character's voice.
   - **Gender & Plurality Resolution (`speaker` vs. `addressee`):** English "you" or gender-neutral verbs must match the **addressee (listener)**, NOT the speaker! Use `speaker`'s gender **ONLY** for 1st-person self-reference ("אני חושב/חושבת", "אני מוכן/מוכנה"). Use `addressee`'s gender and number (or infer the listener from vocatives and adjacent lines in `previous_context`/`target_batch`/`lookahead_context`) for **all 2nd-person forms** ("אתה / את / אתם / אתן", imperatives, adjectives — e.g. a female speaker talking to a man uses feminine 1st-person "אני הולכת" and masculine 2nd-person "אתה בא?"; addressing multiple people or `group (plural)` uses "אתם/אתן"). Ensure full subject-verb-adjective gender and number agreement inside every Hebrew clause.
   - **Locked Proper Nouns, Recurring Titles & Terms:** Render every character name, recurring title/epithet (e.g., occupational or noble forms of address), place name, and historical/technical term strictly according to the Locked Glossary across all batches, correcting obvious ASR phonetic mishearings back to the locked spelling.
   - **On-screen titles & speaker chyrons (`segment_type` = "ON_SCREEN", or any line wrapped in `[...]`):** translate concisely as an official Hebrew broadcast screen title and keep the square brackets, e.g. `[One week later...]` → `[שבוע לאחר מכן...]`, `[Damascus, Syria]` → `[דמשק, סוריה]`, `[DR. JAN FRANCKE — INTERNATIONAL GPR EXPERT]` → `[ד״ר יאן פרנקה — מומחה בינלאומי ל-GPR]`. Write dates/times the way Israeli broadcasters do.{chyron_sdh_note}
   - **Radio & military comms (`segment_type` = "COMMS", or military/radio terminology in any line):** use authentic IDF/tactical Hebrew radio phrasing, short and clipped, e.g. "Copy that" → "קיבלתי" / "רות", "Over" → "עבור", "Over and out" → "סוף", "Roger" → "רות", "Say again" → "חזור", "Sitrep" → "דו"ח מצב", "Do you copy?" → "שומע?". Keep callsigns as transliterated names.
"""

    metric_rules = ""
    if convert_to_metric:
        metric_rules = """
8. **Metric System Conversion (Imperial to Metric):**
   - Automatically convert non-metric measurements (miles, feet, inches, yards, pounds/lbs, ounces, Fahrenheit, gallons, acres, etc.) into their natural, idiomatic Metric system equivalents (km, km/h, meters, cm, kg, grams, Celsius, liters, dunams).
   - Round converted numbers naturally and comfortably for subtitle readability, matching common human speech (e.g., "50 feet" → "15 מטר", "6 feet tall" / "6-foot" → "בגובה 1.80 מ׳", "100 degrees" → "38 מעלות", "90 miles an hour" → "145 קמ״ש", "200 pounds" → "90 ק״ג", "half a mile" → "כ-800 מטר", "10 miles" → "16 ק״מ", "5 acres" → "20 דונם").
   - Do NOT convert figurative idioms or symbolic expressions where the unit is metaphorical rather than a literal measurement (e.g. "give him an inch and he'll take a mile", "miles away" meaning distracted, "walk a mile in my shoes", "million-mile stare").
"""

    sdh_rules = ""
    if remove_sdh_speakers:
        sdh_rules = """
9. **SDH Speaker Labels Stripping:**
   - Input subtitles (SDH / Closed Captions) often include hearing-impaired speaker labels (e.g. `[JOHN] Hello`, `[Sarah]: I'm fine`, `(MAN 1) Stop!`, `NARRATOR: Long ago...`).
   - Use the character's name to determine the speaker's gender (masculine/feminine) for accurate grammatical conjugation in the target language (e.g. `[Sarah] I'm ready.` → feminine `אני מוכנה.`).
   - **Never include the speaker's name or bracketed speaker label in your translation output.** Output ONLY the translated dialogue text (e.g. `[John] Hello!` → `שלום!`, NOT `[ג'ון] שלום!`).
   - If a segment contains two speakers with labels (e.g. `[John] Yes.\\n[Sarah] No.`), format with standard dialogue hyphens without names:
     `- כן.\\n- לא.`
   - Keep square brackets ONLY for genuine on-screen text, dates, and locations (e.g. `[One week later...]` → `[שבוע לאחר מכן...]`, `[Damascus, Syria]` → `[דמשק, סוריה]`).
"""

    addressee_form_rule = ""
    if report_addressee_form:
        addressee_form_rule = (
            "     - **`addressee_form` (in every output item):** report the 2nd-person form you actually wrote in "
            "`text` — `m` (masculine singular), `f` (feminine singular), `pl` (plural) — or an empty "
            "string when the translation has no gender/number-marked 2nd-person form.\n"
        )

    dual_source_rules = ""
    if dual_source_mode and dual_source_pair_mode != "single_source":
        anchor_label = dual_source_anchor_lang or "Arabic"
        if dual_source_pair_mode == "ar_secondary":
            dual_source_rules = f"""
### DUAL-SOURCE PIVOT MODE ({dual_source_primary_lang} + {anchor_label} -> {target_lang.upper()})
- Translate the primary Arabic text (`original_text`) into natural, idiomatic {target_lang}, preserving Arabic 1st/2nd/3rd-person grammatical gender (masculine vs. feminine vs. plural).
- Use the paired {anchor_label} reference (`ar_reference`) as a semantic and terminology cross-check when idiomatic meaning or proper nouns need verification.
"""
        else:
            onscreen_line = (
                f'- For segments marked `AR_ONLY` (`dual_source_kind: "AR_ONLY"`, on-screen text outside English dialogue timestamps), '
                f'translate the {anchor_label} text directly into concise {target_lang} on-screen titles (keep square brackets `[...]` when appropriate for titles or time jumps).\n'
                if dual_source_pair_mode == "en_ar"
                else ""
            )
            dual_source_rules = f"""
### DUAL-SOURCE GENDER ANCHOR MODE ({dual_source_primary_lang} + {anchor_label} -> {target_lang.upper()})
- Translate the {dual_source_primary_lang} text (`original_text`) into natural, idiomatic {target_lang}. Use the paired {anchor_label} translation (`ar_reference`) strictly as a grammatical gender reference (masculine vs. feminine speaker/addressee, 1st/2nd/3rd person inflection) while preserving exact {dual_source_primary_lang} meaning, names, and terminology.
{onscreen_line}"""

    return f"""You are an elite, broadcast-grade audiovisual subtitle translator specializing in translating movies and TV series from {source_lang} to {target_lang}.

### CORE OBJECTIVE
Translate the `target_batch` subtitle segments into natural, idiomatic, spoken {target_lang} suitable for cinema and television (Netflix/HBO broadcast standards), while strictly maintaining a 1-to-1 mapping with the original segment `id`.

### STRICT TECHNICAL RULES
1. **1-to-1 ID Mapping & Read-Only Sliding Context:**
   - You MUST return the exact same number of items as provided in `target_batch`.
   - Every returned item MUST have the exact `id` from `target_batch` and its corresponding translated `text`.
   - NEVER merge two adjacent segments into one item, NEVER shift translations across neighbouring IDs, and NEVER skip or invent an `id`.
   - `previous_context` (the last 3–5 already-translated lines from the preceding batch) and `lookahead_context` are READ-ONLY context: use `previous_context` to preserve ongoing tone, register, terminology, and speaker/listener grammatical gender across batch boundaries, and NEVER emit items from `previous_context` or `lookahead_context`.

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
     - A gender written as `male by voice` / `female by voice` is an acoustic estimate, not a fact: when the line or the adjacent dialogue clearly shows otherwise (the speaker is called "my lady"/"sir" by the others, names or describes themselves), follow the dialogue.
     - `addressee` (when present) is the person or group being spoken TO. Use the **addressee's** gender and number for **all 2nd-person forms** ("you", "your", imperatives, 2nd-person verbs/adjectives: masculine singular, feminine singular, or plural when `addressee` is `group (plural)` or multiple people).
     - `addressee_guess` (given instead of `addressee` when the listener is likely but not confirmed): follow it unless the line itself or the adjacent dialogue clearly points to a different listener.
     - When neither is tagged on a line, infer who the speaker is talking to from vocative names/titles in the line, who spoke immediately before (`previous_context` / earlier in `target_batch`), and who replies immediately after (`target_batch` / `lookahead_context`).
     - **Stay consistent inside a conversation:** keep the same listener gender/number across consecutive lines of one exchange and never flip between masculine and feminine "you" without a reason in the dialogue. When the listener truly cannot be determined, prefer a natural phrasing that avoids gender-marked 2nd-person forms if one exists.
{addressee_form_rule}
5. **Segment Types (`segment_type`):**
   - "DIALOGUE": normal speech.
   - "COMMS": a radio / walkie-talkie transmission — keep it terse, in radio procedure style.
   - "ON_SCREEN": text shown on screen (speaker/expert identification chyron, time jump, location, date, phone text message). Translate it as a short screen title (transliterating person names and translating their title/role) and keep any surrounding square brackets `[...]`. (Note: an inline prefix like `[Travis] Hello` inside a DIALOGUE segment is a speaker tag, NOT an on-screen title.)
   - "FOREIGN": a line spoken in a different language than the rest of the show (e.g. Russian in an English series). Its text is either the show's own subtitle for that line or a transcription in the original language — either way, translate it into the target language as ordinary dialogue (no brackets).

6. **Idiomatic Target-Language Phrasing & Anti-Calque Discipline:**
   - Forbid word-for-word literal calques of source-language idioms, imperatives, and exclamations (e.g. English *"Easy, easy"* when calming someone means *"Slowly / Calm down"*, NEVER *"Simple, simple"*). Always render natural idiomatic {target_lang} appropriate to the scene's genre, historical period, and register.
   - Follow the Locked Glossary strictly for all proper names, recurring titles, places, and domain terminology, and recover obvious ASR phonetic mishearings using scene context and the glossary.
{hebrew_rules}
{metric_rules}
{sdh_rules}
{dual_source_rules}
10. **Output hygiene:** Plain text only. Keep punctuation at the logical end of sentences and do NOT insert Unicode direction marks or manually reverse punctuation — a downstream processor handles right-to-left display. Never put the ASCII double-quote character (") inside a translation — it ends the JSON string and cuts the subtitle off. For Hebrew abbreviations use gershayim ״ and geresh ׳ (ד״ר, אב״כ, נ״צ, כנ״ל, צה״ל, ג׳יימס); for quotations use “ ”.
{context_tail}"""


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
    into bare punctuation/brackets ('}]}'). Concise complete Hebrew sentences (0.20-0.34x English
    character length) are allowed as long as they are not cut off mid-sentence."""
    src_norm = re.sub(r"\s+", " ", source).strip()
    tr_norm = translation.strip()
    if re.search(r"\w", src_norm) and not re.search(r"\w", tr_norm):
        return True
    src = len(src_norm)
    if src < 25:
        return False
    if len(tr_norm) < 0.18 * src:
        return True
    src_ends_punct = src_norm[-1:] in ".!?"
    tr_ends_punct = tr_norm[-1:] in ".!?…\"'”’״׳)]"
    return len(tr_norm) < 0.28 * src and src_ends_punct and not tr_ends_punct


ANTHROPIC_BATCH_SIZE = 50
CAPABLE_DEFAULT_BATCH_SIZE = 35
COMPACT_DEFAULT_BATCH_SIZE = 18
_COMPACT_MODEL_TAG_RE = re.compile(
    r"(?:^|[-_.:/])(mini|nano|luna|lite)(?:$|[-_.:/0-9])",
    re.IGNORECASE,
)


def default_batch_size_for_model(provider: str, model_name: str, configured_batch_size: int = CAPABLE_DEFAULT_BATCH_SIZE) -> int:
    """Choose default translation batch size per provider/model capability tier.

    Capable structured-JSON models (Gemini Flash/Pro/4 and full-size OpenAI GPT-5/6/4.1/4o/o3)
    use 30-35 line batches (`configured_batch_size`, default 35), Anthropic models use
    `ANTHROPIC_BATCH_SIZE` (50), and compact `mini`/`nano`/`luna`/`lite` or local Ollama models
    use smaller 18-line batches so they never merge cues or lose 1:1 segment ID alignment.
    """
    prov = (provider or "google").strip().lower()
    low = (model_name or "").strip().lower()
    if prov == "anthropic":
        return ANTHROPIC_BATCH_SIZE
    is_compact_tier = bool(_COMPACT_MODEL_TAG_RE.search(low))
    if is_compact_tier or prov == "ollama":
        return min(configured_batch_size, COMPACT_DEFAULT_BATCH_SIZE)
    if prov == "openai":
        return min(configured_batch_size, CAPABLE_DEFAULT_BATCH_SIZE)
    return configured_batch_size


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
        prune_glossary: Optional[bool] = None,
        dual_source_mode: bool = False,
        dual_source_primary_lang: str = "English",
        dual_source_anchor_lang: Optional[str] = "Arabic",
        dual_source_pair_mode: str = "en_ar",
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
        else:
            self.batch_size = default_batch_size_for_model(
                self.provider, self.model_name, settings.gemini_batch_size
            )
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
        self.prune_glossary = bool(prune_glossary)
        self.dual_source_mode = bool(dual_source_mode)
        self.dual_source_primary_lang = dual_source_primary_lang
        self.dual_source_anchor_lang = dual_source_anchor_lang
        self.dual_source_pair_mode = dual_source_pair_mode
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
            t_cfg = thinking_config(model, thinking)
            cfg_kwargs: Dict[str, Any] = {
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": default_temperature(model, temperature),
                "thinking_config": t_cfg,
            }
            # Gemini counts thinking tokens inside max_output_tokens; reserve generous headroom
            if getattr(t_cfg, "thinking_budget", None) != 0:
                cfg_kwargs["max_output_tokens"] = 16384
            return await gemini_calls.generate_content(
                self.client,
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
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

    @staticmethod
    def _sample_transcript_for_glossary(lines: List[str], max_chars: int = _GLOSSARY_MAX_CHARS) -> str:
        """Keep the whole transcript when within budget; otherwise sample head, middle, and tail evenly."""
        full = "\n".join(lines)
        if len(full) <= max_chars:
            return full
        head_budget = int(max_chars * 0.45)
        mid_budget = int(max_chars * 0.30)
        tail_budget = max_chars - head_budget - mid_budget - 80
        mid_start = max(0, (len(full) - mid_budget) // 2)
        return (
            full[:head_budget]
            + "\n... [middle excerpt] ...\n"
            + full[mid_start:mid_start + mid_budget]
            + "\n... [ending excerpt] ...\n"
            + full[-tail_budget:]
        )

    async def build_glossary(
        self,
        segments: List[TranslatableSegment],
        source_lang: str,
        target_lang: str,
        context_notes: str = "",
        known_glossary: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        from app.core.glossary_profiles import merge_glossary, parse_user_glossary

        user_glossary = parse_user_glossary(context_notes)
        has_speaker_tags = any(getattr(s, "speaker", None) for s in segments)
        lines = [
            f"{s.index}: "
            + (f"[{s.speaker}] " if getattr(s, "speaker", None) else "")
            + s.original_text.replace(chr(10), " ")
            for s in segments
        ]
        transcript = self._sample_transcript_for_glossary(lines, _GLOSSARY_MAX_CHARS)
        notes = (
            f"\nUser-provided context / glossary / names (MUST be included and respected):\n{context_notes.strip()}\n"
            if context_notes.strip()
            else ""
        )
        speaker_note = (
            " When transcript lines start with [SPEAKER_XX] audio tags that clearly correspond to a character, set `speaker_id` to that tag."
            if has_speaker_tags
            else ""
        )
        if is_same_language(source_lang, target_lang):
            system_prompt = (
                f"You prepare subtitle proofreaders. Read the {target_lang} transcript of a video and produce "
                f"a locked glossary for copy-editing it in {target_lang}: "
                f"(1) all named characters/speakers and recurring titles/nicknames (authoritative {target_lang} spelling, "
                f"gender, and any ASR phonetic mishearings spotted in the transcript), "
                f"(2) recurring places, historical/domain/technical terms, brands, and jargon with their authoritative "
                f"{target_lang} spelling, and (3) concise notes on genre, era, and register. "
                f"Include every proper noun or term mentioned in the user's notes; mark gender 'unknown' when unsure."
                f"{speaker_note}{notes}{_known_glossary_note(known_glossary)}"
            )
        else:
            system_prompt = (
                f"You prepare audiovisual subtitle translators. Read the {source_lang} transcript of a movie/episode and "
                f"produce an authoritative locked glossary for translating it into {target_lang}:\n"
                f"1. `characters`: proper character names AND recurring titles/epithets/occupations used as forms of address "
                f"(e.g. 'Ploughman', 'Sheriff', 'My Lord'), their single locked {target_lang} spelling/rendering, speaker "
                f"`gender` ('male', 'female', 'mixed group', or 'unknown' carefully inferred from dialogue context, pronouns, "
                f"names, and honorific titles such as my lord, sir, father, husband, king, priest, son, brother vs lady, madam, "
                f"mother, wife, queen), and `notes` on role, register, and any suspected ASR mishearings in the transcript.\n"
                f"2. `terms`: recurring places, estates, factions, ranks, call signs, and historical/period/domain/technical "
                f"terms (e.g. 'Domesday', 'Corpus Christi') with one consistent {target_lang} translation.\n"
                f"3. `summary` & `register_notes`: genre, historical setting/era, tone, and idiomatic register guidance.\n"
                f"Include every proper name, title, and term provided in the user's notes. Only include what the transcript "
                f"or user notes support; mark gender 'unknown' when unsure."
                f"{speaker_note}{notes}{_known_glossary_note(known_glossary)}"
            )
        try:
            response = await self._generate(
                transcript, system_prompt, GlossarySchema, temperature=0.1,
                stage="glossary", model=self.glossary_model, thinking=self.glossary_thinking,
                flex=self.glossary_flex,
            )
            glossary = GlossarySchema.model_validate_json(response.text).model_dump()
            merged = merge_glossary(user_glossary, glossary) or glossary
            logger.info(
                f"Glossary built: {len(merged.get('characters', []))} characters, "
                f"{len(merged.get('terms', []))} terms"
            )
            return merged
        except Exception as exc:
            msg = str(exc).lower()
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            same_creds = (self.glossary_provider, self.glossary_key_id) == (self.provider, self.key_id)
            is_auth_error = code in (401, 403) or any(
                tok in msg
                for tok in (
                    "http 401",
                    "http 403",
                    "api key not valid",
                    "api_key_invalid",
                    "invalid_api_key",
                    "incorrect api key",
                    "authentication_error",
                )
            )
            # Only a bad key for the active translation provider stops everything; a problem with
            # a separate glossary *model* must not prevent translation.
            if is_auth_error and same_creds:
                raise TranslationAborted(str(exc)) from exc
            self.last_glossary_error = f"{self.glossary_model}: {exc}"
            logger.warning(f"Glossary pass failed (continuing without it): {exc}")
            return user_glossary

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
        batch_instruction: Optional[str] = None,
    ) -> List[TranslatableSegment]:
        """Translate segments in batches with sliding context and static prefix caching."""
        if not segments:
            return []

        context = context_segments or segments
        position = {seg.index: pos for pos, seg in enumerate(context)}
        total_segments = len(segments)
        from app.providers.translation.glossary_pruner import prune_glossary_for_batch

        do_rtl = apply_rtl and is_rtl(target_lang)
        from app.pipeline.gender_pretag import is_gendered_target_language

        # Gendered 2nd person: ask which form was written, so it can be checked against the listener
        report_forms = is_gendered_target_language(target_lang) and not is_same_language(source_lang, target_lang)

        # Build the static system+glossary prompt ONCE outside the batch loop whenever per-batch
        # pruning is off, guaranteeing byte-for-byte prefix identity across every call in the job.
        static_system_prompt: Optional[str] = None
        if not self.prune_glossary:
            static_system_prompt = build_subtitle_system_prompt(
                source_lang=source_lang,
                target_lang=target_lang,
                context_notes=context_notes,
                max_chars_per_line=self.max_chars_per_line,
                glossary=glossary,
                convert_to_metric=self.convert_to_metric,
                remove_sdh_speakers=self.remove_sdh_speakers,
                report_addressee_form=report_forms,
                static_glossary=True,
                dual_source_mode=self.dual_source_mode,
                dual_source_primary_lang=self.dual_source_primary_lang,
                dual_source_anchor_lang=self.dual_source_anchor_lang,
                dual_source_pair_mode=self.dual_source_pair_mode,
            )

        for start_idx in range(0, total_segments, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_segments)
            current_batch = segments[start_idx:end_idx]

            first_pos = position.get(current_batch[0].index, 0)
            last_pos = position.get(current_batch[-1].index, len(context) - 1)
            prev_context = context[max(0, first_pos - self.prev_context_size):first_pos]
            lookahead_context = context[last_pos + 1:last_pos + 1 + self.lookahead_size]

            if self.prune_glossary:
                batch_glossary = prune_glossary_for_batch(
                    glossary=glossary,
                    batch_texts=current_batch,
                    previous_context=prev_context,
                    lookahead_context=lookahead_context,
                )
                if (
                    glossary
                    and batch_glossary
                    and self.usage is not None
                    and hasattr(self.usage, "record_local_savings")
                ):
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
                    report_addressee_form=report_forms,
                    dual_source_mode=self.dual_source_mode,
                    dual_source_primary_lang=self.dual_source_primary_lang,
                    dual_source_anchor_lang=self.dual_source_anchor_lang,
                    dual_source_pair_mode=self.dual_source_pair_mode,
                )
            else:
                assert static_system_prompt is not None
                system_prompt = static_system_prompt

            try:
                await self._translate_single_batch(
                    target_batch=current_batch,
                    prev_context=prev_context,
                    lookahead_context=lookahead_context,
                    system_prompt=system_prompt,
                    apply_rtl=do_rtl,
                    report_forms=report_forms,
                    batch_instruction=batch_instruction,
                )
            except TranslationAborted:
                for seg in segments[start_idx:]:
                    if not seg.translated_text:
                        seg.translation_failed = True
                raise

            progress_pct = round((end_idx / total_segments) * 100, 2)
            logger.info(
                f"Translated batch {start_idx + 1}-{end_idx}/{total_segments} ({progress_pct}%) "
                f"via {self.provider}/{self.model_name}"
            )

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
        report_forms: bool = False,
        batch_instruction: Optional[str] = None,
    ) -> None:
        """Translate one batch. Missing indices are retried; leftovers are flagged as failed."""
        pending = list(target_batch)
        for seg in pending:
            seg.translation_failed = False
        schema = TranslationBatchWithFormResponseSchema if report_forms else TranslationBatchResponseSchema

        def _ctx_item(
            seg: TranslatableSegment,
            *,
            with_translated: bool = False,
            with_meta: bool = False,
        ) -> Dict[str, Any]:
            item: Dict[str, Any] = {"id": seg.index}
            if with_meta:
                item["segment_type"] = seg.segment_type
                item["duration_sec"] = seg.duration
            item["original_text"] = seg.original_text
            if getattr(seg, "reference_text", None):
                item["ar_reference"] = seg.reference_text
            if getattr(seg, "dual_source_kind", None):
                item["dual_source_kind"] = seg.dual_source_kind
            if with_translated and seg.translated_text:
                item["translated_text"] = strip_directional_marks(seg.translated_text)
            sp = getattr(seg, "speaker_info", None) or getattr(seg, "speaker", None)
            if sp:
                item["speaker"] = sp
            addr = getattr(seg, "addressee_info", None)
            if addr:
                item["addressee"] = addr
            else:
                guess = getattr(seg, "addressee_guess", None)
                if guess:
                    item["addressee_guess"] = guess
            return item

        for attempt in range(1, self.max_retries + 1):
            payload: Dict[str, Any] = {}
            if batch_instruction and batch_instruction.strip():
                payload["batch_instruction"] = batch_instruction.strip()
            payload["previous_context"] = [_ctx_item(seg, with_translated=True) for seg in prev_context]
            payload["target_batch"] = [_ctx_item(seg, with_meta=True) for seg in pending]
            payload["lookahead_context"] = [_ctx_item(seg) for seg in lookahead_context]
            try:
                response = await self._generate(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    system_prompt,
                    schema,
                    temperature=0.2,
                )
                parsed = schema.model_validate_json(response.text)
                expected_ids = [seg.index for seg in pending]
                expected_pos = {sid: pos for pos, sid in enumerate(expected_ids)}
                returned_ids = [item.id for item in parsed.translations]

                # Validate strict 1:1 ID correspondence so merged/shifted cues never corrupt neighbours
                if len(set(returned_ids)) != len(returned_ids):
                    raise ValueError(f"Batch 1:1 ID mismatch: duplicate segment IDs in {returned_ids}")
                unknown_ids = [rid for rid in returned_ids if rid not in expected_pos]
                if unknown_ids:
                    raise ValueError(f"Batch 1:1 ID mismatch: unexpected segment IDs {unknown_ids} (expected {expected_ids})")
                positions = [expected_pos[rid] for rid in returned_ids]
                if positions != sorted(positions):
                    raise ValueError(f"Batch 1:1 ID mismatch: out-of-order segment IDs {returned_ids} (expected {expected_ids})")
                if len(pending) >= 6 and returned_ids != expected_ids:
                    missing_ids = [sid for sid in expected_ids if sid not in set(returned_ids)]
                    raise ValueError(
                        f"Batch 1:1 ID mismatch (expected {len(expected_ids)} items, got {len(returned_ids)}); "
                        f"omitted segment indices: {missing_ids}"
                    )

                translation_map = {item.id: item.text for item in parsed.translations}
                form_map = {
                    item.id: str(getattr(item, "addressee_form", "") or "").strip().lower()
                    for item in parsed.translations
                }

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
                        form = form_map.get(seg.index)
                        seg.addressee_form = form if form in _ADDRESSEE_FORMS else None
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
                        "1:1 id mismatch",
                    )
                )
                if len(pending) > 8 and (is_size_or_timeout or attempt >= self.max_retries):
                    mid = len(pending) // 2
                    left, right = pending[:mid], pending[mid:]
                    logger.info(
                        f"Splitting failed batch of {len(pending)} segments into {len(left)} + {len(right)}"
                    )
                    await self._translate_single_batch(
                        left, prev_context, right[:self.lookahead_size], system_prompt, apply_rtl, report_forms,
                        batch_instruction=batch_instruction,
                    )
                    right_prev = (prev_context + left)[-self.prev_context_size:] if self.prev_context_size > 0 else []
                    await self._translate_single_batch(
                        right, right_prev, lookahead_context, system_prompt, apply_rtl, report_forms,
                        batch_instruction=batch_instruction,
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
