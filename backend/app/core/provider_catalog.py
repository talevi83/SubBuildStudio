"""
Multi-provider AI pricing, capability filtering, suitability warnings, and per-episode/movie cost benchmarks.

Supports Google Gemini, OpenAI, Anthropic Claude, and Ollama (both local $0.00 models and
Ollama Cloud models). Keeps backwards compatibility with ``app.core.model_catalog``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from math import ceil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.provider_keys import ProviderName, PROVIDERS

EPISODE_DURATION_SECONDS = 45 * 60   # 45-minute TV episode
MOVIE_DURATION_SECONDS = 120 * 60    # 2-hour feature film

# Token volume assumptions matching model_catalog.estimate_job_cost
_SEGMENTS_PER_MIN = 14
_TRANSLATION_BATCH = 120
_SYS_PROMPT_TOKENS = 1900
_TR_IN_PER_SEG = 36
_TR_OUT_PER_SEG = 32
_GL_IN_BASE = 350
_GL_IN_PER_SEG = 14
_GL_OUT = 1500
_VISION_FRAME_INTERVAL = 2.0
_VISION_FRAMES_KEPT_RATIO = 0.5
_VISION_MAX_FRAMES = 660
_VISION_FOREIGN_SHARE = 0.10
_VISION_FOREIGN_FPS = 2.0
_VISION_BATCH = 12
_VISION_PROMPT = 450
_VISION_OUT = 150
_VISION_IMG_TOKENS = 560  # medium resolution default (Google Gemini 3)

_EXCLUDE_MODEL_TOKENS = (
    "embedding",
    "embed",
    "-tts",
    "tts-",
    "whisper",
    "transcribe",
    "dall-e",
    "gpt-image",
    "chatgpt-image",
    "-image",
    "nano-banana",
    "gemini-omni",
    "-customtools",
    "imagen",
    "sora",
    "veo",
    "moderation",
    "realtime",
    "computer-use",
    "babbage",
    "davinci",
    "search-preview",
    "search-api",
    "-search",
    "audio-preview",
    "-audio",
    "-live",
    "-coder",
    "codellama",
    "starcoder",
    "codestral",
    "devstral",
    "codegemma",
    "codex-",
    "-codex",
    "robotics",
    "deep-research",
    "stable-diffusion",
    "prompt-generator",
    "gpt-3.5",
    "claude-3-haiku",
    "claude-instant",
    "gemini-1.",
    "gemini-2.",
    "gemini-2.0-flash-lite",
    "gemma-3-1b",
    "gemma-2-2b",
)

_OLLAMA_VISION_HINTS = (
    "vision",
    "-vl",
    "vl-",
    "vl:",
    "qwen2.5vl",
    "qwen3-vl",
    "qwen3.5",
    "llama3.2-vision",
    "llama4",
    "gemma3",
    "gemma4",
    "minicpm-v",
    "pixtral",
    "llava",
    "moondream",
    "bakllava",
    "granite3.2-vision",
)

_SMALL_PARAM_RE = re.compile(r"(?:^|[:\-_])(\d+(?:\.\d+)?)b(?:$|[:\-_])", re.IGNORECASE)


@dataclass(frozen=True)
class PricingRule:
    provider: ProviderName
    prefixes: Tuple[str, ...]
    label: str
    input_per_m: Optional[float]
    output_per_m: Optional[float]
    supports_vision: bool = True
    thinking: Tuple[str, ...] = ()
    default_thinking: str = "default"
    supports_flex: bool = False
    is_local: bool = False
    recommended_for: Tuple[str, ...] = ()
    warnings: Tuple[Tuple[str, str], ...] = ()
    rank: int = 50
    note: str = ""


_RULES: Tuple[PricingRule, ...] = (
    # ---- OpenAI ----
    PricingRule(
        "openai",
        ("gpt-5.4-mini",),
        "GPT-5.4 mini",
        0.75,
        4.50,
        supports_vision=True,
        recommended_for=("translation", "glossary", "vision"),
        rank=11,
        note="Strong quality-to-cost balance for subtitle translation and OCR.",
    ),
    PricingRule(
        "openai",
        ("gpt-5.4-nano",),
        "GPT-5.4 nano",
        0.20,
        1.25,
        supports_vision=True,
        recommended_for=("vision",),
        rank=34,
        note="Ultra-fast low-cost tier suited for on-screen OCR scanning.",
    ),
    PricingRule(
        "openai",
        ("gpt-5.4",),
        "GPT-5.4",
        2.50,
        15.00,
        supports_vision=True,
        recommended_for=("translation", "glossary"),
        rank=16,
        note="Flagship OpenAI 5.4 quality for idioms, tone, and complex dialogue.",
    ),
    PricingRule(
        "openai",
        ("gpt-5-mini", "gpt-5.1-mini", "gpt-5.2-mini", "gpt-5.3-mini", "gpt-5.5-mini"),
        "GPT-5 mini",
        0.25,
        2.00,
        supports_vision=True,
        recommended_for=("translation", "glossary", "vision"),
        rank=10,
        note="Strong quality-to-cost balance for subtitle translation and OCR.",
    ),
    PricingRule(
        "openai",
        ("gpt-5-nano", "gpt-5.1-nano", "gpt-5.2-nano", "gpt-5.3-nano", "gpt-5.5-nano"),
        "GPT-5 nano",
        0.05,
        0.40,
        supports_vision=True,
        recommended_for=("vision",),
        rank=35,
        note="Ultra-fast low-cost tier suited for on-screen OCR scanning.",
    ),
    PricingRule(
        "openai",
        ("gpt-5",),
        "GPT-5",
        1.25,
        10.00,
        supports_vision=True,
        recommended_for=("translation", "glossary"),
        rank=15,
        note="Flagship OpenAI quality for idioms, tone, and complex dialogue.",
    ),
    PricingRule(
        "openai",
        ("gpt-6-luna", "gpt-6.1-luna", "gpt-6.2-luna", "gpt-6-nano", "gpt-6.1-nano", "gpt-6.2-nano"),
        "GPT-6 Luna",
        0.10,
        0.50,
        supports_vision=True,
        recommended_for=("vision",),
        rank=31,
        note="High-volume low-cost GPT-6 tier suited for on-screen OCR scanning.",
    ),
    PricingRule(
        "openai",
        ("gpt-6-mini", "gpt-6.1-mini", "gpt-6.2-mini"),
        "GPT-6 mini",
        0.25,
        2.00,
        supports_vision=True,
        recommended_for=("translation", "glossary", "vision"),
        rank=11,
        note="Fast GPT-6 mini tier for subtitle translation and OCR.",
    ),
    PricingRule(
        "openai",
        ("gpt-6-terra", "gpt-6.1-terra", "gpt-6.2-terra"),
        "GPT-6 Terra",
        1.00,
        5.00,
        supports_vision=True,
        recommended_for=("translation", "glossary", "vision"),
        rank=14,
        note="Balanced GPT-6 visual reasoning and dialogue tier.",
    ),
    PricingRule(
        "openai",
        ("gpt-6-sol", "gpt-6.1-sol", "gpt-6.2-sol"),
        "GPT-6 Sol",
        2.00,
        10.00,
        supports_vision=True,
        recommended_for=("translation", "glossary"),
        rank=18,
        note="Mid-tier balanced GPT-6 model for dialogue translation.",
    ),
    PricingRule(
        "openai",
        ("gpt-6-astra", "gpt-6.1-astra", "gpt-6.2-astra"),
        "GPT-6 Astra",
        10.00,
        50.00,
        supports_vision=True,
        rank=27,
        note="Flagship high-intelligence GPT-6 tier.",
    ),
    PricingRule(
        "openai",
        ("gpt-6",),
        "GPT-6",
        2.00,
        10.00,
        supports_vision=True,
        recommended_for=("translation", "glossary"),
        rank=18,
        note="GPT-6 model family.",
    ),
    PricingRule(
        "openai",
        ("gpt-4.1-mini",),
        "GPT-4.1 mini",
        0.40,
        1.60,
        supports_vision=True,
        recommended_for=("translation", "glossary", "vision"),
        rank=18,
        note="Fast, reliable structured JSON translation and vision.",
    ),
    PricingRule(
        "openai",
        ("gpt-4.1-nano",),
        "GPT-4.1 nano",
        0.10,
        0.40,
        supports_vision=True,
        recommended_for=("vision",),
        rank=40,
        note="Fast low-cost tier suited for on-screen OCR scanning.",
    ),
    PricingRule(
        "openai",
        ("gpt-4.1",),
        "GPT-4.1",
        2.00,
        8.00,
        supports_vision=True,
        rank=22,
    ),
    PricingRule(
        "openai",
        ("gpt-4o-mini",),
        "GPT-4o mini",
        0.15,
        0.60,
        supports_vision=True,
        recommended_for=("vision",),
        rank=32,
        note="OpenAI bills gpt-4o-mini at 2,833 tokens/image (Low/Medium detail) or 14,167 tokens (High). GPT-4.1 mini & Gemini Flash-Lite cost ~4-5x less for frame OCR.",
    ),
    PricingRule(
        "openai",
        ("gpt-4o",),
        "GPT-4o",
        2.50,
        10.00,
        supports_vision=True,
        rank=28,
    ),
    PricingRule(
        "openai",
        ("o4-mini",),
        "o4-mini",
        1.10,
        4.40,
        supports_vision=True,
        rank=30,
    ),
    PricingRule(
        "openai",
        ("o3-mini", "o1-mini"),
        "o-series mini",
        1.10,
        4.40,
        supports_vision=False,
        rank=33,
    ),
    PricingRule(
        "openai",
        ("o3",),
        "o3",
        2.00,
        8.00,
        supports_vision=True,
        rank=29,
    ),
    # ---- Anthropic Claude ----
    PricingRule(
        "anthropic",
        ("claude-sonnet-4-5", "claude-sonnet-4-6", "claude-sonnet-4"),
        "Claude Sonnet 4.5",
        3.00,
        15.00,
        supports_vision=True,
        recommended_for=("translation", "glossary"),
        rank=12,
        note="Top-tier natural Hebrew translation, speaker register, and glossary consistency.",
    ),
    PricingRule(
        "anthropic",
        ("claude-haiku-4-5", "claude-haiku-4"),
        "Claude Haiku 4.5",
        1.00,
        5.00,
        supports_vision=True,
        recommended_for=("translation", "glossary", "vision"),
        rank=16,
        note="Fast and cost-effective Claude model with strong JSON compliance and vision.",
    ),
    PricingRule(
        "anthropic",
        ("claude-opus-4-1", "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4"),
        "Claude Opus 4",
        15.00,
        75.00,
        supports_vision=True,
        rank=26,
        note="Maximum quality for nuanced translation.",
    ),
    PricingRule(
        "anthropic",
        ("claude-3-7-sonnet", "claude-3-5-sonnet"),
        "Claude 3.x Sonnet",
        3.00,
        15.00,
        supports_vision=True,
        rank=24,
    ),
    PricingRule(
        "anthropic",
        ("claude-3-5-haiku",),
        "Claude 3.5 Haiku",
        0.80,
        4.00,
        supports_vision=False,
        rank=34,
        note="Text-only fast Claude model for glossary extraction.",
    ),
    # ---- Fallback Gemini family rules for newly discovered IDs not in model_catalog.CATALOG ----
    PricingRule(
        "google",
        ("gemini-2.5-pro",),
        "Gemini 2.5 Pro",
        1.25,
        10.00,
        supports_vision=True,
        thinking=("low", "medium", "high"),
        default_thinking="medium",
        supports_flex=True,
        rank=25,
    ),
    PricingRule(
        "google",
        ("gemini-2.5-flash-lite",),
        "Gemini 2.5 Flash-Lite",
        0.10,
        0.40,
        supports_vision=True,
        thinking=("off", "low", "medium", "high"),
        default_thinking="off",
        supports_flex=True,
        recommended_for=("vision",),
        rank=38,
    ),
    PricingRule(
        "google",
        ("gemini-2.5-flash",),
        "Gemini 2.5 Flash",
        0.30,
        2.50,
        supports_vision=True,
        thinking=("off", "low", "medium", "high"),
        default_thinking="medium",
        supports_flex=True,
        recommended_for=("translation", "glossary", "vision"),
        rank=14,
    ),
)

_DEFAULT_CATALOG_SEEDS: Dict[ProviderName, Tuple[str, ...]] = {
    "google": (
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-4-argon",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ),
    "openai": (
        "gpt-5-mini",
        "gpt-5",
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-5-nano",
    ),
    "anthropic": (
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-1",
    ),
    "ollama": (
        "qwen2.5:7b",
        "qwen2.5vl:7b",
    ),
}


def default_catalog_seeds(provider: Optional[ProviderName] = None) -> List[Tuple[ProviderName, str]]:
    if provider is not None:
        return [(provider, m) for m in _DEFAULT_CATALOG_SEEDS.get(provider, ())]
    out: List[Tuple[ProviderName, str]] = []
    for p in PROVIDERS:
        out.extend((p, m) for m in _DEFAULT_CATALOG_SEEDS.get(p, ()))
    return out


def infer_provider_from_model(model_id: str) -> Optional[ProviderName]:
    mid = (model_id or "").strip().lower().replace("models/", "")
    if not mid:
        return None
    # Ollama models use `name:tag` syntax (e.g. `gpt-oss:20b`, `gemma4:12b`), `:cloud`/`-cloud`, or `gpt-oss*`.
    # Check Ollama markers before cloud prefixes so `gpt-oss:20b` is never misrouted to OpenAI.
    if ":" in mid or mid.endswith("-cloud") or mid.startswith("gpt-oss"):
        return "ollama"
    if mid.startswith("gemini"):
        return "google"
    if mid.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if mid.startswith("claude-"):
        return "anthropic"
    return None


def is_chat_candidate(provider: ProviderName, model_id: str) -> bool:
    """Return True if ``model_id`` is usable for text/multimodal JSON pipeline calls."""
    mid = (model_id or "").strip().lower().replace("models/", "")
    if not mid:
        return False
    if any(tok in mid for tok in _EXCLUDE_MODEL_TOKENS):
        return False
    if provider == "google":
        from app.core.model_catalog import RETIRED_MODELS

        if mid in RETIRED_MODELS or mid.endswith("-latest"):
            return False
        return mid.startswith("gemini")
    if provider == "openai":
        if mid.startswith("gpt-oss") or ":" in mid:
            return False
        # OpenAI "-pro" variants (gpt-5-pro, gpt-5.4-pro, o1-pro, o3-pro) are /v1/responses-only,
        # unsupported by /v1/chat/completions used by this pipeline.
        if mid.endswith("-pro") or "-pro-" in mid:
            return False
        return mid.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
    if provider == "anthropic":
        return mid.startswith("claude-")
    if provider == "ollama":
        return True
    return True


def _is_ollama_cloud_model(model_id: str, base_url: Optional[str] = None) -> bool:
    mid = (model_id or "").strip().lower()
    url = (base_url or "").strip().lower()
    if ":cloud" in mid or mid.endswith("-cloud"):
        return True
    if url and not any(h in url for h in ("localhost", "127.0.0.1", "[::1]", "host.docker.internal")):
        return True
    return False


def _match_rule(provider: ProviderName, model_id: str) -> Optional[PricingRule]:
    mid = (model_id or "").strip().lower().replace("models/", "")
    best: Optional[PricingRule] = None
    best_len = -1
    is_mini_or_nano = "-mini" in mid or "-nano" in mid or "-luna" in mid
    for rule in _RULES:
        if rule.provider != provider:
            continue
        for prefix in rule.prefixes:
            if mid != prefix and not mid.startswith((prefix + "-", prefix + ".", prefix + ":")):
                continue
            if is_mini_or_nano and not any(t in prefix for t in ("-mini", "-nano", "-luna")):
                continue
            if len(prefix) > best_len:
                best = rule
                best_len = len(prefix)
    return best


def _gemini_catalog_entry(model_id: str):
    from app.core.model_catalog import get_model

    return get_model(model_id)


def _ollama_param_billions(
    model_id: str,
    details: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    m = _SMALL_PARAM_RE.search((model_id or "").lower())
    if not m and isinstance(details, dict):
        param_size = str(details.get("parameter_size") or "").strip().lower().replace(" ", "")
        if param_size:
            m = _SMALL_PARAM_RE.search(param_size)
            if not m and param_size.endswith("m"):
                try:
                    return float(param_size[:-1]) / 1000.0
                except ValueError:
                    pass
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _ollama_supports_vision(model_id: str, capabilities: Optional[Sequence[str]] = None) -> bool:
    if capabilities:
        caps = {c.lower() for c in capabilities}
        if "vision" in caps:
            return True
    mid = (model_id or "").lower()
    return any(hint in mid for hint in _OLLAMA_VISION_HINTS)


def estimated_image_tokens(
    provider: str,
    model_id: str,
    resolution: str = "medium",
) -> int:
    """Estimate per-frame (640x360 JPEG) input tokens using each provider's actual vision billing."""
    from app.core.model_catalog import (
        IMAGE_TOKENS_2_5,
        IMAGE_TOKENS_BY_RESOLUTION,
        family_of,
    )

    prov = (provider or "google").strip().lower()
    mid = (model_id or "").replace("models/", "").strip().lower()
    res = (resolution or "medium").strip().lower()

    if prov == "google":
        if family_of(mid) == "2.5":
            return IMAGE_TOKENS_2_5
        return IMAGE_TOKENS_BY_RESOLUTION.get(res, IMAGE_TOKENS_BY_RESOLUTION["medium"])

    if prov == "openai":
        high = res == "high"
        if mid.startswith("gpt-4o-mini"):
            # OpenAI applies a 33.33x multiplier so gpt-4o-mini image USD matches gpt-4o
            return 14167 if high else 2833
        if mid.startswith(("gpt-4o", "chatgpt-4o")):
            return 425 if high else 85
        if mid.startswith(("gpt-4.1-mini", "gpt-5")) and "mini" in mid:
            return 390 if high else 235
        if mid.startswith(("gpt-4.1-nano", "gpt-5")) and "nano" in mid:
            return 590 if high else 355
        if mid.startswith("gpt-6"):
            if "mini" in mid:
                return 390 if high else 235
            if "nano" in mid or "luna" in mid:
                return 590 if high else 355
        return 320 if high else 160

    if prov == "anthropic":
        # Anthropic bills roughly (width * height) / 750 (~308 tokens for 640x360)
        return 308

    return _VISION_IMG_TOKENS


def estimated_reasoning_tokens(
    provider: str,
    model_id: str,
    thinking: str = "default",
) -> int:
    """Estimate hidden thinking/reasoning output tokens per request."""
    from app.core.model_catalog import estimated_thinking_tokens

    prov = (provider or "google").strip().lower()
    mid = (model_id or "").replace("models/", "").strip().lower()
    if prov == "google":
        return estimated_thinking_tokens(mid, thinking)
    if prov == "openai" and mid.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        lvl = (thinking or "default").strip().lower()
        return {"off": 80, "minimal": 80, "default": 650, "low": 650, "medium": 1800, "high": 4500}.get(lvl, 650)
    return 0


def _benchmark_cost_for_duration(
    duration_seconds: float,
    input_per_m: Optional[float],
    output_per_m: Optional[float],
    task: str = "translation",
    thinking_tokens_per_req: int = 0,
    input_long_per_m: Optional[float] = None,
    output_long_per_m: Optional[float] = None,
    flex_factor: Optional[float] = None,
    provider: ProviderName = "google",
    model_id: str = "",
    resolution: str = "medium",
) -> Optional[float]:
    if input_per_m is None or output_per_m is None:
        return None
    minutes = max(0.0, duration_seconds) / 60.0
    segments = max(1, round(minutes * _SEGMENTS_PER_MIN))

    def _cost(reqs: int, in_per: int, out_per: int) -> float:
        r_in = input_long_per_m if (in_per > 200_000 and input_long_per_m) else input_per_m
        r_out = output_long_per_m if (in_per > 200_000 and output_long_per_m) else output_per_m
        one = in_per / 1e6 * r_in + (out_per + thinking_tokens_per_req) / 1e6 * r_out
        if flex_factor:
            one *= flex_factor
        return one * reqs

    if task == "vision":
        frames = min(_VISION_MAX_FRAMES, int(duration_seconds / _VISION_FRAME_INTERVAL * _VISION_FRAMES_KEPT_RATIO))
        foreign_crops = int(duration_seconds * _VISION_FOREIGN_SHARE * _VISION_FOREIGN_FPS)
        batches = (ceil(frames / _VISION_BATCH) if frames else 0) + (
            ceil(foreign_crops / _VISION_BATCH) if foreign_crops else 0
        )
        img_tokens = estimated_image_tokens(provider, model_id, resolution)
        total = _cost(batches, _VISION_BATCH * img_tokens + _VISION_PROMPT, _VISION_OUT)
        return round(total, 4)

    # Translation + glossary benchmark
    batch_sz = 50 if provider in ("openai", "anthropic") else (40 if provider == "ollama" else _TRANSLATION_BATCH)
    gl_in = _GL_IN_BASE + segments * _GL_IN_PER_SEG
    gl_cost = _cost(1, gl_in, _GL_OUT)
    tr_batches = ceil(segments / batch_sz)
    tr_in = _SYS_PROMPT_TOKENS + 350 + batch_sz * _TR_IN_PER_SEG + 8 * 40
    tr_cost = _cost(tr_batches, tr_in, batch_sz * _TR_OUT_PER_SEG)
    if task == "glossary":
        return round(gl_cost, 4)
    return round(gl_cost + tr_cost, 4)


def pricing_for_model(
    provider: ProviderName,
    model_id: str,
    day: Optional[date] = None,
    base_url: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], bool]:
    """Return ``(input_per_m, output_per_m, input_long_per_m, output_long_per_m, is_local)``."""
    if provider == "google":
        g = _gemini_catalog_entry(model_id)
        if g is not None:
            p = g.price_on(day)
            return p.input, p.output, p.input_long, p.output_long, False
    if provider == "ollama":
        if _is_ollama_cloud_model(model_id, base_url=base_url):
            return None, None, None, None, False
        return 0.0, 0.0, None, None, True
    rule = _match_rule(provider, model_id)
    if rule is not None:
        return rule.input_per_m, rule.output_per_m, None, None, rule.is_local
    return None, None, None, None, False


def build_model_entry(
    provider: ProviderName,
    model_id: str,
    *,
    available: bool = False,
    key_ids: Optional[Iterable[str]] = None,
    base_url: Optional[str] = None,
    capabilities: Optional[Sequence[str]] = None,
    details: Optional[Dict[str, Any]] = None,
    day: Optional[date] = None,
) -> Dict[str, Any]:
    """Build a complete UI/API catalog descriptor for any provider model."""
    from app.core.model_catalog import (
        FLEX_PRICE_FACTOR,
        allowed_thinking,
        estimated_thinking_tokens,
    )

    clean_id = (model_id or "").replace("models/", "").strip()
    in_p, out_p, in_long, out_long, is_local = pricing_for_model(provider, clean_id, day=day, base_url=base_url)

    g_info = _gemini_catalog_entry(clean_id) if provider == "google" else None
    rule = _match_rule(provider, clean_id)

    label = clean_id
    family = provider
    thinking: List[str] = ["default"]
    default_level = "default"
    supports_media_res = False
    supports_flex = False
    preview = False
    note = ""
    rank = 50
    params_b: Optional[float] = None
    recommended_for: List[str] = []
    warnings_by_task: Dict[str, str] = {}
    suitable_tasks: List[str] = ["translation", "glossary"]
    price_change = None

    if g_info is not None:
        label = g_info.label
        family = g_info.family
        thinking = list(allowed_thinking(g_info.id))
        default_level = g_info.default_level
        supports_media_res = g_info.family != "2.5"
        supports_flex = g_info.flex
        preview = g_info.preview
        note = g_info.note
        change = g_info.next_price_change(day)
        if change:
            price_change = {"after": change[0].isoformat(), "input_per_m": change[1].input, "output_per_m": change[1].output}
        if clean_id in ("gemini-3.8-flash", "gemini-3.6-flash"):
            suitable_tasks = ["translation", "glossary", "vision"]
            recommended_for = ["translation", "glossary", "vision"]
            rank = 8 if clean_id == "gemini-3.8-flash" else 20
        elif clean_id in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
            suitable_tasks = ["glossary", "vision"]
            recommended_for = ["vision"] if clean_id == "gemini-3.5-flash-lite" else []
            rank = 30 if clean_id == "gemini-3.5-flash-lite" else 42
        elif clean_id in ("gemini-4-argon", "gemini-3.1-pro-preview"):
            suitable_tasks = ["translation"]
            recommended_for = ["translation"]
            rank = 19
        else:
            suitable_tasks = list(g_info.tasks)
    elif provider == "ollama":
        is_cloud = _is_ollama_cloud_model(clean_id, base_url=base_url)
        label = f"{clean_id} (Cloud)" if is_cloud else clean_id
        has_vision = _ollama_supports_vision(clean_id, capabilities=capabilities)
        params_b = _ollama_param_billions(clean_id, details=details)
        if params_b is not None and params_b < 6.5:
            # Models smaller than ~7B cannot reliably preserve Hebrew grammar & strict JSON batch schemas
            suitable_tasks = ["vision"] if has_vision else []
            rank = 45
        else:
            suitable_tasks = ["translation", "glossary"] + (["vision"] if has_vision else [])
            if any(k in clean_id.lower() for k in ("qwen2.5", "qwen3", "gemma3", "gemma4", "llama3.3", "mistral", "deepseek", "gpt-oss")):
                recommended_for = list(suitable_tasks)
                rank = 17 if is_cloud else 21
        if is_cloud:
            note = "Runs on Ollama Cloud endpoint; billed according to your Ollama Cloud plan."
        else:
            note = "Runs 100% locally via Ollama ($0.00 API cost)."
    elif rule is not None:
        label = rule.label if clean_id == rule.prefixes[0] else f"{rule.label} ({clean_id})"
        thinking = ["default", *rule.thinking]
        default_level = rule.default_thinking
        supports_flex = rule.supports_flex
        supports_media_res = provider == "google" and not clean_id.startswith("gemini-2.")
        note = rule.note
        rank = rule.rank
        recommended_for = list(rule.recommended_for)
        warnings_by_task = dict(rule.warnings)
        low_id = clean_id.lower()
        if provider == "google":
            if "flash-lite" in low_id:
                suitable_tasks = ["glossary", "vision"]
            elif "pro" in low_id:
                suitable_tasks = ["translation"]
            else:
                suitable_tasks = ["translation", "glossary", "vision"]
        elif provider == "openai":
            if "nano" in low_id or "luna" in low_id:
                suitable_tasks = ["vision"]
            elif low_id.startswith("gpt-4o-mini"):
                suitable_tasks = ["glossary", "vision"]
            else:
                suitable_tasks = ["translation", "glossary"] + (["vision"] if rule.supports_vision else [])
        elif provider == "anthropic":
            if "claude-3-5-haiku" in low_id:
                suitable_tasks = ["glossary"]
            elif "opus" in low_id:
                suitable_tasks = ["translation"]
            else:
                suitable_tasks = ["translation", "glossary"] + (["vision"] if rule.supports_vision else [])
        else:
            suitable_tasks = ["translation", "glossary"] + (["vision"] if rule.supports_vision else [])
    else:
        low = clean_id.lower()
        if provider in ("openai", "anthropic", "google") or _ollama_supports_vision(low, capabilities=capabilities):
            suitable_tasks.append("vision")
        note = "Discovered from your API key; pricing not in built-in table."

    suitability_by_task: Dict[str, str] = {}
    for task in ("translation", "glossary", "vision"):
        if task not in suitable_tasks:
            suitability_by_task[task] = "unsuitable"
        elif task in warnings_by_task:
            suitability_by_task[task] = "warning"
        elif task in recommended_for:
            suitability_by_task[task] = "recommended"
        else:
            suitability_by_task[task] = "suitable"

    think_tokens = estimated_reasoning_tokens(provider, clean_id, default_level)
    ep_cost = _benchmark_cost_for_duration(
        EPISODE_DURATION_SECONDS, in_p, out_p, task="translation",
        thinking_tokens_per_req=think_tokens, input_long_per_m=in_long, output_long_per_m=out_long,
        provider=provider, model_id=clean_id,
    )
    mv_cost = _benchmark_cost_for_duration(
        MOVIE_DURATION_SECONDS, in_p, out_p, task="translation",
        thinking_tokens_per_req=think_tokens, input_long_per_m=in_long, output_long_per_m=out_long,
        provider=provider, model_id=clean_id,
    )
    ep_vision = _benchmark_cost_for_duration(
        EPISODE_DURATION_SECONDS, in_p, out_p, task="vision",
        thinking_tokens_per_req=think_tokens, input_long_per_m=in_long, output_long_per_m=out_long,
        provider=provider, model_id=clean_id,
    )
    mv_vision = _benchmark_cost_for_duration(
        MOVIE_DURATION_SECONDS, in_p, out_p, task="vision",
        thinking_tokens_per_req=think_tokens, input_long_per_m=in_long, output_long_per_m=out_long,
        provider=provider, model_id=clean_id,
    )

    return {
        "id": clean_id,
        "provider": provider,
        "label": label,
        "family": family,
        "preview": preview,
        "available": available,
        "key_ids": list(dict.fromkeys(key_ids or [])),
        "is_local": is_local,
        "param_billions": params_b,
        "thinking": thinking,
        "default_level": default_level,
        "input_per_m": in_p,
        "output_per_m": out_p,
        "input_long_per_m": in_long,
        "output_long_per_m": out_long,
        "price_change": price_change,
        "episode_cost_usd": ep_cost,
        "movie_cost_usd": mv_cost,
        "episode_vision_cost_usd": ep_vision,
        "movie_vision_cost_usd": mv_vision,
        "supports_media_resolution": supports_media_res,
        "tasks": suitable_tasks,
        "recommended_for": recommended_for,
        "warnings_by_task": warnings_by_task,
        "suitability_by_task": suitability_by_task,
        "supports_flex": supports_flex,
        "flex_price_factor": FLEX_PRICE_FACTOR if supports_flex else None,
        "rank": rank,
        "note": note,
    }
