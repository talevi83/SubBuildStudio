"""
Unified async JSON / Vision-JSON client supporting Google Gemini, OpenAI, Anthropic, and Ollama.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx
from pydantic import BaseModel

from app.core.model_catalog import UsageTracker
from app.core.provider_keys import OLLAMA_LOCAL_DEFAULT_URL, ProviderName, resolve_secret_key

logger = logging.getLogger(__name__)

ContentPiece = Union[str, Tuple[bytes, str]]  # text or (image_bytes, mime_type)

_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class ProviderSpec:
    provider: ProviderName
    model: str
    api_key: str
    base_url: Optional[str] = None
    key_id: Optional[str] = None
    thinking: str = "default"
    flex: bool = False


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: ProviderName
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


def resolve_provider_spec(
    provider: ProviderName,
    model: str,
    *,
    key_id: Optional[str] = None,
    api_key_override: Optional[str] = None,
    thinking: str = "default",
    flex: bool = False,
) -> ProviderSpec:
    """Resolve a runnable ``ProviderSpec`` from explicit overrides or the key vault."""
    clean_override = (api_key_override or "").strip()
    if clean_override:
        return ProviderSpec(
            provider=provider,
            model=model,
            api_key=clean_override,
            base_url=OLLAMA_LOCAL_DEFAULT_URL if provider == "ollama" else None,
            key_id=key_id,
            thinking=thinking,
            flex=flex if provider == "google" else False,
        )

    resolved = resolve_secret_key(provider, key_id=key_id)
    if resolved is None:
        raise ValueError(f"No API key configured for provider '{provider}'")
    return ProviderSpec(
        provider=provider,
        model=model,
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        key_id=resolved.id,
        thinking=thinking,
        flex=flex if provider == "google" else False,
    )


def _inline_json_schema(schema_cls: type[BaseModel]) -> Dict[str, Any]:
    """Return a flat JSON Schema with Pydantic ``$defs``/``$ref`` references inlined."""
    raw = schema_cls.model_json_schema()
    defs = raw.get("$defs") or raw.get("definitions") or {}

    def _resolve(node: Any, seen: Optional[set[str]] = None) -> Any:
        seen = seen or set()
        if isinstance(node, list):
            return [_resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            name = ref.rsplit("/", 1)[-1]
            if name in defs and name not in seen:
                merged = dict(defs[name])
                for k, v in node.items():
                    if k != "$ref":
                        merged[k] = v
                return _resolve(merged, seen | {name})
        return {
            k: _resolve(v, seen)
            for k, v in node.items()
            if k not in ("$defs", "definitions")
        }

    resolved = _resolve(raw)
    return resolved if isinstance(resolved, dict) else {"type": "object"}


def _openai_strict_json_schema(schema_cls: type[BaseModel]) -> Dict[str, Any]:
    """Build an OpenAI Structured Outputs strict JSON Schema (`additionalProperties: false`, all keys required)."""
    inlined = _inline_json_schema(schema_cls)

    def _strictify(node: Any) -> Any:
        if isinstance(node, list):
            return [_strictify(item) for item in node]
        if not isinstance(node, dict):
            return node
        cleaned = {
            k: _strictify(v)
            for k, v in node.items()
            if k not in ("title", "default")
        }
        props = cleaned.get("properties")
        if cleaned.get("type") == "object" or isinstance(props, dict):
            cleaned["type"] = "object"
            props_dict = props if isinstance(props, dict) else {}
            cleaned["properties"] = props_dict
            cleaned["required"] = list(props_dict.keys())
            cleaned["additionalProperties"] = False
        return cleaned

    out = _strictify(inlined)
    return out if isinstance(out, dict) else {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def _schema_instruction(schema_cls: Optional[type[BaseModel]]) -> str:
    if schema_cls is None:
        return ""
    try:
        schema_json = json.dumps(_inline_json_schema(schema_cls), ensure_ascii=False)
        return (
            "\n\nReturn ONLY a single valid JSON object matching this JSON Schema "
            "(include every required top-level key; do NOT wrap or substitute keys like 'summary' or 'status'; "
            "no markdown fences, no commentary):\n" + schema_json
        )
    except Exception:
        return "\n\nReturn ONLY a single valid JSON object (no markdown fences, no commentary)."


def _strip_json_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text and not text.startswith("{"):
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1].strip()
    return text


def _retry_delay_seconds(resp: Optional[httpx.Response], attempt: int) -> float:
    base = min(12.0, 1.5 * (2 ** attempt))
    if resp is not None and resp.status_code == 429:
        ra = (resp.headers.get("retry-after") or "").strip()
        if ra:
            try:
                return max(1.0, min(20.0, float(ra)))
            except ValueError:
                pass
    return base


async def _post_json_with_retry(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float,
    provider_label: str,
    max_attempts: int = 4,
) -> Dict[str, Any]:
    last_exc_desc: str = ""
    client_timeout = httpx.Timeout(timeout, connect=15.0, read=timeout, write=30.0, pool=15.0)
    async with httpx.AsyncClient(timeout=client_timeout) as client:
        for attempt in range(max_attempts):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in _RETRYABLE_STATUS and attempt < max_attempts - 1:
                    wait_s = _retry_delay_seconds(resp, attempt)
                    logger.warning(f"{provider_label} returned HTTP {resp.status_code}; retrying in {wait_s:.1f}s")
                    await asyncio.sleep(wait_s)
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"{provider_label} HTTP {resp.status_code}: {resp.text[:400]}")
                try:
                    data = resp.json()
                except ValueError:
                    # e.g. a proxy's or captive portal's HTML page answering with 200
                    raise RuntimeError(
                        f"{provider_label} returned a response that is not JSON "
                        f"(HTTP {resp.status_code}, {resp.headers.get('content-type') or 'no content type'}): "
                        f"{resp.text[:200]!r}"
                    ) from None
                if not isinstance(data, dict):
                    raise RuntimeError(f"{provider_label} returned unexpected JSON ({type(data).__name__})")
                return data
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc_desc = str(exc).strip() or type(exc).__name__
                if attempt < max_attempts - 1:
                    wait_s = min(8.0, 1.5 * (2 ** attempt))
                    logger.warning(f"{provider_label} network error ({last_exc_desc}); retrying in {wait_s:.1f}s")
                    await asyncio.sleep(wait_s)
                    continue
                raise RuntimeError(f"{provider_label} request failed ({last_exc_desc})") from exc
    raise RuntimeError(f"{provider_label} request failed ({last_exc_desc})")


async def _call_google(
    spec: ProviderSpec,
    *,
    system_instruction: str,
    pieces: Sequence[ContentPiece],
    schema_cls: Optional[type[BaseModel]],
    temperature: float,
    max_output_tokens: int,
    media_resolution: Optional[str],
    usage: Optional[UsageTracker],
    stage: str,
    flex_state: Any,
    timeout: float,
) -> LLMResponse:
    from google import genai
    from google.genai import types
    from app.core.model_catalog import default_temperature, media_resolution as resolve_media_res, thinking_config
    from app.providers import gemini_calls

    client = genai.Client(api_key=spec.api_key)
    contents: List[Any] = []
    for piece in pieces:
        if isinstance(piece, tuple):
            data, mime = piece
            contents.append(types.Part.from_bytes(data=data, mime_type=mime))
        else:
            contents.append(piece)

    t_cfg = thinking_config(spec.model, spec.thinking)
    # Gemini counts thinking tokens inside max_output_tokens; reserve ample headroom when thinking is on
    effective_max_tokens = (
        max(max_output_tokens, 16384) if getattr(t_cfg, "thinking_budget", None) != 0 else max_output_tokens
    )
    cfg_kwargs: Dict[str, Any] = {
        "system_instruction": system_instruction,
        "response_mime_type": "application/json",
        "max_output_tokens": effective_max_tokens,
        "thinking_config": t_cfg,
    }
    if schema_cls is not None:
        cfg_kwargs["response_schema"] = schema_cls
    temp = default_temperature(spec.model, temperature)
    if temp is not None:
        cfg_kwargs["temperature"] = temp
    if media_resolution:
        m_res = resolve_media_res(spec.model, media_resolution)
        if m_res is not None:
            cfg_kwargs["media_resolution"] = m_res

    config = types.GenerateContentConfig(**cfg_kwargs)
    response = await gemini_calls.generate_content(
        client,
        model=spec.model,
        contents=contents,
        config=config,
        flex=spec.flex,
        usage=usage,
        stage=stage,
        flex_state=flex_state,
        timeout=timeout,
    )
    meta = getattr(response, "usage_metadata", None)
    return LLMResponse(
        text=_strip_json_fences(getattr(response, "text", "") or ""),
        provider="google",
        model=spec.model,
        input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
        thinking_tokens=int(getattr(meta, "thoughts_token_count", 0) or 0),
    )


def _openai_reasoning_effort(model_id: str, thinking: str) -> str:
    lvl = (thinking or "default").strip().lower()
    if model_id.startswith(("gpt-5", "gpt-6")):
        return {
            "off": "minimal",
            "minimal": "minimal",
            "default": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
        }.get(lvl, "low")
    return {
        "off": "low",
        "minimal": "low",
        "default": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }.get(lvl, "low")


async def _call_openai(
    spec: ProviderSpec,
    *,
    system_instruction: str,
    pieces: Sequence[ContentPiece],
    schema_cls: Optional[type[BaseModel]],
    temperature: float,
    max_output_tokens: int,
    media_resolution: Optional[str] = "medium",
    usage: Optional[UsageTracker],
    stage: str,
    timeout: float,
) -> LLMResponse:
    root = (spec.base_url or "https://api.openai.com/v1").rstrip("/")
    url = root if root.endswith("/chat/completions") else f"{root}/chat/completions"
    headers = {"Authorization": f"Bearer {spec.api_key}", "Content-Type": "application/json"}
    image_detail = "high" if (media_resolution or "medium").strip().lower() == "high" else "low"

    has_images = any(isinstance(p, tuple) for p in pieces)
    if has_images:
        user_content: Any = []
        for piece in pieces:
            if isinstance(piece, tuple):
                data, mime = piece
                b64 = base64.b64encode(data).decode("ascii")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": image_detail},
                })
            else:
                user_content.append({"type": "text", "text": piece})
    else:
        user_content = "\n".join(p for p in pieces if isinstance(p, str))

    low_model = spec.model.lower()
    is_reasoning = low_model.startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))
    # OpenAI counts reasoning tokens inside max_completion_tokens; reserve ample headroom
    effective_cap = max(max_output_tokens, 24576) if is_reasoning else max_output_tokens

    sys_text = system_instruction + _schema_instruction(schema_cls)
    response_format: Dict[str, Any] = {"type": "json_object"}
    if schema_cls is not None:
        try:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cls.__name__,
                    "strict": True,
                    "schema": _openai_strict_json_schema(schema_cls),
                },
            }
        except Exception:
            response_format = {"type": "json_object"}

    payload: Dict[str, Any] = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_content},
        ],
        "response_format": response_format,
        "max_completion_tokens": effective_cap,
    }
    if is_reasoning:
        payload["reasoning_effort"] = _openai_reasoning_effort(low_model, spec.thinking)
    else:
        payload["temperature"] = temperature

    try:
        data = await _post_json_with_retry(url, headers, payload, timeout=timeout, provider_label="OpenAI")
    except RuntimeError as exc:
        if response_format.get("type") == "json_schema" and "HTTP 400" in str(exc):
            fallback_payload = {**payload, "response_format": {"type": "json_object"}}
            data = await _post_json_with_retry(url, headers, fallback_payload, timeout=timeout, provider_label="OpenAI")
        else:
            raise
    choices = data.get("choices") or []
    choice0 = choices[0] if choices else {}
    text = ((choice0.get("message") or {}).get("content") or "") if choice0 else ""
    finish_reason = (choice0.get("finish_reason") or "").lower()
    raw_usage = data.get("usage") or {}
    in_tok = int(raw_usage.get("prompt_tokens") or 0)
    out_tok = int(raw_usage.get("completion_tokens") or 0)
    think_tok = int(((raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
    if usage is not None:
        usage.record_counts(stage, "openai", spec.model, in_tok, max(0, out_tok - think_tok), think_tok)
    if not text.strip() and finish_reason == "length":
        raise RuntimeError(
            f"OpenAI {spec.model} exhausted max_completion_tokens ({effective_cap}) before emitting JSON output"
        )
    return LLMResponse(
        text=_strip_json_fences(text),
        provider="openai",
        model=spec.model,
        input_tokens=in_tok,
        output_tokens=max(0, out_tok - think_tok),
        thinking_tokens=think_tok,
    )


_ANTHROPIC_MAJOR_VERSION_RE = re.compile(r"^claude-[a-z]+-(\d+)(?:$|[.\-])")


def _anthropic_supports_temperature(model_id: str) -> bool:
    """Return False for newer Claude models where Anthropic deprecated sampling parameters."""
    mid = (model_id or "").strip().lower()
    if not mid:
        return True
    if mid.startswith((
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-haiku-5",
        "claude-fable",
        "claude-opus-4-7",
        "claude-opus-4-8",
    )):
        return False
    m = _ANTHROPIC_MAJOR_VERSION_RE.match(mid)
    if m:
        try:
            if int(m.group(1)) >= 5:
                return False
        except ValueError:
            pass
    return True


_ANTHROPIC_EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Lowest-thinking request fields per Claude family, verified against the Anthropic thinking docs:
#   * Haiku 5.x, Sonnet 5 and Opus 5 accept `thinking: {"type": "disabled"}` (at effort <= high);
#   * Sonnet 5.5 rejects "disabled"; its lowest setting is `{"type": "between_tools"}`;
#   * Opus 5.5 and Fable/Mythos always think, so only a low `output_config.effort` reduces it.
# Claude 4.x and older do not think unless asked, so they get no extra field.


def anthropic_thinking_fields(model_id: str, level: str = "default") -> Dict[str, Any]:
    """Extra Messages API fields that keep a structured JSON call from spending output on thinking.

    Claude 5.x models think by default ("adaptive thinking", medium effort) and bill those tokens as
    output even though the text is never returned: in job_320de88c99 that was ~80% of the
    303,894 translation output tokens. `level` is the user's thinking setting: "default" (and
    "off"/"none"/"minimal") asks for the lowest thinking the model allows; "low".."max" pass an
    explicit effort and leave adaptive thinking on.
    """
    mid = (model_id or "").strip().lower()
    lvl = (level or "default").strip().lower()
    if not mid.startswith("claude-"):
        return {}
    always_thinks = mid.startswith(("claude-opus-5-5", "claude-fable", "claude-mythos"))
    sonnet_55 = mid.startswith("claude-sonnet-5-5")
    gen5 = mid.startswith(("claude-haiku-5", "claude-sonnet-5", "claude-opus-5")) or always_thinks
    if not gen5:
        return {}
    if lvl in _ANTHROPIC_EFFORTS:
        return {"output_config": {"effort": lvl}}
    if always_thinks:
        return {"output_config": {"effort": "low"}}
    if sonnet_55:
        return {"thinking": {"type": "between_tools"}}
    return {"thinking": {"type": "disabled"}}


def _anthropic_system_blocks(system_text: str, schema_text: str) -> List[Dict[str, Any]]:
    """System prompt as content blocks with a prompt-cache breakpoint after the static prefix.

    When `prune_glossary=False` (`static_glossary=True`), the entire system prompt + show notes +
    locked glossary has no `PER_REQUEST_CONTEXT_HEADING`, so 100% of it sits in `blocks[0]` under
    `cache_control: {"type": "ephemeral"}`. When per-request context is used, `blocks[0]` caches
    the static rules + JSON schema and `blocks[1]` carries the uncached per-batch suffix.
    """
    from app.providers.translation.gemini_provider import split_cacheable_system_prompt

    static, dynamic = split_cacheable_system_prompt(system_text)
    blocks: List[Dict[str, Any]] = [
        {"type": "text", "text": static + schema_text, "cache_control": {"type": "ephemeral"}},
    ]
    if dynamic.strip():
        blocks.append({"type": "text", "text": dynamic})
    return blocks


# Anthropic prompt-cache pricing relative to the base input rate (5-minute cache)
ANTHROPIC_CACHE_WRITE_FACTOR = 1.25
ANTHROPIC_CACHE_READ_FACTOR = 0.10


async def _call_anthropic(
    spec: ProviderSpec,
    *,
    system_instruction: str,
    pieces: Sequence[ContentPiece],
    schema_cls: Optional[type[BaseModel]],
    temperature: float,
    max_output_tokens: int,
    usage: Optional[UsageTracker],
    stage: str,
    timeout: float,
) -> LLMResponse:
    root = (spec.base_url or "https://api.anthropic.com").rstrip("/")
    url = f"{root}/v1/messages"
    headers = {
        "x-api-key": spec.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    content_blocks: List[Dict[str, Any]] = []
    for piece in pieces:
        if isinstance(piece, tuple):
            data, mime = piece
            b64 = base64.b64encode(data).decode("ascii")
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
        else:
            content_blocks.append({"type": "text", "text": piece})

    payload: Dict[str, Any] = {
        "model": spec.model,
        "max_tokens": max_output_tokens,
        "system": _anthropic_system_blocks(system_instruction, _schema_instruction(schema_cls)),
        "messages": [{"role": "user", "content": content_blocks}],
    }
    payload.update(anthropic_thinking_fields(spec.model, spec.thinking))
    if _anthropic_supports_temperature(spec.model):
        payload["temperature"] = temperature
    try:
        data = await _post_json_with_retry(url, headers, payload, timeout=timeout, provider_label="Anthropic")
    except RuntimeError as exc:
        if "temperature" in payload and "HTTP 400" in str(exc) and "temperature" in str(exc).lower():
            retry_payload = {k: v for k, v in payload.items() if k != "temperature"}
            data = await _post_json_with_retry(url, headers, retry_payload, timeout=timeout, provider_label="Anthropic")
        else:
            raise
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
    thinking_blocks = sum(
        1 for b in blocks if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
    )
    raw_usage = data.get("usage") or {}
    uncached_in = int(raw_usage.get("input_tokens") or 0)
    cache_write = int(raw_usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(raw_usage.get("cache_read_input_tokens") or 0)
    in_tok = uncached_in + cache_write + cache_read  # the whole prompt, however it was billed
    out_tok = int(raw_usage.get("output_tokens") or 0)  # includes any thinking tokens
    if thinking_blocks:
        logger.info(
            f"Anthropic {spec.model} ({stage}) returned {thinking_blocks} thinking block(s); "
            f"{out_tok} output tokens include the thinking"
        )
    if data.get("stop_reason") == "max_tokens":
        logger.warning(f"Anthropic {spec.model} ({stage}) stopped at max_tokens={max_output_tokens}")
    if usage is not None:
        cost = _anthropic_request_cost(spec.model, uncached_in, cache_write, cache_read, out_tok)
        usage.record_counts(stage, "anthropic", spec.model, in_tok, out_tok, 0, cost_usd=cost)
    return LLMResponse(
        text=_strip_json_fences(text),
        provider="anthropic",
        model=spec.model,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def _anthropic_request_cost(
    model: str, uncached_in: int, cache_write: int, cache_read: int, out_tok: int
) -> Optional[float]:
    """USD for one Anthropic response, pricing cache writes/reads at their own rates."""
    from app.core.model_catalog import request_cost

    effective_in = uncached_in + cache_write * ANTHROPIC_CACHE_WRITE_FACTOR + cache_read * ANTHROPIC_CACHE_READ_FACTOR
    # Long-context pricing is decided by the real prompt size, not the discounted one
    total_in = uncached_in + cache_write + cache_read
    full = request_cost(model, total_in, out_tok, provider="anthropic")
    if full is None:
        return None
    base_in_only = request_cost(model, total_in, 0, provider="anthropic") or 0.0
    rate_in = base_in_only / total_in if total_in else 0.0
    return (full - base_in_only) + effective_in * rate_in


async def _call_ollama(
    spec: ProviderSpec,
    *,
    system_instruction: str,
    pieces: Sequence[ContentPiece],
    schema_cls: Optional[type[BaseModel]],
    temperature: float,
    max_output_tokens: int = 8192,
    usage: Optional[UsageTracker],
    stage: str,
    timeout: float,
) -> LLMResponse:
    root = (spec.base_url or OLLAMA_LOCAL_DEFAULT_URL).rstrip("/")
    url = f"{root}/api/chat"
    headers = {"Content-Type": "application/json"}
    if spec.api_key:
        headers["Authorization"] = f"Bearer {spec.api_key}"

    text_parts: List[str] = []
    images_b64: List[str] = []
    for piece in pieces:
        if isinstance(piece, tuple):
            data, _mime = piece
            images_b64.append(base64.b64encode(data).decode("ascii"))
        else:
            text_parts.append(piece)

    user_content = "\n".join(text_parts)
    user_msg: Dict[str, Any] = {"role": "user", "content": user_content}
    if images_b64:
        user_msg["images"] = images_b64

    sys_content = system_instruction + _schema_instruction(schema_cls)
    est_prompt_tokens = max(1024, (len(sys_content) + len(user_content)) // 3)
    low_model = spec.model.lower()
    is_thinking_ollama = any(k in low_model for k in ("gpt-oss", "qwen3", "deepseek-r1", "qwq"))
    effective_predict = max(max_output_tokens, 16384) if is_thinking_ollama else max_output_tokens
    # Prevent Ollama's default 2k/4k num_ctx from silently truncating the system prompt & schema
    num_ctx = min(32768, max(16384, est_prompt_tokens + effective_predict + 1024))

    fmt: Any = "json"
    if schema_cls is not None:
        try:
            fmt = _inline_json_schema(schema_cls)
        except Exception:
            fmt = "json"

    payload: Dict[str, Any] = {
        "model": spec.model,
        "stream": False,
        "format": fmt,
        "messages": [
            {"role": "system", "content": sys_content},
            user_msg,
        ],
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": effective_predict,
        },
    }
    try:
        data = await _post_json_with_retry(url, headers, payload, timeout=timeout, provider_label="Ollama")
    except RuntimeError as exc:
        if isinstance(fmt, dict) and "HTTP 400" in str(exc):
            payload = {**payload, "format": "json"}
            data = await _post_json_with_retry(url, headers, payload, timeout=timeout, provider_label="Ollama")
        else:
            raise
    msg = data.get("message") or {}
    text = str(msg.get("content") or "")
    if not text.strip() and msg.get("thinking"):
        raise RuntimeError(
            f"Ollama {spec.model} spent the response budget on thinking without emitting JSON output"
        )
    in_tok = int(data.get("prompt_eval_count") or 0)
    out_tok = int(data.get("eval_count") or 0)
    if usage is not None:
        usage.record_counts(stage, "ollama", spec.model, in_tok, out_tok, 0)
    return LLMResponse(
        text=_strip_json_fences(text),
        provider="ollama",
        model=spec.model,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


async def call_text_json(
    spec: ProviderSpec,
    *,
    system_instruction: str,
    prompt: str,
    schema_cls: Optional[type[BaseModel]] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
    usage: Optional[UsageTracker] = None,
    stage: str = "translation",
    flex_state: Any = None,
    timeout: Optional[float] = None,
) -> LLMResponse:
    if timeout is None:
        is_slow_provider = spec.provider == "ollama" or (
            spec.provider == "openai" and spec.model.lower().startswith(("gpt-5", "o1", "o3", "o4"))
        )
        timeout = 240.0 if is_slow_provider else 150.0
    return await call_vision_json(
        spec,
        system_instruction=system_instruction,
        pieces=[prompt],
        schema_cls=schema_cls,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        media_resolution=None,
        usage=usage,
        stage=stage,
        flex_state=flex_state,
        timeout=timeout,
    )


async def call_vision_json(
    spec: ProviderSpec,
    *,
    system_instruction: str,
    pieces: Sequence[ContentPiece],
    schema_cls: Optional[type[BaseModel]] = None,
    temperature: float = 0.1,
    max_output_tokens: int = 4096,
    media_resolution: Optional[str] = "medium",
    usage: Optional[UsageTracker] = None,
    stage: str = "onscreen",
    flex_state: Any = None,
    timeout: float = 90.0,
) -> LLMResponse:
    if spec.provider == "google":
        return await _call_google(
            spec,
            system_instruction=system_instruction,
            pieces=pieces,
            schema_cls=schema_cls,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            media_resolution=media_resolution,
            usage=usage,
            stage=stage,
            flex_state=flex_state,
            timeout=timeout,
        )
    if spec.provider == "openai":
        return await _call_openai(
            spec,
            system_instruction=system_instruction,
            pieces=pieces,
            schema_cls=schema_cls,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            media_resolution=media_resolution,
            usage=usage,
            stage=stage,
            timeout=timeout,
        )
    if spec.provider == "anthropic":
        return await _call_anthropic(
            spec,
            system_instruction=system_instruction,
            pieces=pieces,
            schema_cls=schema_cls,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            usage=usage,
            stage=stage,
            timeout=timeout,
        )
    if spec.provider == "ollama":
        return await _call_ollama(
            spec,
            system_instruction=system_instruction,
            pieces=pieces,
            schema_cls=schema_cls,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            usage=usage,
            stage=stage,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported provider '{spec.provider}'")
