"""
Unified async JSON / Vision-JSON client supporting Google Gemini, OpenAI, Anthropic, and Ollama.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
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

    cfg_kwargs: Dict[str, Any] = {
        "system_instruction": system_instruction,
        "response_mime_type": "application/json",
        "max_output_tokens": max_output_tokens,
        "thinking_config": thinking_config(spec.model, spec.thinking),
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
    if model_id.startswith("gpt-5"):
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
    is_reasoning = low_model.startswith(("o1", "o3", "o4", "gpt-5"))
    # OpenAI counts reasoning tokens inside max_completion_tokens; reserve ample headroom
    effective_cap = max(max_output_tokens, 24576) if is_reasoning else max_output_tokens

    sys_text = system_instruction + _schema_instruction(schema_cls)
    payload: Dict[str, Any] = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": effective_cap,
    }
    if is_reasoning:
        payload["reasoning_effort"] = _openai_reasoning_effort(low_model, spec.thinking)
    else:
        payload["temperature"] = temperature

    data = await _post_json_with_retry(url, headers, payload, timeout=timeout, provider_label="OpenAI")
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
        "temperature": temperature,
        "system": system_instruction + _schema_instruction(schema_cls),
        "messages": [{"role": "user", "content": content_blocks}],
    }
    data = await _post_json_with_retry(url, headers, payload, timeout=timeout, provider_label="Anthropic")
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
    raw_usage = data.get("usage") or {}
    in_tok = int(raw_usage.get("input_tokens") or 0)
    out_tok = int(raw_usage.get("output_tokens") or 0)
    if usage is not None:
        usage.record_counts(stage, "anthropic", spec.model, in_tok, out_tok, 0)
    return LLMResponse(
        text=_strip_json_fences(text),
        provider="anthropic",
        model=spec.model,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


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
