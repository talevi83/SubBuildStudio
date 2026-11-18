import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import settings
from app.core.model_catalog import estimate_job_cost
from app.core.model_discovery import (
    discover_for_key,
    get_key_discovered_count,
    get_live_providers,
    get_merged_catalog,
    refresh_all_models,
    reset_discovery_cache,
)
from app.core.provider_keys import (
    add_key,
    delete_key,
    is_provider_configured,
    list_keys,
    provider_summary,
    resolve_secret_key,
    update_key,
)
from app.core.user_settings import (
    AppSettings,
    load_settings,
    save_settings,
    sync_defaults_with_discovered,
)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class SettingsResponse(BaseModel):
    settings: AppSettings
    catalog: List[Dict[str, Any]]
    providers: Dict[str, Any] = Field(default_factory=dict)
    keys: List[Dict[str, Any]] = Field(default_factory=list)
    pricing_source: str = "https://ai.google.dev/gemini-api/docs/pricing"
    pricing_checked: str = "2026-10"


def _enriched_keys(provider: Optional[str] = None) -> List[Dict[str, Any]]:
    keys = list_keys(provider)
    for item in keys:
        item["discovered_count"] = get_key_discovered_count(item["provider"], item["id"])
    return keys


def _enriched_providers() -> Dict[str, Any]:
    summary = provider_summary()
    live = get_live_providers()
    for prov, entry in summary.items():
        entry["discovered"] = prov in live
    return summary


def _build_settings_response(chosen: Optional[AppSettings] = None) -> SettingsResponse:
    return SettingsResponse(
        settings=chosen or load_settings(),
        catalog=get_merged_catalog(),
        providers=_enriched_providers(),
        keys=_enriched_keys(),
    )


class EstimateRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    duration_seconds: float = Field(ge=0, le=7 * 24 * 3600)
    enable_translation: bool = True
    detect_onscreen_text: bool = False
    has_forced_track: bool = False
    sync_lines: int = Field(default=0, ge=0)  # "sync existing subtitles": lines in the file (Gemini line matching)
    cost_saver_mode: Optional[bool] = None
    # Preview unsaved choices from the Settings screen
    settings: Optional[AppSettings] = None


@router.get("", response_model=SettingsResponse)
async def get_settings():
    from app.core.model_discovery import ensure_initial_discovery

    await ensure_initial_discovery()
    return _build_settings_response()


@router.put("", response_model=SettingsResponse)
def put_settings(new: Dict[str, Any]):
    try:
        validated = AppSettings.model_validate(new)
    except (ValidationError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _build_settings_response(save_settings(validated))


@router.post("/estimate")
def estimate(req: EstimateRequest):
    from app.providers.translation.gemini_provider import build_subtitle_system_prompt

    prompt_tokens = len(build_subtitle_system_prompt("English", "Hebrew")) // 3
    chosen = req.settings or load_settings()
    saver = req.cost_saver_mode if req.cost_saver_mode is not None else chosen.cost_saver_mode
    return estimate_job_cost(
        duration_seconds=req.duration_seconds,
        enable_translation=req.enable_translation,
        detect_onscreen_text=req.detect_onscreen_text,
        app_settings=chosen,
        glossary_pass=settings.gemini_glossary_pass,
        translation_batch_size=settings.gemini_batch_size,
        system_prompt_tokens=prompt_tokens,
        frame_interval=settings.onscreen_frame_interval,
        max_frames=settings.onscreen_max_frames,
        vision_batch_size=settings.onscreen_batch_size,
        has_forced_track=req.has_forced_track,
        sync_lines=req.sync_lines,
        cost_saver_mode=saver,
    )


class CheckRequest(BaseModel):
    settings: Optional[AppSettings] = None
    api_key: Optional[str] = None


# Flex can be slow by design; give the test request longer than a Standard one
_FLEX_CHECK_TIMEOUT = 90


def _make_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


@router.post("/check")
async def check_models(req: CheckRequest):
    """Send a tiny probe request to each configured step's model/provider to verify key access."""
    chosen = req.settings or load_settings()
    steps = {"translation": chosen.translation, "glossary": chosen.glossary, "vision": chosen.vision}

    # Pre-resolve credentials so missing keys surface immediately as HTTP 400 when nothing works
    explicit_override = (req.api_key or "").strip()
    if explicit_override and all(s.provider == "google" for s in steps.values()):
        google_client = _make_client(explicit_override)
    else:
        google_client = None

    any_key = False
    for step, call in steps.items():
        if explicit_override and call.provider == "google":
            any_key = True
            break
        if resolve_secret_key(call.provider, key_id=call.key_id, allow_local_ollama_fallback=True) is not None:
            if call.provider == "ollama" or resolve_secret_key(call.provider, key_id=call.key_id, allow_local_ollama_fallback=False) is not None:
                any_key = True
                break
    if not any_key:
        raise HTTPException(status_code=400, detail="No AI API key configured")

    async def probe(step: str, call):
        if call.provider == "google" and call.key_id is None:
            key = explicit_override
            if not key:
                vault = resolve_secret_key("google", key_id=None, allow_local_ollama_fallback=False)
                key = vault.api_key if vault else (settings.gemini_api_key if settings.is_gemini_configured else "")
            if not key:
                return step, call.model, "No Google Gemini API key configured"

            from google.genai import types
            from app.core.model_catalog import media_resolution, thinking_config
            from app.providers.gemini_calls import flex_config

            client = google_client or _make_client(key)
            contents: Any = "Reply with the single word OK."
            config = types.GenerateContentConfig(
                max_output_tokens=512, thinking_config=thinking_config(call.model, call.thinking))
            if step == "vision":
                contents = [types.Part.from_bytes(data=_probe_png(), mime_type="image/png"),
                            "What color is this image? Reply with one word."]
                config.media_resolution = media_resolution(call.model, chosen.vision_media_resolution)
            if call.flex:
                config = flex_config(config)
            limit = _FLEX_CHECK_TIMEOUT if call.flex else 30
            try:
                await asyncio.wait_for(asyncio.to_thread(
                    client.models.generate_content, model=call.model, contents=contents, config=config,
                ), timeout=limit)
                return step, call.model, None
            except asyncio.TimeoutError:
                return step, call.model, (
                    f"Flex did not answer within {limit}s (Google may be busy); "
                    "jobs retry and then fall back to Standard" if call.flex else f"No answer within {limit}s")
            except Exception as e:
                return step, call.model, (str(e)[:400] or type(e).__name__)

        # Non-Google provider or explicit vault key_id
        from app.providers.llm import ProviderSpec, call_text_json, call_vision_json

        resolved = resolve_secret_key(call.provider, key_id=call.key_id, allow_local_ollama_fallback=(call.provider == "ollama"))
        if resolved is None or (call.provider != "ollama" and not resolved.api_key):
            return step, call.model, f"No API key configured for provider '{call.provider}'"

        spec = ProviderSpec(
            provider=call.provider,
            model=call.model,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            key_id=resolved.id,
            thinking=call.thinking,
            flex=call.flex,
        )
        try:
            if step == "vision":
                await call_vision_json(
                    spec,
                    system_instruction="You check image accessibility. Return JSON {\"ok\": true}.",
                    pieces=[(_probe_png(), "image/png"), "What color is this image? Reply in JSON: {\"color\": \"red\"}."],
                    max_output_tokens=128,
                    media_resolution=chosen.vision_media_resolution,
                    stage="check",
                    timeout=30.0,
                )
            else:
                await call_text_json(
                    spec,
                    system_instruction="Reply ONLY with JSON {\"status\": \"ok\"}.",
                    prompt="Ping check. Reply with {\"status\": \"ok\"}.",
                    max_output_tokens=64,
                    stage="check",
                    timeout=30.0,
                )
            return step, call.model, None
        except Exception as e:
            return step, call.model, (str(e)[:400] or type(e).__name__)

    outcomes = await asyncio.gather(*(probe(step, call) for step, call in steps.items()))
    results: Dict[str, Dict[str, Any]] = {}
    for step, model, error in outcomes:
        entry = results.setdefault(model, {"ok": True, "error": None})
        if error and entry["ok"]:
            entry.update(ok=False, error=f"{step}: {error}")
    return {"results": results, "steps": {step: {"model": m, "ok": e is None, "error": e} for step, m, e in outcomes}}


def _probe_png(width: int = 32, height: int = 32) -> bytes:
    """A tiny solid-red PNG built in memory (no image library needed)."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b""))


class OllamaSetupRequest(BaseModel):
    model: Optional[str] = None


class ApiKeyRequest(BaseModel):
    api_key: str


class ApiKeyResponse(BaseModel):
    is_gemini_configured: bool
    masked_api_key: Optional[str] = None


class ProviderKeyCreateRequest(BaseModel):
    provider: str
    api_key: str = ""
    label: Optional[str] = None
    base_url: Optional[str] = None
    is_default: Optional[bool] = None
    verify: bool = True


class ProviderKeyUpdateRequest(BaseModel):
    label: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    is_default: Optional[bool] = None


@router.get("/api-key", response_model=ApiKeyResponse)
def get_api_key_status():
    summary = provider_summary()
    google_key = resolve_secret_key("google", allow_local_ollama_fallback=False)
    return ApiKeyResponse(
        is_gemini_configured=is_provider_configured("google"),
        masked_api_key=settings.masked_api_key or (summary["google"].get("masked_default_key") if google_key else None),
    )


@router.put("/api-key", response_model=ApiKeyResponse)
async def update_api_key(req: ApiKeyRequest):
    from app.api.routes_system import reset_gemini_models_cache
    from app.core.config import save_gemini_api_key

    save_gemini_api_key(req.api_key)
    reset_gemini_models_cache()
    reset_discovery_cache("google")
    if os.environ.get("SUBBUILD_OFFLINE_DISCOVERY") != "1" and settings.is_gemini_configured:
        resolved = resolve_secret_key("google", allow_local_ollama_fallback=False)
        if resolved:
            await discover_for_key(resolved, force=True)
            sync_defaults_with_discovered()
    summary = provider_summary()
    return ApiKeyResponse(
        is_gemini_configured=is_provider_configured("google"),
        masked_api_key=settings.masked_api_key or summary["google"].get("masked_default_key"),
    )


@router.get("/keys")
def get_provider_keys(provider: Optional[str] = None):
    try:
        keys = _enriched_keys(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"keys": keys, "providers": _enriched_providers()}


@router.post("/keys")
async def create_provider_key(req: ProviderKeyCreateRequest):
    try:
        created = add_key(
            provider=req.provider,
            api_key=req.api_key,
            label=req.label,
            base_url=req.base_url,
            is_default=req.is_default,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    discovery_error: Optional[str] = None
    reset_discovery_cache(created["provider"], key_id=created["id"])
    if created["provider"] == "ollama":
        reset_discovery_cache("ollama", key_id="local_ollama_default")
    if req.verify and os.environ.get("SUBBUILD_OFFLINE_DISCOVERY") != "1":
        resolved = resolve_secret_key(created["provider"], key_id=created["id"], allow_local_ollama_fallback=False)
        if resolved:
            _, discovery_error = await discover_for_key(resolved, force=True)
            if not discovery_error:
                sync_defaults_with_discovered()

    all_keys = _enriched_keys()
    return {
        "key": next((k for k in all_keys if k["id"] == created["id"]), created),
        "keys": all_keys,
        "providers": _enriched_providers(),
        "catalog": get_merged_catalog(),
        "settings": load_settings(),
        "discovery_error": discovery_error,
    }


@router.put("/keys/{key_id}")
async def edit_provider_key(key_id: str, req: ProviderKeyUpdateRequest):
    try:
        updated = update_key(
            key_id,
            label=req.label,
            api_key=req.api_key,
            base_url=req.base_url,
            is_default=req.is_default,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if req.api_key is not None or req.base_url is not None:
        reset_discovery_cache(updated["provider"], key_id=key_id)
        if os.environ.get("SUBBUILD_OFFLINE_DISCOVERY") != "1":
            resolved = resolve_secret_key(updated["provider"], key_id=key_id, allow_local_ollama_fallback=False)
            if resolved:
                await discover_for_key(resolved, force=True)
                sync_defaults_with_discovered()

    all_keys = _enriched_keys()
    return {
        "key": next((k for k in all_keys if k["id"] == key_id), updated),
        "keys": all_keys,
        "providers": _enriched_providers(),
        "catalog": get_merged_catalog(),
        "settings": load_settings(),
    }


@router.delete("/keys/{key_id}")
def remove_provider_key(key_id: str):
    existing = next((k for k in list_keys() if k["id"] == key_id), None)
    if not existing or not delete_key(key_id):
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    reset_discovery_cache(existing["provider"], key_id=key_id)
    return {
        "deleted": True,
        "keys": _enriched_keys(),
        "providers": _enriched_providers(),
        "catalog": get_merged_catalog(),
        "settings": load_settings(),
    }


@router.post("/providers/refresh")
async def refresh_provider_models():
    """Query all configured AI providers + local Ollama now and refresh recommendations."""
    result = await refresh_all_models(force=True)
    catalog = get_merged_catalog()
    updated_settings = sync_defaults_with_discovered(catalog)
    return {
        "settings": updated_settings,
        "catalog": catalog,
        "providers": _enriched_providers(),
        "keys": _enriched_keys(),
        "errors": result.get("errors", {}),
        "discovered_counts": result.get("discovered_counts", {}),
    }


@router.get("/ollama/status")
async def ollama_status():
    from app.core.ollama_setup import get_ollama_status

    return await get_ollama_status()


@router.post("/ollama/setup")
async def ollama_setup(req: Optional[OllamaSetupRequest] = None):
    from app.core.ollama_setup import RECOMMENDED_LOCAL_MODEL, start_ollama_setup

    model = (req.model if req and req.model else RECOMMENDED_LOCAL_MODEL).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?", model):
        raise HTTPException(status_code=400, detail=f"Invalid Ollama model name: {model}")
    return await start_ollama_setup(model=model)

