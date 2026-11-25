import asyncio
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlmodel import Session, select
from app.core.config import settings
from app.core.database import get_session
from app.core.hardware import get_hardware_info
from app.core.updater import (
    apply_update,
    check_for_updates,
    get_cached_update_status,
    get_current_version_info,
)
from app.models.db_models import Job
from app.models.schemas import GpuLibsStatusSchema, SystemInfoResponse, UpdateApplyResponse, UpdateStatusSchema

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["System"])

_gemini_models_cache: Optional[List[str]] = None
_gemini_models_cache_at = 0.0


def _fetch_gemini_models() -> List[str]:
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    names = []
    for m in client.models.list():
        name = (getattr(m, "name", "") or "").replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if name.startswith("gemini") and (not actions or "generateContent" in actions):
            if not any(x in name for x in ("embedding", "tts", "image", "audio", "live")):
                names.append(name)
    return sorted(set(names), reverse=True)


async def _gemini_models() -> List[str]:
    """Live model list from the API (cached for an hour), falling back to the configured list."""
    global _gemini_models_cache, _gemini_models_cache_at
    configured = list(dict.fromkeys([settings.gemini_model, *settings.gemini_models]))
    if not settings.is_gemini_configured:
        return configured
    if _gemini_models_cache is None or time.time() - _gemini_models_cache_at > 3600:
        try:
            _gemini_models_cache = await asyncio.wait_for(asyncio.to_thread(_fetch_gemini_models), timeout=5)
            _gemini_models_cache_at = time.time()
        except Exception as e:
            logger.info(f"Could not list Gemini models ({e}); using configured list")
            return configured
    return list(dict.fromkeys(configured + (_gemini_models_cache or [])))


def reset_gemini_models_cache() -> None:
    global _gemini_models_cache, _gemini_models_cache_at
    _gemini_models_cache = None
    _gemini_models_cache_at = 0.0


@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Retrieve server hardware capabilities and configuration status."""
    from app.core.provider_keys import PROVIDERS, is_provider_configured, provider_summary, resolve_secret_key

    hw, version_info = await asyncio.gather(
        asyncio.to_thread(get_hardware_info),
        asyncio.to_thread(get_current_version_info),
    )
    summary = provider_summary()
    google_configured = is_provider_configured("google")
    any_configured = any(is_provider_configured(p) for p in PROVIDERS)
    google_key = resolve_secret_key("google", allow_local_ollama_fallback=False)
    masked = settings.masked_api_key or (summary["google"].get("masked_default_key") if google_key else None)
    cached_update = get_cached_update_status()

    return SystemInfoResponse(
        is_gemini_configured=google_configured,
        is_ai_configured=any_configured,
        masked_api_key=masked,
        gemini_model=settings.gemini_model,
        gemini_models=await _gemini_models(),
        providers=summary,
        transcription_providers=["faster_whisper", "whisperx"],
        whisperx_available=bool(hw.get("whisperx_available")),
        vocal_isolation_available=bool(hw.get("vocal_isolation_available")),
        whisper_default_model=settings.whisper_default_model,
        whisper_models=settings.whisper_models,
        max_cps_warning=settings.max_cps_warning,
        hardware=hw,
        version=version_info["current_version"],
        update=UpdateStatusSchema(**cached_update) if cached_update else None,
    )


@router.get("/update/status", response_model=UpdateStatusSchema)
async def get_update_status():
    """Return cached update status or perform a quick remote version check."""
    cached = get_cached_update_status()
    if cached is not None:
        return UpdateStatusSchema(**cached)
    data = await asyncio.to_thread(check_for_updates)
    return UpdateStatusSchema(**data)


@router.post("/update/check", response_model=UpdateStatusSchema)
async def check_system_updates():
    """Check GitHub Releases / remote Git tags and origin/main for newer versions."""
    data = await asyncio.to_thread(check_for_updates)
    return UpdateStatusSchema(**data)


@router.post("/update/apply", response_model=UpdateApplyResponse)
async def apply_system_update(session: Session = Depends(get_session)):
    """Apply an available Git update when no subtitling job is running."""
    from app.pipeline.orchestrator import ACTIVE_STATUSES

    active_job = session.exec(select(Job).where(Job.status.in_(ACTIVE_STATUSES))).first()
    if active_job is not None:
        raise HTTPException(status_code=409, detail="Cannot update while a job is running")

    res = await asyncio.to_thread(apply_update, rebuild_frontend=True)
    return UpdateApplyResponse(**res)


@router.get("/gpu-libs", response_model=GpuLibsStatusSchema)
def get_gpu_libs_status():
    """The Linux AppImage's optional NVIDIA libraries: offered when an NVIDIA card is found."""
    from app.core import gpu_runtime

    return GpuLibsStatusSchema(**gpu_runtime.status())


@router.post("/gpu-libs/install", response_model=GpuLibsStatusSchema)
def install_gpu_libs():
    """Start downloading the NVIDIA libraries (~1.5 GB) in the background; poll GET /gpu-libs."""
    from app.core import gpu_runtime

    if not gpu_runtime.is_supported():
        raise HTTPException(status_code=400, detail="GPU libraries are downloaded only by the Linux AppImage")
    return GpuLibsStatusSchema(**gpu_runtime.start_install())


@router.get("/log")
def download_server_log():
    """The backend's own log (previous rotation included), for troubleshooting."""
    from datetime import datetime

    from app.core.job_logging import server_log_path

    current = server_log_path()
    parts = []
    previous = current.with_name(current.name + ".1")
    for path in (previous, current):
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    content = "".join(parts) or "(The server log is empty.)\n"
    name = f"subbuild-server-{datetime.now():%Y%m%d-%H%M}.log"
    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-cache"},
    )
