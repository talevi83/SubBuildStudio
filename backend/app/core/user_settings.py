"""
User-editable AI settings (Settings screen), stored in storage/settings.json.

Environment variables (.env) provide the initial defaults; once saved from the UI the file wins.
"""
import json
import logging
import threading
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.core.config import settings
from app.core.model_catalog import (
    CATALOG, RETIRED_MODELS, TASKS, allowed_thinking, resolve_model, validate_flex, validate_thinking,
)
from app.core.provider_keys import ProviderName

logger = logging.getLogger(__name__)
_lock = threading.Lock()


_NON_GOOGLE_THINKING = ("default", "off", "minimal", "low", "medium", "high")


class GeminiCallSettings(BaseModel):
    provider: ProviderName = "google"
    model: str
    key_id: Optional[str] = None
    thinking: str = "default"
    # Flex service tier (Google Gemini): 50% cheaper, slower and sheddable
    flex: bool = False

    @model_validator(mode="after")
    def _check_thinking(self):
        if self.provider == "google":
            validate_thinking(self.model, self.thinking)
            validate_flex(self.model, self.flex)
        else:
            if self.flex:
                raise ValueError(f"Flex service tier is only supported for Google Gemini, not '{self.provider}'")
            norm = (self.thinking or "default").strip().lower()
            if norm not in _NON_GOOGLE_THINKING:
                raise ValueError(f"Unsupported thinking level '{self.thinking}' for provider '{self.provider}'")
            self.thinking = norm
        return self


ModelCallSettings = GeminiCallSettings


class AppSettings(BaseModel):
    translation: GeminiCallSettings
    glossary: GeminiCallSettings
    vision: GeminiCallSettings = Field(description="Reads on-screen text from video frames")
    vision_media_resolution: Literal["low", "medium", "high"] = "medium"
    onscreen_position: Literal[
        "top_center", "top_left", "top_right", "middle_left", "middle_right", "middle_center"
    ] = "top_center"
    convert_to_metric: bool = False
    remove_sdh_speakers: bool = False
    skip_end_credits: bool = True
    cost_saver_mode: bool = False

    @model_validator(mode="after")
    def _check_tasks(self):
        for task in TASKS:
            call = getattr(self, task)
            if call.provider == "google":
                if call.model in RETIRED_MODELS:
                    raise ValueError(f"{call.model} is no longer available; use {RETIRED_MODELS[call.model]}")
                info = CATALOG.get(call.model)
                if info is not None and task not in info.tasks:
                    raise ValueError(f"{info.label} is not offered for {task}")
        return self


# Cheapest sensible thinking level per task: routine batch translation and vision OCR use
# minimal/low so thinking tokens (billed as output) do not dominate job cost, while the single
# whole-show glossary pass keeps "default" reasoning depth for character & gender analysis.
_PREFERRED_THINKING = {
    "translation": ("minimal", "low", "off"),
    "glossary": ("default",),
    "vision": ("minimal", "low", "off"),
}


def _thinking_for(model: str, task: str, current: str = "default") -> str:
    allowed = allowed_thinking(model)
    if current != "default" and current in allowed:
        return current
    prefs = _PREFERRED_THINKING.get(task, ("low",))
    return next((lvl for lvl in prefs if lvl in allowed), "default")


def preferred_thinking(provider: str, model: str, task: str, current: str = "default") -> str:
    """Return the recommended per-stage thinking level for `(provider, model, task)`."""
    prov = (provider or "google").strip().lower()
    if prov == "google":
        return _thinking_for(model, task, current=current)
    if current and current != "default" and current in _NON_GOOGLE_THINKING:
        return current
    low = (model or "").strip().lower()
    if prov == "openai" and low.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        if task == "glossary":
            return "default"
        return "minimal" if low.startswith(("gpt-5", "gpt-6")) else "low"
    return "default"


def _defaults() -> AppSettings:
    configured = {"translation": settings.gemini_model, "glossary": settings.gemini_model,
                  "vision": settings.gemini_vision_model}
    calls = {}
    for task, model in configured.items():
        model = resolve_model(model, task)
        calls[task] = GeminiCallSettings(model=model, thinking=_thinking_for(model, task))
    return AppSettings(
        cost_saver_mode=settings.cost_saver_mode,
        onscreen_position=getattr(settings, "onscreen_position", "top_center"),
        convert_to_metric=getattr(settings, "convert_to_metric", False),
        remove_sdh_speakers=getattr(settings, "remove_sdh_speakers", False),
        skip_end_credits=getattr(settings, "skip_end_credits", True),
        **calls,
    )


def _migrate(raw: dict) -> bool:
    """Swap retired/unsuitable Google models in saved settings for working ones. Returns True if changed."""
    changed = False
    for task in TASKS:
        call = raw.get(task)
        if not isinstance(call, dict):
            continue
        provider = (call.get("provider") or "google").strip().lower()
        if provider != "google":
            continue
        old = call.get("model", "")
        new = resolve_model(old, task)
        if new != old:
            logger.info(f"Settings: {task} model {old} is unavailable or unsuitable; using {new}")
            call["model"] = new
            call["thinking"] = _thinking_for(new, task)
            changed = True
    return changed


def sync_defaults_with_discovered(catalog: Optional[list] = None) -> AppSettings:
    """
    When `settings.json` does not exist yet (or the active task provider has no configured key
    while live discovery found suitable models), update defaults cleanly without overwriting
    valid user-selected providers/models.
    """
    from app.core.model_discovery import get_merged_catalog, recommend_best_models_by_task
    from app.core.provider_keys import is_provider_configured

    items = catalog if catalog is not None else get_merged_catalog()
    live_items = [m for m in items if m.get("available")]
    if not live_items:
        return load_settings()

    recs = recommend_best_models_by_task(live_items)
    if not recs:
        return load_settings()

    current = load_settings()
    path = _path()
    has_saved_file = path.exists()
    available_pairs = {(m["provider"], m["id"]) for m in live_items}

    updated_tasks = {}
    changed = False
    for task in TASKS:
        cur_call = getattr(current, task)
        same_prov_live = [
            m for m in live_items
            if m["provider"] == cur_call.provider and task in (m.get("tasks") or [])
        ]
        # Treat local Ollama as active whenever its local daemon returned live models,
        # even if the user didn't save an explicit Ollama entry in the key vault.
        prov_configured = is_provider_configured(cur_call.provider) or (
            cur_call.provider == "ollama" and any(m["provider"] == "ollama" for m in live_items)
        )
        cur_is_available = (cur_call.provider, cur_call.model) in available_pairs

        if has_saved_file:
            # Never overwrite an explicitly saved provider (e.g. local Ollama) back to Google at startup.
            # Only swap the model ID within the SAME provider when that provider returned live models
            # and the saved model ID is unavailable for this task.
            if not same_prov_live or cur_is_available:
                continue
            chosen = same_prov_live[0]
            updated_tasks[task] = GeminiCallSettings(
                provider=cur_call.provider,
                model=chosen["id"],
                key_id=cur_call.key_id,
                thinking=preferred_thinking(cur_call.provider, chosen["id"], task),
                flex=False,
            )
            changed = True
            continue

        rec = recs.get(task)
        if not rec:
            continue
        if (not prov_configured) or (same_prov_live and not cur_is_available):
            chosen = same_prov_live[0] if (prov_configured and same_prov_live) else None
            target_prov = chosen["provider"] if chosen else rec["provider"]
            target_model = chosen["id"] if chosen else rec["model"]
            if target_prov != cur_call.provider or target_model != cur_call.model:
                updated_tasks[task] = GeminiCallSettings(
                    provider=target_prov,
                    model=target_model,
                    key_id=None,
                    thinking=preferred_thinking(target_prov, target_model, task),
                    flex=False,
                )
                changed = True

    if not changed:
        return current
    new_settings = current.model_copy(update=updated_tasks)
    if has_saved_file:
        return save_settings(new_settings)
    return new_settings


def _path():
    return settings.storage_dir / "settings.json"


def load_settings() -> AppSettings:
    with _lock:
        path = _path()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                changed = _migrate(raw)
                loaded = AppSettings.model_validate(raw)
                if changed:
                    _write(path, loaded)
                return loaded
            except Exception as e:
                logger.warning(f"Ignoring invalid {path.name} ({e}); using defaults")
        return _defaults()


def _write(path, new: AppSettings) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(new.model_dump(), indent=2), encoding="utf-8")
    tmp.replace(path)


def save_settings(new: AppSettings) -> AppSettings:
    with _lock:
        _write(_path(), new)
    return new
