"""
Multi-provider API key vault stored in ``storage/provider_keys.json`` with private-file ACLs.

Supports multiple saved keys per AI provider (``google``, ``openai``, ``anthropic``, ``ollama``)
with user-defined labels, default selection, masked API previews, and backwards-compatible
fallback to ``GEMINI_API_KEY`` from ``.env``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from app.core.config import settings
from app.core.private_files import write_private_text

logger = logging.getLogger(__name__)

ProviderName = Literal["google", "openai", "anthropic", "ollama"]
PROVIDERS: tuple[ProviderName, ...] = ("google", "openai", "anthropic", "ollama")

OLLAMA_LOCAL_DEFAULT_URL = "http://localhost:11434"
OLLAMA_CLOUD_DEFAULT_URL = "https://ollama.com"

_lock = threading.Lock()


@dataclass(frozen=True)
class ResolvedProviderKey:
    """Internal-only resolved credential for executing provider calls."""

    id: str
    provider: ProviderName
    label: str
    api_key: str
    base_url: Optional[str] = None
    source: str = "vault"  # "vault" | "env" | "local_default"


def _vault_path() -> Path:
    return settings.storage_dir / "provider_keys.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_secret(value: Optional[str]) -> str:
    return re.sub(r"[\r\n\x00]+", "", value or "").strip()


def validate_provider(provider: str) -> ProviderName:
    norm = (provider or "").strip().lower()
    if norm not in PROVIDERS:
        raise ValueError(f"Unsupported AI provider '{provider}'. Supported: {', '.join(PROVIDERS)}")
    return norm  # type: ignore[return-value]


def normalize_base_url(provider: ProviderName, base_url: Optional[str]) -> Optional[str]:
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        return OLLAMA_LOCAL_DEFAULT_URL if provider == "ollama" else None
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid base URL '{base_url}': must start with http:// or https://")
    return raw


def mask_secret(secret: Optional[str]) -> Optional[str]:
    clean = _clean_secret(secret)
    if not clean:
        return None
    if len(clean) <= 8:
        return "•" * len(clean)
    return f"{clean[:4]}••••••••{clean[-4:]}"


def _backup_path() -> Path:
    return _vault_path().with_name("provider_keys.json.bak")


def _parse_vault(path: Path) -> Optional[List[Dict[str, Any]]]:
    """The rows of a vault file, or None when it can't be read as one."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("keys"), list):
        return [row for row in data["keys"] if isinstance(row, dict)]
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return None


def _read_raw() -> List[Dict[str, Any]]:
    path = _vault_path()
    if not path.exists():
        return []
    rows = _parse_vault(path)
    if rows is not None:
        return rows
    backup = _backup_path()
    rows = _parse_vault(backup) if backup.exists() else None
    if rows is not None:
        logger.error(f"{path.name} is unreadable; using the previous copy ({backup.name})")
        return rows
    logger.error(f"{path.name} is unreadable and there is no usable backup; saved API keys are unavailable")
    return []


def _set_aside_corrupt_vault(path: Path) -> None:
    """Keep an unreadable vault for manual recovery instead of overwriting it with the next save."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"provider_keys.corrupt-{stamp}.json")
    try:
        os.replace(path, target)
        logger.error(f"Moved the unreadable {path.name} to {target.name}")
    except OSError as e:
        logger.warning(f"Could not set aside the unreadable {path.name}: {e}")


def _write_raw(rows: List[Dict[str, Any]]) -> None:
    path = _vault_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _parse_vault(path) is not None:
            # The last good version: _read_raw falls back to it if the vault is ever damaged
            try:
                write_private_text(_backup_path(), path.read_text(encoding="utf-8"))
            except OSError as e:
                logger.warning(f"Could not back up {path.name}: {e}")
        else:
            _set_aside_corrupt_vault(path)
    payload = json.dumps({"version": 1, "keys": rows}, indent=2)
    write_private_text(path, payload)


def _public_record(row: Dict[str, Any], source: str = "vault") -> Dict[str, Any]:
    provider = row["provider"]
    secret = row.get("api_key") or ""
    base_url = row.get("base_url")
    return {
        "id": row["id"],
        "provider": provider,
        "label": row.get("label") or "Default",
        "masked_key": mask_secret(secret),
        "has_key": bool(secret),
        "base_url": base_url,
        "is_default": bool(row.get("is_default", False)),
        "source": source,
        "created_at": row.get("created_at"),
        "verified_at": row.get("verified_at"),
        "last_error": row.get("last_error"),
    }


def _env_google_row() -> Optional[Dict[str, Any]]:
    if not settings.is_gemini_configured:
        return None
    return {
        "id": "env_google",
        "provider": "google",
        "label": "Environment (.env)",
        "api_key": (settings.gemini_api_key or "").strip(),
        "base_url": None,
        "is_default": True,
        "created_at": None,
        "verified_at": None,
        "last_error": None,
    }


def list_keys(provider: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return masked metadata for all saved keys (never exposes plaintext secrets)."""
    filter_provider = validate_provider(provider) if provider else None
    with _lock:
        rows = _read_raw()
    out = [
        _public_record(r, source="vault")
        for r in rows
        if r.get("provider") in PROVIDERS and (filter_provider is None or r.get("provider") == filter_provider)
    ]
    if filter_provider in (None, "google"):
        has_vault_google = any(r["provider"] == "google" for r in out)
        env_row = _env_google_row()
        if env_row:
            env_public = _public_record(env_row, source="env")
            env_public["is_default"] = not has_vault_google
            out.append(env_public)
    return out


def add_key(
    provider: str,
    api_key: str = "",
    label: Optional[str] = None,
    base_url: Optional[str] = None,
    is_default: Optional[bool] = None,
) -> Dict[str, Any]:
    """Store a new API key (or Ollama local/cloud endpoint) for ``provider``."""
    prov = validate_provider(provider)
    secret = _clean_secret(api_key)
    norm_url = normalize_base_url(prov, base_url)
    if prov != "ollama" and not secret:
        raise ValueError(f"API key is required for provider '{prov}'")
    if prov == "ollama" and norm_url == OLLAMA_CLOUD_DEFAULT_URL and not secret:
        raise ValueError("API key is required for Ollama Cloud (https://ollama.com)")

    clean_label = (label or "").strip() or (
        "Ollama Cloud" if prov == "ollama" and norm_url == OLLAMA_CLOUD_DEFAULT_URL
        else ("Local Ollama" if prov == "ollama" and not secret else "Default")
    )

    with _lock:
        rows = _read_raw()
        existing_for_provider = [r for r in rows if r.get("provider") == prov]
        make_default = bool(is_default) if is_default is not None else (len(existing_for_provider) == 0)
        if make_default:
            for r in rows:
                if r.get("provider") == prov:
                    r["is_default"] = False

        record: Dict[str, Any] = {
            "id": f"key_{uuid.uuid4().hex[:10]}",
            "provider": prov,
            "label": clean_label[:80],
            "api_key": secret,
            "base_url": norm_url,
            "is_default": make_default,
            "created_at": _now_iso(),
            "verified_at": None,
            "last_error": None,
        }
        rows.append(record)
        _write_raw(rows)
        return _public_record(record)


def update_key(
    key_id: str,
    label: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    is_default: Optional[bool] = None,
) -> Dict[str, Any]:
    """Update a stored key's label, secret, base URL, or default status."""
    with _lock:
        rows = _read_raw()
        target = next((r for r in rows if r.get("id") == key_id), None)
        if target is None:
            raise KeyError(f"API key '{key_id}' not found")
        prov = validate_provider(target["provider"])

        if label is not None:
            clean_label = label.strip()
            if clean_label:
                target["label"] = clean_label[:80]
        if api_key is not None:
            secret = _clean_secret(api_key)
            if prov != "ollama" and not secret:
                raise ValueError(f"API key cannot be empty for provider '{prov}'")
            target["api_key"] = secret
            target["verified_at"] = None
            target["last_error"] = None
        if base_url is not None:
            target["base_url"] = normalize_base_url(prov, base_url)
        if is_default:
            for r in rows:
                if r.get("provider") == prov:
                    r["is_default"] = (r.get("id") == key_id)
        elif is_default is False:
            target["is_default"] = False
            # Ensure at least one key remains default for this provider if any exist
            same = [r for r in rows if r.get("provider") == prov]
            if same and not any(r.get("is_default") for r in same):
                same[0]["is_default"] = True

        _write_raw(rows)
        return _public_record(target)


def delete_key(key_id: str) -> bool:
    """Delete a stored key (also supports clearing the virtual `.env` Google fallback)."""
    if key_id == "env_google":
        from app.core.config import save_gemini_api_key

        save_gemini_api_key("")
        return True

    with _lock:
        rows = _read_raw()
        target = next((r for r in rows if r.get("id") == key_id), None)
        if target is None:
            return False
        prov = target.get("provider")
        was_default = bool(target.get("is_default"))
        rows = [r for r in rows if r.get("id") != key_id]
        if was_default:
            same = [r for r in rows if r.get("provider") == prov]
            if same and not any(r.get("is_default") for r in same):
                same[0]["is_default"] = True
        _write_raw(rows)
        return True


def mark_key_status(key_id: str, ok: bool, error: Optional[str] = None) -> None:
    if not key_id or key_id in ("env_google", "local_ollama_default"):
        return
    with _lock:
        rows = _read_raw()
        target = next((r for r in rows if r.get("id") == key_id), None)
        if target is None:
            return
        if ok:
            target["verified_at"] = _now_iso()
            target["last_error"] = None
        else:
            target["last_error"] = (error or "Verification failed")[:300]
        _write_raw(rows)


def resolve_secret_key(
    provider: str,
    key_id: Optional[str] = None,
    allow_local_ollama_fallback: bool = True,
) -> Optional[ResolvedProviderKey]:
    """Resolve the active credential for ``provider`` (explicit key_id -> default -> .env)."""
    prov = validate_provider(provider)
    with _lock:
        rows = [r for r in _read_raw() if r.get("provider") == prov]

    if key_id and key_id != "env_google":
        match = next((r for r in rows if r.get("id") == key_id), None)
        if match:
            return ResolvedProviderKey(
                id=match["id"],
                provider=prov,
                label=match.get("label") or "Default",
                api_key=match.get("api_key") or "",
                base_url=match.get("base_url"),
                source="vault",
            )
        return None

    if key_id == "env_google" and prov == "google":
        env_row = _env_google_row()
        if env_row:
            return ResolvedProviderKey(
                id="env_google",
                provider="google",
                label=env_row["label"],
                api_key=env_row["api_key"],
                source="env",
            )
        return None

    chosen = next((r for r in rows if r.get("is_default")), None) or (rows[0] if rows else None)
    if chosen:
        return ResolvedProviderKey(
            id=chosen["id"],
            provider=prov,
            label=chosen.get("label") or "Default",
            api_key=chosen.get("api_key") or "",
            base_url=chosen.get("base_url"),
            source="vault",
        )

    if prov == "google":
        env_row = _env_google_row()
        if env_row:
            return ResolvedProviderKey(
                id="env_google",
                provider="google",
                label=env_row["label"],
                api_key=env_row["api_key"],
                source="env",
            )

    if prov == "ollama" and allow_local_ollama_fallback:
        return ResolvedProviderKey(
            id="local_ollama_default",
            provider="ollama",
            label="Local Ollama",
            api_key="",
            base_url=OLLAMA_LOCAL_DEFAULT_URL,
            source="local_default",
        )
    return None


def resolve_all_keys_for_provider(provider: str) -> List[ResolvedProviderKey]:
    """Return every configured key for ``provider`` (vault keys + `.env` if distinct)."""
    prov = validate_provider(provider)
    with _lock:
        rows = [r for r in _read_raw() if r.get("provider") == prov]
    out = [
        ResolvedProviderKey(
            id=r["id"],
            provider=prov,
            label=r.get("label") or "Default",
            api_key=r.get("api_key") or "",
            base_url=r.get("base_url"),
            source="vault",
        )
        for r in rows
    ]
    if prov == "google":
        env_row = _env_google_row()
        if env_row and not any(k.api_key == env_row["api_key"] for k in out):
            out.append(
                ResolvedProviderKey(
                    id="env_google",
                    provider="google",
                    label=env_row["label"],
                    api_key=env_row["api_key"],
                    source="env",
                )
            )
    return out


def is_provider_configured(provider: str) -> bool:
    prov = validate_provider(provider)
    if prov == "ollama":
        # Explicitly saved Ollama keys/endpoints count as configured; local daemon status is checked separately.
        return len(resolve_all_keys_for_provider("ollama")) > 0
    return resolve_secret_key(prov, allow_local_ollama_fallback=False) is not None


def provider_summary() -> Dict[str, Dict[str, Any]]:
    """Summary per provider: key count, configured flag, and active default key preview."""
    all_keys = list_keys()
    summary: Dict[str, Dict[str, Any]] = {}
    for prov in PROVIDERS:
        keys = [k for k in all_keys if k["provider"] == prov]
        default_key = next((k for k in keys if k["is_default"]), None) or (keys[0] if keys else None)
        summary[prov] = {
            "provider": prov,
            "configured": bool(keys),
            "key_count": len(keys),
            "default_key_id": default_key["id"] if default_key else None,
            "masked_default_key": default_key["masked_key"] if default_key else None,
            "default_base_url": default_key["base_url"] if default_key else (OLLAMA_LOCAL_DEFAULT_URL if prov == "ollama" else None),
        }
    return summary
