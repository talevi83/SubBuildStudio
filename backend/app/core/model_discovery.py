"""
Live model discovery across AI providers (Google Gemini, OpenAI, Anthropic Claude, Ollama local & cloud).

Queries each configured key's provider endpoint at startup (in background) and when keys change,
evaluates task suitability + red warnings + per-episode/movie cost estimates, and caches results.
"""
from __future__ import annotations

import asyncio
import errno
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.core.provider_catalog import (
    build_model_entry,
    default_catalog_seeds,
    is_chat_candidate,
)
from app.core.provider_keys import (
    OLLAMA_LOCAL_DEFAULT_URL,
    PROVIDERS,
    ProviderName,
    ResolvedProviderKey,
    mark_key_status,
    resolve_all_keys_for_provider,
)

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 3600.0
_FETCH_TIMEOUT_SECONDS = 5.0
_OLLAMA_LOCAL_TIMEOUT_SECONDS = 1.0

# (provider, key_id) -> {"fetched_at": float, "models": List[Tuple[str, Dict[str, Any]]], "error": Optional[str]}
_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}


_PROVIDER_DISPLAY_NAMES: Dict[str, str] = {
    "google": "Google Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "ollama": "Ollama",
}

# Windows socket errors that mean "something on this computer refused the connection"
# (10013 = WSAEACCES: firewall / antivirus / network filter blocking this process).
_WIN_BLOCKED_ERRORS = {10013}
# Windows socket errors that mean the host could not be reached / resolved.
_WIN_UNREACHABLE_ERRORS = {10050, 10051, 10060, 10061, 10065, 11001, 11002, 11004}


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def describe_discovery_error(provider: str, exc: BaseException) -> str:
    """Turn a raw discovery exception into a message a non-technical user can act on.

    Network-level failures (blocked socket, no connection, timeout) get a plain-language
    explanation; anything else (HTTP 401/403 for a bad key, etc.) keeps its original text.
    """
    name = _PROVIDER_DISPLAY_NAMES.get(provider, provider)
    chain = list(_iter_exception_chain(exc))
    raw_text = " ".join(str(e) for e in chain)

    blocked = "WinError 10013" in raw_text or any(
        isinstance(e, OSError)
        and (
            getattr(e, "winerror", None) in _WIN_BLOCKED_ERRORS
            or (getattr(e, "winerror", None) is None and e.errno in (errno.EACCES, errno.EPERM))
        )
        for e in chain
    )
    if blocked:
        return (
            f"SubBuild Studio was blocked from connecting to {name}. "
            "A firewall, antivirus or network filter on this computer is likely blocking the app - "
            "allow SubBuild Studio through it and try again. (Windows error 10013)"
        )

    if any(isinstance(e, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)) for e in chain):
        return (
            f"{name} did not respond in time. "
            "Check your internet connection (or VPN / proxy) and try again."
        )

    unreachable = any(
        isinstance(e, (httpx.ConnectError, httpx.NetworkError, ConnectionError))
        or (isinstance(e, OSError) and getattr(e, "winerror", None) in _WIN_UNREACHABLE_ERRORS)
        for e in chain
    )
    if unreachable:
        return (
            f"Could not connect to {name}. "
            "Check your internet connection (or VPN / proxy) and try again."
        )

    return str(exc)[:300] or type(exc).__name__


def reset_discovery_cache(provider: Optional[str] = None, key_id: Optional[str] = None) -> None:
    """Clear cached discovery entries (for all providers, a provider, or a single key)."""
    if provider is None and key_id is None:
        _cache.clear()
        return
    for key in [
        k for k in _cache
        if (provider is None or k[0] == provider) and (key_id is None or k[1] == key_id)
    ]:
        _cache.pop(key, None)


def get_key_discovered_count(provider: str, key_id: str) -> Optional[int]:
    """Return number of chat-candidate models cached for ``(provider, key_id)``, or None if unqueried/failed."""
    entry = _cache.get((provider, key_id))
    if not entry or entry.get("error") is not None:
        return None
    return sum(
        1
        for mid, _ in (entry.get("models") or [])
        if is_chat_candidate(provider, (mid or "").replace("models/", "").strip())
    )


def get_live_providers() -> set[str]:
    """Return provider names that have at least one successful cached live discovery entry."""
    return {
        prov
        for (prov, _kid), entry in _cache.items()
        if entry and entry.get("error") is None
    }


def _fetch_google_models_sync(api_key: str) -> List[Tuple[str, Dict[str, Any]]]:
    from google import genai

    client = genai.Client(api_key=api_key)
    seen: Dict[str, Dict[str, Any]] = {}
    for m in client.models.list():
        name = (getattr(m, "name", "") or "").replace("models/", "").strip()
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        if not is_chat_candidate("google", name):
            continue
        seen[name] = {}
    return [(k, seen[k]) for k in sorted(seen.keys(), reverse=True)]


async def _fetch_openai_models(api_key: str, base_url: Optional[str]) -> List[Tuple[str, Dict[str, Any]]]:
    import re

    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{root}/models" if not root.endswith("/models") else root
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
    raw = data.get("data") if isinstance(data, dict) else None
    candidates: List[str] = []
    for item in raw or []:
        mid = str(item.get("id") or "").strip() if isinstance(item, dict) else ""
        if mid and is_chat_candidate("openai", mid):
            candidates.append(mid)

    candidate_set = {m.lower() for m in candidates}
    date_suffix_re = re.compile(r"-\d{4}-\d{2}-\d{2}$")
    seen: set[str] = set()
    out: List[Tuple[str, Dict[str, Any]]] = []
    for mid in candidates:
        low = mid.lower()
        base = date_suffix_re.sub("", low)
        if base != low and base in candidate_set:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append((mid, {}))
    return sorted(out, key=lambda t: t[0], reverse=True)


async def _fetch_anthropic_models(api_key: str, base_url: Optional[str]) -> List[Tuple[str, Dict[str, Any]]]:
    root = (base_url or "https://api.anthropic.com").rstrip("/")
    url = f"{root}/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS) as client:
        resp = await client.get(url, headers=headers, params={"limit": 100})
        if resp.status_code >= 400:
            raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
    raw = data.get("data") if isinstance(data, dict) else None
    out: List[Tuple[str, Dict[str, Any]]] = []
    for item in raw or []:
        mid = str(item.get("id") or "").strip() if isinstance(item, dict) else ""
        if mid and is_chat_candidate("anthropic", mid):
            out.append((mid, {}))
    return out


async def _fetch_ollama_show(
    client: httpx.AsyncClient,
    root: str,
    name: str,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    """Best-effort Ollama /api/show lookup for model capabilities (vision/thinking/tools) & details."""
    try:
        resp = await client.post(f"{root}/api/show", json={"model": name}, headers=headers)
        if resp.status_code >= 400:
            return {}
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


async def _fetch_ollama_models(
    api_key: str,
    base_url: Optional[str],
    *,
    is_local_probe: bool = False,
) -> List[Tuple[str, Dict[str, Any]]]:
    root = (base_url or OLLAMA_LOCAL_DEFAULT_URL).rstrip("/")
    url = f"{root}/api/tags"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timeout = _OLLAMA_LOCAL_TIMEOUT_SECONDS if is_local_probe and not api_key else _FETCH_TIMEOUT_SECONDS
    items: List[Tuple[str, Dict[str, Any], List[str]]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        raw = data.get("models") if isinstance(data, dict) else None
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if not name or not is_chat_candidate("ollama", name):
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            caps = item.get("capabilities") if isinstance(item.get("capabilities"), list) else []
            items.append((name, details, caps))

        show_indices = [
            idx
            for idx, (_name, details, caps) in enumerate(items)
            if not caps or not details.get("parameter_size")
        ]
        if show_indices:
            show_results = await asyncio.gather(
                *(_fetch_ollama_show(client, root, items[idx][0], headers) for idx in show_indices),
                return_exceptions=True,
            )
            for idx, res in zip(show_indices, show_results):
                if not isinstance(res, dict) or not res:
                    continue
                name, details, caps = items[idx]
                show_caps = res.get("capabilities")
                if isinstance(show_caps, list) and show_caps and not caps:
                    caps = show_caps
                show_details = res.get("details")
                if isinstance(show_details, dict) and show_details:
                    details = {**details, **show_details}
                items[idx] = (name, details, caps)

    return [
        (name, {"capabilities": caps, "details": details, "base_url": root})
        for name, details, caps in items
    ]


async def discover_for_key(
    key: ResolvedProviderKey,
    *,
    force: bool = False,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], Optional[str]]:
    """Return ``(models, error)`` for a single resolved provider credential."""
    cache_key = (key.provider, key.id)
    cached = _cache.get(cache_key)
    now = time.time()
    if not force and cached and (now - cached["fetched_at"] < _CACHE_TTL_SECONDS):
        return cached["models"], cached["error"]

    try:
        if key.provider == "google":
            models = await asyncio.wait_for(
                asyncio.to_thread(_fetch_google_models_sync, key.api_key),
                timeout=_FETCH_TIMEOUT_SECONDS,
            )
        elif key.provider == "openai":
            models = await _fetch_openai_models(key.api_key, key.base_url)
        elif key.provider == "anthropic":
            models = await _fetch_anthropic_models(key.api_key, key.base_url)
        elif key.provider == "ollama":
            models = await _fetch_ollama_models(
                key.api_key,
                key.base_url,
                is_local_probe=(key.source == "local_default"),
            )
        else:
            models = []
        error = None
        mark_key_status(key.id, ok=True)
    except Exception as exc:
        models = []
        error = describe_discovery_error(key.provider, exc)
        if key.source != "local_default":
            logger.info(
                f"Model discovery for {key.provider} ({key.label}) failed: {error} "
                f"[raw: {type(exc).__name__}: {str(exc)[:300]}]"
            )
            mark_key_status(key.id, ok=False, error=error)

    _cache[cache_key] = {"fetched_at": now, "models": models, "error": error}
    return models, error


from weakref import WeakKeyDictionary

_loop_locks: WeakKeyDictionary = WeakKeyDictionary()
_initial_discovery_done: bool = False
_saw_cache: bool = False
_attempted_signature: Optional[Tuple[Any, ...]] = None


def _discovery_signature() -> Tuple[Any, ...]:
    return (
        os.environ.get("SUBBUILD_SKIP_LOCAL_OLLAMA_PROBE"),
        tuple((prov, k.id) for prov in PROVIDERS for k in resolve_all_keys_for_provider(prov)),
    )


async def ensure_initial_discovery(*, force: bool = False) -> None:
    """Wait once for initial live provider discovery if startup refresh has not completed yet."""
    global _initial_discovery_done, _saw_cache, _attempted_signature
    if os.environ.get("SUBBUILD_OFFLINE_DISCOVERY") == "1":
        return
    if _saw_cache and not _cache:
        _saw_cache = False
        _initial_discovery_done = False
    sig = _discovery_signature()
    if not force and _initial_discovery_done and _attempted_signature == sig:
        return
    loop = asyncio.get_running_loop()
    lock = _loop_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _loop_locks[loop] = lock
    async with lock:
        if _saw_cache and not _cache:
            _saw_cache = False
            _initial_discovery_done = False
        sig = _discovery_signature()
        if not force and _initial_discovery_done and _attempted_signature == sig:
            return
        try:
            await refresh_all_models(force=force)
            from app.core.user_settings import sync_defaults_with_discovered

            sync_defaults_with_discovered(get_merged_catalog())
        finally:
            _initial_discovery_done = True
            _attempted_signature = sig
            if _cache:
                _saw_cache = True


async def refresh_all_models(
    *,
    force: bool = False,
    include_local_ollama_probe: bool = True,
) -> Dict[str, Any]:
    """Refresh live model lists for all configured provider keys across all providers."""
    global _initial_discovery_done, _saw_cache, _attempted_signature
    if os.environ.get("SUBBUILD_OFFLINE_DISCOVERY") == "1":
        return {"errors": {}, "discovered_counts": {}}

    skip_local_probe = os.environ.get("SUBBUILD_SKIP_LOCAL_OLLAMA_PROBE") == "1"
    tasks = []
    task_meta: List[ResolvedProviderKey] = []
    for prov in PROVIDERS:
        keys = resolve_all_keys_for_provider(prov)
        if prov == "ollama" and not keys and include_local_ollama_probe and not skip_local_probe:
            keys = [
                ResolvedProviderKey(
                    id="local_ollama_default",
                    provider="ollama",
                    label="Local Ollama",
                    api_key="",
                    base_url=OLLAMA_LOCAL_DEFAULT_URL,
                    source="local_default",
                )
            ]
        for k in keys:
            task_meta.append(k)
            tasks.append(discover_for_key(k, force=force))

    results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
    errors: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for key_obj, res in zip(task_meta, results):
        if isinstance(res, Exception):
            errors[key_obj.id] = str(res)
            continue
        models, err = res
        counts[key_obj.provider] = counts.get(key_obj.provider, 0) + len(models)
        if err and key_obj.source != "local_default":
            errors[key_obj.id] = err
    _initial_discovery_done = True
    _attempted_signature = _discovery_signature()
    if _cache:
        _saw_cache = True
    return {"errors": errors, "discovered_counts": counts}


def get_merged_catalog(
    *,
    include_unconfigured: bool = True,
) -> List[Dict[str, Any]]:
    """Build the unified multi-provider catalog.

    For any provider whose live API key(s) have already been queried, ONLY the models
    actually returned for those keys are included (never unverified static seeds).
    """
    # (provider, model_id) -> {"key_ids": [...], "available": bool, "capabilities": [...], "base_url": str|None}
    merged: Dict[Tuple[ProviderName, str], Dict[str, Any]] = {}
    providers_with_live_cache: set[ProviderName] = set()

    for prov in PROVIDERS:
        keys = resolve_all_keys_for_provider(prov)
        if prov == "ollama" and not keys:
            keys = [
                ResolvedProviderKey(
                    id="local_ollama_default",
                    provider="ollama",
                    label="Local Ollama",
                    api_key="",
                    base_url=OLLAMA_LOCAL_DEFAULT_URL,
                    source="local_default",
                )
            ]
        for k in keys:
            entry = _cache.get((prov, k.id))
            if not entry:
                continue
            if entry.get("error") is None:
                providers_with_live_cache.add(prov)
            for mid, meta in entry.get("models", []):
                clean_mid = (mid or "").replace("models/", "").strip()
                if not clean_mid or not is_chat_candidate(prov, clean_mid):
                    continue
                bucket = merged.setdefault(
                    (prov, clean_mid),
                    {
                        "available": True,
                        "key_ids": [],
                        "capabilities": meta.get("capabilities"),
                        "details": meta.get("details"),
                        "base_url": meta.get("base_url") or k.base_url,
                    },
                )
                bucket["available"] = True
                if k.id not in bucket["key_ids"]:
                    bucket["key_ids"].append(k.id)
                if meta.get("capabilities") and not bucket.get("capabilities"):
                    bucket["capabilities"] = meta.get("capabilities")
                if meta.get("details") and not bucket.get("details"):
                    bucket["details"] = meta.get("details")
                if k.base_url:
                    bucket["base_url"] = k.base_url

    if include_unconfigured:
        for prov, mid in default_catalog_seeds():
            # Never mix static unverified seeds into a provider that already returned live key models
            if prov in providers_with_live_cache:
                continue
            merged.setdefault((prov, mid), {
                "available": False,
                "key_ids": [],
                "capabilities": None,
                "details": None,
                "base_url": OLLAMA_LOCAL_DEFAULT_URL if prov == "ollama" else None,
            })

    entries = [
        build_model_entry(
            prov,
            mid,
            available=bool(meta["available"]),
            key_ids=meta["key_ids"],
            base_url=meta.get("base_url"),
            capabilities=meta.get("capabilities"),
            details=meta.get("details"),
        )
        for (prov, mid), meta in merged.items()
    ]

    # Sort: available live models first, then lower rank (recommended first), then provider/label
    entries.sort(
        key=lambda e: (
            0 if e["available"] else 1,
            e.get("rank", 50),
            PROVIDERS.index(e["provider"]),
            e["label"].lower(),
        )
    )
    return entries


def recommend_best_models_by_task(
    catalog: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, str]]:
    """Pick the best available model+provider per pipeline task from discovered models."""
    items = catalog if catalog is not None else get_merged_catalog()
    live = [m for m in items if m.get("available")]
    pool = live if live else items

    out: Dict[str, Dict[str, str]] = {}
    for task in ("translation", "glossary", "vision"):
        candidates = [m for m in pool if task in (m.get("tasks") or [])]
        if not candidates:
            continue
        # Prefer recommended -> suitable -> warning, then lowest rank
        suitability_order = {"recommended": 0, "suitable": 1, "warning": 2, "unsuitable": 9}
        candidates.sort(
            key=lambda m: (
                suitability_order.get((m.get("suitability_by_task") or {}).get(task, "suitable"), 5),
                m.get("rank", 50),
            )
        )
        best = candidates[0]
        out[task] = {"provider": best["provider"], "model": best["id"]}
    return out
