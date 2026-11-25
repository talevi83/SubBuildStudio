"""Tests for multi-provider key vault, live model discovery, pricing/suitability, and mixed-provider execution."""
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core import model_discovery as md
from app.core import provider_keys as pk
from app.core.user_settings import load_settings
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "gemini_api_key", None)
    monkeypatch.setenv("SUBBUILD_OFFLINE_DISCOVERY", "1")
    md.reset_discovery_cache()
    with TestClient(app) as c:
        yield c
    md.reset_discovery_cache()


def test_multi_key_vault_crud_and_secret_masking(client: TestClient):
    # Cloud providers require a non-empty key
    assert client.post("/api/settings/keys", json={"provider": "openai", "api_key": ""}).status_code == 400
    assert client.post("/api/settings/keys", json={"provider": "ollama", "base_url": "ftp://bad"}).status_code == 400
    assert client.post("/api/settings/keys", json={
        "provider": "ollama", "base_url": "https://ollama.com", "api_key": "",
    }).status_code == 400

    # Add 2 keys for OpenAI, 1 for Anthropic, 1 for Ollama Cloud
    r1 = client.post("/api/settings/keys", json={
        "provider": "openai", "label": "Personal", "api_key": "sk-proj-secret-1111aaaa", "verify": False,
    })
    assert r1.status_code == 200
    k1 = r1.json()["key"]
    assert k1["is_default"] is True
    assert "1111aaaa" not in (k1["masked_key"] or "") or k1["masked_key"] == "sk-p••••••••aaaa"

    r2 = client.post("/api/settings/keys", json={
        "provider": "openai", "label": "Work", "api_key": "sk-proj-secret-2222bbbb", "is_default": True, "verify": False,
    })
    assert r2.status_code == 200
    k2 = r2.json()["key"]
    assert k2["is_default"] is True

    client.post("/api/settings/keys", json={
        "provider": "anthropic", "label": "Claude Main", "api_key": "sk-ant-secret-9999zzzz", "verify": False,
    })
    client.post("/api/settings/keys", json={
        "provider": "ollama", "label": "Ollama Cloud", "base_url": "https://ollama.com", "api_key": "olc-secret-5555wwww", "verify": False,
    })

    # Plaintext secrets must never leak via GET endpoints
    keys_res = client.get("/api/settings/keys").json()
    dumped = json.dumps(keys_res)
    for raw_secret in ("sk-proj-secret-1111aaaa", "sk-proj-secret-2222bbbb", "sk-ant-secret-9999zzzz", "olc-secret-5555wwww"):
        assert raw_secret not in dumped

    assert keys_res["providers"]["openai"]["key_count"] == 2
    assert keys_res["providers"]["openai"]["default_key_id"] == k2["id"]
    assert keys_res["providers"]["anthropic"]["configured"] is True
    assert keys_res["providers"]["ollama"]["configured"] is True

    # Deleting current default promotes remaining OpenAI key
    assert client.delete(f"/api/settings/keys/{k2['id']}").status_code == 200
    after_del = client.get("/api/settings/keys").json()
    assert after_del["providers"]["openai"]["key_count"] == 1
    assert after_del["providers"]["openai"]["default_key_id"] == k1["id"]

    # System info reflects multi-provider AI readiness even without Gemini
    sys_info = client.get("/api/system/info").json()
    assert sys_info["is_gemini_configured"] is False
    assert sys_info["is_ai_configured"] is True


def test_live_discovery_filtering_suitability_and_benchmarks(client: TestClient, monkeypatch):
    monkeypatch.setenv("SUBBUILD_OFFLINE_DISCOVERY", "0")

    k_oai1 = client.post("/api/settings/keys", json={
        "provider": "openai", "label": "OAI Primary", "api_key": "sk-test-openai-1111", "verify": False,
    }).json()["key"]
    k_oai2 = client.post("/api/settings/keys", json={
        "provider": "openai", "label": "OAI VisionOnly", "api_key": "sk-test-openai-2222", "verify": False,
    }).json()["key"]
    client.post("/api/settings/keys", json={
        "provider": "anthropic", "label": "Ant", "api_key": "sk-ant-test-1234", "verify": False,
    })

    async def fake_fetch_openai(api_key: str, base_url=None):
        if api_key == "sk-test-openai-1111":
            return [
                ("gpt-5-mini", {}),
                ("whisper-1", {}),
                ("text-embedding-3-large", {}),
                ("gpt-3.5-turbo", {}),
            ]
        return [("gpt-5-nano", {})]

    monkeypatch.setattr(md, "_fetch_openai_models", fake_fetch_openai)
    monkeypatch.setattr(md, "_fetch_anthropic_models", AsyncMock(return_value=[
        ("claude-sonnet-4-5-20250929", {}),
        ("claude-3-5-haiku-20241022", {}),
    ]))
    monkeypatch.setattr(md, "_fetch_ollama_models", AsyncMock(return_value=[
        ("qwen2.5:14b", {"base_url": "http://localhost:11434"}),
        ("qwen2.5:1.5b", {"base_url": "http://localhost:11434"}),
        ("qwen2.5-coder:7b", {"base_url": "http://localhost:11434"}),
        ("qwen2.5vl:7b", {"base_url": "http://localhost:11434", "capabilities": ["completion", "vision"]}),
    ]))

    res = client.post("/api/settings/providers/refresh")
    assert res.status_code == 200
    body = res.json()
    catalog = {(m["provider"], m["id"]): m for m in body["catalog"]}

    # Non-chat, legacy, and code-only models excluded
    assert ("openai", "whisper-1") not in catalog
    assert ("openai", "text-embedding-3-large") not in catalog
    assert ("openai", "gpt-3.5-turbo") not in catalog
    assert ("ollama", "qwen2.5-coder:7b") not in catalog

    # Unverified static seeds are NOT mixed into providers whose keys were queried live
    assert ("openai", "gpt-5") not in catalog
    assert ("anthropic", "claude-haiku-4-5") not in catalog

    # Per-key scoping: k_oai1 only sees gpt-5-mini; k_oai2 only sees gpt-5-nano
    mini = catalog[("openai", "gpt-5-mini")]
    assert mini["available"] is True
    assert mini["key_ids"] == [k_oai1["id"]]
    assert mini["tasks"] == ["translation", "glossary", "vision"]
    assert mini["episode_cost_usd"] is not None and mini["movie_cost_usd"] > mini["episode_cost_usd"] > 0
    assert "translation" in mini["recommended_for"]

    nano = catalog[("openai", "gpt-5-nano")]
    assert nano["key_ids"] == [k_oai2["id"]]
    # Ultra-small nano is strictly vision-only (excluded from translation & glossary)
    assert nano["tasks"] == ["vision"]

    haiku35 = catalog[("anthropic", "claude-3-5-haiku-20241022")]
    # Text-only haiku 3.5 is strictly glossary-only (no vision, no flagship translation)
    assert haiku35["tasks"] == ["glossary"]

    sonnet = catalog[("anthropic", "claude-sonnet-4-5-20250929")]
    assert sonnet["input_per_m"] == 3.0 and sonnet["output_per_m"] == 15.0
    assert sonnet["tasks"] == ["translation", "glossary", "vision"]

    q14 = catalog[("ollama", "qwen2.5:14b")]
    assert q14["tasks"] == ["translation", "glossary"]  # No vision capability

    q_small = catalog[("ollama", "qwen2.5:1.5b")]
    assert q_small["is_local"] is True
    assert q_small["episode_cost_usd"] == 0.0
    assert q_small["tasks"] == []  # <6.5B non-vision model cannot serve any pipeline step

    q_vl = catalog[("ollama", "qwen2.5vl:7b")]
    assert q_vl["tasks"] == ["translation", "glossary", "vision"]
    assert q_vl["episode_cost_usd"] == 0.0

    keys_by_id = {k["id"]: k for k in body["keys"]}
    assert keys_by_id[k_oai1["id"]]["discovered_count"] == 1
    assert keys_by_id[k_oai2["id"]]["discovered_count"] == 1
    assert body["providers"]["openai"]["discovered"] is True

    # Auto-selected defaults picked live available models instead of unconfigured Google
    cur = body["settings"]
    assert cur["translation"]["provider"] in ("openai", "anthropic", "ollama")


def test_mixed_provider_settings_estimate_and_check(client: TestClient, monkeypatch):
    client.post("/api/settings/keys", json={
        "provider": "openai", "label": "OAI", "api_key": "sk-oai-9999", "verify": False,
    })
    client.post("/api/settings/keys", json={
        "provider": "anthropic", "label": "Ant", "api_key": "sk-ant-9999", "verify": False,
    })

    mixed = client.get("/api/settings").json()["settings"]
    mixed["translation"] = {"provider": "openai", "model": "gpt-5-mini", "thinking": "default", "flex": False}
    mixed["glossary"] = {"provider": "anthropic", "model": "claude-haiku-4-5", "thinking": "default", "flex": False}
    mixed["vision"] = {"provider": "ollama", "model": "qwen2.5vl:7b", "thinking": "default", "flex": False}
    saved = client.put("/api/settings", json=mixed)
    assert saved.status_code == 200
    assert saved.json()["settings"]["translation"]["provider"] == "openai"
    assert saved.json()["settings"]["glossary"]["provider"] == "anthropic"
    assert saved.json()["settings"]["vision"]["provider"] == "ollama"

    # Mixed estimate across OpenAI + Anthropic + local Ollama ($0.00 vision)
    est = client.post("/api/settings/estimate", json={
        "duration_seconds": 45 * 60, "enable_translation": True, "detect_onscreen_text": True,
    }).json()
    assert est["stages"]["translation"]["provider"] == "openai"
    assert est["stages"]["glossary"]["provider"] == "anthropic"
    assert est["stages"]["onscreen"]["provider"] == "ollama"
    assert est["stages"]["onscreen"]["cost_usd"] == 0.0
    assert est["total_cost_usd"] > 0.0

    # /api/settings/check probes each step against its respective provider
    probed = []

    async def fake_vision(spec, **kwargs):
        probed.append((spec.provider, spec.model, kwargs.get("stage")))
        return type("R", (), {"text": "{\"ok\": true}"})()

    monkeypatch.setattr("app.providers.llm.call_vision_json", fake_vision)
    monkeypatch.setattr("app.providers.llm.call_text_json", fake_vision)
    chk = client.post("/api/settings/check", json={})
    assert chk.status_code == 200
    steps = chk.json()["steps"]
    assert steps["translation"]["ok"] is True
    assert steps["glossary"]["ok"] is True
    assert steps["vision"]["ok"] is True
    assert {(p, m) for p, m, _ in probed} == {
        ("openai", "gpt-5-mini"),
        ("anthropic", "claude-haiku-4-5"),
        ("ollama", "qwen2.5vl:7b"),
    }


@pytest.mark.asyncio
async def test_mixed_provider_translation_and_vision_runtime(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.core.model_catalog import UsageTracker
    from app.providers.llm.client import LLMResponse
    from app.providers.translation.base import TranslatableSegment
    from app.providers.translation.gemini_provider import GeminiTranslator
    from app.pipeline.onscreen_vision import _gemini_detect
    from app.pipeline.frame_sampling import Frame

    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    pk.add_key("openai", api_key="sk-test-oai", label="OAI")
    pk.add_key("anthropic", api_key="sk-test-ant", label="Ant")

    calls = []

    async def fake_post(url, headers, payload, timeout, provider_label, max_attempts=4):
        calls.append((provider_label, url, payload["model"]))
        if provider_label == "Anthropic":
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "summary": "Sci-fi", "characters": [], "terms": [], "register_notes": "",
                })}],
                "usage": {"input_tokens": 120, "output_tokens": 45},
            }
        if provider_label == "OpenAI":
            return {
                "choices": [{"message": {"content": json.dumps({
                    "translations": [{"index": 1, "translated_text": "שלום עולם"}],
                })}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 30},
            }
        return {
            "message": {"content": json.dumps({
                "detections": [{"timestamp": 2.0, "text": "ONE WEEK LATER", "kind": "time_jump"}],
            })},
            "prompt_eval_count": 300,
            "eval_count": 20,
        }

    monkeypatch.setattr("app.providers.llm.client._post_json_with_retry", fake_post)

    usage = UsageTracker()
    translator = GeminiTranslator(
        provider="openai",
        model_name="gpt-5-mini",
        glossary_provider="anthropic",
        glossary_model="claude-haiku-4-5",
        usage=usage,
    )
    segs = [TranslatableSegment(index=1, start_time=0.0, end_time=2.0, original_text="Hello world")]
    glossary = await translator.build_glossary(segs, "English", "Hebrew")
    assert glossary["summary"] == "Sci-fi"

    translated = await translator.translate_segments(segs, "English", "Hebrew", glossary=glossary)
    assert translated[0].translated_text == "שלום עולם"

    img_path = tmp_path / "f_001.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0dummy")
    detections = await _gemini_detect(
        client=None,
        model="qwen2.5vl:7b",
        batch=[Frame(time=2.0, path=img_path)],
        usage=usage,
        provider="ollama",
    )
    assert len(detections) == 1
    assert detections[0].text == "ONE WEEK LATER"

    snap = usage.to_dict()
    assert snap["stages"]["glossary"]["input_tokens"] == 120
    assert snap["stages"]["translation"]["input_tokens"] == 200
    assert snap["stages"]["onscreen"]["cost_usd"] == 0.0
    assert snap["total_cost_usd"] > 0


def test_openai_vision_tokens_and_episode_vs_movie_benchmarks():
    from app.core.provider_catalog import build_model_entry, estimated_image_tokens

    # gpt-4o-mini carries OpenAI's 33.33x multiplier (2833 low/medium vs 14167 high)
    assert estimated_image_tokens("openai", "gpt-4o-mini", "medium") == 2833
    assert estimated_image_tokens("openai", "gpt-4o-mini", "high") == 14167
    assert estimated_image_tokens("openai", "gpt-4.1-mini", "medium") == 235
    assert estimated_image_tokens("google", "gemini-3.5-flash-lite", "medium") == 560

    entry = build_model_entry("openai", "gpt-4o-mini", available=True)
    # Benchmarks must reflect OpenAI image tokens AND distinguish 45m episode vs 120m movie
    assert entry["episode_vision_cost_usd"] > 0.15
    assert entry["movie_vision_cost_usd"] > entry["episode_vision_cost_usd"]


@pytest.mark.asyncio
async def test_openai_vision_detail_and_reasoning_headroom(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.providers.llm import ProviderSpec, call_text_json, call_vision_json
    from app.providers.llm.client import _post_json_with_retry

    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    seen_payloads = []

    async def fake_post(url, headers, payload, timeout, provider_label, max_attempts=4):
        seen_payloads.append((payload, timeout))
        if payload.get("model") == "gpt-5-mini":
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": "{\"ok\": true}"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 20}},
            }
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": "{\"detections\": []}"}}],
            "usage": {"prompt_tokens": 2900, "completion_tokens": 10},
        }

    monkeypatch.setattr("app.providers.llm.client._post_json_with_retry", fake_post)

    # Medium resolution maps to OpenAI detail="low" (saves 5x on gpt-4o-mini), High maps to "high"
    spec_vis = ProviderSpec(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    await call_vision_json(spec_vis, system_instruction="sys", pieces=["frame", (b"\xff\xd8\xff\xe0", "image/jpeg")], media_resolution="medium")
    await call_vision_json(spec_vis, system_instruction="sys", pieces=["frame", (b"\xff\xd8\xff\xe0", "image/jpeg")], media_resolution="high")
    assert seen_payloads[0][0]["messages"][1]["content"][1]["image_url"]["detail"] == "low"
    assert seen_payloads[1][0]["messages"][1]["content"][1]["image_url"]["detail"] == "high"

    # OpenAI reasoning models receive reasoning_effort="low" + >=24576 max_completion_tokens + extended timeout
    spec_tr = ProviderSpec(provider="openai", model="gpt-5-mini", api_key="sk-test", thinking="default")
    await call_text_json(spec_tr, system_instruction="sys", prompt="hello")
    tr_payload, tr_timeout = seen_payloads[2]
    assert tr_payload["reasoning_effort"] == "low"
    assert tr_payload["max_completion_tokens"] >= 24576
    assert tr_timeout >= 240.0

    # Timeout exception with empty str(exc) reports exception class name instead of blank "()"
    import httpx

    async def fake_timeout_post(self, url, **kwargs):
        raise httpx.ReadTimeout("")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_timeout_post)
    with pytest.raises(RuntimeError, match="ReadTimeout"):
        await _post_json_with_retry("https://example.com", {}, {}, timeout=1.0, provider_label="Ollama", max_attempts=1)


@pytest.mark.asyncio
async def test_translator_adaptive_batch_split_and_provider_batch_sizes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app.core.config import settings
    from app.providers.translation.base import TranslatableSegment
    from app.providers.translation.gemini_provider import GeminiTranslator

    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    pk.add_key("openai", api_key="sk-test-oai", label="OAI")

    tr_oai_mini = GeminiTranslator(provider="openai", model_name="gpt-5-mini")
    tr_oai_full = GeminiTranslator(provider="openai", model_name="gpt-5")
    tr_ollama = GeminiTranslator(provider="ollama", model_name="gemma4:12b")
    assert tr_oai_mini.batch_size == 18
    assert tr_oai_full.batch_size == 35
    assert tr_ollama.batch_size == 18

    # Simulate a 30-segment batch failing with truncated empty JSON when size > 20, then succeeding when split to 15+15
    calls = []

    async def fake_generate(self, contents, system_prompt, schema, temperature, **kwargs):
        payload = json.loads(contents)
        batch = payload["target_batch"]
        calls.append(len(batch))
        if len(batch) > 20:
            # Emulate gpt-5-mini hitting max_completion_tokens -> empty text -> Pydantic EOF error
            return SimpleNamespace(text="")
        return SimpleNamespace(
            text=json.dumps({
                "translations": [{"index": item["id"], "translated_text": f"שורה {item['id']}"} for item in batch]
            })
        )

    monkeypatch.setattr(GeminiTranslator, "_generate", fake_generate)

    segs = [TranslatableSegment(index=i, start_time=float(i), end_time=i + 1.0, original_text=f"Line {i}") for i in range(1, 31)]
    tr30 = GeminiTranslator(provider="openai", model_name="gpt-5-mini", batch_size=30)
    res = await tr30.translate_segments(segs, "English", "Hebrew")
    assert calls == [30, 15, 15]
    assert all(not s.translation_failed and s.translated_text == f"שורה {s.index}" for s in res)


@pytest.mark.asyncio
async def test_ollama_passes_num_ctx_and_inlined_schema_with_fallback(monkeypatch):
    from app.providers.llm import ProviderSpec, call_text_json
    from app.providers.translation.gemini_provider import TranslationBatchResponseSchema

    seen = []

    async def fake_post(url, headers, payload, timeout, provider_label, max_attempts=4):
        seen.append(payload)
        if len(seen) == 2:
            # First call on older mock rejects dict format with HTTP 400 -> fallback to "json"
            raise RuntimeError("Ollama HTTP 400: invalid format")
        return {
            "message": {"content": json.dumps({"translations": []})},
            "prompt_eval_count": 500,
            "eval_count": 10,
        }

    monkeypatch.setattr("app.providers.llm.client._post_json_with_retry", fake_post)
    spec = ProviderSpec(provider="ollama", model="gemma4:12b", api_key="")

    await call_text_json(spec, system_instruction="sys", prompt="prompt", schema_cls=TranslationBatchResponseSchema)
    first = seen[0]
    assert first["options"]["num_ctx"] >= 16384
    assert first["options"]["num_predict"] == 8192
    assert isinstance(first["format"], dict)
    assert "$defs" not in first["format"]
    assert "translations" in (first["format"].get("properties") or {})

    # Fallback path when Ollama returns HTTP 400 for schema dict
    await call_text_json(spec, system_instruction="sys", prompt="prompt", schema_cls=TranslationBatchResponseSchema)
    assert isinstance(seen[1]["format"], dict)
    assert seen[2]["format"] == "json"


def test_saved_local_ollama_settings_not_overwritten_by_startup_discovery(client: TestClient, monkeypatch):
    from app.core.config import settings
    from app.core.user_settings import AppSettings, GeminiCallSettings, load_settings, save_settings, sync_defaults_with_discovered

    monkeypatch.setattr(settings, "gemini_api_key", "AIzaSyDummyEnvKeyForTest")
    saved = AppSettings(
        translation=GeminiCallSettings(provider="ollama", model="gemma4:12b", thinking="default", flex=False),
        glossary=GeminiCallSettings(provider="ollama", model="gemma4:12b", thinking="default", flex=False),
        vision=GeminiCallSettings(provider="google", model="gemini-3.5-flash-lite", thinking="default", flex=True),
    )
    save_settings(saved)

    # 1) Even without an explicit Ollama vault key, live Google + local Ollama discovery preserves Ollama choices
    catalog_both = [
        {"provider": "google", "id": "gemini-3.8-flash", "available": True, "tasks": ["translation", "glossary", "vision"], "suitability_by_task": {"translation": "recommended", "glossary": "recommended", "vision": "recommended"}, "rank": 8},
        {"provider": "google", "id": "gemini-3.5-flash-lite", "available": True, "tasks": ["glossary", "vision"], "suitability_by_task": {"vision": "recommended"}, "rank": 30},
        {"provider": "ollama", "id": "gemma4:12b", "available": True, "tasks": ["translation", "glossary"], "suitability_by_task": {"translation": "suitable", "glossary": "suitable"}, "rank": 30},
    ]
    after_live = sync_defaults_with_discovered(catalog_both)
    assert after_live.translation.provider == "ollama" and after_live.translation.model == "gemma4:12b"
    assert after_live.glossary.provider == "ollama" and after_live.glossary.model == "gemma4:12b"
    assert after_live.vision.provider == "google" and after_live.vision.model == "gemini-3.5-flash-lite"

    # 2) Even if Ollama isn't running yet at startup, saved provider in settings.json MUST NOT revert to Google
    catalog_google_only = catalog_both[:2]
    after_offline = sync_defaults_with_discovered(catalog_google_only)
    assert after_offline.translation.provider == "ollama" and after_offline.translation.model == "gemma4:12b"
    assert load_settings().translation.provider == "ollama"


def test_gpt_oss_and_tagged_ollama_models_never_misrouted_to_openai(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.core.provider_catalog import infer_provider_from_model, is_chat_candidate
    from app.core.user_settings import AppSettings, GeminiCallSettings, save_settings
    from app.pipeline.context import PipelineContext
    from app.pipeline.stages.translate_stage import TranslateStage

    assert infer_provider_from_model("gpt-oss:20b") == "ollama"
    assert infer_provider_from_model("gpt-oss:120b-cloud") == "ollama"
    assert infer_provider_from_model("gpt-5-mini") == "openai"
    assert is_chat_candidate("openai", "gpt-oss:20b") is False
    assert is_chat_candidate("ollama", "gpt-oss:20b") is True

    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    save_settings(AppSettings(
        translation=GeminiCallSettings(provider="ollama", model="gpt-oss:20b", thinking="default", flex=False),
        glossary=GeminiCallSettings(provider="ollama", model="gemma4:12b", thinking="default", flex=False),
        vision=GeminiCallSettings(provider="google", model="gemini-3.5-flash-lite", thinking="default", flex=True),
    ))
    ctx = PipelineContext(job_id="j_oss", file_name="v.mkv", video_path="v.mkv", gemini_model="gpt-oss:20b")
    translator = TranslateStage()._make_translator(ctx)
    assert translator.provider == "ollama"
    assert translator.model_name == "gpt-oss:20b"


def test_strict_candidate_filtering_and_openai_pricing_rules():
    from app.core.provider_catalog import (
        build_model_entry,
        estimated_image_tokens,
        estimated_reasoning_tokens,
        is_chat_candidate,
        pricing_for_model,
    )

    # Retired, image-gen, omni, customtools, latest, -pro (/v1/responses-only), -codex, search-api, stable-diffusion excluded
    for excluded_prov, excluded_id in [
        ("google", "gemini-2.5-flash"),
        ("google", "gemini-2.0-flash"),
        ("google", "gemini-3.5-flash"),
        ("google", "gemini-flash-latest"),
        ("google", "gemini-2.5-flash-image"),
        ("google", "nano-banana-pro-preview"),
        ("google", "gemini-omni-2"),
        ("google", "gemini-2.5-flash-customtools"),
        ("openai", "gpt-5-pro"),
        ("openai", "gpt-5.4-pro"),
        ("openai", "o1-pro"),
        ("openai", "o3-pro"),
        ("openai", "gpt-5.3-codex"),
        ("openai", "gpt-5-search-api"),
        ("ollama", "brxce/stable-diffusion-prompt-generator:latest"),
    ]:
        assert is_chat_candidate(excluded_prov, excluded_id) is False, f"{excluded_prov}/{excluded_id} should be excluded"

    for allowed_prov, allowed_id in [
        ("google", "gemini-3.8-flash"),
        ("openai", "gpt-5.4-mini"),
        ("openai", "gpt-5.4-nano"),
        ("openai", "gpt-6-mini"),
        ("anthropic", "claude-sonnet-4-5"),
        ("ollama", "gemma4:12b"),
        ("ollama", "gpt-oss:20b"),
    ]:
        assert is_chat_candidate(allowed_prov, allowed_id) is True, f"{allowed_prov}/{allowed_id} should be kept"

    # gpt-5.4-mini & gpt-5.4-nano must not fall through to generic Flagship GPT-5 pricing
    in_54m, out_54m, *_ = pricing_for_model("openai", "gpt-5.4-mini")
    assert (in_54m, out_54m) == (0.75, 4.50)
    in_54n, out_54n, *_ = pricing_for_model("openai", "gpt-5.4-nano")
    assert (in_54n, out_54n) == (0.20, 1.25)
    in_6m, out_6m, *_ = pricing_for_model("openai", "gpt-6-mini")
    assert (in_6m, out_6m) == (0.25, 2.00)

    entry_54m = build_model_entry("openai", "gpt-5.4-mini", available=True)
    assert entry_54m["label"] == "GPT-5.4 mini"
    assert set(entry_54m["tasks"]) == {"translation", "glossary", "vision"}

    entry_54n = build_model_entry("openai", "gpt-5.4-nano", available=True)
    assert entry_54n["label"] == "GPT-5.4 nano"
    assert entry_54n["tasks"] == ["vision"]

    assert estimated_image_tokens("openai", "gpt-5.4-mini", "medium") == 235
    assert estimated_image_tokens("openai", "gpt-5.4-nano", "medium") == 355
    assert estimated_reasoning_tokens("openai", "gpt-6-mini", "low") == 650


@pytest.mark.asyncio
async def test_openai_dated_snapshots_and_ollama_show_discovery(monkeypatch):
    import httpx
    from app.core import model_discovery as md
    from app.core.provider_catalog import build_model_entry

    class FakeResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    async def fake_get(self, url, **kwargs):
        if "/models" in str(url):
            return FakeResp(200, {
                "data": [
                    {"id": "gpt-5.4-mini"},
                    {"id": "gpt-5.4-mini-2026-03-17"},
                    {"id": "gpt-5-mini-2025-08-07"},
                    {"id": "gpt-5.4-pro"},
                    {"id": "gpt-4o-2024-05-13"},  # base gpt-4o omitted -> kept!
                ]
            })
        if "/api/tags" in str(url):
            return FakeResp(200, {
                "models": [
                    {"name": "gemma4:12b", "details": {"parameter_size": "12.2B"}},
                    {"name": "custom-small:latest", "details": {"parameter_size": "3.8B"}},
                    {"name": "gpt-oss:20b", "details": {"parameter_size": "20.9B"}},
                    {"name": "brxce/stable-diffusion-prompt-generator:latest", "details": {}},
                ]
            })
        raise AssertionError(f"Unexpected GET {url}")

    async def fake_post(self, url, **kwargs):
        model_req = (kwargs.get("json") or {}).get("model")
        if model_req == "gemma4:12b":
            return FakeResp(200, {"capabilities": ["completion", "vision", "thinking"]})
        if model_req == "custom-small:latest":
            return FakeResp(200, {"capabilities": ["completion"]})
        return FakeResp(200, {"capabilities": ["completion", "tools"]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    oai_models = await md._fetch_openai_models("sk-test", None)
    oai_ids = [mid for mid, _ in oai_models]
    assert "gpt-5.4-mini" in oai_ids
    assert "gpt-5.4-mini-2026-03-17" not in oai_ids  # deduped because base gpt-5.4-mini present
    assert "gpt-5-mini-2025-08-07" in oai_ids  # kept because base not present
    assert "gpt-4o-2024-05-13" in oai_ids
    assert "gpt-5.4-pro" not in oai_ids  # excluded /v1/responses-only model

    ollama_models = dict(await md._fetch_ollama_models("", "http://127.0.0.1:11434"))
    assert "brxce/stable-diffusion-prompt-generator:latest" not in ollama_models
    assert "vision" in (ollama_models["gemma4:12b"].get("capabilities") or [])

    gemma_entry = build_model_entry(
        "ollama", "gemma4:12b", available=True,
        capabilities=ollama_models["gemma4:12b"]["capabilities"],
        details=ollama_models["gemma4:12b"]["details"],
    )
    assert "vision" in gemma_entry["tasks"]
    assert "translation" in gemma_entry["recommended_for"]

    small_entry = build_model_entry(
        "ollama", "custom-small:latest", available=True,
        capabilities=ollama_models["custom-small:latest"]["capabilities"],
        details=ollama_models["custom-small:latest"]["details"],
    )
    assert small_entry["tasks"] == []  # 3.8B via details.parameter_size -> not suitable for translation

    oss_entry = build_model_entry(
        "ollama", "gpt-oss:20b", available=True,
        capabilities=ollama_models["gpt-oss:20b"]["capabilities"],
        details=ollama_models["gpt-oss:20b"]["details"],
    )
    assert "translation" in oss_entry["recommended_for"]


def test_get_settings_awaits_initial_discovery_race(client: TestClient, monkeypatch):
    from app.core import model_discovery as md

    monkeypatch.setenv("SUBBUILD_OFFLINE_DISCOVERY", "0")
    md.reset_discovery_cache()

    calls = 0

    async def fake_fetch_ollama(api_key, base_url=None, *, is_local_probe=False):
        nonlocal calls
        calls += 1
        return [("gemma4:12b", {"capabilities": ["completion", "vision"], "details": {"parameter_size": "12.2B"}, "base_url": "http://127.0.0.1:11434"})]

    monkeypatch.setattr(md, "_fetch_ollama_models", fake_fetch_ollama)

    res1 = client.get("/api/settings")
    assert res1.status_code == 200
    gemma = next((m for m in res1.json()["catalog"] if m["id"] == "gemma4:12b"), None)
    assert gemma is not None and gemma["available"] is True
    assert "vision" in gemma["tasks"]
    assert calls == 1

    # Second call uses cached initial discovery result without re-probing
    res2 = client.get("/api/settings")
    assert res2.status_code == 200
    assert calls == 1


@pytest.mark.asyncio
async def test_anthropic_temperature_omission_and_http400_retry(monkeypatch):
    from app.providers.llm.client import ProviderSpec, call_text_json

    sent_payloads = []

    async def fake_post(url, headers, payload, timeout, provider_label, max_attempts=4):
        sent_payloads.append(dict(payload))
        if payload.get("model") == "claude-future-custom" and "temperature" in payload:
            raise RuntimeError(
                'Anthropic HTTP 400: {"type":"error","error":{"type":"invalid_request_error",'
                '"message":"\'temperature\' is deprecated for this model."}}'
            )
        return {
            "content": [{"type": "text", "text": json.dumps({"status": "ok"})}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    monkeypatch.setattr("app.providers.llm.client._post_json_with_retry", fake_post)

    # Known Claude 5.5 omits temperature upfront
    res_55 = await call_text_json(
        ProviderSpec(provider="anthropic", model="claude-sonnet-5-5", api_key="sk-ant-1"),
        system_instruction="sys",
        prompt="hi",
    )
    assert json.loads(res_55.text) == {"status": "ok"}
    assert "temperature" not in sent_payloads[-1]

    # Claude 4.5 includes temperature
    sent_payloads.clear()
    await call_text_json(
        ProviderSpec(provider="anthropic", model="claude-sonnet-4-5", api_key="sk-ant-1"),
        system_instruction="sys",
        prompt="hi",
        temperature=0.2,
    )
    assert sent_payloads[-1]["temperature"] == 0.2

    # Unrecognized Anthropic model rejecting temperature retries once without temperature
    sent_payloads.clear()
    res_future = await call_text_json(
        ProviderSpec(provider="anthropic", model="claude-future-custom", api_key="sk-ant-1"),
        system_instruction="sys",
        prompt="hi",
        temperature=0.2,
    )
    assert json.loads(res_future.text) == {"status": "ok"}
    assert len(sent_payloads) == 2
    assert "temperature" in sent_payloads[0]
    assert "temperature" not in sent_payloads[1]


def test_2026_models_pricing_and_promo_schedule():
    from datetime import date
    from app.core.provider_catalog import build_model_entry

    sonnet55 = build_model_entry("anthropic", "claude-sonnet-5-5", available=True)
    assert sonnet55["input_per_m"] == 2.00 and sonnet55["output_per_m"] == 10.00
    opus55 = build_model_entry("anthropic", "claude-opus-5-5", available=True)
    assert opus55["input_per_m"] == 4.00 and opus55["output_per_m"] == 20.00
    opus48 = build_model_entry("anthropic", "claude-opus-4-8", available=True)
    assert opus48["input_per_m"] == 5.00 and opus48["output_per_m"] == 25.00
    haiku55 = build_model_entry("anthropic", "claude-haiku-5-5", available=True)
    assert haiku55["input_per_m"] == 0.10 and haiku55["output_per_m"] == 0.50
    fable52 = build_model_entry("anthropic", "claude-fable-5-2", available=True)
    assert fable52["input_per_m"] == 8.00 and fable52["output_per_m"] == 40.00

    luna56 = build_model_entry("openai", "gpt-5.6-luna", available=True)
    assert luna56["input_per_m"] == 0.20 and luna56["output_per_m"] == 1.20
    assert luna56["tasks"] == ["vision"]

    hydrogen = build_model_entry("google", "gemini-4-hydrogen", available=True)
    assert hydrogen["input_per_m"] == 1.50 and hydrogen["output_per_m"] == 7.50
    neon = build_model_entry("google", "gemini-4-neon", available=True)
    assert neon["input_per_m"] == 3.00 and neon["output_per_m"] == 15.00

    carbon_promo = build_model_entry("google", "gemini-4-carbon", available=True, day=date(2026, 11, 25))
    assert carbon_promo["input_per_m"] == 2.00 and carbon_promo["output_per_m"] == 10.00
    assert carbon_promo["price_change"]["after"] == "2026-11-30"
    carbon_std = build_model_entry("google", "gemini-4-carbon", available=True, day=date(2026, 12, 1))
    assert carbon_std["input_per_m"] == 4.00 and carbon_std["output_per_m"] == 20.00
