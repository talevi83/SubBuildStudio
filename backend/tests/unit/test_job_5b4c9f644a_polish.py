"""Unit tests for job_5b4c9f644a polish fixes: 2-minute Flex timeout/guard and long-film cluster re-split."""
import time
import types

import numpy as np
import pytest

import app.core.model_catalog as mc
import app.providers.gemini_calls as gc
from app.pipeline import diarization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flex_tier(config) -> str | None:
    tier = getattr(config, "service_tier", None)
    return getattr(tier, "value", tier)


def _voices(
    n_speakers: int,
    per_speaker: int,
    shared: float,
    noise_var: float,
    step: float,
    seed: int = 11,
):
    """Build synthetic 512-dim speaker embeddings sharing a common room component.

    Cross-voice window distance ~= 1 - shared / (1 + noise_var),
    within-voice window distance ~= 1 - 1 / (1 + noise_var),
    centroid distance ~= 1 - shared.
    """
    rng = np.random.default_rng(seed)
    dim = 512
    common = rng.normal(size=dim)
    common /= np.linalg.norm(common)
    embs, truth, windows = [], [], []
    t = 0.0
    for k in range(n_speakers):
        u = rng.normal(size=dim)
        u -= (u @ common) * common
        u /= np.linalg.norm(u)
        voice = np.sqrt(shared) * common + np.sqrt(1.0 - shared) * u
        for _ in range(per_speaker):
            noise = rng.normal(size=dim)
            noise *= np.sqrt(noise_var) / np.linalg.norm(noise)
            v = voice + noise
            embs.append(v / np.linalg.norm(v))
            truth.append(k)
            windows.append((t, t + 1.6))
            t += step
    return np.vstack(embs).astype(np.float32), truth, windows


# ---------------------------------------------------------------------------
# 1. Flex timeout (2 min) + asyncio safety guard + immediate 504 fallback
# ---------------------------------------------------------------------------

def test_flex_timeout_bounded_around_two_minutes_above_slow_threshold():
    assert 60_000 < gc.FLEX_TIMEOUT_MS <= 120_000
    assert gc.FLEX_TIMEOUT_MS / 1000.0 > gc.FLEX_SLOW_SECONDS


@pytest.mark.asyncio
async def test_hung_flex_request_trips_client_guard_and_falls_back_once(monkeypatch):
    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(gc.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(gc, "FLEX_CLIENT_GUARD_SECONDS", 0.05)

    tiers_called = []

    class _HungFlexClient:
        def __init__(self):
            self.models = self

        def generate_content(self, model, contents, config):
            tier = _flex_tier(config)
            tiers_called.append(tier)
            if tier == "flex":
                time.sleep(0.3)
            meta = types.SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                thoughts_token_count=0,
                traffic_type="ON_DEMAND",
            )
            return types.SimpleNamespace(text="{}", usage_metadata=meta)

    from google.genai import types as gt

    usage = mc.UsageTracker()
    state = gc.FlexState()
    await gc.generate_content(
        _HungFlexClient(),
        model="gemini-3.8-flash",
        contents="hi",
        config=gt.GenerateContentConfig(temperature=0.2),
        flex=True,
        usage=usage,
        stage="translation",
        flex_state=state,
    )
    assert tiers_called == ["flex", None]
    assert usage.stages["translation"].flex_fallbacks == 1


@pytest.mark.asyncio
async def test_flex_504_deadline_exceeded_falls_back_without_retry(monkeypatch):
    sleeps = []

    async def _rec_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(gc.asyncio, "sleep", _rec_sleep)

    tiers_called = []

    class _DeadlineClient:
        def __init__(self):
            self.models = self

        def generate_content(self, model, contents, config):
            tier = _flex_tier(config)
            tiers_called.append(tier)
            if tier == "flex":
                raise Exception("504 DEADLINE_EXCEEDED. Deadline expired before operation could complete.")
            meta = types.SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                thoughts_token_count=0,
                traffic_type="ON_DEMAND",
            )
            return types.SimpleNamespace(text="{}", usage_metadata=meta)

    from google.genai import types as gt

    usage = mc.UsageTracker()
    await gc.generate_content(
        _DeadlineClient(),
        model="gemini-3.8-flash",
        contents="hi",
        config=gt.GenerateContentConfig(temperature=0.2),
        flex=True,
        usage=usage,
        stage="translation",
    )
    assert tiers_called == ["flex", None]
    assert sleeps == []
    assert usage.stages["translation"].flex_fallbacks == 1


# ---------------------------------------------------------------------------
# 2. Long-film mega-cluster re-split
# ---------------------------------------------------------------------------

def test_long_film_splits_oversized_speaker_clusters_into_pure_voices():
    emb, truth, windows = _voices(4, 60, shared=0.61, noise_var=0.6, step=12.0, seed=11)
    assert (windows[-1][1] - windows[0][0]) >= diarization.LONG_RECORDING_SECONDS

    labels = diarization.assign_window_speakers(emb, windows)
    unique_labels = {lbl for lbl in labels if lbl}
    assert len(unique_labels) == 4
    for lbl in unique_labels:
        assert len({truth[i] for i, cur in enumerate(labels) if cur == lbl}) == 1


def test_same_embeddings_on_short_span_remain_merged_without_resplit():
    emb, _, windows = _voices(4, 60, shared=0.61, noise_var=0.6, step=2.0, seed=11)
    assert (windows[-1][1] - windows[0][0]) < diarization.LONG_RECORDING_SECONDS

    labels = diarization.assign_window_speakers(emb, windows)
    assert len({lbl for lbl in labels if lbl}) < 4


def test_single_voice_over_long_film_stays_one_speaker():
    emb, _, windows = _voices(1, 240, shared=0.61, noise_var=0.6, step=12.0, seed=11)
    assert (windows[-1][1] - windows[0][0]) >= diarization.LONG_RECORDING_SECONDS

    labels = diarization.assign_window_speakers(emb, windows)
    assert len({lbl for lbl in labels if lbl}) == 1


def test_resplit_oversized_clusters_noop_below_long_recording_threshold():
    emb, _, windows = _voices(4, 20, shared=0.61, noise_var=0.6, step=2.0, seed=11)
    initial = ["SPEAKER_01"] * len(windows)
    result = diarization._resplit_oversized_clusters(emb, initial, windows)
    assert result == initial
