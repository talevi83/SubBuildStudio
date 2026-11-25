"""Tests for local ONNX acoustic speaker diarization, gender/glossary sync, retroactive APIs, and speaker exports."""
import io
import json
import types
import uuid
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from app.pipeline import diarization, speech_regions
from app.pipeline.context import PipelineContext, RawSegment, TranscribedWord
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.stages.diarize_stage import DiarizeStage
from app.pipeline.stages.translate_stage import TranslateStage
from app.providers.translation import gemini_provider as gp
from app.providers.translation.base import TranslatableSegment


def test_extract_fbank_80_produces_finite_cmn_features():
    sr = 16000
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
    sine = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    feats = diarization.extract_fbank_80(sine, sample_rate=sr)
    assert feats.dtype == np.float32
    assert feats.ndim == 2 and feats.shape[1] == 80 and feats.shape[0] > 90
    assert np.all(np.isfinite(feats))
    # Per-utterance CMN: each frequency bin has near-zero mean across time
    assert np.max(np.abs(feats.mean(axis=0))) < 1e-4

    # Empty input returns safe finite placeholder
    empty_feats = diarization.extract_fbank_80(np.array([], dtype=np.float32))
    assert empty_feats.shape == (1, 80)
    assert np.all(np.isfinite(empty_feats))


def test_cluster_embeddings_separates_speakers_and_absorbs_singletons():
    rng = np.random.default_rng(7)
    base_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    base_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    base_c = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # Chronological appearance: A, A, B, B, slightly noisy A-ish singleton
    noisy_a = np.array([0.82, 0.05, 0.0, 0.56], dtype=np.float32)  # cos dist from A ~ 0.18 (< 0.60 absorb)
    emb2 = np.vstack([
        base_a + rng.normal(0, 0.02, 4),
        base_a + rng.normal(0, 0.02, 4),
        base_b + rng.normal(0, 0.02, 4),
        base_b + rng.normal(0, 0.02, 4),
        noisy_a,
    ]).astype(np.float32)
    labels2 = diarization.cluster_embeddings(emb2, threshold=0.35)
    assert labels2 == ["SPEAKER_01", "SPEAKER_01", "SPEAKER_02", "SPEAKER_02", "SPEAKER_01"]

    # Three distinct speakers with explicit min_speakers/max_speakers constraints
    emb3 = np.vstack([base_a, base_b, base_c, base_a]).astype(np.float32)
    assert diarization.cluster_embeddings(emb3, threshold=0.45) == [
        "SPEAKER_01",
        "SPEAKER_02",
        "SPEAKER_03",
        "SPEAKER_01",
    ]
    assert len(set(diarization.cluster_embeddings(emb3, threshold=0.45, max_speakers=2))) == 2
    assert len(set(diarization.cluster_embeddings(np.vstack([base_a, base_a * 0.99]), threshold=0.95, min_speakers=2))) == 2


def test_a_lone_window_is_a_speaker_in_a_short_clip_and_noise_in_a_long_one():
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    # 0.70 from the others: beyond the 0.66 clustering threshold, within the 0.72 noise limit
    lone = np.array([0.3, 0.3, 0.9055, 0.0], dtype=np.float32)
    assert 0.66 < 1 - float(lone @ b / np.linalg.norm(lone)) < 0.72

    # A two-line conversation: one window for one voice, three for the other
    short = np.vstack([lone, b, b * 0.99, b * 1.01]).astype(np.float32)
    assert diarization.cluster_embeddings(short, threshold=0.66) == [
        "SPEAKER_01", "SPEAKER_02", "SPEAKER_02", "SPEAKER_02",
    ]
    # A long recording: the same lone window is absorbed as noise
    long = np.vstack([np.tile(a, (5, 1)), np.tile(b, (5, 1)), lone]).astype(np.float32)
    assert len(set(diarization.cluster_embeddings(long, threshold=0.66))) == 2


def test_split_raw_segments_by_speaker_and_assign_skips_onscreen():
    timeline = [
        {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_01"},
        {"start": 1.2, "end": 2.5, "speaker": "SPEAKER_02"},
    ]
    raw = [
        RawSegment(
            index=1,
            start_time=0.0,
            end_time=2.5,
            text="Wait right here. Let me help.",
            words=[
                TranscribedWord("Wait", 0.0, 0.3),
                TranscribedWord(" right", 0.3, 0.6),
                TranscribedWord(" here.", 0.6, 1.0),
                TranscribedWord(" Let", 1.3, 1.6),
                TranscribedWord(" me", 1.6, 1.9),
                TranscribedWord(" help.", 1.9, 2.4),
            ],
        )
    ]
    split = diarization.split_raw_segments_by_speaker(raw, timeline)
    assert len(split) == 2
    assert split[0].speaker == "SPEAKER_01"
    assert split[0].text == "Wait right here."
    assert split[1].speaker == "SPEAKER_02"
    assert split[1].text == "Let me help."

    segs = [
        {"start_time": 0.1, "end_time": 1.0, "segment_type": "DIALOGUE"},
        {"start_time": 1.3, "end_time": 2.2, "segment_type": "ON_SCREEN", "speaker": "SPEAKER_02"},
        {"start_time": 1.4, "end_time": 2.4, "segment_type": "COMMS"},
    ]
    diarization.assign_speakers_to_segments(segs, timeline)
    assert segs[0]["speaker"] == "SPEAKER_01"
    assert segs[1]["speaker"] is None
    assert segs[2]["speaker"] == "SPEAKER_02"


@pytest.mark.asyncio
async def test_diarize_stage_offline_missing_model_warns_non_fatally(tmp_path, monkeypatch):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    monkeypatch.setattr(diarization, "model_path", lambda: tmp_path / "missing.onnx")
    monkeypatch.setenv("SUBBUILD_OFFLINE", "1")

    ctx = PipelineContext(
        job_id="job_offline_diar",
        video_path="clip.mp4",
        file_name="clip.mp4",
        audio_wav_path=str(wav),
        enable_diarization=True,
        raw_segments=[RawSegment(index=1, start_time=0.0, end_time=2.0, text="Hello there.")],
    )
    stage = DiarizeStage()
    await stage.execute(ctx)
    assert any("skipped" in w.lower() for w in ctx.warnings)
    assert ctx.speakers == {}


@pytest.mark.asyncio
async def test_glossary_speaker_sync_and_translation_payload(monkeypatch):
    seen_glossary_prompt = []
    seen_batch_payloads = []

    def fake_behavior(model, contents, config):
        if contents.startswith("{"):
            payload = json.loads(contents)
            seen_batch_payloads.append(payload)
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "translations": [
                            {"index": item["id"], "translated_text": f"ת{item['id']}"}
                            for item in payload["target_batch"]
                        ]
                    }
                )
            )
        seen_glossary_prompt.append(contents)
        return types.SimpleNamespace(
            text=json.dumps(
                {
                    "summary": "Podcast interview",
                    "characters": [
                        {
                            "name": "Sarah",
                            "target_name": "שרה",
                            "gender": "female",
                            "notes": "Host",
                            "speaker_id": "SPEAKER_01",
                        }
                    ],
                    "terms": [],
                    "register_notes": "Casual",
                }
            )
        )

    monkeypatch.setattr(gp.asyncio, "sleep", lambda _: None)
    translator = gp.GeminiTranslator(api_key="test-key", batch_size=10)
    translator.client = types.SimpleNamespace(
        models=types.SimpleNamespace(generate_content=lambda model, contents, config: fake_behavior(model, contents, config))
    )

    ctx = PipelineContext(
        job_id="job_gloss_sync",
        video_path="clip.mp4",
        file_name="clip.mp4",
        enable_translation=True,
        target_language="Hebrew",
        source_language="en",
        speakers={"SPEAKER_01": {"label": "Speaker 1", "gender": "unknown", "color": "#38BDF8"}},
        refined_segments=[
            {
                "index": 1,
                "start_time": 0.0,
                "end_time": 2.0,
                "original_text": "I am ready to begin now.",
                "translated_text": None,
                "translation_failed": False,
                "segment_type": "DIALOGUE",
                "speaker": "SPEAKER_01",
            }
        ],
    )
    monkeypatch.setattr(settings, "gemini_glossary_pass", True)
    await TranslateStage(translator=translator).execute(ctx)

    # Glossary input prefixed [SPEAKER_01]
    assert seen_glossary_prompt and "[SPEAKER_01] I am ready" in seen_glossary_prompt[0]
    # Character speaker_id synced name & female gender back into ctx.speakers
    assert ctx.speakers["SPEAKER_01"]["label"] == "שרה"
    assert ctx.speakers["SPEAKER_01"]["gender"] == "female"
    # Translation batch received enriched speaker identity (even off-screen)
    assert seen_batch_payloads[0]["target_batch"][0]["speaker"] == "SPEAKER_01 (שרה, female)"


def test_orchestrator_schedules_diarizing_between_transcribing_and_refining():
    orch = PipelineOrchestrator("job_sched")
    try:
        ctx_on = PipelineContext(job_id="job_sched", video_path="x.mp4", file_name="x.mp4", enable_diarization=True)
        stage_names_on = [st.value for st, _ in orch._stages(ctx_on)]
        assert stage_names_on.index("TRANSCRIBING") + 1 == stage_names_on.index("DIARIZING")
        assert stage_names_on.index("DIARIZING") < stage_names_on.index("REFINING")

        ctx_off = PipelineContext(job_id="job_sched", video_path="x.mp4", file_name="x.mp4", enable_diarization=False)
        stage_names_off = [st.value for st, _ in orch._stages(ctx_off)]
        assert "DIARIZING" not in stage_names_off
    finally:
        from app.pipeline.orchestrator import active_pipelines
        active_pipelines.pop("job_sched", None)


def test_speakers_put_post_diarize_history_and_colored_exports(tmp_path, monkeypatch):
    video = tmp_path / "Panel.mp4"
    video.write_bytes(b"video-bytes")
    fake_onnx = tmp_path / "cam++.onnx"
    fake_onnx.write_bytes(b"0" * 1024)
    job_id = f"job_{uuid.uuid4().hex[:10]}"

    monkeypatch.setattr(speech_regions, "_extract", lambda *a, **kw: None)
    monkeypatch.setattr(diarization, "ensure_diarization_model", lambda *a, **kw: fake_onnx)
    monkeypatch.setattr(
        diarization,
        "diarize_audio_spans",
        lambda *a, **kw: (
            [
                {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
                {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_02"},
            ],
            {
                "SPEAKER_01": {"label": "Speaker 1", "gender": "unknown", "color": "#38BDF8"},
                "SPEAKER_02": {"label": "Speaker 2", "gender": "unknown", "color": "#F472B6"},
            },
        ),
    )

    with TestClient(app) as client:
        with Session(engine) as session:
            session.add(
                Job(
                    id=job_id,
                    video_path=str(video),
                    file_name="Panel.mp4",
                    status=JobStatus.COMPLETED,
                    source_language="en",
                    target_language="Hebrew",
                    video_duration=10.0,
                )
            )
            session.add(
                SubtitleSegmentRecord(
                    job_id=job_id, index=1, start_time=0.0, end_time=2.0, original_text="First line", translated_text="שורה ראשונה"
                )
            )
            session.add(
                SubtitleSegmentRecord(
                    job_id=job_id, index=2, start_time=2.0, end_time=4.0, original_text="Second line", translated_text="שורה שנייה"
                )
            )
            session.commit()

        # POST /diarize updates segments, speakers, and edit history
        r_diar = client.post(f"/api/jobs/{job_id}/diarize", json={})
        assert r_diar.status_code == 200, r_diar.text
        body = r_diar.json()
        assert body["enable_diarization"] is True
        assert [s["speaker"] for s in body["segments"]] == ["SPEAKER_01", "SPEAKER_02"]
        hist = client.get(f"/api/jobs/{job_id}/history").json()
        assert any(e["action"] == "diarize" for e in hist["entries"])

        # PUT /speakers edits labels/genders/colors
        r_put = client.put(
            f"/api/jobs/{job_id}/speakers",
            json={
                "speakers": {
                    "SPEAKER_01": {"label": "Host Dana", "gender": "female", "color": "#FF00AA"},
                    "SPEAKER_02": {"label": "Guest Avi", "gender": "male", "color": "#00FFAA"},
                }
            },
        )
        assert r_put.status_code == 200
        assert r_put.json()["speakers"]["SPEAKER_01"]["label"] == "Host Dana"

        # Speaker colors and custom labels propagate into ASS, FCPXML, MD, DOCX
        r_ass = client.get(f"/api/jobs/{job_id}/export/ass?use_translated=true&color_speakers=true")
        assert r_ass.status_code == 200
        assert "Style: Speaker_SPEAKER_01" in r_ass.text
        assert "Host Dana" in r_ass.text

        r_xml = client.get(f"/api/jobs/{job_id}/export/fcpxml?use_translated=true")
        assert r_xml.status_code == 200
        assert "Host Dana" in r_xml.text and "Guest Avi" in r_xml.text

        r_md = client.get(f"/api/jobs/{job_id}/export/markdown?use_translated=true")
        assert r_md.status_code == 200
        assert "**Host Dana**" in r_md.text and "**Guest Avi**" in r_md.text

        r_docx = client.get(f"/api/jobs/{job_id}/export/docx?use_translated=true")
        assert r_docx.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r_docx.content), "r") as zf:
            doc_xml = zf.read("word/document.xml").decode("utf-8")
            assert "Host Dana:" in doc_xml


def test_cluster_embeddings_vectorized_speed_progress_and_cancel():
    import threading
    import time
    from app.pipeline.context import PipelineCancelled

    rng = np.random.default_rng(42)
    bases = np.eye(4, 16, dtype=np.float32)
    rows = []
    truth = []
    for idx in range(600):
        spk = idx % 4
        rows.append(bases[spk] + rng.normal(0, 0.02, 16).astype(np.float32))
        truth.append(spk)
    emb = np.vstack(rows)

    progress_events: list[float] = []
    started = time.monotonic()
    labels = diarization.cluster_embeddings(
        emb,
        threshold=0.45,
        on_progress=progress_events.append,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 3.0
    assert len(labels) == 600
    assert len(set(labels)) == 4
    for spk in range(4):
        group_labels = {labels[i] for i, t in enumerate(truth) if t == spk}
        assert len(group_labels) == 1

    assert progress_events
    assert progress_events[-1] == 100.0
    assert progress_events == sorted(progress_events)

    cancel_flag = threading.Event()
    cancel_flag.set()
    with pytest.raises(PipelineCancelled):
        diarization.cluster_embeddings(emb[:10], cancel_flag=cancel_flag)


@pytest.mark.asyncio
async def test_orchestrator_progress_callback_tracks_stage_progress_percent(monkeypatch):
    from app.pipeline.orchestrator import active_pipelines

    job_id = "job_stage_pct_test"
    orch = PipelineOrchestrator(job_id)
    broadcasts = []
    try:
        monkeypatch.setattr(orch, "_update_job", lambda **kw: None)

        async def _fake_broadcast(payload):
            broadcasts.append(payload)

        monkeypatch.setattr(orch, "_broadcast", _fake_broadcast)
        cb = orch._progress_callback(JobStatus.DIARIZING, base_pct=63.0, share=0.10)
        await cb(42.5, None, None, "Identifying voices across timeline...", None)

        assert PipelineOrchestrator.get_stage_progress(job_id) == 42.5
        assert len(broadcasts) == 1
        assert broadcasts[0]["event"] == "STAGE_PROGRESS"
        assert broadcasts[0]["stage"] == "DIARIZING"
        assert broadcasts[0]["stage_progress_percent"] == 42.5
        assert broadcasts[0]["progress_percent"] == round(63.0 + 42.5 * 0.10, 1)
    finally:
        active_pipelines.pop(job_id, None)
        assert PipelineOrchestrator.get_stage_progress(job_id) is None


def test_diarize_audio_spans_two_stage_clustering_and_prune_unused(tmp_path):
    import wave

    sr = 16000
    dur = 60.0
    wav_path = tmp_path / "episode.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * int(sr * dur))

    # Build spans: 3 speakers across 15 long (>=1.3s) turns + 3 short (<0.8s) turns + 1 distant noise shard
    spans = []
    expected_vec_by_idx = {}
    base_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    base_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    base_c = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    noise = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    t = 1.0
    for i in range(15):
        spans.append((t, t + 1.5))
        expected_vec_by_idx[len(spans) - 1] = [base_a, base_b, base_c][i % 3]
        t += 2.2
    # Short utterances (<1.1s) should still project onto the 3 reliable speaker centroids
    for v in (base_a, base_b, base_c):
        spans.append((t, t + 0.5))
        expected_vec_by_idx[len(spans) - 1] = v
        t += 1.2
    # Single distant noise shard (>=0.80 cosine distance from A/B/C) must not create a 4th speaker
    spans.append((t, t + 0.6))
    expected_vec_by_idx[len(spans) - 1] = noise

    class FakeEmbedder:
        def __init__(self):
            self.call_idx = 0

        def embed_samples(self, _chunk, sample_rate=16000):
            v = expected_vec_by_idx[self.call_idx]
            self.call_idx += 1
            return v

    timeline, speakers = diarization.diarize_audio_spans(
        wav_path,
        spans,
        embedder=FakeEmbedder(),
        threshold=0.66,
    )
    assert len(speakers) == 3
    assert set(speakers.keys()) == {"SPEAKER_01", "SPEAKER_02", "SPEAKER_03"}
    # Short utterances at the end got assigned cleanly to the 3 voices (no 4th noise speaker)
    assert all(item["speaker"] in speakers for item in timeline)

    # prune_unused_speakers removes 0-line speakers while adding defaults for uncatalogued IDs
    pruned = diarization.prune_unused_speakers(
        {
            "SPEAKER_01": {"label": "Alex", "gender": "female", "color": "#111111"},
            "SPEAKER_02": {"label": "Ghost", "gender": "unknown", "color": "#222222"},
        },
        [
            {"speaker": "SPEAKER_01", "original_text": "Hi"},
            {"speaker": "SPEAKER_99", "original_text": "Hey"},
        ],
    )
    assert "SPEAKER_02" not in pruned
    assert pruned["SPEAKER_01"]["label"] == "Alex"
    assert "SPEAKER_99" in pruned


@pytest.mark.asyncio
async def test_deterministic_conversational_character_matching():
    from app.pipeline.speaker_match import compute_conversational_clues, match_speakers_to_characters

    glossary = {
        "characters": [
            {"name": "Alex Mack", "target_name": "אלכס מק", "gender": "female", "notes": "Lead", "speaker_id": ""},
            {"name": "Annie Mack", "target_name": "אייני מק", "gender": "female", "notes": "Sister", "speaker_id": ""},
            {"name": "Daniel Atron", "target_name": "דניאל אטרון", "gender": "male", "notes": "Exec", "speaker_id": ""},
            {"name": "Vince", "target_name": "וינס", "gender": "male", "notes": "Security", "speaker_id": ""},
        ]
    }
    speakers = {
        "SPEAKER_01": {"label": "Speaker 1", "gender": "unknown", "color": "#38BDF8"},
        "SPEAKER_02": {"label": "Speaker 2", "gender": "unknown", "color": "#F472B6"},
        "SPEAKER_03": {"label": "Speaker 3", "gender": "unknown", "color": "#4ADE80"},
        "SPEAKER_04": {"label": "Speaker 4", "gender": "unknown", "color": "#FBBF24"},
    }
    segments = [
        {"index": 1, "start_time": 1.0, "end_time": 2.5, "original_text": "You sent for me, Daniel Atron?", "speaker": "SPEAKER_02"},
        {"index": 2, "start_time": 2.8, "end_time": 4.5, "original_text": "Vince, I just got off the phone.", "speaker": "SPEAKER_03"},
        {"index": 3, "start_time": 5.0, "end_time": 6.2, "original_text": "Thank you, Daniel Atron.", "speaker": "SPEAKER_02"},
        {"index": 4, "start_time": 6.5, "end_time": 8.0, "original_text": "Good work, Vince.", "speaker": "SPEAKER_03"},
        {"index": 5, "start_time": 10.0, "end_time": 11.5, "original_text": "I'm Annie, remember?", "speaker": "SPEAKER_01"},
        {"index": 6, "start_time": 12.0, "end_time": 13.0, "original_text": "Alex?", "speaker": "SPEAKER_01"},
        {"index": 7, "start_time": 13.3, "end_time": 14.2, "original_text": "Yes?", "speaker": "SPEAKER_04"},
        {"index": 8, "start_time": 15.0, "end_time": 16.0, "original_text": "Hey Alex!", "speaker": "SPEAKER_01"},
        {"index": 9, "start_time": 16.3, "end_time": 17.5, "original_text": "I am right here.", "speaker": "SPEAKER_04"},
    ]
    clues, det = compute_conversational_clues(segments, speakers, glossary["characters"])
    assert clues["SPEAKER_01"]
    assert len(det) >= 3

    summary = await match_speakers_to_characters(segments, speakers, glossary, use_ai=False)
    assert summary["method"] == "deterministic"
    assert summary["matched"] == 4
    assert speakers["SPEAKER_01"]["label"] == "אייני מק"
    assert speakers["SPEAKER_01"]["gender"] == "female"
    assert speakers["SPEAKER_02"]["label"] == "וינס"
    assert speakers["SPEAKER_03"]["label"] == "דניאל אטרון"
    assert speakers["SPEAKER_04"]["label"] == "אלכס מק"
    by_name = {c["name"]: c for c in glossary["characters"]}
    assert by_name["Alex Mack"]["speaker_id"] == "SPEAKER_04"
    assert by_name["Annie Mack"]["speaker_id"] == "SPEAKER_01"


def test_match_speakers_and_merge_endpoints_with_ai_and_auto_merge(tmp_path, monkeypatch):
    from app.providers import llm as llm_mod
    from app.providers.llm.client import LLMResponse, ProviderSpec

    video = tmp_path / "Show.mp4"
    video.write_bytes(b"video")
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    monkeypatch.delenv("SUBBUILD_OFFLINE", raising=False)

    monkeypatch.setattr(
        llm_mod,
        "resolve_provider_spec",
        lambda provider, model, **kw: ProviderSpec(provider="google", model="gemini-3-flash-preview", api_key="fake-key"),
    )

    async def fake_call_text_json(*a, **kw):
        payload = {
            "matches": [
                {
                    "speaker_id": "SPEAKER_01",
                    "character_name": "Alex Mack",
                    "target_name": "אלכס מק",
                    "gender": "female",
                    "confidence": "high",
                    "evidence": "Replied to 'Alex?' and narrated in 1st person",
                    "merge_into_speaker_id": "",
                },
                {
                    "speaker_id": "SPEAKER_02",
                    "character_name": "Alex Mack",
                    "target_name": "אלכס מק",
                    "gender": "female",
                    "confidence": "high",
                    "evidence": "Same character voice split across scenes",
                    "merge_into_speaker_id": "SPEAKER_01",
                },
                {
                    "speaker_id": "SPEAKER_03",
                    "character_name": "Annie Mack",
                    "target_name": "אייני מק",
                    "gender": "female",
                    "confidence": "medium",
                    "evidence": "Addressed as Annie before replying",
                    "merge_into_speaker_id": "",
                },
            ]
        }
        return LLMResponse(text=json.dumps(payload), provider="google", model="gemini-3-flash-preview")

    monkeypatch.setattr(llm_mod, "call_text_json", fake_call_text_json)

    with TestClient(app) as client:
        with Session(engine) as session:
            session.add(
                Job(
                    id=job_id,
                    video_path=str(video),
                    file_name="Show.mp4",
                    status=JobStatus.COMPLETED,
                    source_language="en",
                    target_language="Hebrew",
                    enable_diarization=True,
                    speakers_json=json.dumps({
                        "SPEAKER_01": {"label": "Speaker 1", "gender": "unknown", "color": "#38BDF8"},
                        "SPEAKER_02": {"label": "Speaker 2", "gender": "unknown", "color": "#F472B6"},
                        "SPEAKER_03": {"label": "Speaker 3", "gender": "unknown", "color": "#4ADE80"},
                    }),
                    glossary_json=json.dumps({
                        "summary": "Teen sci-fi show",
                        "characters": [
                            {"name": "Alex Mack", "target_name": "אלכס מק", "gender": "female", "notes": "", "speaker_id": ""},
                            {"name": "Annie Mack", "target_name": "אייני מק", "gender": "female", "notes": "", "speaker_id": ""},
                        ],
                        "terms": [],
                        "register_notes": "",
                    }),
                )
            )
            session.add(SubtitleSegmentRecord(job_id=job_id, index=1, start_time=1.0, end_time=2.5, original_text="I am fine.", speaker="SPEAKER_01"))
            session.add(SubtitleSegmentRecord(job_id=job_id, index=2, start_time=3.0, end_time=4.5, original_text="Really fine.", speaker="SPEAKER_01"))
            session.add(SubtitleSegmentRecord(job_id=job_id, index=3, start_time=5.0, end_time=6.5, original_text="Wait up!", speaker="SPEAKER_02"))
            session.add(SubtitleSegmentRecord(job_id=job_id, index=4, start_time=7.0, end_time=8.5, original_text="Come on, Alex!", speaker="SPEAKER_03"))
            session.commit()

        r_match = client.post(f"/api/jobs/{job_id}/match-speakers", json={})
        assert r_match.status_code == 200, r_match.text
        body = r_match.json()
        # SPEAKER_02 high-confidence duplicate auto-merged into SPEAKER_01
        assert set(body["speakers"].keys()) == {"SPEAKER_01", "SPEAKER_03"}
        assert body["speakers"]["SPEAKER_01"]["label"] == "אלכס מק"
        assert body["speakers"]["SPEAKER_01"]["confidence"] == "high"
        assert body["speakers"]["SPEAKER_03"]["confidence"] == "medium"
        assert [s["speaker"] for s in body["segments"]] == ["SPEAKER_01", "SPEAKER_01", "SPEAKER_01", "SPEAKER_03"]
        chars = {c["name"]: c for c in body["glossary"]["characters"]}
        assert chars["Alex Mack"]["speaker_id"] == "SPEAKER_01"
        assert chars["Annie Mack"]["speaker_id"] == "SPEAKER_03"

        # Manual merge / unassign endpoint
        r_merge = client.post(
            f"/api/jobs/{job_id}/speakers/merge",
            json={"source_speaker_ids": ["SPEAKER_03"], "target_speaker_id": None},
        )
        assert r_merge.status_code == 200, r_merge.text
        merged_body = r_merge.json()
        assert list(merged_body["speakers"].keys()) == ["SPEAKER_01"]
        assert merged_body["segments"][3]["speaker"] is None
        assert merged_body["glossary"]["characters"][1]["speaker_id"] == ""


def test_infer_segment_addressees_covers_dyadic_vocative_honorific_and_group():
    from app.pipeline.speaker_match import infer_segment_addressees

    speakers = {
        "SPEAKER_01": {"label": "דנה", "gender": "female", "color": "#F472B6"},
        "SPEAKER_02": {"label": "אבי", "gender": "male", "color": "#38BDF8"},
        "SPEAKER_03": {"label": "שרה", "gender": "female", "color": "#4ADE80"},
    }
    characters = [
        {"name": "Dana", "target_name": "דנה", "gender": "female", "speaker_id": "SPEAKER_01"},
        {"name": "Avi", "target_name": "אבי", "gender": "male", "speaker_id": "SPEAKER_02"},
        {"name": "Sarah", "target_name": "שרה", "gender": "female", "speaker_id": "SPEAKER_03"},
    ]
    segments = [
        # Dyadic A (female) <-> B (male) exchange
        {"index": 1, "start_time": 1.0, "end_time": 2.5, "original_text": "Are you coming with me?", "speaker": "SPEAKER_01"},
        {"index": 2, "start_time": 3.0, "end_time": 4.2, "original_text": "Yes, wait a second.", "speaker": "SPEAKER_02"},
        # Plural group address overrides next single speaker
        {"index": 3, "start_time": 4.5, "end_time": 6.0, "original_text": "You guys need to see this.", "speaker": "SPEAKER_01"},
        {"index": 4, "start_time": 6.3, "end_time": 7.5, "original_text": "Listen up, everyone!", "speaker": "SPEAKER_02"},
        # Named vocative overrides adjacent speaker
        {"index": 5, "start_time": 8.0, "end_time": 9.5, "original_text": "Sarah, are you ready?", "speaker": "SPEAKER_01"},
        {"index": 6, "start_time": 10.0, "end_time": 11.2, "original_text": "I'm ready.", "speaker": "SPEAKER_03"},
        # Honorifics without explicit speaker tags
        {"index": 7, "start_time": 20.0, "end_time": 21.0, "original_text": "Yes, sir.", "speaker": None},
        {"index": 8, "start_time": 25.0, "end_time": 26.0, "original_text": "Thank you, ma'am.", "speaker": None},
    ]

    addressees = infer_segment_addressees(segments, speakers, characters)
    assert addressees[1] == "SPEAKER_02 (אבי, male)"
    assert addressees[2] == "SPEAKER_01 (דנה, female)"
    assert addressees[3] == "group (plural)"
    assert addressees[4] == "group (plural)"
    assert addressees[5] == "SPEAKER_03 (שרה, female)"
    assert addressees[6] == "SPEAKER_01 (דנה, female)"
    assert addressees[7] == "male"
    assert addressees[8] == "female"


@pytest.mark.asyncio
async def test_translate_stage_passes_speaker_and_addressee_to_translator_and_lookahead(monkeypatch):
    from app.providers.translation.glossary_pruner import prune_glossary_for_batch

    seen_instructions = []
    seen_payloads = []

    def fake_generate(model, contents, config):
        seen_instructions.append(config.system_instruction)
        payload = json.loads(contents)
        seen_payloads.append(payload)
        return types.SimpleNamespace(
            text=json.dumps({
                "translations": [
                    {"index": item["id"], "translated_text": f"תרגום מלא {item['id']}"}
                    for item in payload["target_batch"]
                ]
            })
        )

    monkeypatch.setattr(gp.asyncio, "sleep", lambda _: None)
    monkeypatch.setattr(settings, "gemini_glossary_pass", False)

    translator = gp.GeminiTranslator(api_key="test-key", batch_size=1, prev_context_size=1, lookahead_size=1)
    translator.client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=fake_generate))

    glossary = {
        "summary": "Drama",
        "characters": [
            {"name": f"Extra{i}", "target_name": f"ניצב{i}", "gender": "male", "notes": ""}
            for i in range(1, 8)
        ] + [
            {"name": "Sarah", "target_name": "שרה", "gender": "female", "notes": "Lead", "speaker_id": "SPEAKER_01"},
            {"name": "David", "target_name": "דוד", "gender": "male", "notes": "Partner", "speaker_id": "SPEAKER_02"},
        ],
        "terms": [],
        "register_notes": "",
    }
    speakers = {
        "SPEAKER_01": {"label": "שרה", "gender": "female", "color": "#F472B6", "character_name": "Sarah"},
        "SPEAKER_02": {"label": "דוד", "gender": "male", "color": "#38BDF8", "character_name": "David"},
    }
    ctx = PipelineContext(
        job_id="job_addr_test",
        video_path="clip.mp4",
        file_name="clip.mp4",
        enable_translation=True,
        target_language="Hebrew",
        source_language="en",
        glossary=glossary,
        speakers=speakers,
        refined_segments=[
            {
                "index": 1,
                "start_time": 0.0,
                "end_time": 2.0,
                "original_text": "Are you coming with me?",
                "translated_text": None,
                "translation_failed": False,
                "segment_type": "DIALOGUE",
                "speaker": "SPEAKER_01",
            },
            {
                "index": 2,
                "start_time": 2.5,
                "end_time": 4.0,
                "original_text": "Yes, I am ready.",
                "translated_text": None,
                "translation_failed": False,
                "segment_type": "DIALOGUE",
                "speaker": "SPEAKER_02",
            },
        ],
    )

    await TranslateStage(translator=translator).execute(ctx)

    assert len(seen_payloads) == 2
    # Batch 1: female speaker talking to male addressee, with lookahead showing male responder
    batch1 = seen_payloads[0]
    assert batch1["target_batch"][0]["speaker"] == "SPEAKER_01 (שרה, female)"
    assert batch1["target_batch"][0]["addressee"] == "SPEAKER_02 (דוד, male)"
    assert batch1["lookahead_context"][0]["speaker"] == "SPEAKER_02 (דוד, male)"
    assert batch1["lookahead_context"][0]["addressee"] == "SPEAKER_01 (שרה, female)"

    # Batch 2: previous_context preserves both speaker and addressee
    batch2 = seen_payloads[1]
    assert batch2["previous_context"][0]["speaker"] == "SPEAKER_01 (שרה, female)"
    assert batch2["previous_context"][0]["addressee"] == "SPEAKER_02 (דוד, male)"
    assert batch2["target_batch"][0]["speaker"] == "SPEAKER_02 (דוד, male)"
    assert batch2["target_batch"][0]["addressee"] == "SPEAKER_01 (שרה, female)"

    # System prompt clearly distinguishes speaker (1st person) vs addressee (2nd person)
    assert "ONLY for 1st-person" in seen_instructions[0]
    assert "addressee" in seen_instructions[0]
    assert "group (plural)" in seen_instructions[0]

    # Non-core characters active as speaker/addressee are kept by glossary_pruner even when their name isn't spoken
    pruned = prune_glossary_for_batch(
        glossary,
        batch_texts=[
            TranslatableSegment(
                index=1,
                start_time=0.0,
                end_time=2.0,
                original_text="Are you coming with me?",
                speaker="SPEAKER_01",
                speaker_info="SPEAKER_01 (שרה, female)",
                addressee_info="SPEAKER_02 (דוד, male)",
            )
        ],
    )
    kept_names = {c["name"] for c in pruned["characters"]}
    assert "Sarah" in kept_names and "David" in kept_names
