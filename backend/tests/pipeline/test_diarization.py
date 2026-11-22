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
                            {"index": item["index"], "translated_text": f"ת{item['index']}"}
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
