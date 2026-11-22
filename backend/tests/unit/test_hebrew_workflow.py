"""Tests for Hebrew-to-Hebrew transcription, segmentation, same-language polish, QA, and RTL export."""
from types import SimpleNamespace as NS
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from app.pipeline.context import RawSegment, TranscribedWord
from app.pipeline.qa import check_rows
from app.pipeline.segmentation import _tokens_from_raw, refine_raw_segments
from app.providers.translation.gemini_provider import (
    build_subtitle_system_prompt,
    is_same_language,
)
from app.utils.text_rules import ABBREVIATIONS, BREAK_BEFORE_WORDS, SENTENCE_END, WEAK_END_WORDS


def _unspaced_words(words: list[str], start: float, step: float = 0.3) -> list[TranscribedWord]:
    out = []
    t = start
    for w in words:
        out.append(TranscribedWord(word=w, start=t, end=t + step - 0.05))
        t += step
    return out


def test_hebrew_segment_boundary_inserts_space_while_cjk_remains_unspaced():
    raw_he = [
        RawSegment(index=1, start_time=0.0, end_time=0.5, text="שלום", words=[TranscribedWord("שלום", 0.0, 0.5)]),
        RawSegment(index=2, start_time=0.6, end_time=1.0, text="עולם.", words=[TranscribedWord("עולם.", 0.6, 1.0)]),
    ]
    toks_he = _tokens_from_raw(raw_he)
    assert "".join(t.text for t in toks_he) == "שלום עולם."

    raw_cjk = [
        RawSegment(index=1, start_time=0.0, end_time=0.5, text="你好", words=[TranscribedWord("你好", 0.0, 0.5)]),
        RawSegment(index=2, start_time=0.6, end_time=1.0, text="世界", words=[TranscribedWord("世界", 0.6, 1.0)]),
    ]
    toks_cjk = _tokens_from_raw(raw_cjk)
    assert "".join(t.text for t in toks_cjk) == "你好世界"


def test_hebrew_text_rules_and_line_breaks():
    assert SENTENCE_END.search("זה עובד״!")
    assert 'ד"ר' in ABBREVIATIONS and "ד״ר" in ABBREVIATIONS and "וכו׳" in ABBREVIATIONS
    assert "של" in WEAK_END_WORDS and "על" in WEAK_END_WORDS
    assert "אבל" in BREAK_BEFORE_WORDS and "כאשר" in BREAK_BEFORE_WORDS

    long_he = (
        "אנחנו חייבים להספיק לסיים את כל המשימות החשובות של הפרויקט לפני שמגיע הערב "
        "אבל יש לנו עוד הרבה עבודה לעשות לפני שנוכל לשלוח את הגרסה הסופית."
    )
    words = []
    t = 0.0
    for w in long_he.split():
        words.append(TranscribedWord(word=" " + w, start=t, end=t + 0.25))
        t += 0.3
    raw = [RawSegment(index=1, start_time=0.0, end_time=t, text=long_he, words=words)]
    refined = refine_raw_segments(raw, max_chars_per_line=42)
    assert len(refined) >= 2
    joined = " ".join(r["original_text"].replace("\n", " ") for r in refined)
    assert joined == long_he


def test_same_language_detection_and_proofread_prompt():
    assert is_same_language("he", "Hebrew") is True
    assert is_same_language("Hebrew", "עברית") is True
    assert is_same_language("en", "Hebrew") is False
    assert is_same_language("auto", "Hebrew") is False

    prompt = build_subtitle_system_prompt("he", "Hebrew", max_chars_per_line=42)
    assert "proofreader" in prompt.lower()
    assert "Hebrew Proofreading & Orthography" in prompt
    assert "DO NOT rewrite" in prompt


def test_qa_untranslated_job_skips_missing_and_checks_original_hebrew_agreement():
    rows = [
        NS(id=1, index=1, original_text="שלום עולם", translated_text=None, segment_type="DIALOGUE", translation_failed=False),
        NS(id=2, index=2, original_text="את תהיה שם", translated_text="", segment_type="DIALOGUE", translation_failed=False),
    ]
    untranslated_issues = check_rows(rows, glossary=None, translation_enabled=False)
    assert [(i.segment_id, i.kind) for i in untranslated_issues] == [(2, "agreement")]

    # When translation IS enabled, missing translation (or failed flag even with text) is flagged
    rows_enabled = [
        NS(id=1, index=1, original_text="Hello", translated_text=None, segment_type="DIALOGUE", translation_failed=False),
        NS(id=2, index=2, original_text="World", translated_text="stale", segment_type="DIALOGUE", translation_failed=True),
    ]
    enabled_issues = check_rows(rows_enabled, glossary=None, translation_enabled=True)
    assert [(i.segment_id, i.kind) for i in enabled_issues] == [(1, "missing"), (2, "missing")]


def test_ivrit_model_registered_and_original_hebrew_save_next_to_video(tmp_path):
    assert "ivrit-ai/whisper-large-v3-turbo-ct2" in settings.whisper_models

    video = tmp_path / "Podcast.mp4"
    video.write_bytes(b"x")
    job_id = f"job_{uuid.uuid4().hex[:10]}"

    with TestClient(app) as client:
        with Session(engine) as session:
            session.add(
                Job(
                    id=job_id,
                    video_path=str(video),
                    file_name="Podcast.mp4",
                    status=JobStatus.COMPLETED,
                    source_language="he",
                    detected_language="he",
                    enable_translation=False,
                    target_language="Hebrew",
                    whisper_model="ivrit-ai/whisper-large-v3-turbo-ct2",
                )
            )
            session.add(
                SubtitleSegmentRecord(
                    job_id=job_id,
                    index=1,
                    start_time=0.5,
                    end_time=3.0,
                    original_text="שלום לכולם, ברוכים הבאים!",
                    translated_text=None,
                )
            )
            session.commit()

        resp = client.post(
            f"/api/jobs/{job_id}/export/save-next-to-video",
            json={"format": "srt", "use_translated": False, "apply_rtl": True},
        )
        assert resp.status_code == 200, resp.text
        target = tmp_path / "Podcast.he.srt"
        assert resp.json()["saved"] is True
        assert resp.json()["path"] == str(target)
        text = target.read_text(encoding="utf-8-sig")
        assert "שלום לכולם" in text
        assert "\u200f" in text  # RTL mark applied on original Hebrew subtitles
