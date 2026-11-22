"""Unit and API tests for Advanced SubStation Alpha (.ass) serialization, styling, and preview frames."""
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core import video_embed as ve
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from app.utils.ass_formatter import AssStyleConfig, escape_ass_text, format_ass_timestamp, serialize_to_ass
from app.utils.rtl_fixer import RLE, RLM, PDF


def test_format_ass_timestamp_carries_centiseconds_cleanly():
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(59.999) == "0:01:00.00"
    assert format_ass_timestamp(3723.456) == "1:02:03.46"


def test_escape_ass_text_handles_braces_backslashes_and_newlines():
    escaped = escape_ass_text("Hello {world}\\path\nLine 2")
    assert "\\{world\\}" in escaped
    assert "\\N" in escaped
    assert "{world}" not in escaped.replace("\\{world\\}", "")


def test_serialize_to_ass_styles_segment_types_and_rtl():
    segments = [
        {
            "index": 1,
            "start_time": 0.5,
            "end_time": 2.5,
            "original_text": "One week later",
            "translated_text": "[שבוע לאחר מכן]",
            "segment_type": "ON_SCREEN",
        },
        {
            "index": 2,
            "start_time": 3.0,
            "end_time": 5.0,
            "original_text": "Alpha actual, copy?",
            "translated_text": "אלפא קודקוד, שומע?",
            "segment_type": "COMMS",
        },
        {
            "index": 3,
            "start_time": 6.0,
            "end_time": 8.0,
            "original_text": "Hello friend!",
            "translated_text": "שלום חבר!",
            "segment_type": "DIALOGUE",
        },
    ]
    ass = serialize_to_ass(
        segments,
        use_translated=True,
        apply_rtl=True,
        onscreen_position="top_right",
        style=AssStyleConfig(font_family="Rubik", font_size_px=22, text_color="#FCD34D", background_opacity=60, bold=True),
    )
    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass
    assert "Style: Default,Rubik," in ass
    assert "Style: OnScreen," in ass
    assert ",9,40,40,36,1" in ass  # top_right alignment = 9
    assert "Style: Comms," in ass
    assert ",OnScreen," in ass
    assert ",Comms," in ass
    assert f"{RLE}{RLM}שלום חבר!{RLM}{PDF}" in ass


def test_serialize_to_ass_bilingual_and_speaker_colors():
    segments = [
        {
            "index": 1,
            "start_time": 1.0,
            "end_time": 3.0,
            "original_text": "Are you ready?",
            "translated_text": "את מוכנה?",
            "segment_type": "DIALOGUE",
            "speaker": "SPEAKER_01",
        },
        {
            "index": 2,
            "start_time": 3.2,
            "end_time": 5.0,
            "original_text": "Yes I am.",
            "translated_text": "כן, אני מוכנה.",
            "segment_type": "DIALOGUE",
            "speaker": "SPEAKER_02",
        },
    ]
    speakers = {
        "SPEAKER_01": {"id": "SPEAKER_01", "label": "Dan", "color": "#38BDF8"},
        "SPEAKER_02": {"id": "SPEAKER_02", "label": "Sarah", "color": "#F472B6"},
    }
    ass = serialize_to_ass(
        segments,
        use_translated=True,
        apply_rtl=True,
        bilingual=True,
        style=AssStyleConfig(color_speakers=True),
        speakers=speakers,
    )
    assert "Style: Speaker_SPEAKER_01," in ass
    assert "Style: Speaker_SPEAKER_02," in ass
    assert ",Speaker_SPEAKER_01,Dan," in ass
    assert ",Speaker_SPEAKER_02,Sarah," in ass
    assert "{\\rOriginalSub}" in ass
    assert "{\\rSpeaker_SPEAKER_01}" in ass


def test_serialize_to_ass_word_and_karaoke_highlights():
    segs = [
        {
            "index": 1,
            "start_time": 1.0,
            "end_time": 3.0,
            "original_text": "Hello big world",
            "translated_text": None,
            "segment_type": "DIALOGUE",
        }
    ]
    ass_word = serialize_to_ass(
        segs,
        use_translated=False,
        apply_rtl=False,
        style=AssStyleConfig(highlight_mode="word", highlight_color="#FACC15"),
    )
    dialogue_lines = [ln for ln in ass_word.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue_lines) == 3
    assert any("{\\1c&H15CCFA&\\b1}big{" in ln for ln in dialogue_lines)

    ass_karaoke = serialize_to_ass(
        segs,
        use_translated=False,
        apply_rtl=False,
        style=AssStyleConfig(highlight_mode="karaoke", highlight_color="#FACC15"),
    )
    assert "{\\kf" in ass_karaoke


def test_build_burn_command_omits_force_style_for_ass_input():
    probe = {"streams": [{"index": 0, "codec_type": "video"}, {"index": 1, "codec_type": "audio", "codec_name": "aac"}]}
    cmd = ve.build_burn_command("M.mkv", "subs.ass", "out", probe, ve.EmbedOptions(mode="burn", subtitle_format="ass"), "libx264")
    vf = cmd[cmd.index("-vf") + 1]
    assert "subtitles=subs.ass" in vf
    assert "force_style" not in vf


def test_export_ass_and_preview_frame_endpoints(tmp_path, monkeypatch):
    video = tmp_path / "Sample.mp4"
    video.write_bytes(b"video")
    job_id = f"job_{uuid.uuid4().hex[:8]}"

    with TestClient(app) as client:
        with Session(engine) as session:
            session.add(
                Job(
                    id=job_id,
                    video_path=str(video),
                    file_name=video.name,
                    status=JobStatus.COMPLETED,
                    source_language="en",
                    target_language="Hebrew",
                    video_duration=12.0,
                )
            )
            session.add(
                SubtitleSegmentRecord(
                    job_id=job_id,
                    index=1,
                    start_time=1.0,
                    end_time=4.0,
                    original_text="Welcome back!",
                    translated_text="ברוכים השבים!",
                    segment_type="DIALOGUE",
                )
            )
            session.commit()

        resp = client.get(
            f"/api/jobs/{job_id}/export/ass",
            params={
                "use_translated": "true",
                "apply_rtl": "true",
                "font_family": "Rubik",
                "font_size_px": 24,
                "bold": "true",
                "margin_v": 30,
            },
        )
        assert resp.status_code == 200
        assert "text/x-ssa" in resp.headers["content-type"]
        assert "Sample.he.ass" in resp.headers["content-disposition"]
        text = resp.content.decode("utf-8-sig")
        assert "Style: Default,Rubik," in text
        assert "ברוכים השבים!" in text

        # Mock grab_frame so preview-frame endpoint can be tested without invoking ffmpeg
        from app.pipeline import speaker_detect as sd
        monkeypatch.setattr(sd, "grab_frame", lambda _path, t, _ff="ffmpeg": b"\xff\xd8\xff\xe0MOCKJPEG")
        frame_resp = client.get(f"/api/jobs/{job_id}/preview-frame")
        assert frame_resp.status_code == 200
        assert frame_resp.headers["content-type"] == "image/jpeg"
        assert frame_resp.content.startswith(b"\xff\xd8\xff\xe0")
