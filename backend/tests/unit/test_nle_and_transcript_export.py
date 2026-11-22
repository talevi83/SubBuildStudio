"""Tests for word timing persistence, FCPXML NLE export, and Markdown / OOXML DOCX transcripts."""
import io
import uuid
import xml.etree.ElementTree as ET
import zipfile

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from app.pipeline.context import RawSegment, TranscribedWord
from app.pipeline.orchestrator import PipelineOrchestrator, active_pipelines
from app.pipeline.segmentation import refine_raw_segments
from app.utils.nle_export import serialize_to_fcpxml
from app.utils.transcript_export import serialize_to_docx, serialize_to_markdown


def test_words_json_persisted_and_maintained_across_segment_operations(tmp_path):
    raw = [
        RawSegment(
            index=1,
            start_time=1.0,
            end_time=2.5,
            text="Hello world today.",
            words=[
                TranscribedWord("Hello", 1.0, 1.4),
                TranscribedWord(" world", 1.5, 1.9),
                TranscribedWord(" today.", 2.0, 2.4),
            ],
        ),
        RawSegment(
            index=2,
            start_time=4.0,
            end_time=5.0,
            text="Next line.",
            words=[
                TranscribedWord("Next", 4.0, 4.4),
                TranscribedWord(" line.", 4.5, 5.0),
            ],
        ),
    ]
    refined = refine_raw_segments(raw)
    assert refined[0]["words"] is not None
    assert [w["w"] for w in refined[0]["words"]] == ["Hello", "world", "today."]

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    video = tmp_path / "Clip.mp4"
    video.write_bytes(b"v")

    with TestClient(app) as client:
        with Session(engine) as session:
            session.add(Job(id=job_id, video_path=str(video), file_name="Clip.mp4", status=JobStatus.COMPLETED))
            session.commit()
        orch = PipelineOrchestrator(job_id)
        try:
            orch._save_segments(refined)
        finally:
            active_pipelines.pop(job_id, None)

        # GET returns parsed `words` list
        got = client.get(f"/api/jobs/{job_id}").json()
        assert len(got["segments"]) == 2
        seg1_id = got["segments"][0]["id"]
        seg2_id = got["segments"][1]["id"]
        assert [w["w"] for w in got["segments"][0]["words"]] == ["Hello", "world", "today."]

        # Shift moves word timestamps along with segment timestamps
        shifted = client.post(f"/api/jobs/{job_id}/segments/shift", json={"offset_seconds": 2.0}).json()
        assert shifted["segments"][0]["words"][0]["s"] == pytest_approx_or_round(3.0)

        # Split partitions words_json around split_time
        split_res = client.post(f"/api/jobs/{job_id}/segments/{seg1_id}/split", json={"split_time": 3.7}).json()
        assert len(split_res["segments"]) == 3
        left_seg, right_seg = split_res["segments"][0], split_res["segments"][1]
        assert [w["w"] for w in left_seg["words"]] == ["Hello", "world"]
        assert [w["w"] for w in right_seg["words"]] == ["today."]

        # Merge re-concatenates consecutive word lists
        merge_res = client.post(
            f"/api/jobs/{job_id}/segments/merge",
            json={"segment_ids": [left_seg["id"], right_seg["id"]]},
        ).json()
        assert len(merge_res["segments"]) == 2
        merged_seg = merge_res["segments"][0]
        assert [w["w"] for w in merged_seg["words"]] == ["Hello", "world", "today."]

        # Editing original_text to a different word count invalidates stale word timestamps
        patched = client.patch(
            f"/api/jobs/{job_id}/segments/{merged_seg['id']}",
            json={"original_text": "Completely rewritten sentence with more words"},
        ).json()
        assert patched["words"] is None

        # Moving clears stale word timings on touched rows
        move_res = client.post(
            f"/api/jobs/{job_id}/segments/move",
            json={"segment_id": merged_seg["id"], "to_position": 1},
        ).json()
        assert all(s["words"] is None for s in move_res["segments"])
        assert seg2_id != merged_seg["id"]


def pytest_approx_or_round(val: float) -> float:
    return round(val, 3)


def test_serialize_to_fcpxml_valid_xml_lanes_and_markers():
    segs = [
        {"index": 1, "start_time": 0.0, "end_time": 2.0, "original_text": "One & <Two>", "translated_text": "שלום & ברכה", "segment_type": "DIALOGUE", "speaker": "SPEAKER_01"},
        {"index": 2, "start_time": 2.2, "end_time": 4.0, "original_text": "Copy that", "translated_text": "רות עבור", "segment_type": "COMMS", "speaker": "SPEAKER_02"},
        {"index": 3, "start_time": 4.2, "end_time": 6.0, "original_text": "TEL AVIV", "translated_text": "[תל אביב]", "segment_type": "ON_SCREEN", "speaker": None},
    ]
    speakers = {
        "SPEAKER_01": {"label": "Dana", "gender": "female", "color": "#38BDF8"},
        "SPEAKER_02": {"label": "Avi", "gender": "male", "color": "#F472B6"},
    }
    xml_out = serialize_to_fcpxml(segs, title="Test & Show", duration=10.0, fps=25.0, use_translated=True, apply_rtl=False, speakers=speakers)
    assert xml_out.startswith("<?xml")
    assert "<!DOCTYPE fcpxml>" in xml_out

    # Strip DOCTYPE before ElementTree parsing if needed or parse directly (ET supports <!DOCTYPE fcpxml>)
    root = ET.fromstring(xml_out)
    assert root.tag == "fcpxml"
    assert root.attrib["version"] == "1.9"
    fmt = root.find(".//format")
    assert fmt is not None and fmt.attrib["frameDuration"] == "100/2500s"
    titles = root.findall(".//title")
    assert len(titles) == 3
    assert [t.attrib["lane"] for t in titles] == ["1", "1", "2"]
    markers = root.findall(".//marker")
    assert len(markers) == 3
    assert "Dana" in markers[0].attrib["value"]
    assert "COMMS" in markers[1].attrib["value"]
    assert "ON_SCREEN" in markers[2].attrib["value"]


def test_serialize_to_markdown_and_docx_with_rtl_and_sanitization():
    segs = [
        {"index": 1, "start_time": 0.0, "end_time": 2.0, "original_text": "First part", "translated_text": "חלק ראשון", "segment_type": "DIALOGUE", "speaker": "SPEAKER_01"},
        {"index": 2, "start_time": 2.4, "end_time": 4.0, "original_text": "Second part", "translated_text": "חלק שני", "segment_type": "DIALOGUE", "speaker": "SPEAKER_01"},
        {"index": 3, "start_time": 5.0, "end_time": 7.0, "original_text": "Reply\x00Bad", "translated_text": "תשובה\x07!", "segment_type": "DIALOGUE", "speaker": "SPEAKER_02"},
        {"index": 4, "start_time": 8.0, "end_time": 10.0, "original_text": "[CITY HALL]", "translated_text": "[בית העירייה]", "segment_type": "ON_SCREEN", "speaker": None},
    ]
    speakers = {"SPEAKER_01": {"label": "Host"}, "SPEAKER_02": {"label": "Guest"}}

    md = serialize_to_markdown(segs, title="Podcast 1", language="Hebrew", duration=65.0, use_translated=True, bilingual=True, speakers=speakers)
    assert "# Podcast 1" in md
    assert "[00:00:00] **Host**" in md
    assert "חלק ראשון חלק שני" in md  # consecutive same-speaker turns grouped!
    assert "*First part Second part*" in md
    assert "> [00:00:08] **[ON_SCREEN]** [בית העירייה]" in md

    docx_bytes = serialize_to_docx(segs, title="פרק 1", language="Hebrew", duration=65.0, use_translated=True, rtl=True, speakers=speakers)
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zf:
        names = set(zf.namelist())
        assert {"[Content_Types].xml", "_rels/.rels", "word/_rels/document.xml.rels", "word/styles.xml", "word/document.xml"} <= names
        doc_xml = zf.read("word/document.xml").decode("utf-8")
        ET.fromstring(doc_xml)  # valid XML even with \x00 / \x07 in input
        assert "<w:bidi/>" in doc_xml
        assert "<w:rtl/>" in doc_xml
        assert "Host:" in doc_xml


def test_creator_export_http_endpoints(tmp_path):
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    video = tmp_path / "Interview.mp4"
    video.write_bytes(b"v")

    with TestClient(app) as client:
        with Session(engine) as session:
            session.add(
                Job(
                    id=job_id,
                    video_path=str(video),
                    file_name="Interview.mp4",
                    status=JobStatus.COMPLETED,
                    source_language="he",
                    detected_language="he",
                    target_language="Hebrew",
                    video_duration=12.0,
                )
            )
            session.add(
                SubtitleSegmentRecord(
                    job_id=job_id,
                    index=1,
                    start_time=0.0,
                    end_time=2.5,
                    original_text="ברוכים הבאים לתוכנית",
                    translated_text=None,
                    segment_type="DIALOGUE",
                )
            )
            session.commit()

        r_xml = client.get(f"/api/jobs/{job_id}/export/fcpxml?use_translated=false&fps=25")
        assert r_xml.status_code == 200
        assert "application/xml" in r_xml.headers["content-type"]
        assert ".fcpxml" in r_xml.headers["content-disposition"]
        assert "<fcpxml" in r_xml.text

        r_md = client.get(f"/api/jobs/{job_id}/export/markdown?use_translated=false")
        assert r_md.status_code == 200
        assert "text/markdown" in r_md.headers["content-type"]
        assert ".md" in r_md.headers["content-disposition"]
        assert "ברוכים הבאים לתוכנית" in r_md.text

        r_docx = client.get(f"/api/jobs/{job_id}/export/docx?use_translated=false")
        assert r_docx.status_code == 200
        assert "wordprocessingml.document" in r_docx.headers["content-type"]
        assert ".docx" in r_docx.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(r_docx.content), "r") as zf:
            doc_xml = zf.read("word/document.xml").decode("utf-8")
            assert "<w:bidi/>" in doc_xml
