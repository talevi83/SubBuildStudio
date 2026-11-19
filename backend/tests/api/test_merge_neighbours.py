"""Merging subtitle lines: only neighbours of one lane (merging #1 with #4 would overlap #2 and #3)."""
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.database import engine
from app.models.db_models import SubtitleSegmentRecord
from tests.fixtures.builders import create_job_with_segments


def _segments(client: TestClient, job_id: str):
    return client.get(f"/api/jobs/{job_id}").json()["segments"]


def test_merging_lines_that_are_not_neighbours_is_refused(client: TestClient):
    job_id = create_job_with_segments(["a", "b", "c", "d"])
    segs = _segments(client, job_id)
    res = client.post(f"/api/jobs/{job_id}/segments/merge", json={"segment_ids": [segs[0]["id"], segs[3]["id"]]})
    assert res.status_code == 400
    assert "neighbouring" in res.json()["detail"]
    assert [s["original_text"] for s in _segments(client, job_id)] == ["a", "b", "c", "d"]  # nothing changed


def test_merging_neighbours_works_in_any_click_order(client: TestClient):
    job_id = create_job_with_segments(["a", "b", "c", "d"])
    segs = _segments(client, job_id)
    res = client.post(f"/api/jobs/{job_id}/segments/merge",
                      json={"segment_ids": [segs[2]["id"], segs[1]["id"]]})
    assert res.status_code == 200
    assert [s["original_text"] for s in res.json()["segments"]] == ["a", "b\nc", "d"]


def test_an_on_screen_title_between_two_dialogue_lines_does_not_block_their_merge(client: TestClient):
    job_id = create_job_with_segments(["a", "b"])
    with Session(engine) as session:  # a title shown at the same time, between the two lines
        session.add(SubtitleSegmentRecord(job_id=job_id, index=3, start_time=3.6, end_time=3.9,
                                          original_text="SIGN", segment_type="ON_SCREEN"))
        session.commit()
    dialogue = [s for s in _segments(client, job_id) if s["segment_type"] != "ON_SCREEN"]
    res = client.post(f"/api/jobs/{job_id}/segments/merge", json={"segment_ids": [s["id"] for s in dialogue]})
    assert res.status_code == 200
    texts = sorted(s["original_text"] for s in res.json()["segments"])
    assert texts == ["SIGN", "a\nb"]


def test_dialogue_and_on_screen_text_are_not_merged_together(client: TestClient):
    job_id = create_job_with_segments(["a"])
    with Session(engine) as session:
        session.add(SubtitleSegmentRecord(job_id=job_id, index=2, start_time=4.0, end_time=5.0,
                                          original_text="SIGN", segment_type="ON_SCREEN"))
        session.commit()
    ids = [s["id"] for s in _segments(client, job_id)]
    assert client.post(f"/api/jobs/{job_id}/segments/merge", json={"segment_ids": ids}).status_code == 400
