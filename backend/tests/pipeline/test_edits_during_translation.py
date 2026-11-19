"""Lines the user edits while a translation runs must not be overwritten by that run."""
import asyncio
import time

import pytest
from sqlmodel import Session, select

from app.core.database import engine, init_db
from app.models.db_models import JobStatus, SubtitleSegmentRecord
from app.pipeline.context import PipelineCancelled, PipelineContext
from app.pipeline.orchestrator import PipelineOrchestrator, active_pipelines
from app.providers.translation.base import TranslatableSegment
from tests.fixtures.builders import create_job_with_segments


def _rows(job_id):
    with Session(engine) as session:
        rows = session.exec(select(SubtitleSegmentRecord).where(SubtitleSegmentRecord.job_id == job_id)).all()
        return sorted(rows, key=lambda r: r.index)


def _user_edit(row_id, **fields):
    with Session(engine) as session:
        row = session.get(SubtitleSegmentRecord, row_id)
        for k, v in fields.items():
            setattr(row, k, v)
        session.add(row)
        session.commit()


def _batch(rows, texts):
    return [
        TranslatableSegment(index=r.index, start_time=r.start_time, end_time=r.end_time,
                            original_text=r.original_text, translated_text=t, translation_failed=False)
        for r, t in zip(rows, texts)
    ]


@pytest.fixture
def translating_job():
    init_db()  # also when this file runs on its own (the client fixture usually creates the tables)
    job_id = create_job_with_segments(["one", "two", "three"], translated=True, status=JobStatus.TRANSLATING)
    orch = PipelineOrchestrator(job_id=job_id, mode="translate")
    assert orch._build_context() is not None
    yield job_id, orch
    active_pipelines.pop(job_id, None)


@pytest.mark.asyncio
async def test_a_translation_edited_by_the_user_is_kept(translating_job):
    job_id, orch = translating_job
    rows = _rows(job_id)
    _user_edit(rows[0].id, translated_text="USER FIX")  # typed while the batch was being translated
    await orch._persist_translations(_batch(rows, ["machine 1", "machine 2", "machine 3"]))
    after = _rows(job_id)
    assert [r.translated_text for r in after] == ["USER FIX", "machine 2", "machine 3"]
    seg = next(s for s in orch._ctx.refined_segments if s["id"] == rows[0].id)
    assert seg["translated_text"] == "USER FIX"  # the run's own copy follows the user


@pytest.mark.asyncio
async def test_a_changed_source_line_without_translation_is_left_for_retranslate_failed(translating_job):
    job_id, orch = translating_job
    rows = _rows(job_id)
    _user_edit(rows[1].id, original_text="two (fixed)", translated_text=None)
    await orch._persist_translations(_batch(rows, ["m1", "m2", "m3"]))
    after = _rows(job_id)
    assert after[1].translated_text is None and after[1].translation_failed is True
    assert after[0].translated_text == "m1"


@pytest.mark.asyncio
async def test_a_retried_batch_does_not_overwrite_the_users_edit_the_second_time(translating_job):
    job_id, orch = translating_job
    rows = _rows(job_id)
    _user_edit(rows[0].id, translated_text="USER FIX")
    await orch._persist_translations(_batch(rows[:1], ["machine"]))
    await orch._persist_translations(_batch(rows[:1], ["machine, retried"]))
    assert _rows(job_id)[0].translated_text == "USER FIX"


@pytest.mark.asyncio
async def test_later_batches_still_translate_lines_the_run_wrote_itself(translating_job):
    job_id, orch = translating_job
    rows = _rows(job_id)
    await orch._persist_translations(_batch(rows[:1], ["first try"]))
    await orch._persist_translations(_batch(rows[:1], ["second try"]))  # e.g. a retried batch
    assert _rows(job_id)[0].translated_text == "second try"


@pytest.mark.asyncio
async def test_the_broadcast_carries_what_was_stored(translating_job, monkeypatch):
    job_id, orch = translating_job
    sent = []

    async def capture(payload):
        sent.append(payload)

    monkeypatch.setattr(orch, "_broadcast", capture)
    rows = _rows(job_id)
    _user_edit(rows[2].id, translated_text="mine")
    await orch._persist_translations(_batch(rows, ["a", "b", "c"]))
    batch = sent[-1]["segments_batch"]
    assert [s["translated_text"] for s in batch] == ["a", "b", "mine"]
    assert [s["id"] for s in batch] == [r.id for r in rows]


def test_edits_survive_when_a_split_saves_the_whole_list(translating_job):
    job_id, orch = translating_job
    rows = _rows(job_id)
    _user_edit(rows[0].id, translated_text="kept")
    segments = [dict(s) for s in orch._ctx.refined_segments]
    for s in segments:
        s["translated_text"] = "from the run"
    assert orch._adopt_user_edits(segments) == 1
    orch._save_segments(segments)
    assert [r.translated_text for r in _rows(job_id)] == ["kept", "from the run", "from the run"]


@pytest.mark.asyncio
async def test_run_cancellable_stops_a_slow_request_quickly():
    ctx = PipelineContext(job_id="x", video_path="x.mkv", file_name="x.mkv")
    finished = asyncio.Event()

    async def slow_request():
        try:
            await asyncio.sleep(60)
        finally:
            finished.set()

    async def cancel_soon():
        await asyncio.sleep(0.1)
        ctx.cancel_flag.set()

    began = time.monotonic()
    asyncio.ensure_future(cancel_soon())
    with pytest.raises(PipelineCancelled):
        await ctx.run_cancellable(slow_request(), poll_seconds=0.05)
    assert time.monotonic() - began < 2
    assert finished.is_set()  # the request itself was cancelled, not left running


@pytest.mark.asyncio
async def test_run_cancellable_returns_the_result():
    ctx = PipelineContext(job_id="x", video_path="x.mkv", file_name="x.mkv")

    async def quick():
        return 42

    assert await ctx.run_cancellable(quick()) == 42
