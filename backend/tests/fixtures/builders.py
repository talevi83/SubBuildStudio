"""Database job/segment builders and HTTP polling helpers for backend tests."""
import time
import uuid
from typing import Sequence, Tuple

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.database import engine
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord


def insert_job(status: JobStatus = JobStatus.COMPLETED, **kwargs) -> str:
    """Create and persist a single Job record in the isolated test database."""
    job_id = kwargs.pop("id", None) or f"test_{uuid.uuid4().hex[:8]}"
    video_path = kwargs.pop("video_path", "x.mkv")
    file_name = kwargs.pop("file_name", "x.mkv")
    with Session(engine) as session:
        session.add(
            Job(
                id=job_id,
                video_path=video_path,
                file_name=file_name,
                status=status,
                **kwargs,
            )
        )
        session.commit()
    return job_id


def create_job_with_segments(
    texts: Sequence[str],
    *,
    translated: bool = False,
    translated_prefix: str = "old",
    **job_kwargs,
) -> str:
    """Create a Job with sequentially timed SubtitleSegmentRecord rows."""
    job_id = job_kwargs.pop("id", None) or f"test_{uuid.uuid4().hex[:8]}"
    video_path = job_kwargs.pop("video_path", "x.mkv")
    file_name = job_kwargs.pop("file_name", "x.mkv")
    status = job_kwargs.pop("status", JobStatus.COMPLETED)
    with Session(engine) as session:
        session.add(
            Job(
                id=job_id,
                video_path=video_path,
                file_name=file_name,
                status=status,
                **job_kwargs,
            )
        )
        session.commit()
        for i, text in enumerate(texts, start=1):
            session.add(
                SubtitleSegmentRecord(
                    job_id=job_id,
                    index=i,
                    start_time=i * 2.0,
                    end_time=i * 2.0 + 1.5,
                    original_text=text,
                    translated_text=f"{translated_prefix} {i}" if translated else None,
                )
            )
        session.commit()
    return job_id


def wait_for_job(
    client: TestClient,
    job_id: str,
    statuses: Tuple[str, ...] = ("COMPLETED", "FAILED", "CANCELLED"),
    timeout: float = 10.0,
) -> dict:
    """Poll GET /api/jobs/{job_id} until it reaches one of the target statuses."""
    deadline = time.time() + timeout
    job = None
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job['status'] if job else 'unknown'}")
