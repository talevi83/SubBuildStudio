import logging
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Generator
from sqlalchemy import event, inspect, text
from sqlmodel import create_engine, Session, SQLModel
from app.core.config import settings

logger = logging.getLogger(__name__)

# SQLite connection configuration
connect_args = {"check_same_thread": False}


def _adopt_legacy_database() -> None:
    """The app used to be called SubForge Studio and kept its data in ``subforge.db``.

    Keep existing projects: if the default ``subbuild.db`` doesn't exist yet but the old file
    does, rename it (and its SQLite sidecar files) before the engine opens anything. Also
    migrate an older ``DATABASE_URL=.../subforge.db`` from ``.env`` so it switches to ``subbuild.db``.
    """
    prefix = "sqlite:///"
    url = settings.database_url
    if url.startswith(prefix) and url.endswith("/subforge.db"):
        url = url[:-len("subforge.db")] + "subbuild.db"
        settings.database_url = url
        try:
            from app.core.config import env_file_path

            env_path = env_file_path()
            if env_path.is_file():
                content = env_path.read_text(encoding="utf-8", errors="replace")
                updated = re.sub(
                    r"^(\s*DATABASE_URL\s*=[^\r\n]*?)subforge\.db",
                    r"\g<1>subbuild.db",
                    content,
                    flags=re.MULTILINE,
                )
                if updated != content:
                    env_path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            logger.warning(f"Could not update legacy DATABASE_URL in .env: {exc}")

    if not url.startswith(prefix) or not url.endswith("/subbuild.db"):
        return
    new = Path(url[len(prefix):])
    old = new.with_name("subforge.db")
    if new.exists() or not old.is_file():
        return
    try:
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(old) + suffix)
            if src.exists():
                src.rename(Path(str(new) + suffix))
        logger.info("Renamed legacy database subforge.db -> subbuild.db")
    except OSError as exc:  # fall back to a fresh DB rather than failing startup
        logger.warning(f"Could not adopt legacy database {old}: {exc}")


_adopt_legacy_database()
engine = create_engine(settings.database_url, echo=False, connect_args=connect_args)


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record) -> None:
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.close()
    except Exception:
        pass

# Columns added after the first release. SQLModel.create_all() never alters existing
# tables, so older databases get these through a tiny additive migration.
_ADDED_COLUMNS = {
    "jobs": {
        "is_upload": "BOOLEAN NOT NULL DEFAULT 0",
        "warning_message": "VARCHAR",
        "detected_language": "VARCHAR",
        "glossary_json": "VARCHAR",
        "comms_boost": "BOOLEAN NOT NULL DEFAULT 0",
        "use_embedded_subtitles": "BOOLEAN NOT NULL DEFAULT 1",
        "selected_subtitle_stream": "INTEGER",
        "detect_onscreen_text": "BOOLEAN NOT NULL DEFAULT 0",
        "usage_json": "VARCHAR",
        "glossary_profile": "VARCHAR",
        "job_kind": "VARCHAR NOT NULL DEFAULT 'transcribe'",
        "sync_file_name": "VARCHAR",
        "sync_source_json": "VARCHAR",
        "sync_report_json": "VARCHAR",
        "onscreen_sensitivity": "INTEGER NOT NULL DEFAULT 2",
        "high_speech_sensitivity": "BOOLEAN NOT NULL DEFAULT 0",
        "allow_onscreen_overlap": "BOOLEAN NOT NULL DEFAULT 1",
        "onscreen_position": "VARCHAR NOT NULL DEFAULT 'top_center'",
        "convert_to_metric": "BOOLEAN NOT NULL DEFAULT 0",
        "remove_sdh_speakers": "BOOLEAN NOT NULL DEFAULT 0",
        "skip_end_credits": "BOOLEAN NOT NULL DEFAULT 1",
        "auto_save_srt": "BOOLEAN NOT NULL DEFAULT 0",
        "supplement_with_whisper": "BOOLEAN NOT NULL DEFAULT 0",
        "enable_diarization": "BOOLEAN NOT NULL DEFAULT 0",
        "speakers_json": "VARCHAR",
        "history_cursor": "INTEGER",
        "condition_on_previous_text": "BOOLEAN NOT NULL DEFAULT 0",
        "initial_prompt": "VARCHAR",
        "transcription_provider": "VARCHAR NOT NULL DEFAULT 'faster_whisper'",
        "vocal_isolation": "BOOLEAN NOT NULL DEFAULT 0",
        "vocal_isolation_mode": "VARCHAR NOT NULL DEFAULT 'off'",
        "comms_boost_mode": "VARCHAR NOT NULL DEFAULT 'off'",
        "enable_dual_source_pivot": "BOOLEAN NOT NULL DEFAULT 0",
        "dual_source_en_srt": "VARCHAR",
        "dual_source_ar_srt": "VARCHAR",
        "dual_source_fallback_languages_json": "VARCHAR",
        "dual_source_smart_asr_fallback": "BOOLEAN NOT NULL DEFAULT 1",
        "dual_source_report_json": "VARCHAR",
    },
    "subtitle_segments": {
        "translation_failed": "BOOLEAN NOT NULL DEFAULT 0",
        "segment_type": "VARCHAR NOT NULL DEFAULT 'DIALOGUE'",
        "words_json": "VARCHAR",
        "speaker": "VARCHAR",
        "addressee_json": "VARCHAR",
    },
}


_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}_")


def _backfill_uploads(conn) -> None:
    """Jobs created before is_upload existed: mark browser uploads and drop the random
    "1a2b3c4d_" prefix from their display name (it also leaked into exported file names)."""
    for job_id, video_path, file_name in conn.execute(text("SELECT id, video_path, file_name FROM jobs")).fetchall():
        if not video_path:
            continue
        p = PureWindowsPath(video_path) if "\\" in video_path else PurePosixPath(video_path)
        # Only a display flag: deleting files always re-checks the real location first.
        in_uploads = p.parent.name.lower() == "uploads" and _UPLOAD_PREFIX.match(p.name) is not None
        if in_uploads:
            conn.execute(
                text("UPDATE jobs SET is_upload = 1, file_name = :name WHERE id = :id"),
                {"name": _UPLOAD_PREFIX.sub("", file_name or "", count=1) or file_name, "id": job_id},
            )


def _migrate() -> None:
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            if not inspector.has_table(table):
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl in columns.items():
                if name not in existing:
                    logger.info(f"Migrating database: adding {table}.{name}")
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    if (table, name) == ("jobs", "is_upload"):
                        try:
                            _backfill_uploads(conn)
                        except Exception as e:  # cosmetic; never block startup
                            logger.warning(f"Upload backfill skipped: {e}")
        # API keys used to be stored in plain text; scrub any that are left over.
        if inspector.has_table("jobs"):
            conn.execute(text("UPDATE jobs SET gemini_api_key_override = NULL WHERE gemini_api_key_override IS NOT NULL"))


def init_db() -> None:
    """Initialize SQLite database tables and apply additive migrations."""
    import app.models.db_models  # noqa: F401  (register tables)
    SQLModel.metadata.create_all(engine)
    _migrate()


def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session per request."""
    with Session(engine) as session:
        yield session
