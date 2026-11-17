"""Databases created while the app was still called SubForge Studio must survive the rename."""
import sqlite3

import pytest

from app.core import config, database


@pytest.fixture
def default_db(tmp_path, monkeypatch):
    new = tmp_path / "subbuild.db"
    monkeypatch.setattr(config.settings, "database_url", f"sqlite:///{new.as_posix()}")
    return new


def _make_db(path, marker):
    con = sqlite3.connect(path)
    con.execute("create table t (v text)")
    con.execute("insert into t values (?)", (marker,))
    con.commit()
    con.close()


def test_old_database_is_renamed(default_db):
    old = default_db.with_name("subforge.db")
    _make_db(old, "kept")
    database._adopt_legacy_database()
    assert default_db.is_file() and not old.exists()
    assert sqlite3.connect(default_db).execute("select v from t").fetchone() == ("kept",)


def test_existing_new_database_is_never_overwritten(default_db):
    old = default_db.with_name("subforge.db")
    _make_db(old, "old")
    _make_db(default_db, "new")
    database._adopt_legacy_database()
    assert old.is_file()
    assert sqlite3.connect(default_db).execute("select v from t").fetchone() == ("new",)


def test_custom_database_url_is_left_alone(tmp_path, monkeypatch):
    custom = tmp_path / "mine.db"
    old = tmp_path / "subforge.db"
    _make_db(old, "x")
    monkeypatch.setattr(config.settings, "database_url", f"sqlite:///{custom.as_posix()}")
    database._adopt_legacy_database()
    assert old.is_file() and not custom.exists()


def test_legacy_subforge_url_is_migrated_and_renamed(tmp_path, monkeypatch):
    old = tmp_path / "subforge.db"
    new = tmp_path / "subbuild.db"
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=abc\nDATABASE_URL=sqlite:///./storage/subforge.db\n", encoding="utf-8")
    _make_db(old, "migrated")

    monkeypatch.setattr(config.settings, "database_url", f"sqlite:///{old.as_posix()}")
    monkeypatch.setattr(config, "env_file_path", lambda: env_file)
    database._adopt_legacy_database()

    assert config.settings.database_url == f"sqlite:///{new.as_posix()}"
    assert new.is_file() and not old.exists()
    assert sqlite3.connect(new).execute("select v from t").fetchone() == ("migrated",)
    assert "DATABASE_URL=sqlite:///./storage/subbuild.db" in env_file.read_text(encoding="utf-8")


def test_tests_use_isolated_database():
    assert "subbuild_test_" in config.settings.database_url


def test_migration_adds_missing_columns(tmp_path):
    from sqlalchemy import create_engine, inspect, text

    old = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with old.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE jobs (id VARCHAR PRIMARY KEY, video_path VARCHAR, "
                "file_name VARCHAR, gemini_api_key_override VARCHAR)"
            )
        )
        conn.execute(text("CREATE TABLE subtitle_segments (id INTEGER PRIMARY KEY, job_id VARCHAR)"))
        conn.execute(text("INSERT INTO jobs VALUES ('j', 'x', 'x.mkv', 'secret')"))
        conn.execute(
            text(
                "INSERT INTO jobs VALUES ('u', 'C:\\Tools\\app\\storage\\uploads\\1a2b3c4d_Show [720p].mkv', "
                "'1a2b3c4d_Show [720p].mkv', NULL)"
            )
        )
    original = database.engine
    database.engine = old
    try:
        database._migrate()
    finally:
        database.engine = original
    cols = {c["name"] for c in inspect(old).get_columns("jobs")}
    assert {"is_upload", "warning_message", "detected_language", "glossary_json"} <= cols
    assert "translation_failed" in {c["name"] for c in inspect(old).get_columns("subtitle_segments")}
    with old.begin() as conn:
        assert conn.execute(text("SELECT gemini_api_key_override FROM jobs WHERE id='j'")).scalar() is None
        row = conn.execute(text("SELECT is_upload, file_name FROM jobs WHERE id='u'")).one()
        assert row == (1, "Show [720p].mkv")
        assert conn.execute(text("SELECT is_upload FROM jobs WHERE id='j'")).scalar() == 0
