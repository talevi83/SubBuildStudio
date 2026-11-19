"""Unit tests for helpers added by the security / bug review fixes."""
import os
import stat
import threading
import time
from pathlib import Path

import pytest

from app.utils.filenames import defuse_reserved, is_reserved_device_name, safe_basename


# ------------------------------------------------------------------ file names

@pytest.mark.parametrize("name", ["CON", "con", "NUL.json", "aux.he.srt", "COM1", "lpt9.txt", "PRN ", "COM¹.mkv"])
def test_device_names_are_recognised(name):
    assert is_reserved_device_name(name)


@pytest.mark.parametrize("name", ["CONSOLE", "Movie", "nullify.srt", "COM10", "my con.mkv", "", "LPT"])
def test_ordinary_names_are_not_device_names(name):
    assert not is_reserved_device_name(name)


def test_defuse_reserved_keeps_the_extension():
    assert defuse_reserved("NUL.json") == "NUL_.json"
    assert defuse_reserved("con.he.srt") == "con_.he.srt"
    assert defuse_reserved("Movie.mkv") == "Movie.mkv"


def test_safe_basename():
    assert safe_basename("..\\..\\Windows\\CON.mkv") == "CON_.mkv"
    assert safe_basename("/etc/passwd") == "passwd"
    assert safe_basename('a:b*c?.mkv') == "a_b_c_.mkv"
    assert safe_basename("  ..  ", fallback="x") == "x"
    assert safe_basename(None, fallback="upload.bin") == "upload.bin"
    assert len(safe_basename("a" * 500)) == 200


def test_glossary_profile_names():
    from app.core.glossary_profiles import clean_name

    assert clean_name("NUL") == "NUL_"
    assert clean_name("..") == ""
    assert clean_name("The Office (US).") == "The Office (US)"
    assert clean_name("a/b\\c") == "abc"


def test_embed_output_name_is_safe():
    from app.core.video_embed import output_name

    assert output_name("CON.mkv", "he", "soft", ".mkv") == "CON_.he.subs.mkv"
    assert output_name("..\\x\\Movie.mkv", "he", "burn", ".mp4") == "Movie.he.burned.mp4"


# ------------------------------------------------------------------ executables

def _make_exe(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + (".exe" if os.name == "nt" else ""))
    path.write_bytes(b"")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_find_executable_ignores_the_current_folder(tmp_path, monkeypatch):
    from app.core.executables import find_executable

    planted = _make_exe(tmp_path / "videos", "ffmpeg")
    monkeypatch.chdir(planted.parent)
    # "" and "." and relative entries all mean "the current folder"
    assert find_executable("ffmpeg", path=os.pathsep.join(["", ".", "videos", "rel" + os.sep + "bin"])) is None
    real = _make_exe(tmp_path / "tools", "ffmpeg")
    found = find_executable("ffmpeg", path=os.pathsep.join([".", str(real.parent)]))
    assert found is not None and Path(found) == real


def test_bare_name_fallback_when_missing(monkeypatch):
    from app.core import executables

    monkeypatch.setattr(executables, "find_executable", lambda name, path=None: None)
    assert executables.executable("ffprobe") == "ffprobe"


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_windows_create_process_skips_the_current_folder():
    import app.core.executables  # noqa: F401

    assert os.environ.get("NoDefaultCurrentDirectoryInExePath") == "1"


# ------------------------------------------------------------------ private files

def test_write_private_text_is_owner_only(tmp_path):
    from app.core.private_files import restrict_to_owner, write_private_text

    path = tmp_path / ".session-token"
    write_private_text(path, "secret")
    assert path.read_text(encoding="utf-8") == "secret"
    if os.name == "nt":
        assert restrict_to_owner(path) is True
    else:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    write_private_text(path, "second")  # rewriting keeps working
    assert path.read_text(encoding="utf-8") == "second"


def test_saving_the_api_key_writes_a_private_env(tmp_path, monkeypatch):
    from app.core import config

    env = tmp_path / ".env"
    env.write_text("PORT=8000\n", encoding="utf-8")
    monkeypatch.setattr(config, "env_file_path", lambda: env)
    calls = []
    import app.core.private_files as pf
    monkeypatch.setattr(pf, "restrict_to_owner", lambda p: calls.append(Path(p)) or True)
    old = config.settings.gemini_api_key
    try:
        config.save_gemini_api_key("AIzaTest")
    finally:
        config.settings.gemini_api_key = old
    assert "GEMINI_API_KEY=AIzaTest" in env.read_text(encoding="utf-8")
    # The new content is written to a locked-down temp file next to .env that then replaces it
    assert len(calls) == 1 and calls[0].parent == env.parent and calls[0].name.startswith("..env.")
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]  # no temp file left behind


def test_launcher_token_file_is_private(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "subbuild_launcher_private", Path(__file__).resolve().parents[3] / "packaging" / "launcher.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    import app.core.private_files as pf

    calls = []
    monkeypatch.setattr(pf, "restrict_to_owner", lambda p: calls.append(Path(p).name) or True)
    token = launcher.new_session_token(tmp_path)
    assert (tmp_path / ".session-token").read_text(encoding="utf-8") == token
    assert len(calls) == 1 and calls[0].startswith("..session-token.")
    assert sorted(p.name for p in tmp_path.iterdir()) == [".session-token"]


# ------------------------------------------------------------------ ollama

def test_ollama_needs_the_exact_model():
    from app.core.ollama_setup import has_model

    assert has_model(["qwen2.5:7b"], "qwen2.5:7b")
    assert has_model(["Qwen2.5:7B"], "qwen2.5:7b")
    assert has_model(["llama3:latest"], "llama3")
    assert has_model(["llama3"], "llama3:latest")
    assert not has_model(["qwen2.5:0.5b"], "qwen2.5:7b")
    assert not has_model(["qwen2.5-coder:7b"], "qwen2.5:7b")
    assert not has_model([], "qwen2.5:7b")


# ------------------------------------------------------------------ bounded in-memory state

def test_job_locks_are_dropped_when_unused():
    from app.core import edit_history as history

    with history.job_lock("job_tmp_lock"):
        assert "job_tmp_lock" in history._locks
    assert "job_tmp_lock" not in history._locks


def test_job_lock_still_serialises():
    from app.core import edit_history as history

    order = []
    entered = threading.Event()

    def first():
        with history.job_lock("job_serial"):
            entered.set()
            time.sleep(0.2)
            order.append("first")

    t = threading.Thread(target=first)
    t.start()
    entered.wait(2)
    with history.job_lock("job_serial"):
        order.append("second")
    t.join()
    assert order == ["first", "second"]
    assert "job_serial" not in history._locks


def test_upload_hash_cache_is_bounded(monkeypatch):
    from app.api import routes_media as rm

    monkeypatch.setattr(rm, "_hash_cache", {})
    monkeypatch.setattr(rm, "_HASH_CACHE_MAX", 3)
    for i in range(5):
        rm._cache_hash(f"k{i}", (i, i, "h"))
    assert list(rm._hash_cache) == ["k2", "k3", "k4"]


def test_finished_embed_tasks_are_forgotten_after_a_while():
    from app.core import video_embed as ve

    old = ve.EmbedTask(id="old1", job_id="j", mode="soft", target=Path("a.mkv"), in_app_storage=False,
                       state="done", finished_at=1.0)
    running = ve.EmbedTask(id="run1", job_id="j", mode="soft", target=Path("b.mkv"), in_app_storage=False)
    ve._tasks.update({old.id: old, running.id: running})
    try:
        ve._prune_finished(now=1.0 + ve._FINISHED_TTL_SECONDS + 1)
        assert "old1" not in ve._tasks and "run1" in ve._tasks
    finally:
        ve._tasks.pop("old1", None)
        ve._tasks.pop("run1", None)


def test_running_task_for_target(tmp_path):
    from app.core import video_embed as ve

    target = tmp_path / "Movie.he.subs.mkv"
    t = ve.EmbedTask(id="tgt1", job_id="a", mode="soft", target=target, in_app_storage=False)
    ve._tasks[t.id] = t
    try:
        assert ve.running_task_for_target(tmp_path / "." / "Movie.he.subs.mkv") is t
        assert ve.running_task_for_target(tmp_path / "Other.mkv") is None
        t.state = "done"
        assert ve.running_task_for_target(target) is None
    finally:
        ve._tasks.pop("tgt1", None)


def test_range_parsing_of_an_empty_file():
    from app.api.routes_media import _parse_range

    assert _parse_range("bytes=-10", 0) is None
    assert _parse_range("bytes=0-", 0) is None
