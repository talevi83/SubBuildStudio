"""Unit tests for the Git & Release auto-updater (app.core.updater)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core import updater

pytestmark = pytest.mark.unit


def _make_fake_repo(tmp_path: Path, version: str = "1.5.0") -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "requirements.txt").write_text("fastapi==0.100\n", encoding="utf-8")
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text('{"name":"frontend"}\n', encoding="utf-8")
    (tmp_path / "frontend" / "dist").mkdir()
    (tmp_path / "frontend" / "dist" / "index.html").write_text("<html/>", encoding="utf-8")
    return tmp_path


def test_parse_semver():
    assert updater.parse_semver("v1.5.0") == (1, 5, 0)
    assert updater.parse_semver("1.5.0") == (1, 5, 0)
    assert updater.parse_semver("  v2.10.3 ") == (2, 10, 3)
    assert updater.parse_semver("main") is None
    assert updater.parse_semver("") is None
    assert updater.parse_semver("v1.5") is None


def test_get_current_version_info_prefers_highest_semver(tmp_path: Path):
    repo = _make_fake_repo(tmp_path, version="1.5.0")

    def fake_run_git(_root, *args, timeout=4.0):
        cmd = " ".join(args)
        if cmd.startswith("describe"):
            return 0, "v1.6.0", ""
        if cmd == "rev-parse --short HEAD":
            return 0, "abc1234", ""
        if cmd == "rev-parse HEAD":
            return 0, "abc1234" + "0" * 33, ""
        if cmd == "rev-parse --abbrev-ref HEAD":
            return 0, "main", ""
        return 1, "", ""

    with patch.object(updater, "_run_git", side_effect=fake_run_git):
        info = updater.get_current_version_info(repo)

    assert info["current_version"] == "1.6.0"
    assert info["current_commit"] == "abc1234"
    assert info["branch"] == "main"


def test_check_for_updates_detects_newer_release_tag(tmp_path: Path):
    repo = _make_fake_repo(tmp_path, version="1.5.0")
    head_sha = "1111111" + "0" * 33
    rel_sha = "2222222" + "0" * 33

    def fake_run_git(_root, *args, timeout=4.0):
        cmd = " ".join(args)
        if cmd.startswith("describe"):
            return 0, "v1.5.0", ""
        if cmd == "rev-parse --short HEAD":
            return 0, "1111111", ""
        if cmd == "rev-parse HEAD":
            return 0, head_sha, ""
        if cmd == "rev-parse --abbrev-ref HEAD":
            return 0, "main", ""
        if cmd.startswith("ls-remote"):
            return (
                0,
                f"{head_sha}\trefs/heads/main\n"
                f"aaaaaaa0000000\trefs/tags/v1.5.0\n"
                f"{rel_sha}\trefs/tags/v1.6.0^{{}}\n",
                "",
            )
        return 1, "", ""

    with (
        patch.object(updater, "_run_git", side_effect=fake_run_git),
        patch.object(updater, "_fetch_github_release_metadata", return_value=("https://example.com/v1.6.0", "New UI")),
    ):
        res = updater.check_for_updates(repo)

    assert res["update_available"] is True
    assert res["update_kind"] == "release"
    assert res["latest_version"] == "1.6.0"
    assert res["latest_commit"] == "2222222"
    assert res["target_ref"] == "v1.6.0"
    assert res["release_url"] == "https://example.com/v1.6.0"
    assert res["release_notes"] == "New UI"


def test_check_for_updates_ignores_unreleased_commits_without_newer_release_tag(tmp_path: Path):
    repo = _make_fake_repo(tmp_path, version="1.5.0")
    local_sha = "1111111" + "0" * 33

    def fake_run_git(_root, *args, timeout=4.0):
        cmd = " ".join(args)
        if cmd.startswith("describe"):
            return 0, "v1.5.0", ""
        if cmd == "rev-parse --short HEAD":
            return 0, "1111111", ""
        if cmd == "rev-parse HEAD":
            return 0, local_sha, ""
        if cmd == "rev-parse --abbrev-ref HEAD":
            return 0, "main", ""
        if cmd.startswith("ls-remote"):
            return 0, f"{local_sha}\trefs/tags/v1.5.0\n", ""
        return 1, "", ""

    with patch.object(updater, "_run_git", side_effect=fake_run_git):
        res = updater.check_for_updates(repo)

    assert res["update_available"] is False
    assert res["update_kind"] is None
    assert res["commits_behind"] == 0
    assert res["latest_version"] == "1.5.0"
    assert res["target_ref"] is None


def test_check_for_updates_handles_offline_gracefully(tmp_path: Path):
    repo = _make_fake_repo(tmp_path, version="1.5.0")

    def fake_run_git(_root, *args, timeout=4.0):
        cmd = " ".join(args)
        if cmd.startswith("describe"):
            return 0, "v1.5.0", ""
        if cmd == "rev-parse --short HEAD":
            return 0, "1111111", ""
        if cmd == "rev-parse HEAD":
            return 0, "1111111" + "0" * 33, ""
        if cmd == "rev-parse --abbrev-ref HEAD":
            return 0, "main", ""
        if cmd.startswith("ls-remote"):
            return 128, "", "fatal: unable to access origin: Could not resolve host"
        return 1, "", ""

    with patch.object(updater, "_run_git", side_effect=fake_run_git):
        res = updater.check_for_updates(repo)

    assert res["update_available"] is False
    assert "Could not resolve host" in (res["error"] or "")


def test_apply_update_skips_feature_branch_in_auto_mode_and_dirty_worktree(tmp_path: Path):
    repo = _make_fake_repo(tmp_path, version="1.5.0")

    release_status = {
        "current_version": "1.5.0",
        "current_commit": "1111111",
        "branch": "feat/custom-work",
        "update_available": True,
        "update_kind": "release",
        "latest_version": "1.6.0",
        "latest_commit": "4444444",
        "commits_behind": 0,
        "target_ref": "v1.6.0",
        "release_url": None,
        "release_notes": None,
        "error": None,
    }
    with patch.object(updater, "check_for_updates", return_value=release_status):
        res = updater.apply_update(repo, auto_mode=True)
    assert res["updated"] is False
    assert res["skipped_reason"] == "feature_branch"

    main_status = {**release_status, "branch": "main"}
    with (
        patch.object(updater, "check_for_updates", return_value=main_status),
        patch.object(updater, "_run_git", return_value=(0, " M backend/app/main.py", "")),
    ):
        res_dirty = updater.apply_update(repo, auto_mode=False)
    assert res_dirty["updated"] is False
    assert res_dirty["skipped_reason"] == "dirty_worktree"


def test_apply_update_pulls_and_installs_changed_requirements(tmp_path: Path):
    repo = _make_fake_repo(tmp_path, version="1.5.0")
    pull_calls = []

    release_status = {
        "current_version": "1.5.0",
        "current_commit": "1111111",
        "branch": "main",
        "update_available": True,
        "update_kind": "release",
        "latest_version": "1.6.0",
        "latest_commit": "5555555",
        "commits_behind": 0,
        "target_ref": "v1.6.0",
        "release_url": None,
        "release_notes": None,
        "error": None,
    }

    state = {"pulled": False}

    def fake_run_git(root, *args, timeout=4.0):
        cmd = " ".join(args)
        pull_calls.append(cmd)
        if cmd.startswith("status"):
            return 0, "", ""
        if cmd.startswith("fetch"):
            return 0, "", ""
        if cmd.startswith("merge --ff-only v1.6.0"):
            state["pulled"] = True
            (root / "VERSION").write_text("1.6.0\n", encoding="utf-8")
            (root / "backend" / "requirements.txt").write_text("fastapi==0.101\n", encoding="utf-8")
            return 0, "Updating 1111111..5555555", ""
        if cmd.startswith("describe"):
            return 0, "v1.6.0" if state["pulled"] else "v1.5.0", ""
        if cmd == "rev-parse --short HEAD":
            return 0, "5555555" if state["pulled"] else "1111111", ""
        if cmd == "rev-parse HEAD":
            return 0, ("5555555" if state["pulled"] else "1111111") + "0" * 33, ""
        if cmd == "rev-parse --abbrev-ref HEAD":
            return 0, "main", ""
        return 0, "", ""

    with (
        patch.object(updater, "check_for_updates", return_value=release_status),
        patch.object(updater, "_run_git", side_effect=fake_run_git),
        patch.object(updater, "_install_backend_requirements", return_value=True) as mock_pip,
    ):
        res = updater.apply_update(repo, auto_mode=True, rebuild_frontend=False)

    assert res["updated"] is True
    assert res["previous_version"] == "1.5.0"
    assert res["new_version"] == "1.6.0"
    assert res["new_commit"] == "5555555"
    assert res["requirements_updated"] is True
    assert res["restart_required"] is True
    mock_pip.assert_called_once_with(repo)


# ------------------------------------------------------------------ hardening (tag origin, rollback)

_RELEASE = {
    "current_version": "1.5.0",
    "current_commit": "1111111",
    "branch": "main",
    "update_available": True,
    "update_kind": "release",
    "latest_version": "1.6.0",
    "latest_commit": "5555555",
    "commits_behind": 0,
    "target_ref": "v1.6.0",
    "release_url": None,
    "release_notes": None,
    "error": None,
}
_OLD_SHA = "1111111" + "0" * 33


def _git_fake(calls, *, tag_on_main=True, change_requirements=False, change_frontend=False):
    state = {"pulled": False}

    def fake_run_git(root, *args, timeout=4.0):
        cmd = " ".join(args)
        calls.append(cmd)
        if cmd.startswith("merge-base --is-ancestor"):
            return (0, "", "") if tag_on_main else (1, "", "")
        if cmd.startswith("merge --ff-only"):
            state["pulled"] = True
            if change_requirements:
                (root / "backend" / "requirements.txt").write_text("fastapi==0.101\n", encoding="utf-8")
            return 0, "", ""
        if cmd.startswith("reset --keep"):
            state["pulled"] = False
            return 0, "", ""
        if cmd.startswith("diff --name-only"):
            return 0, ("frontend/src/App.tsx" if change_frontend else ""), ""
        if cmd == "rev-parse HEAD":
            return 0, ("5555555" + "0" * 33) if state["pulled"] else _OLD_SHA, ""
        if cmd == "rev-parse --short HEAD":
            return 0, "5555555" if state["pulled"] else "1111111", ""
        if cmd == "rev-parse --abbrev-ref HEAD":
            return 0, "main", ""
        return 0, "", ""

    return fake_run_git


def test_a_release_tag_that_is_not_on_main_is_not_installed(tmp_path: Path):
    repo = _make_fake_repo(tmp_path)
    calls = []
    with (
        patch.object(updater, "check_for_updates", return_value=_RELEASE),
        patch.object(updater, "_run_git", side_effect=_git_fake(calls, tag_on_main=False)),
    ):
        res = updater.apply_update(repo, auto_mode=True, rebuild_frontend=False)
    assert res["updated"] is False
    assert res["skipped_reason"] == "tag_not_on_main"
    assert not any(c.startswith("merge --ff-only") for c in calls)


def test_a_failed_dependency_install_rolls_the_code_back(tmp_path: Path):
    repo = _make_fake_repo(tmp_path)
    calls = []
    with (
        patch.object(updater, "check_for_updates", return_value=_RELEASE),
        patch.object(updater, "_run_git", side_effect=_git_fake(calls, change_requirements=True)),
        patch.object(updater, "_install_backend_requirements", return_value=False) as pip,
    ):
        res = updater.apply_update(repo, auto_mode=True, rebuild_frontend=False)
    assert res["updated"] is False
    assert res["skipped_reason"] == "dependency_install_failed"
    assert f"reset --keep {_OLD_SHA}" in calls
    assert pip.call_count == 2  # the new requirements (failed), then the old ones again


def test_a_frontend_build_that_wiped_dist_rolls_back_and_rebuilds_the_old_interface(tmp_path: Path):
    repo = _make_fake_repo(tmp_path)
    calls = []
    results = iter([False, True])

    def build(root, run_npm_install):
        ok = next(results)
        if not ok:  # vite empties frontend/dist first, then fails
            (root / "frontend" / "dist" / "index.html").unlink()
        return ok

    with (
        patch.object(updater, "check_for_updates", return_value=_RELEASE),
        patch.object(updater, "_run_git", side_effect=_git_fake(calls, change_frontend=True)),
        patch.object(updater, "_rebuild_frontend", side_effect=build) as mock_build,
    ):
        res = updater.apply_update(repo, auto_mode=True, rebuild_frontend=True)
    assert res["updated"] is False
    assert res["skipped_reason"] == "frontend_build_failed"
    assert f"reset --keep {_OLD_SHA}" in calls
    assert mock_build.call_count == 2


def test_a_frontend_build_that_could_not_start_keeps_the_update(tmp_path: Path):
    repo = _make_fake_repo(tmp_path)  # e.g. npm is not installed: the previous frontend/dist is untouched
    calls = []
    with (
        patch.object(updater, "check_for_updates", return_value=_RELEASE),
        patch.object(updater, "_run_git", side_effect=_git_fake(calls, change_frontend=True)),
        patch.object(updater, "_rebuild_frontend", return_value=False),
    ):
        res = updater.apply_update(repo, auto_mode=True, rebuild_frontend=True)
    assert res["updated"] is True and res["frontend_rebuilt"] is False
    assert not any(c.startswith("reset") for c in calls)


def test_release_notes_are_fetched_for_the_tag_being_offered():
    assert updater.GITHUB_RELEASE_BY_TAG_API.format(tag="v2.0.1").endswith("/releases/tags/v2.0.1")


def test_read_version_file(tmp_path: Path):
    assert updater.read_version_file(_make_fake_repo(tmp_path, version="2.0.1")) == "2.0.1"
    assert updater.read_version_file(tmp_path / "missing") == "0.0.0"


def test_signed_tags_are_required_only_when_turned_on(tmp_path: Path, monkeypatch):
    repo = _make_fake_repo(tmp_path)
    monkeypatch.delenv("SUBBUILD_REQUIRE_SIGNED_TAGS", raising=False)
    assert updater.require_signed_tags(repo) is False
    (repo / "backend" / ".env").write_text("PORT=8000\nSUBBUILD_REQUIRE_SIGNED_TAGS=true  # on\n", encoding="utf-8")
    assert updater.require_signed_tags(repo) is True
    monkeypatch.setenv("SUBBUILD_REQUIRE_SIGNED_TAGS", "0")  # the environment wins over .env
    assert updater.require_signed_tags(repo) is False


@pytest.mark.parametrize("signed", [True, False])
def test_an_unsigned_tag_is_not_installed_when_signatures_are_required(tmp_path: Path, monkeypatch, signed):
    repo = _make_fake_repo(tmp_path)
    monkeypatch.setenv("SUBBUILD_REQUIRE_SIGNED_TAGS", "1")
    calls = []
    fake = _git_fake(calls)

    def run_git(root, *args, timeout=4.0):
        if args and args[0] == "verify-tag":
            calls.append(" ".join(args))
            return (0, "", "") if signed else (1, "", "error: no signature found")
        return fake(root, *args, timeout=timeout)

    with (
        patch.object(updater, "check_for_updates", return_value=_RELEASE),
        patch.object(updater, "_run_git", side_effect=run_git),
    ):
        res = updater.apply_update(repo, auto_mode=True, rebuild_frontend=False)
    assert "verify-tag v1.6.0" in calls
    assert res["updated"] is signed
    assert any(c.startswith("merge --ff-only") for c in calls) is signed
    if not signed:
        assert res["skipped_reason"] == "unsigned_tag"


# ------------------------------------------------------------------ VERSION comes from the tags

_PLACEHOLDER = "$Format:%(describe:tags=true,match=v[0-9]*)$\n"


@pytest.mark.parametrize("content, expected", [
    ("v2.1.0\n", "2.1.0"),                # zip of a release (GitHub fills in the tag)
    ("v2.1.0-3-gabc1234\n", "2.1.0"),     # zip of a later commit
    ("2.0.1\n", "2.0.1"),                 # an older, hand-written file
])
def test_version_file_as_filled_in_by_github(tmp_path: Path, content, expected):
    (tmp_path / "VERSION").write_text(content, encoding="utf-8")
    assert updater.read_version_file(tmp_path) == expected
    assert updater.get_current_version_info(tmp_path)["current_version"] == expected


def test_in_a_git_checkout_the_placeholder_is_ignored_and_the_tag_wins(tmp_path: Path):
    repo = _make_fake_repo(tmp_path)
    (repo / "VERSION").write_text(_PLACEHOLDER, encoding="utf-8")

    def fake_git(root, *args, timeout=4.0):
        return (0, "v2.1.0", "") if args[:1] == ("describe",) else (0, "", "")

    with patch.object(updater, "_run_git", side_effect=fake_git):
        assert updater.read_version_file(repo) == "2.1.0"
        assert updater.get_current_version_info(repo)["current_version"] == "2.1.0"


def test_without_tags_or_a_filled_in_file_the_version_is_unknown(tmp_path: Path):
    (tmp_path / "VERSION").write_text(_PLACEHOLDER, encoding="utf-8")
    assert updater.read_version_file(tmp_path) == "0.0.0"
    with patch.object(updater, "_run_git", return_value=(128, "", "not a git repository")):
        assert updater.get_current_version_info(tmp_path)["current_version"] == "0.0.0"


def _git_version() -> tuple:
    import re
    import subprocess

    out = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout
    match = re.search(r"(\d+)\.(\d+)", out)
    return tuple(int(x) for x in match.groups()) if match else (0, 0)


@pytest.mark.skipif(_git_version() < (2, 35), reason="%(describe:tags=...) needs git 2.35+ (GitHub's is newer)")
def test_a_release_zip_carries_its_version(tmp_path: Path):
    import io
    import subprocess
    import tarfile

    repo_root = Path(updater.get_repo_root())
    assert (repo_root / "VERSION").read_text(encoding="utf-8") == _PLACEHOLDER
    assert "VERSION export-subst" in (repo_root / ".gitattributes").read_text(encoding="utf-8")

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], capture_output=True, check=True).stdout

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "VERSION").write_text(_PLACEHOLDER, encoding="utf-8")
    (tmp_path / ".gitattributes").write_text("VERSION export-subst\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    git("tag", "v2.1.0")  # lightweight, like the tags GitHub's website creates
    archive = tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", "v2.1.0")))
    extracted = tmp_path / "zip"
    archive.extractall(extracted, filter="data")
    assert updater.read_version_file(extracted) == "2.1.0"


# ------------------------------------------------------------------ non-git (ZIP release) updater


def _make_non_git_install(tmp_path: Path, version: str = "1.5.0") -> Path:
    root = tmp_path / "app"
    root.mkdir()
    (root / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    (root / "backend").mkdir()
    (root / "backend" / "requirements.txt").write_text("fastapi==0.100\n", encoding="utf-8")
    (root / "backend" / ".env").write_text("SECRET=keep\n", encoding="utf-8")
    (root / "frontend").mkdir()
    (root / "frontend" / "package.json").write_text('{"name":"frontend"}\n', encoding="utf-8")
    (root / "frontend" / "dist").mkdir()
    (root / "frontend" / "dist" / "index.html").write_text("<html/>", encoding="utf-8")
    return root


def _build_release_zip(zip_path: Path, top_dir: str, files: dict[str, str]) -> Path:
    import zipfile

    with zipfile.ZipFile(zip_path, "w") as zf:
        for rel, content in files.items():
            zf.writestr(f"{top_dir}/{rel}", content)
    return zip_path


def test_check_for_updates_without_git_uses_github_tags_api(tmp_path: Path):
    root = _make_non_git_install(tmp_path, version="1.5.0")
    tags = {
        "v1.5.0": "1111111" + "0" * 33,
        "v1.6.0": "2222222" + "0" * 33,
    }
    with (
        patch.object(updater, "_fetch_github_tags", return_value=(tags, None)),
        patch.object(
            updater,
            "_fetch_github_release_metadata",
            return_value=("https://example.com/v1.6.0", "ZIP release notes"),
        ),
    ):
        res = updater.check_for_updates(root)

    assert res["error"] is None
    assert res["update_available"] is True
    assert res["update_kind"] == "release"
    assert res["latest_version"] == "1.6.0"
    assert res["latest_commit"] == "2222222"
    assert res["target_ref"] == "v1.6.0"
    assert res["release_url"] == "https://example.com/v1.6.0"
    assert res["release_notes"] == "ZIP release notes"


def test_check_for_updates_without_git_handles_offline_gracefully(tmp_path: Path):
    root = _make_non_git_install(tmp_path, version="1.5.0")
    with patch.object(updater, "_fetch_github_tags", return_value=(None, "URLError: offline")):
        res = updater.check_for_updates(root)

    assert res["update_available"] is False
    assert "offline" in (res["error"] or "")


def test_apply_update_without_git_downloads_zip_and_preserves_user_files(tmp_path: Path, monkeypatch):
    import io

    monkeypatch.delenv("SUBBUILD_REQUIRE_SIGNED_TAGS", raising=False)
    root = _make_non_git_install(tmp_path, version="1.5.0")
    zip_file = _build_release_zip(
        tmp_path / "release.zip",
        "SubBuildStudio-1.6.0",
        {
            "VERSION": "v1.6.0\n",
            "backend/requirements.txt": "fastapi==0.101\n",
            "backend/app/new_file.py": "VALUE = 1\n",
            "backend/.env": "SECRET=overwritten\n",
        },
    )

    class _FakeZipResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    release_status = {
        "current_version": "1.5.0",
        "current_commit": "",
        "branch": "",
        "update_available": True,
        "update_kind": "release",
        "latest_version": "1.6.0",
        "latest_commit": "5555555",
        "commits_behind": 0,
        "target_ref": "v1.6.0",
        "release_url": None,
        "release_notes": None,
        "error": None,
    }

    with (
        patch.object(updater, "check_for_updates", return_value=release_status),
        patch.object(updater, "_verify_github_tag_on_main", return_value=(True, "origin/main")),
        patch.object(updater.urllib.request, "urlopen", return_value=_FakeZipResp(zip_file.read_bytes())),
        patch.object(updater, "_install_backend_requirements", return_value=True) as mock_pip,
    ):
        res = updater.apply_update(root, auto_mode=True, rebuild_frontend=False)

    assert res["updated"] is True
    assert res["previous_version"] == "1.5.0"
    assert res["new_version"] == "1.6.0"
    assert res["new_commit"] == "5555555"
    assert res["requirements_updated"] is True
    assert res["restart_required"] is True
    mock_pip.assert_called_once_with(root)
    assert (root / "backend" / "app" / "new_file.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (root / "backend" / ".env").read_text(encoding="utf-8") == "SECRET=keep\n"


def test_apply_update_without_git_rolls_back_on_failed_dependency_install(tmp_path: Path, monkeypatch):
    import zipfile

    monkeypatch.delenv("SUBBUILD_REQUIRE_SIGNED_TAGS", raising=False)
    root = _make_non_git_install(tmp_path, version="1.5.0")
    zip_file = _build_release_zip(
        tmp_path / "release.zip",
        "SubBuildStudio-1.6.0",
        {
            "VERSION": "v1.6.0\n",
            "backend/requirements.txt": "fastapi==0.101\n",
            "backend/app/new_file.py": "VALUE = 1\n",
        },
    )

    def fake_extract(_tag: str, temp_dir: Path, timeout: float = 60.0):
        dest = temp_dir / "extracted"
        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(dest)
        return dest / "SubBuildStudio-1.6.0", None

    release_status = {
        "current_version": "1.5.0",
        "current_commit": "",
        "branch": "",
        "update_available": True,
        "update_kind": "release",
        "latest_version": "1.6.0",
        "latest_commit": "5555555",
        "commits_behind": 0,
        "target_ref": "v1.6.0",
        "release_url": None,
        "release_notes": None,
        "error": None,
    }

    with (
        patch.object(updater, "check_for_updates", return_value=release_status),
        patch.object(updater, "_verify_github_tag_on_main", return_value=(True, "origin/main")),
        patch.object(updater, "_download_and_extract_release_zip", side_effect=fake_extract),
        patch.object(updater, "_install_backend_requirements", return_value=False) as mock_pip,
    ):
        res = updater.apply_update(root, auto_mode=True, rebuild_frontend=False)

    assert res["updated"] is False
    assert res["skipped_reason"] == "dependency_install_failed"
    assert mock_pip.call_count == 2
    assert (root / "VERSION").read_text(encoding="utf-8") == "1.5.0\n"
    assert (root / "backend" / "requirements.txt").read_text(encoding="utf-8") == "fastapi==0.100\n"
    assert not (root / "backend" / "app" / "new_file.py").exists()


def test_apply_update_without_git_skips_tag_not_on_main_and_unsigned_tag(tmp_path: Path, monkeypatch):
    root = _make_non_git_install(tmp_path, version="1.5.0")
    release_status = {
        "current_version": "1.5.0",
        "current_commit": "",
        "branch": "",
        "update_available": True,
        "update_kind": "release",
        "latest_version": "1.6.0",
        "latest_commit": "5555555",
        "commits_behind": 0,
        "target_ref": "v1.6.0",
        "release_url": None,
        "release_notes": None,
        "error": None,
    }

    with (
        patch.object(updater, "check_for_updates", return_value=release_status),
        patch.object(updater, "_verify_github_tag_on_main", return_value=(False, "origin/main")),
    ):
        res_branch = updater.apply_update(root, auto_mode=True, rebuild_frontend=False)
    assert res_branch["updated"] is False
    assert res_branch["skipped_reason"] == "tag_not_on_main"

    monkeypatch.setenv("SUBBUILD_REQUIRE_SIGNED_TAGS", "1")
    with (
        patch.object(updater, "check_for_updates", return_value=release_status),
        patch.object(updater, "_verify_github_tag_on_main", return_value=(True, "origin/main")),
        patch.object(updater, "verify_release_tag", return_value=(False, "bad sig")),
    ):
        res_sig = updater.apply_update(root, auto_mode=True, rebuild_frontend=False)
    assert res_sig["updated"] is False
    assert res_sig["skipped_reason"] == "unsigned_tag"


def test_apply_update_in_an_installer_copy_points_to_the_new_installer(tmp_path: Path, monkeypatch):
    """A copy from the Windows installer has no Node.js and must not be patched in place."""
    monkeypatch.delenv("SUBBUILD_REQUIRE_SIGNED_TAGS", raising=False)
    root = _make_non_git_install(tmp_path, version="1.5.0")
    (root / updater.INSTALLER_MARKER).write_text("windows", encoding="ascii")
    release_status = {
        "current_version": "1.5.0",
        "current_commit": "",
        "branch": "",
        "update_available": True,
        "update_kind": "release",
        "latest_version": "1.6.0",
        "latest_commit": "5555555",
        "commits_behind": 0,
        "target_ref": "v1.6.0",
        "release_url": "https://github.com/talevi83/SubBuildStudio/releases/tag/v1.6.0",
        "release_notes": None,
        "error": None,
    }

    with (
        patch.object(updater, "check_for_updates", return_value=release_status),
        patch.object(updater, "_apply_zip_update") as mock_zip,
        patch.object(updater.urllib.request, "urlopen") as mock_urlopen,
    ):
        res = updater.apply_update(root, auto_mode=False)

    assert updater.is_installer_managed(root) is True
    assert res["updated"] is False
    assert res["skipped_reason"] == "installer_managed"
    assert "installer" in res["message"]
    mock_zip.assert_not_called()
    mock_urlopen.assert_not_called()
    assert (root / "VERSION").read_text(encoding="utf-8").strip().endswith("1.5.0")


def test_a_source_copy_is_not_installer_managed(tmp_path: Path):
    root = _make_non_git_install(tmp_path, version="1.5.0")
    assert updater.is_installer_managed(root) is False
