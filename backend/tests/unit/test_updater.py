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


def test_check_for_updates_falls_back_to_origin_main_commits(tmp_path: Path):
    repo = _make_fake_repo(tmp_path, version="1.5.0")
    local_sha = "1111111" + "0" * 33
    remote_sha = "3333333" + "0" * 33

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
            return 0, f"{remote_sha}\trefs/heads/main\n{local_sha}\trefs/tags/v1.5.0\n", ""
        if cmd.startswith("fetch"):
            return 0, "", ""
        if cmd.startswith("rev-list --count"):
            return 0, "3", ""
        return 1, "", ""

    with patch.object(updater, "_run_git", side_effect=fake_run_git):
        res = updater.check_for_updates(repo)

    assert res["update_available"] is True
    assert res["update_kind"] == "commit"
    assert res["commits_behind"] == 3
    assert res["latest_version"] == "1.5.0 (+3)"
    assert res["target_ref"] == "origin/main"


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

    commit_status = {
        "current_version": "1.5.0",
        "current_commit": "1111111",
        "branch": "feat/custom-work",
        "update_available": True,
        "update_kind": "commit",
        "latest_version": "1.5.0 (+2)",
        "latest_commit": "4444444",
        "commits_behind": 2,
        "target_ref": "origin/main",
        "release_url": None,
        "release_notes": None,
        "error": None,
    }
    with patch.object(updater, "check_for_updates", return_value=commit_status):
        res = updater.apply_update(repo, auto_mode=True)
    assert res["updated"] is False
    assert res["skipped_reason"] == "feature_branch"

    main_status = {**commit_status, "branch": "main"}
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
