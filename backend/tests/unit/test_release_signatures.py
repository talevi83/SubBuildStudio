"""Release tags are checked against the installed release-signers file with real git + ssh-keygen."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.core import updater

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("ssh-keygen") is None, reason="needs git and ssh-keygen"
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env, check=False)


def _key(folder: Path, name: str) -> Path:
    key = folder / name
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", str(key)], check=True)
    return key


@pytest.fixture
def repo(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)  # the updater's own git calls ignore the user's config
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("SUBBUILD_REQUIRE_SIGNED_TAGS", raising=False)
    root = tmp_path / "app"
    root.mkdir()
    assert _git(root, "init", "-q").returncode == 0
    for k, v in (("user.name", "Tal"), ("user.email", "tal@example.com"), ("gpg.format", "ssh")):
        _git(root, "config", k, v)
    (root / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    _git(root, "add", ".")
    assert _git(root, "commit", "-q", "-m", "init").returncode == 0
    keys = tmp_path / "keys"
    keys.mkdir()
    return root, _key(keys, "release"), _key(keys, "stranger")


def _trust(root: Path, key: Path) -> None:
    public = key.with_suffix(".pub").read_text(encoding="utf-8").split()
    (root / updater.RELEASE_SIGNERS_FILE).write_text(
        f'# comment lines are fine\ntal@example.com namespaces="git" {public[0]} {public[1]}\n', encoding="utf-8")


def _signed_tag(root: Path, tag: str, key: Path) -> None:
    res = _git(root, "-c", f"user.signingkey={key}", "tag", "-s", tag, "-m", tag)
    assert res.returncode == 0, res.stderr


def test_a_tag_signed_with_the_release_key_is_accepted(repo):
    root, release_key, _ = repo
    _trust(root, release_key)
    _signed_tag(root, "v1.1.0", release_key)
    ok, detail = updater.verify_release_tag(root, "v1.1.0")
    assert ok, detail
    assert updater.require_signed_tags(root) is True  # on by itself once release-signers exists


def test_an_unsigned_tag_is_refused(repo):
    root, release_key, _ = repo
    _trust(root, release_key)
    _git(root, "tag", "-a", "v1.1.0", "-m", "made on the GitHub website")
    assert updater.verify_release_tag(root, "v1.1.0")[0] is False
    _git(root, "tag", "v1.1.1")  # lightweight, like the tags GitHub creates for a release
    assert updater.verify_release_tag(root, "v1.1.1")[0] is False


def test_a_tag_signed_with_any_other_key_is_refused(repo):
    root, release_key, stranger_key = repo
    _trust(root, release_key)
    _signed_tag(root, "v1.1.0", stranger_key)
    assert updater.verify_release_tag(root, "v1.1.0")[0] is False


def test_without_release_signers_nothing_is_required_unless_turned_on(repo, monkeypatch):
    root, _, _ = repo
    assert updater.require_signed_tags(root) is False
    monkeypatch.setenv("SUBBUILD_REQUIRE_SIGNED_TAGS", "true")
    assert updater.require_signed_tags(root) is True


def test_signatures_can_be_turned_off_explicitly(repo, monkeypatch):
    root, release_key, _ = repo
    _trust(root, release_key)
    monkeypatch.setenv("SUBBUILD_REQUIRE_SIGNED_TAGS", "false")
    assert updater.require_signed_tags(root) is False


def test_the_repository_trusts_the_maintainers_key():
    signers = Path(updater.get_repo_root()) / updater.RELEASE_SIGNERS_FILE
    lines = [ln for ln in signers.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    assert lines and all(' namespaces="git" ssh-' in ln for ln in lines)
