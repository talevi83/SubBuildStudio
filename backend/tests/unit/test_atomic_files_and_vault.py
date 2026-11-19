"""Secrets and glossaries are replaced in one step; a damaged key vault never silently loses the keys."""
import json
import os
import stat

import pytest

import app.core.private_files as pf
import app.core.provider_keys as pk


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "provider_keys.json"
    pf.write_private_text(path, "good")

    def boom(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(pf.os, "fsync", boom)
    with pytest.raises(OSError):
        pf.write_private_text(path, "half-written")
    assert path.read_text(encoding="utf-8") == "good"
    assert [p.name for p in tmp_path.iterdir()] == ["provider_keys.json"]  # temp file removed


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_new_file_is_owner_only_from_the_first_byte(tmp_path, monkeypatch):
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(stat.S_IMODE(os.stat(src).st_mode))
        real_replace(src, dst)

    monkeypatch.setattr(pf.os, "replace", spy)
    pf.write_private_text(tmp_path / ".env", "GEMINI_API_KEY=x")
    assert seen == [0o600]


def test_write_text_atomic(tmp_path):
    target = tmp_path / "sub" / "profile.json"
    pf.write_text_atomic(target, "{}")
    pf.write_text_atomic(target, '{"a": 1}')
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in target.parent.iterdir()] == ["profile.json"]


@pytest.fixture
def vault(tmp_path, monkeypatch):
    path = tmp_path / "provider_keys.json"
    monkeypatch.setattr(pk, "_vault_path", lambda: path)
    return path


def _rows(n):
    return [{"id": f"k{i}", "provider": "openai", "api_key": f"sk-{i}"} for i in range(n)]


def test_a_damaged_vault_falls_back_to_the_last_good_copy(vault):
    pk._write_raw(_rows(1))
    pk._write_raw(_rows(2))  # the 1-key version is now the backup
    vault.write_text('{"version": 1, "keys": [', encoding="utf-8")  # cut off mid-write
    assert [r["id"] for r in pk._read_raw()] == ["k0"]


def test_saving_over_a_damaged_vault_keeps_it_for_recovery(vault):
    vault.write_text("not json", encoding="utf-8")
    pk._write_raw(_rows(1))
    corrupt = [p for p in vault.parent.iterdir() if p.name.startswith("provider_keys.corrupt-")]
    assert len(corrupt) == 1 and corrupt[0].read_text(encoding="utf-8") == "not json"
    assert [r["id"] for r in pk._read_raw()] == ["k0"]
