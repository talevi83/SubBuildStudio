"""The key vault is encrypted at rest when the OS offers it (Windows DPAPI), and migrates transparently."""
import json

import pytest

import app.core.provider_keys as pk
import app.core.secret_store as ss


@pytest.fixture
def vault(tmp_path, monkeypatch):
    path = tmp_path / "provider_keys.json"
    monkeypatch.setattr(pk, "_vault_path", lambda: path)
    return path


@pytest.fixture
def fake_dpapi(monkeypatch):
    """A reversible stand-in for DPAPI (the real one only exists on Windows)."""
    monkeypatch.setattr(ss, "available", lambda: True)
    monkeypatch.setattr(ss, "protect", lambda data: b"SEALED:" + data[::-1])

    def unprotect(data):
        if not data.startswith(b"SEALED:"):
            raise ss.SecretStoreError("not ours")
        return data[len(b"SEALED:"):][::-1]

    monkeypatch.setattr(ss, "unprotect", unprotect)


ROWS = [{"id": "k1", "provider": "openai", "api_key": "sk-secret-123", "label": "Work"}]


def test_keys_are_not_stored_in_plain_text(vault, fake_dpapi):
    pk._write_raw(ROWS)
    text = vault.read_text(encoding="utf-8")
    assert "sk-secret-123" not in text
    assert json.loads(text)["protection"] == "dpapi"
    assert pk._read_raw() == ROWS


def test_a_plain_text_vault_is_encrypted_on_startup_backup_included(vault, fake_dpapi, monkeypatch):
    monkeypatch.setattr(ss, "available", lambda: False)
    pk._write_raw(ROWS)
    pk._write_raw(ROWS)  # leaves a plain-text .bak
    monkeypatch.setattr(ss, "available", lambda: True)
    assert pk.encrypt_vault_at_rest() is True
    for path in (vault, pk._backup_path()):
        assert "sk-secret-123" not in path.read_text(encoding="utf-8")
    assert pk._read_raw() == ROWS
    assert pk.encrypt_vault_at_rest() is False  # already done


def test_a_vault_from_another_account_is_reported_not_silently_used(vault, fake_dpapi, caplog):
    vault.write_text(json.dumps({"version": 2, "protection": "dpapi", "data": "bm90IG91cnM="}), encoding="utf-8")
    assert pk._read_raw() == []
    assert "could not be decrypted" in caplog.text


def test_without_dpapi_the_vault_stays_readable_json(vault, monkeypatch):
    monkeypatch.setattr(ss, "available", lambda: False)
    pk._write_raw(ROWS)
    assert json.loads(vault.read_text(encoding="utf-8"))["keys"] == ROWS
    assert pk.encrypt_vault_at_rest() is False


def test_secret_store_refuses_cleanly_off_windows():
    if ss.os.name == "nt":
        pytest.skip("Windows has DPAPI")
    assert ss.available() is False
    with pytest.raises(ss.SecretStoreError):
        ss.unprotect(b"x")
