"""Encryption at rest for saved API keys, tied to the signed-in Windows account (DPAPI).

The key vault (storage/provider_keys.json) is already readable by the current user only (see
private_files.py). DPAPI adds that a copy of the file - a backup, a synced folder, a disk image, the
portable folder carried to another PC - is useless: only this Windows account on this machine can
decrypt it. No password to remember and nothing to install (crypt32 ships with Windows).

Elsewhere (macOS / Linux) `available()` is False and the vault stays a 0600 JSON file.
"""
import ctypes
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Mixed into the encryption: another program of the same user that uses DPAPI without this exact
# value can't simply decrypt the vault with CryptUnprotectData.
_ENTROPY = b"SubBuild Studio provider key vault v1"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class SecretStoreError(Exception):
    """The vault could not be encrypted or decrypted (e.g. it was made by another Windows account)."""


if os.name == "nt":  # pragma: no cover - exercised on Windows only
    from ctypes import wintypes

    class _DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def _blob(data: bytes):
        buffer = ctypes.create_string_buffer(data, len(data))
        return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer

    def _crypt(func_name: str, data: bytes) -> bytes:
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        data_in, _keep_in = _blob(data)
        entropy, _keep_entropy = _blob(_ENTROPY)
        data_out = _DataBlob()
        if func_name == "protect":
            ok = crypt32.CryptProtectData(
                ctypes.byref(data_in), "SubBuild Studio", ctypes.byref(entropy), None, None,
                _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(data_out),
            )
        else:
            ok = crypt32.CryptUnprotectData(
                ctypes.byref(data_in), None, ctypes.byref(entropy), None, None,
                _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(data_out),
            )
        if not ok:
            raise SecretStoreError(f"Windows could not {func_name} the key vault (error {ctypes.GetLastError()})")
        try:
            return ctypes.string_at(data_out.pbData, data_out.cbData)
        finally:
            kernel32.LocalFree(data_out.pbData)


_available: Optional[bool] = None


def available() -> bool:
    """True when the vault can be encrypted on this machine (Windows with DPAPI)."""
    global _available
    if _available is None:
        if os.name != "nt":
            _available = False
        else:  # pragma: no cover - Windows only
            try:
                _available = unprotect(protect(b"probe")) == b"probe"
            except Exception as e:
                logger.warning(f"DPAPI is not usable; the key vault stays unencrypted: {e}")
                _available = False
    return _available


def protect(data: bytes) -> bytes:
    if os.name != "nt":
        raise SecretStoreError("Encryption at rest is only available on Windows")
    return _crypt("protect", data)  # pragma: no cover


def unprotect(data: bytes) -> bytes:
    if os.name != "nt":
        raise SecretStoreError("This key vault was encrypted on Windows and can only be read there")
    return _crypt("unprotect", data)  # pragma: no cover
