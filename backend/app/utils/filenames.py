"""Safe file names from text the client sends (upload names, display names, profile names).

Windows maps the device names CON, PRN, AUX, NUL, COM1-9 and LPT1-9 to devices even with an
extension ("NUL.json", "con.he.srt"): writing there silently discards the data, and reading CON
waits for console input. Such a name gets a trailing underscore.
"""
import re
from typing import Optional

UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])$", re.IGNORECASE)


def is_reserved_device_name(name: str) -> bool:
    """True for "CON", "nul.txt", "COM1.tar.gz"... (what precedes the first dot, trailing spaces ignored)."""
    base = (name or "").split(".", 1)[0].rstrip(" ")
    return bool(_RESERVED.match(base))


def defuse_reserved(name: str) -> str:
    """ "NUL.json" -> "NUL_.json"; other names are returned unchanged."""
    if not is_reserved_device_name(name):
        return name
    head, dot, tail = name.partition(".")
    return f"{head.rstrip(' ')}_{dot}{tail}"


def safe_basename(raw: Optional[str], fallback: str = "file", max_len: int = 200) -> str:
    """A client-supplied name reduced to one safe file name: no folders, no reserved characters,
    no leading/trailing dots or spaces, no Windows device names."""
    name = (raw or "").replace("\\", "/").split("/")[-1]
    name = UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    name = defuse_reserved(name[:max_len])
    return name or fallback
