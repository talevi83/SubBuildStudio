"""Find helper programs (ffmpeg, ffprobe) without picking them up from the current folder.

On Windows both ``shutil.which`` (Python < 3.12) and ``CreateProcess`` with a bare program name
look in the current directory first, so an ``ffmpeg.exe`` dropped next to a video or into the
folder the app was started from would run instead of the real one. Here:

* only absolute PATH entries are searched (an empty or relative entry means "the current folder");
* ``NoDefaultCurrentDirectoryInExePath`` is set for this process, which makes ``CreateProcess``
  (and ``shutil.which`` on 3.12+) skip the current folder when a bare name is run anyway.
"""
import os
from typing import Optional

if os.name == "nt":
    os.environ.setdefault("NoDefaultCurrentDirectoryInExePath", "1")

_WINDOWS_EXTS = (".exe", ".com")


def find_executable(name: str, path: Optional[str] = None) -> Optional[str]:
    """Absolute path of `name` on PATH, or None. Never resolves to the current folder implicitly."""
    search = os.environ.get("PATH", "") if path is None else path
    if os.name == "nt":
        lower = name.lower()
        candidates = [name] if lower.endswith(_WINDOWS_EXTS) else [name + ext for ext in _WINDOWS_EXTS]
    else:
        candidates = [name]
    for entry in search.split(os.pathsep):
        entry = entry.strip().strip('"')
        if not entry or not os.path.isabs(entry):
            continue
        for candidate in candidates:
            full = os.path.join(entry, candidate)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                return full
    return None


def executable(name: str) -> str:
    """`find_executable` or the bare name (then the launch fails with a clear 'not found')."""
    return find_executable(name) or name
