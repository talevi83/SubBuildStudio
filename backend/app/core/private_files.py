"""Files only the current user may read: the session token and the .env with the Gemini API key.

The session token is meant to keep other Windows accounts on the same PC out of the API (see
app/core/auth.py). That only holds if they can't simply read the token file: in the portable
layout, or with the app on a second drive (D:\\, I:\\ ...), the folder usually inherits ACLs that
let every signed-in user read it. Writing such a file:

* Windows: the inherited permissions are removed and only the current user gets access (icacls);
* elsewhere: mode 0600.

The secret goes into a new temporary file next to the target that is locked down *before* anything
is written into it; the finished file then replaces the target in one step (os.replace), so a crash
or a full disk mid-write never leaves a half-written vault or .env behind.
Locking down is best effort: a failure is logged, never raised (the app must still start).
"""
import functools
import logging
import os
import secrets
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _current_user_sid() -> Optional[str]:
    try:
        out = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    # "desktop-1\\tal","S-1-5-21-..."
    parts = [p.strip().strip('"') for p in out.split(",")]
    sid = parts[-1] if parts else ""
    return sid if sid.startswith("S-1-") else None


def _windows_account() -> Optional[str]:
    sid = _current_user_sid()
    if sid:
        return f"*{sid}"
    user = os.environ.get("USERNAME")
    domain = os.environ.get("USERDOMAIN")
    if not user:
        return None
    return f"{domain}\\{user}" if domain else user


def restrict_to_owner(path: Path) -> bool:
    """Leave only the current user with access to `path`. True when that worked."""
    try:
        if os.name == "nt":
            account = _windows_account()
            if not account:
                return False
            res = subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{account}:F"],
                capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if res.returncode != 0:
                logger.warning(f"Could not restrict access to {path.name}: {(res.stderr or res.stdout).strip()}")
                return False
            return True
        os.chmod(path, 0o600)
        return True
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"Could not restrict access to {path.name}: {e}")
        return False


def _temp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")


def _replace_with(path: Path, tmp: Path, text: str) -> None:
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # the file keeps the temp file's (locked-down) permissions
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_text_atomic(path: Path, text: str) -> None:
    """Replace `path` with `text` in one step: readers see the old or the new file, never half of one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_with(path, _temp_sibling(path), text)


def write_private_text(path: Path, text: str) -> None:
    """Write `text` to `path` so that only the current user can read it (see the module doc)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_sibling(path)
    if os.name == "nt":
        tmp.touch()
    else:
        os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    restrict_to_owner(tmp)
    _replace_with(path, tmp, text)
