"""Files only the current user may read: the session token and the .env with the Gemini API key.

The session token is meant to keep other Windows accounts on the same PC out of the API (see
app/core/auth.py). That only holds if they can't simply read the token file: in the portable
layout, or with the app on a second drive (D:\\, I:\\ ...), the folder usually inherits ACLs that
let every signed-in user read it. Writing such a file:

* Windows: the inherited permissions are removed and only the current user gets access (icacls);
* elsewhere: mode 0600.

The file is created empty and locked down *before* the secret is written into it.
Best effort: a failure is logged, never raised (the app must still start).
"""
import functools
import logging
import os
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


def write_private_text(path: Path, text: str) -> None:
    """Write `text` to `path` so that only the current user can read it (see the module doc)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if os.name == "nt":
            path.touch()
        else:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600))
    restrict_to_owner(path)
    path.write_text(text, encoding="utf-8")
