"""
Git & GitHub Release auto-updater for SubBuild Studio.

Supports both:
- Pre-launch CLI checks invoked by `run.bat` / `run.sh` (`python -m app.core.updater --auto --no-frontend-build`)
- Live status & update endpoints exposed via `GET/POST /api/system/update/*`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

GITHUB_REPO = "talevi83/SubBuildStudio"
GITHUB_RELEASE_BY_TAG_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{{tag}}"
GITHUB_TAGS_API = f"https://api.github.com/repos/{GITHUB_REPO}/tags?per_page=100"
GITHUB_COMPARE_API = f"https://api.github.com/repos/{GITHUB_REPO}/compare/{{base}}...{{head}}"
GITHUB_TAG_REF_API = f"https://api.github.com/repos/{GITHUB_REPO}/git/ref/tags/{{tag}}"
GITHUB_TAG_OBJ_API = f"https://api.github.com/repos/{GITHUB_REPO}/git/tags/{{sha}}"
GITHUB_RELEASE_ZIP_URL = f"https://api.github.com/repos/{GITHUB_REPO}/zipball/refs/tags/{{tag}}"
# Release tags are only trusted when they point at a commit that is already on one of these branches
# (an accidental or rogue tag on some other branch is never installed automatically).
RELEASE_BRANCHES = ("origin/main", "origin/master")
_SEMVER_RE = re.compile(r"^\s*v?(\d+)\.(\d+)\.(\d+)\s*$")
_DESCRIBE_COMMIT_RE = re.compile(r"^\s*v?\d+\.\d+\.\d+(?:-\d+-g([0-9a-fA-F]{7,40}))")
_PROTECTED_ZIP_PATHS = (
    ".git",
    ".env",
    "backend/.env",
    "backend/.session-token",
    "backend/storage",
    "backend/.venv",
    ".venv",
    "storage",
    "data",
    "frontend/node_modules",
    "node_modules",
    "frontend/dist",
    "webview",
    "window.json",
    "launcher.log",
)

# Written by the Windows installer build (packaging/windows/stage.ps1, "windows") and the macOS app
# build (packaging/macos/build-app.sh, "macos"). Such a copy has no Node.js to rebuild the interface
# and its Python packages may be in use (the macOS app is also signed and often read-only), so it is
# updated by installing the new download rather than by overlaying a release ZIP.
INSTALLER_MARKER = "INSTALLER"

_cached_update_status: Optional[Dict[str, Any]] = None


def parse_semver(tag_or_ver: str) -> Optional[Tuple[int, int, int]]:
    """Parse a release tag (`v1.5.0`) or bare version (`1.5.0`) into `(major, minor, patch)`."""
    if not tag_or_ver:
        return None
    m = _SEMVER_RE.match(tag_or_ver.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def get_repo_root() -> Path:
    """Return the root directory of the SubBuild Studio repository."""
    return Path(__file__).resolve().parents[3]


def _subprocess_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _run_cmd(
    args: list[str],
    cwd: Path,
    timeout: float = 30.0,
) -> Tuple[int, str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=_subprocess_flags(),
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout:.0f}s: {' '.join(args[:3])}"
    except Exception as exc:
        return 1, "", str(exc)


def _run_git(repo_root: Path, *args: str, timeout: float = 4.0) -> Tuple[int, str, str]:
    return _run_cmd(["git", "-C", str(repo_root), *args], cwd=repo_root, timeout=timeout)


def _file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_semver(ver: Tuple[int, int, int]) -> str:
    return f"{ver[0]}.{ver[1]}.{ver[2]}"


_LEADING_SEMVER_RE = re.compile(r"^\s*v?(\d+\.\d+\.\d+)")


def _version_from_file(root: Path) -> str:
    """The version written in `VERSION` ("2.1.0"), or "" when it holds none.

    The file is never edited by hand. It contains a `$Format:%(describe...)$` placeholder that Git
    fills in when GitHub builds a release's zip (export-subst in .gitattributes): "v2.1.0", or
    "v2.1.0-3-gabc1234" for a zip of a later commit. In a Git checkout the placeholder stays as it is
    and the version comes from the tags instead.
    """
    try:
        raw = (root / "VERSION").read_text(encoding="utf-8")
    except OSError:
        return ""
    match = _LEADING_SEMVER_RE.match(raw)
    return match.group(1) if match else ""


def _commit_from_version_file(root: Path) -> str:
    """Commit hash embedded in `VERSION` (e.g., `v2.1.0-3-gabc1234` -> `abc1234`), or ''."""
    try:
        raw = (root / "VERSION").read_text(encoding="utf-8")
    except OSError:
        return ""
    match = _DESCRIBE_COMMIT_RE.match(raw)
    return match.group(1) if match else ""


def read_version_file(repo_root: Optional[Path] = None) -> str:
    """The installed version without network access: from `VERSION` (zip downloads), else the
    newest tag of the Git checkout, else "0.0.0"."""
    root = repo_root or get_repo_root()
    version = _version_from_file(root)
    if version:
        return version
    if (root / ".git").exists():
        rc, out, _ = _run_git(root, "describe", "--tags", "--abbrev=0", timeout=2.0)
        if rc == 0 and parse_semver(out):
            return _format_semver(parse_semver(out))  # type: ignore[arg-type]
    return "0.0.0"


def get_current_version_info(repo_root: Optional[Path] = None) -> Dict[str, str]:
    """Read local `VERSION` and Git HEAD/tag metadata without network access."""
    root = repo_root or get_repo_root()

    file_ver_raw = _version_from_file(root)

    git_tag = ""
    short_sha = ""
    full_sha = ""
    branch = ""
    if (root / ".git").exists():
        rc_tag, out_tag, _ = _run_git(root, "describe", "--tags", "--abbrev=0", timeout=2.0)
        if rc_tag == 0 and out_tag:
            git_tag = out_tag

        rc_short, out_short, _ = _run_git(root, "rev-parse", "--short", "HEAD", timeout=2.0)
        if rc_short == 0 and out_short:
            short_sha = out_short

        rc_full, out_full, _ = _run_git(root, "rev-parse", "HEAD", timeout=2.0)
        if rc_full == 0 and out_full:
            full_sha = out_full

        rc_br, out_br, _ = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD", timeout=2.0)
        if rc_br == 0 and out_br:
            branch = out_br

    if not short_sha:
        version_commit = _commit_from_version_file(root)
        if version_commit:
            short_sha = version_commit[:7]
            full_sha = full_sha or version_commit

    file_tuple = parse_semver(file_ver_raw)
    tag_tuple = parse_semver(git_tag)
    if file_tuple and tag_tuple:
        current_version = _format_semver(max(file_tuple, tag_tuple))
    elif file_tuple:
        current_version = _format_semver(file_tuple)
    elif tag_tuple:
        current_version = _format_semver(tag_tuple)
    else:
        current_version = "0.0.0"

    return {
        "current_version": current_version,
        "current_commit": short_sha,
        "full_commit": full_sha,
        "branch": branch,
    }


def _github_get_json(url: str, timeout: float = 4.0) -> Any:
    """Perform a GitHub REST API GET request and return parsed JSON."""
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "SubBuild-Studio-Updater"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _fetch_github_tags(timeout: float = 4.0) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Fetch remote semver tags and their commit SHAs via the GitHub API."""
    try:
        data = _github_get_json(GITHUB_TAGS_API, timeout=timeout)
        if not isinstance(data, list):
            return None, "Unexpected response from GitHub tags API"
        tag_shas: Dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if parse_semver(name) is None:
                continue
            sha = str((item.get("commit") or {}).get("sha") or "")
            tag_shas[name] = sha
        return tag_shas, None
    except Exception as exc:
        return None, str(exc) or "Unable to contact GitHub API"


def _fetch_github_release_metadata(tag: str, timeout: float = 2.0) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort query for the release page URL & concise notes of `tag`; (None, None) on error."""
    try:
        data = _github_get_json(
            GITHUB_RELEASE_BY_TAG_API.format(tag=urllib.parse.quote(tag, safe="")),
            timeout=timeout,
        )
        html_url = data.get("html_url")
        body = (data.get("body") or "").strip()
        if len(body) > 500:
            body = body[:497].rstrip() + "..."
        return (html_url or None), (body or None)
    except Exception:
        return None, None


def get_cached_update_status() -> Optional[Dict[str, Any]]:
    return dict(_cached_update_status) if _cached_update_status is not None else None


def check_for_updates(repo_root: Optional[Path] = None, timeout: float = 4.0) -> Dict[str, Any]:
    """
    Compare local installation against published `origin` semantic Release tags (`vX.Y.Z`).
    Ignores raw branch commits. Never raises on network/offline errors; caches result for `/api/system/info`.
    """
    global _cached_update_status

    root = repo_root or get_repo_root()
    info = get_current_version_info(root)
    current_version = info["current_version"]
    current_commit = info["current_commit"]

    result: Dict[str, Any] = {
        "current_version": current_version,
        "current_commit": current_commit,
        "branch": info["branch"],
        "update_available": False,
        "update_kind": None,
        "latest_version": current_version,
        "latest_commit": current_commit,
        "commits_behind": 0,
        "target_ref": None,
        "release_url": None,
        "release_notes": None,
        "error": None,
    }

    if (root / ".git").exists():
        rc, out, err = _run_git(root, "ls-remote", "--tags", "origin", timeout=timeout)
        if rc != 0:
            result["error"] = err or "Unable to contact remote repository"
            logger.info("Updater check failed (v%s): %s", current_version, result["error"])
            _cached_update_status = dict(result)
            return result

        # Parse remote semver release tags (preferring dereferenced annotated ^{} SHAs)
        tag_shas: Dict[str, str] = {}
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sha, ref = parts[0], parts[1]
            if not ref.startswith("refs/tags/"):
                continue
            raw_tag = ref[len("refs/tags/") :]
            if raw_tag.endswith("^{}"):
                tag_name = raw_tag[:-3]
                if parse_semver(tag_name) is not None:
                    tag_shas[tag_name] = sha
            elif parse_semver(raw_tag) is not None:
                tag_shas.setdefault(raw_tag, sha)
    else:
        fetched_tags, err = _fetch_github_tags(timeout=timeout)
        if fetched_tags is None:
            result["error"] = err or "Unable to contact remote repository"
            logger.info("Updater check failed (v%s): %s", current_version, result["error"])
            _cached_update_status = dict(result)
            return result
        tag_shas = fetched_tags

    best_tag: Optional[str] = None
    best_tuple: Optional[Tuple[int, int, int]] = None
    best_tag_sha = ""
    for tag_name, sha in tag_shas.items():
        parsed = parse_semver(tag_name)
        if parsed and (best_tuple is None or parsed > best_tuple):
            best_tuple = parsed
            best_tag = tag_name
            best_tag_sha = sha

    current_tuple = parse_semver(current_version) or (0, 0, 0)
    if best_tuple is not None and best_tag is not None and best_tuple > current_tuple:
        latest_ver = _format_semver(best_tuple)
        release_url, release_notes = _fetch_github_release_metadata(best_tag, timeout=min(timeout, 2.0))
        if not release_url:
            release_url = f"https://github.com/{GITHUB_REPO}/releases/tag/{best_tag}"
        result.update(
            {
                "update_available": True,
                "update_kind": "release",
                "latest_version": latest_ver,
                "latest_commit": best_tag_sha[:7],
                "target_ref": best_tag,
                "release_url": release_url,
                "release_notes": release_notes,
            }
        )
        logger.info(
            "Updater check (v%s, branch=%s): newer release %s (%s) available",
            current_version,
            info["branch"] or "detached",
            best_tag,
            best_tag_sha[:7],
        )
    else:
        logger.info(
            "Updater check (v%s, branch=%s): up to date (latest remote tag: %s)",
            current_version,
            info["branch"] or "detached",
            best_tag or "none",
        )

    _cached_update_status = dict(result)
    return result


def _find_venv_python(root: Path) -> str:
    if sys.platform == "win32":
        win_py = root / "backend" / ".venv" / "Scripts" / "python.exe"
        if win_py.is_file():
            return str(win_py)
    else:
        nix_py = root / "backend" / ".venv" / "bin" / "python"
        if nix_py.is_file():
            return str(nix_py)
    return sys.executable


def _install_backend_requirements(root: Path) -> bool:
    req_file = root / "backend" / "requirements.txt"
    if not req_file.is_file():
        return False
    venv_py = _find_venv_python(root)
    uv_bin = shutil.which("uv")
    if uv_bin:
        rc, _, err = _run_cmd(
            [uv_bin, "pip", "install", "-r", str(req_file), "--python", venv_py],
            cwd=root,
            timeout=300.0,
        )
    else:
        rc, _, err = _run_cmd(
            [venv_py, "-m", "pip", "install", "-r", str(req_file)],
            cwd=root,
            timeout=300.0,
        )
    if rc != 0:
        logger.warning(f"Updater: dependency update failed: {err}")
        return False
    return True


def _frontend_needs_rebuild(root: Path, old_rev: str, new_rev: str, pkg_changed: bool) -> bool:
    dist_index = root / "frontend" / "dist" / "index.html"
    if not dist_index.is_file() or pkg_changed:
        return True
    if old_rev and new_rev and old_rev != new_rev:
        rc, changed, _ = _run_git(root, "diff", "--name-only", f"{old_rev}..{new_rev}", "--", "frontend/")
        if rc == 0 and changed.strip():
            return True
    return False


def _rebuild_frontend(root: Path, run_npm_install: bool) -> bool:
    fe_dir = root / "frontend"
    if not (fe_dir / "package.json").is_file():
        return False
    npm_cmd = shutil.which("npm.cmd" if sys.platform == "win32" else "npm") or "npm"
    npx_cmd = shutil.which("npx.cmd" if sys.platform == "win32" else "npx") or "npx"

    if run_npm_install or not (fe_dir / "node_modules").is_dir():
        rc_i, _, err_i = _run_cmd([npm_cmd, "install"], cwd=fe_dir, timeout=300.0)
        if rc_i != 0:
            logger.warning(f"Updater: npm install failed: {err_i}")
            return False

    rc_b, _, err_b = _run_cmd([npx_cmd, "vite", "build"], cwd=fe_dir, timeout=180.0)
    if rc_b != 0:
        logger.warning(f"Updater: frontend build failed: {err_b}")
        return False
    return True


_TRUE = {"1", "true", "yes", "on"}


_FALSE = {"0", "false", "no", "off"}
# The SSH keys allowed to sign releases (git's "allowed signers" format), at the root of the app.
# The copy already installed is the one a new release is checked against, so a release can't swap
# in its own key: a compromised GitHub account can push code and tags, but can't sign them.
RELEASE_SIGNERS_FILE = "release-signers"


def require_signed_tags(root: Path) -> bool:
    """Whether a release must carry a valid signature before it is installed.

    SUBBUILD_REQUIRE_SIGNED_TAGS (environment, then backend/.env - read directly: this module also
    runs as a launch-time CLI, before the app's settings exist) decides when it is set; otherwise
    signatures are required as soon as the installed copy has a release-signers file.
    """
    value = os.environ.get("SUBBUILD_REQUIRE_SIGNED_TAGS")
    if value is None:
        try:
            for line in (root / "backend" / ".env").read_text(encoding="utf-8", errors="replace").splitlines():
                key, sep, raw = line.partition("=")
                if sep and key.strip().upper() == "SUBBUILD_REQUIRE_SIGNED_TAGS":
                    value = raw.split("#", 1)[0].strip().strip("'\"")
        except OSError:
            value = None
    value = (value or "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return (root / RELEASE_SIGNERS_FILE).is_file()


def _verify_github_tag_on_main(tag: str, timeout: float = 10.0) -> Tuple[Optional[bool], str]:
    """Check via GitHub Compare API whether `tag` is an ancestor of `main` or `master`."""
    encoded_tag = urllib.parse.quote(tag, safe="")
    for branch in ("main", "master"):
        url = GITHUB_COMPARE_API.format(base=branch, head=encoded_tag)
        try:
            data = _github_get_json(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            logger.warning("Updater: compare check failed for %s (%s): %s", branch, tag, exc)
            continue
        except Exception as exc:
            logger.warning("Updater: compare check failed for %s (%s): %s", branch, tag, exc)
            continue
        if isinstance(data, dict):
            status = data.get("status")
            if status in ("behind", "identical"):
                return True, f"origin/{branch}"
            if status in ("ahead", "diverged"):
                return False, f"origin/{branch}"
    return None, ""


def _verify_ssh_signature_payload(signers_file: Path, payload: str, signature: str) -> Tuple[bool, str]:
    """Verify a Git SSH signature (`payload` + `signature`) against `signers_file` using `ssh-keygen`."""
    ssh_keygen = shutil.which("ssh-keygen")
    if not ssh_keygen:
        return False, "ssh-keygen not found on PATH"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sig_path = tmp_dir / "tag.sig"
        sig_path.write_bytes(signature.encode("utf-8"))
        rc, out, err = _run_cmd(
            [ssh_keygen, "-Y", "find-principals", "-f", str(signers_file), "-s", str(sig_path)],
            cwd=tmp_dir,
            timeout=10.0,
        )
        principals = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if rc != 0 or not principals:
            return False, (err or out or "No matching principal in release-signers").strip()
        principal = principals[0]
        try:
            proc = subprocess.run(
                [
                    ssh_keygen,
                    "-Y",
                    "verify",
                    "-f",
                    str(signers_file),
                    "-I",
                    principal,
                    "-n",
                    "git",
                    "-s",
                    str(sig_path),
                ],
                input=payload.encode("utf-8"),
                capture_output=True,
                timeout=10.0,
                creationflags=_subprocess_flags(),
            )
            detail = (
                proc.stderr.decode("utf-8", errors="replace")
                or proc.stdout.decode("utf-8", errors="replace")
            ).strip()
            return proc.returncode == 0, detail
        except Exception as exc:
            return False, str(exc)


def _verify_github_release_tag_signature(root: Path, tag: str, timeout: float = 15.0) -> Tuple[bool, str]:
    """Verify a release tag's Git signature via GitHub's Git Tags API (and local `release-signers`)."""
    try:
        ref_data = _github_get_json(
            GITHUB_TAG_REF_API.format(tag=urllib.parse.quote(tag, safe="")),
            timeout=timeout,
        )
        obj = (ref_data.get("object") or {}) if isinstance(ref_data, dict) else {}
        if obj.get("type") != "tag" or not obj.get("sha"):
            return False, "Tag is not a signed annotated tag"
        tag_data = _github_get_json(
            GITHUB_TAG_OBJ_API.format(sha=urllib.parse.quote(str(obj["sha"]), safe="")),
            timeout=timeout,
        )
    except Exception as exc:
        return False, str(exc) or "Failed to fetch tag signature from GitHub API"

    ver = (tag_data.get("verification") or {}) if isinstance(tag_data, dict) else {}
    sig = (ver.get("signature") or "").strip()
    payload = ver.get("payload") or ""
    if not sig or not payload:
        return False, "error: no signature found"
    if f"\ntag {tag}\n" not in payload:
        return False, "error: tag payload mismatch"

    signers = root / RELEASE_SIGNERS_FILE
    if signers.is_file():
        return _verify_ssh_signature_payload(signers, payload, sig)
    return bool(ver.get("verified")), str(ver.get("reason") or "")


def verify_release_tag(root: Path, tag: str) -> Tuple[bool, str]:
    """(signature ok, verification detail) for `tag`, checked against the installed release-signers file
    (or, without one, against this machine's own git signing configuration / GitHub verification)."""
    if not (root / ".git").exists():
        return _verify_github_release_tag_signature(root, tag)
    signers = root / RELEASE_SIGNERS_FILE
    args = ["verify-tag", tag]
    if signers.is_file():
        args = ["-c", f"gpg.ssh.allowedSignersFile={signers.resolve().as_posix()}", *args]
    rc, out, err = _run_git(root, *args, timeout=15.0)
    return rc == 0, (err or out).strip()


def is_installer_managed(root: Path) -> bool:
    """True for a copy installed by the Windows installer or the macOS app (see INSTALLER_MARKER)."""
    return (root / INSTALLER_MARKER).is_file()


def _release_branch(root: Path) -> Optional[str]:
    for ref in RELEASE_BRANCHES:
        rc, _, _ = _run_git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", timeout=4.0)
        if rc == 0:
            return ref
    return None


def _is_protected_rel_path(rel_posix: str) -> bool:
    clean = rel_posix.strip("/")
    return any(clean == p or clean.startswith(f"{p}/") for p in _PROTECTED_ZIP_PATHS)


def _download_and_extract_release_zip(
    tag: str, temp_dir: Path, timeout: float = 60.0
) -> Tuple[Optional[Path], Optional[str]]:
    """Download the GitHub release ZIP for `tag` and safely extract it into `temp_dir`."""
    zip_path = temp_dir / "release.zip"
    extract_dir = temp_dir / "extracted"
    url = GITHUB_RELEASE_ZIP_URL.format(tag=urllib.parse.quote(tag, safe=""))
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "SubBuild-Studio-Updater"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(zip_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)
        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_root_resolved = extract_dir.resolve()
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                target = (extract_dir / info.filename).resolve()
                if not target.is_relative_to(extract_root_resolved):
                    return None, f"Unsafe archive path: {info.filename}"
            zf.extractall(extract_dir)
    except Exception as exc:
        return None, str(exc) or "Failed to download release archive."

    top_entries = list(extract_dir.iterdir())
    top_dirs = [p for p in top_entries if p.is_dir()]
    if len(top_entries) == 1 and len(top_dirs) == 1:
        return top_dirs[0], None
    return extract_dir, None


def _restore_zip_backup(
    root: Path,
    backup_dir: Path,
    backed_up_files: list[Path],
    added_files: list[Path],
) -> None:
    """Restore backed-up files and remove newly added files after a failed ZIP update."""
    for rel in backed_up_files:
        src = backup_dir / rel
        dst = root / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for rel in added_files:
        dst = root / rel
        try:
            if dst.is_file():
                dst.unlink()
        except OSError:
            pass


def _finalize_update_success(
    root: Path,
    status: Dict[str, Any],
    prev_version: str,
    target_ref: str,
    req_updated: bool,
    fe_rebuilt: bool,
) -> Dict[str, Any]:
    global _cached_update_status

    after_info = get_current_version_info(root)
    _cached_update_status = {
        "current_version": after_info["current_version"],
        "current_commit": after_info["current_commit"],
        "branch": after_info["branch"],
        "update_available": False,
        "update_kind": None,
        "latest_version": after_info["current_version"],
        "latest_commit": after_info["current_commit"],
        "commits_behind": 0,
        "target_ref": None,
        "release_url": status.get("release_url"),
        "release_notes": status.get("release_notes"),
        "error": None,
    }

    logger.info(
        "Updater applied release %s: v%s -> v%s (%s), reqs=%s, frontend=%s",
        target_ref,
        prev_version,
        after_info["current_version"],
        after_info["current_commit"],
        req_updated,
        fe_rebuilt,
    )

    return {
        "updated": True,
        "previous_version": prev_version,
        "new_version": after_info["current_version"],
        "new_commit": after_info["current_commit"],
        "requirements_updated": req_updated,
        "frontend_rebuilt": fe_rebuilt,
        "restart_required": True,
        "skipped_reason": None,
        "message": f"Updated to v{after_info['current_version']} ({after_info['current_commit']}).",
    }


def _apply_zip_update(
    root: Path,
    status: Dict[str, Any],
    base_reply: Dict[str, Any],
    rebuild_frontend: bool,
) -> Dict[str, Any]:
    """Download and overlay a verified GitHub Release ZIP when `.git` is absent."""
    target_ref = str(status["target_ref"])
    prev_version = status["current_version"]

    on_main, release_branch = _verify_github_tag_on_main(target_ref)
    if on_main is None:
        reply = dict(base_reply)
        reply["skipped_reason"] = "no_release_branch"
        reply["message"] = "Skipped update: the remote main branch could not be found."
        logger.warning("Updater apply skipped (target=%s): no main or master branch on GitHub", target_ref)
        return reply
    if not on_main:
        reply = dict(base_reply)
        reply["skipped_reason"] = "tag_not_on_main"
        reply["message"] = f"Skipped update: release {target_ref} is not part of {release_branch}."
        logger.warning("Updater apply skipped: %s is not an ancestor of %s", target_ref, release_branch)
        return reply

    if require_signed_tags(root):
        signed, detail = verify_release_tag(root, target_ref)
        if not signed:
            reply = dict(base_reply)
            reply["skipped_reason"] = "unsigned_tag"
            reply["message"] = f"Skipped update: release {target_ref} has no valid signature."
            logger.warning("Updater apply skipped: signature check for %s failed: %s", target_ref, detail)
            return reply
        logger.info("Updater: release %s is signed by a trusted key (%s)", target_ref, detail[:200])

    req_file = root / "backend" / "requirements.txt"
    pkg_file = root / "frontend" / "package.json"
    req_before = _file_sha256(req_file)
    pkg_before = _file_sha256(pkg_file)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        extracted_root, err = _download_and_extract_release_zip(target_ref, tmp_dir)
        if extracted_root is None:
            reply = dict(base_reply)
            reply["message"] = err or "Failed to download release archive."
            logger.warning("Updater ZIP download failed for %s: %s", target_ref, reply["message"])
            return reply

        backup_dir = tmp_dir / "backup"
        backed_up_files: list[Path] = []
        added_files: list[Path] = []
        fe_changed = False

        for src_file in extracted_root.rglob("*"):
            if not src_file.is_file():
                continue
            rel = src_file.relative_to(extracted_root)
            rel_posix = rel.as_posix()
            if _is_protected_rel_path(rel_posix):
                continue
            dst_file = root / rel
            src_bytes = src_file.read_bytes()
            file_changed = (not dst_file.is_file()) or (dst_file.read_bytes() != src_bytes)
            if not file_changed:
                continue
            if rel_posix.startswith("frontend/"):
                fe_changed = True
            if dst_file.is_file():
                bak = backup_dir / rel
                bak.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst_file, bak)
                backed_up_files.append(rel)
            else:
                added_files.append(rel)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            # copyfile updates mtime to now so run.bat/run.sh rebuilds frontend/dist when changed
            shutil.copyfile(src_file, dst_file)
            shutil.copymode(src_file, dst_file)

        version_rel = Path("VERSION")
        version_file = root / version_rel
        if version_rel not in backed_up_files and version_rel not in added_files:
            if version_file.is_file():
                bak_ver = backup_dir / version_rel
                bak_ver.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(version_file, bak_ver)
                backed_up_files.append(version_rel)
            else:
                added_files.append(version_rel)
        latest_ver = status["latest_version"]
        latest_commit = status.get("latest_commit") or ""
        ver_text = f"v{latest_ver}-0-g{latest_commit}\n" if latest_commit else f"v{latest_ver}\n"
        version_file.write_text(ver_text, encoding="utf-8")

        req_after = _file_sha256(req_file)
        pkg_after = _file_sha256(pkg_file)
        req_changed = req_before != req_after and req_after is not None
        pkg_changed = pkg_before != pkg_after

        req_updated = False
        if req_changed:
            req_updated = _install_backend_requirements(root)
            if not req_updated:
                _restore_zip_backup(root, backup_dir, backed_up_files, added_files)
                _install_backend_requirements(root)
                reply = dict(base_reply)
                reply["skipped_reason"] = "dependency_install_failed"
                reply["message"] = (
                    "Update failed while installing the new Python packages; kept the previous version. "
                    "Details are in the server log."
                )
                logger.warning("Updater ZIP update rolled back (%s): dependency_install_failed", target_ref)
                return reply

        fe_rebuilt = False
        if rebuild_frontend and (
            not (root / "frontend" / "dist" / "index.html").is_file() or pkg_changed or fe_changed
        ):
            fe_rebuilt = _rebuild_frontend(root, run_npm_install=pkg_changed)
            if not fe_rebuilt and not (root / "frontend" / "dist" / "index.html").is_file():
                _restore_zip_backup(root, backup_dir, backed_up_files, added_files)
                if req_changed:
                    _install_backend_requirements(root)
                _rebuild_frontend(root, run_npm_install=pkg_changed)
                reply = dict(base_reply)
                reply["skipped_reason"] = "frontend_build_failed"
                reply["message"] = (
                    "Update failed while building the new interface; kept the previous version. "
                    "Details are in the server log."
                )
                logger.warning("Updater ZIP update rolled back (%s): frontend_build_failed", target_ref)
                return reply

    return _finalize_update_success(root, status, prev_version, target_ref, req_updated, fe_rebuilt)


def _roll_back(
    root: Path,
    old_commit: str,
    base_reply: Dict[str, Any],
    reason: str,
    reinstall_requirements: bool,
    rebuild_frontend_after: bool,
    pkg_changed: bool,
) -> Dict[str, Any]:
    """Return the working copy to `old_commit` after a failed update step (best effort)."""
    reply = dict(base_reply)
    reply["skipped_reason"] = reason
    what = "installing the new Python packages" if reason == "dependency_install_failed" else "building the new interface"
    if not old_commit:
        reply["message"] = f"Update failed while {what}, and the previous version is unknown. Reinstall SubBuild Studio."
        logger.error("Updater: %s and no previous commit to roll back to", reason)
        return reply
    rc, _, err = _run_git(root, "reset", "--keep", old_commit, timeout=30.0)
    if rc != 0:
        reply["message"] = (f"Update failed while {what}, and going back to the previous version failed too "
                            f"({err or 'git reset'}). Run 'git reset --keep {old_commit[:12]}' in the app folder.")
        logger.error("Updater: rollback to %s failed: %s", old_commit[:12], err)
        return reply
    if reinstall_requirements:
        _install_backend_requirements(root)
    if rebuild_frontend_after:
        _rebuild_frontend(root, run_npm_install=pkg_changed)
    reply["message"] = f"Update failed while {what}; kept the previous version. Details are in the server log."
    logger.warning("Updater: %s; rolled back to %s", reason, old_commit[:12])
    return reply


def apply_update(
    repo_root: Optional[Path] = None,
    auto_mode: bool = False,
    rebuild_frontend: bool = True,
) -> Dict[str, Any]:
    """
    Update the installation to the newer published Release tag (`vX.Y.Z`) via Git fast-forward
    (or GitHub Release ZIP download when `.git` is absent), then sync Python dependencies and
    rebuild `frontend/dist` when changed.
    """
    root = repo_root or get_repo_root()
    status = check_for_updates(root)
    prev_version = status["current_version"]
    prev_commit = status["current_commit"]

    base_reply: Dict[str, Any] = {
        "updated": False,
        "previous_version": prev_version,
        "new_version": prev_version,
        "new_commit": prev_commit,
        "requirements_updated": False,
        "frontend_rebuilt": False,
        "restart_required": False,
        "skipped_reason": None,
        "message": "",
    }

    if status.get("error"):
        base_reply["message"] = str(status["error"])
        return base_reply

    if not status.get("update_available") or not status.get("target_ref"):
        base_reply["message"] = "Already up to date."
        return base_reply

    if is_installer_managed(root):
        base_reply["skipped_reason"] = "installer_managed"
        if sys.platform == "darwin":
            base_reply["message"] = (
                "This is the macOS app: download the new SubBuild Studio .dmg and drag the app to "
                "Applications again. Projects and settings are kept."
            )
        else:
            base_reply["message"] = (
                "This copy was installed with the Windows installer: download the new installer "
                "(SubBuild-Studio-Setup.exe) and run it. Projects and settings are kept."
            )
        logger.info("Updater apply skipped (target=%s): installer_managed", status.get("target_ref"))
        return base_reply

    if not (root / ".git").exists():
        return _apply_zip_update(root, status, base_reply, rebuild_frontend=rebuild_frontend)

    branch = status.get("branch") or ""
    target_ref = str(status["target_ref"])

    if auto_mode and branch not in ("main", "master"):
        base_reply["skipped_reason"] = "feature_branch"
        base_reply["message"] = f"Skipped automatic release update on branch '{branch}'."
        logger.info("Updater apply skipped (branch=%s): %s", branch, base_reply["skipped_reason"])
        return base_reply

    # Protect modified tracked files in developer checkouts
    st_rc, st_out, _ = _run_git(root, "status", "--porcelain", "--untracked-files=no", timeout=4.0)
    if st_rc == 0 and st_out.strip():
        base_reply["skipped_reason"] = "dirty_worktree"
        base_reply["message"] = "Skipped update because tracked files have local uncommitted changes."
        logger.info("Updater apply skipped (target=%s): dirty_worktree", target_ref)
        return base_reply

    req_file = root / "backend" / "requirements.txt"
    pkg_file = root / "frontend" / "package.json"
    req_before = _file_sha256(req_file)
    pkg_before = _file_sha256(pkg_file)
    old_full_commit = get_current_version_info(root)["full_commit"]

    # Fetch remote tags cleanly
    fetch_rc, _, fetch_err = _run_git(root, "fetch", "--tags", "--quiet", "origin", timeout=30.0)
    if fetch_rc != 0:
        base_reply["message"] = fetch_err or "Failed to fetch updates from origin."
        logger.warning("Updater fetch failed for %s: %s", target_ref, base_reply["message"])
        return base_reply

    release_branch = _release_branch(root)
    if release_branch is None:
        base_reply["skipped_reason"] = "no_release_branch"
        base_reply["message"] = "Skipped update: the remote main branch could not be found."
        logger.warning("Updater apply skipped (target=%s): no origin/main or origin/master", target_ref)
        return base_reply
    anc_rc, _, _ = _run_git(root, "merge-base", "--is-ancestor", target_ref, release_branch, timeout=15.0)
    if anc_rc != 0:
        base_reply["skipped_reason"] = "tag_not_on_main"
        base_reply["message"] = f"Skipped update: release {target_ref} is not part of {release_branch}."
        logger.warning("Updater apply skipped: %s is not an ancestor of %s", target_ref, release_branch)
        return base_reply
    if require_signed_tags(root):
        signed, detail = verify_release_tag(root, target_ref)
        if not signed:
            base_reply["skipped_reason"] = "unsigned_tag"
            base_reply["message"] = f"Skipped update: release {target_ref} has no valid signature."
            logger.warning("Updater apply skipped: git verify-tag %s failed: %s", target_ref, detail)
            return base_reply
        logger.info("Updater: release %s is signed by a trusted key (%s)", target_ref, detail[:200])

    pull_rc, pull_out, pull_err = _run_git(root, "merge", "--ff-only", target_ref, timeout=30.0)

    if pull_rc != 0:
        base_reply["skipped_reason"] = "non_fast_forward"
        base_reply["message"] = pull_err or pull_out or "Fast-forward update failed."
        logger.warning("Updater fast-forward merge failed for %s: %s", target_ref, base_reply["message"])
        return base_reply

    req_after = _file_sha256(req_file)
    pkg_after = _file_sha256(pkg_file)
    after_info = get_current_version_info(root)
    new_full_commit = after_info["full_commit"]

    req_changed = req_before != req_after and req_after is not None
    pkg_changed = pkg_before != pkg_after
    req_updated = False
    if req_changed:
        req_updated = _install_backend_requirements(root)
        if not req_updated:
            # New code without its dependencies would not start: go back to the version that worked
            return _roll_back(root, old_full_commit, base_reply, "dependency_install_failed",
                              reinstall_requirements=True, rebuild_frontend_after=False, pkg_changed=False)

    fe_rebuilt = False
    if rebuild_frontend and _frontend_needs_rebuild(root, old_full_commit, new_full_commit, pkg_changed):
        fe_rebuilt = _rebuild_frontend(root, run_npm_install=pkg_changed)
        if not fe_rebuilt and not (root / "frontend" / "dist" / "index.html").is_file():
            # vite empties frontend/dist before building: a build that failed midway leaves no UI at
            # all. (When it failed before starting - e.g. npm missing - the previous UI still works.)
            return _roll_back(root, old_full_commit, base_reply, "frontend_build_failed",
                              reinstall_requirements=req_changed, rebuild_frontend_after=True,
                              pkg_changed=pkg_changed)

    return _finalize_update_success(root, status, prev_version, target_ref, req_updated, fe_rebuilt)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SubBuild Studio Git Auto-Updater")
    parser.add_argument("--auto", action="store_true", help="Check and apply fast-forward updates automatically")
    parser.add_argument("--check", action="store_true", help="Check only (do not apply)")
    parser.add_argument(
        "--no-frontend-build",
        action="store_true",
        help="Skip rebuilding frontend/dist (launcher script builds it on timestamp delta)",
    )
    args = parser.parse_args(argv)

    try:
        info = get_current_version_info()
        print(f"      Current version: v{info['current_version']} ({info['current_commit'] or 'local'}, branch: {info['branch'] or 'detached'})")
        status = check_for_updates()
        if status.get("error"):
            print(f"      Update check skipped ({status['error']}).")
            return 0

        if not status.get("update_available"):
            print("      SubBuild Studio is up to date.")
            return 0

        print(f"      Update available: v{status['latest_version']} [{status['update_kind']}]")
        if args.check:
            return 0

        if args.auto:
            res = apply_update(auto_mode=True, rebuild_frontend=not args.no_frontend_build)
            print(f"      {res['message']}")
    except Exception as exc:
        print(f"      Update check failed safely ({exc}); continuing launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
