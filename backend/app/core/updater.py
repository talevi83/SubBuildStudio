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
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

GITHUB_REPO = "talevi83/SubBuildStudio"
GITHUB_RELEASE_BY_TAG_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{{tag}}"
# Release tags are only trusted when they point at a commit that is already on one of these branches
# (an accidental or rogue tag on some other branch is never installed automatically).
RELEASE_BRANCHES = ("origin/main", "origin/master")
_SEMVER_RE = re.compile(r"^\s*v?(\d+)\.(\d+)\.(\d+)\s*$")

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


def read_version_file(repo_root: Optional[Path] = None) -> str:
    """The version in the `VERSION` file ("2.0.1"), or "0.0.0" when it is missing (no Git calls)."""
    try:
        raw = ((repo_root or get_repo_root()) / "VERSION").read_text(encoding="utf-8").strip().lstrip("vV")
    except OSError:
        return "0.0.0"
    return raw if parse_semver(raw) else "0.0.0"


def get_current_version_info(repo_root: Optional[Path] = None) -> Dict[str, str]:
    """Read local `VERSION` and Git HEAD/tag metadata without network access."""
    root = repo_root or get_repo_root()

    file_ver_raw = ""
    version_file = root / "VERSION"
    if version_file.is_file():
        try:
            file_ver_raw = version_file.read_text(encoding="utf-8").strip().lstrip("vV")
        except OSError:
            file_ver_raw = ""

    git_tag = ""
    rc_tag, out_tag, _ = _run_git(root, "describe", "--tags", "--abbrev=0", timeout=2.0)
    if rc_tag == 0 and out_tag:
        git_tag = out_tag

    _, short_sha, _ = _run_git(root, "rev-parse", "--short", "HEAD", timeout=2.0)
    _, full_sha, _ = _run_git(root, "rev-parse", "HEAD", timeout=2.0)
    rc_br, branch, _ = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD", timeout=2.0)
    if rc_br != 0:
        branch = ""

    file_tuple = parse_semver(file_ver_raw)
    tag_tuple = parse_semver(git_tag)
    if file_tuple and tag_tuple:
        current_version = _format_semver(max(file_tuple, tag_tuple))
    elif file_tuple:
        current_version = _format_semver(file_tuple)
    elif tag_tuple:
        current_version = _format_semver(tag_tuple)
    else:
        current_version = file_ver_raw or git_tag.lstrip("vV") or "0.0.0"

    return {
        "current_version": current_version,
        "current_commit": short_sha,
        "full_commit": full_sha,
        "branch": branch,
    }


def _fetch_github_release_metadata(tag: str, timeout: float = 2.0) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort query for the release page URL & concise notes of `tag`; (None, None) on error."""
    try:
        req = urllib.request.Request(
            GITHUB_RELEASE_BY_TAG_API.format(tag=urllib.parse.quote(tag, safe="")),
            headers={"Accept": "application/vnd.github+json", "User-Agent": "SubBuild-Studio-Updater"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
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

    if not (root / ".git").exists():
        result["error"] = "Not a Git checkout"
        logger.info("Updater check skipped: %s", result["error"])
        _cached_update_status = dict(result)
        return result

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


def _release_branch(root: Path) -> Optional[str]:
    for ref in RELEASE_BRANCHES:
        rc, _, _ = _run_git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", timeout=4.0)
        if rc == 0:
            return ref
    return None


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
    Fast-forward the Git working copy to the newer published Release tag (`vX.Y.Z`),
    then sync Python dependencies and rebuild `frontend/dist` when changed.
    """
    global _cached_update_status

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
