"""Every ffmpeg / ffprobe input is limited to local files and supported containers."""
import ast
import shutil
from pathlib import Path

import pytest

from app.core.ffmpeg_safety import ALLOWED_DEMUXERS, INPUT_RESTRICTIONS, safe_input, safe_probe_input

APP_DIR = Path(__file__).resolve().parents[1] / "app"
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffmpeg not installed")


def test_safe_input_puts_restrictions_right_before_the_input():
    assert safe_input("/v/movie.mkv") == [*INPUT_RESTRICTIONS, "-i", "/v/movie.mkv"]
    assert safe_probe_input("/v/movie.mkv") == [*INPUT_RESTRICTIONS, "/v/movie.mkv"]
    assert INPUT_RESTRICTIONS[:2] == ["-protocol_whitelist", "file"]
    for name in ("hls", "concat", "dash", "image2", "subfile", "http"):
        assert name not in ALLOWED_DEMUXERS.split(",")


def test_no_ffmpeg_input_is_built_without_the_restrictions():
    """A bare "-i" anywhere in the app means a new ffmpeg call skipped safe_input()."""
    offenders = []
    for path in APP_DIR.rglob("*.py"):
        if path.name == "ffmpeg_safety.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value == "-i":
                offenders.append(f"{path.relative_to(APP_DIR)}:{node.lineno}")
    assert offenders == []


def test_embed_commands_restrict_both_inputs():
    from app.core import video_embed as ve

    probe = {"streams": [{"index": 0, "codec_type": "video"}, {"index": 1, "codec_type": "audio", "codec_name": "aac"}]}
    soft = ve.build_soft_command("/v/movie.mkv", "/tmp/subs.srt", "/v/out.mkv", probe, ve.EmbedOptions())
    burn = ve.build_burn_command("/v/movie.mkv", "subs.srt", "/v/out.mp4", probe, ve.EmbedOptions(mode="burn"), "libx264")
    for cmd in (soft, burn):
        for i, arg in enumerate(cmd):
            if arg == "-i":
                assert cmd[i - 4:i] == INPUT_RESTRICTIONS


@needs_ffmpeg
@pytest.mark.asyncio
@pytest.mark.parametrize("name,content", [
    ("playlist.mp4", "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttp://127.0.0.1:9/x.ts\n#EXT-X-ENDLIST\n"),
    ("script.mkv", "ffconcat version 1.0\nfile /etc/hostname\n"),
])
async def test_disguised_playlists_and_scripts_are_refused(tmp_path, name, content):
    from app.pipeline.stages.probe_stage import probe_media_file

    fake = tmp_path / name
    fake.write_text(content)
    with pytest.raises(Exception):
        await probe_media_file(str(fake))


@needs_ffmpeg
@pytest.mark.asyncio
async def test_real_media_still_probes(tmp_path):
    import subprocess

    from app.pipeline.stages.probe_stage import probe_media_file

    clip = tmp_path / "clip.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=1:s=160x120", "-f", "lavfi",
                    "-i", "sine=d=1", "-shortest", "-c:v", "mpeg4", "-c:a", "aac", "-y", str(clip)], check=True)
    res = await probe_media_file(str(clip))
    assert res.duration and res.duration > 0.5
