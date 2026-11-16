"""Input restrictions for every ffmpeg / ffprobe call on a file the user brought in.

Videos usually come from the internet, and ffmpeg picks the demuxer from the file's contents,
not its extension: a file named ``episode.mp4`` can really be an HLS playlist that makes ffmpeg
open network addresses (telling the sender the user's IP and that the file was opened), or a
rarely used container whose parser has a memory-safety bug. Two input options close that off:

* ``-protocol_whitelist file``: local files only - no http(s), tcp, concat, data, subfile...
* ``-format_whitelist``: only the containers SubBuild actually supports (the extensions in
  ``routes_media.ALLOWED_MEDIA_EXTENSIONS``, plus the subtitle formats used as a second input).

Both go right before the ``-i`` they apply to (ffprobe: before the path).
"""
from typing import List

# Demuxer names as ffmpeg reports them. One ffmpeg demuxer can carry several names
# ("mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm"); listing any one of them allows it.
ALLOWED_DEMUXERS = ",".join([
    "matroska", "webm",              # .mkv .webm
    "mov", "mp4", "m4a", "3gp",      # .mp4 .m4v .mov .m4a .3gp
    "avi",                           # .avi .divx
    "mpegts",                        # .ts .m2ts .mts
    "mpeg", "mpegvideo",             # .mpg .mpeg .vob (program stream / raw MPEG video)
    "asf",                           # .wmv .wma .asf
    "flv",                           # .flv
    "ogg",                           # .ogv .ogg .opus
    "wav", "w64",                    # .wav
    "mp3", "aac", "flac", "ac3", "eac3",
    "srt", "ass", "webvtt",          # the subtitle file added as a second input when embedding
])

INPUT_RESTRICTIONS: List[str] = [
    "-protocol_whitelist", "file",
    "-format_whitelist", ALLOWED_DEMUXERS,
]


def safe_input(path: str) -> List[str]:
    """``[...restrictions, "-i", path]`` for an ffmpeg command line."""
    return [*INPUT_RESTRICTIONS, "-i", str(path)]


def safe_probe_input(path: str) -> List[str]:
    """``[...restrictions, path]`` for an ffprobe command line."""
    return [*INPUT_RESTRICTIONS, str(path)]
