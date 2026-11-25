"""Transcription provider registry and factory helpers."""
import logging
from typing import Any, Optional, Tuple

from app.providers.transcription.base import BaseTranscriber
from app.providers.transcription.whisperx_provider import (
    WhisperXProvider,
    WhisperXUnavailableError,
    is_whisperx_available,
)

logger = logging.getLogger(__name__)

TRANSCRIPTION_PROVIDERS = ("faster_whisper", "whisperx")


def create_transcriber(
    provider: str = "faster_whisper",
    *,
    allow_fallback: bool = True,
    **kwargs: Any,
) -> Tuple[BaseTranscriber, str, Optional[str]]:
    """Instantiate a transcription provider by name with optional fallback when WhisperX is absent.

    Returns ``(transcriber_instance, active_provider_name, optional_warning_message)``.
    """
    normalized = (provider or "faster_whisper").strip().lower()
    if normalized not in TRANSCRIPTION_PROVIDERS:
        raise ValueError(f"Unknown transcription provider: {provider!r}")

    fw_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in {
            "model_size_or_path",
            "device",
            "compute_type",
            "vad_filter",
            "batch_size",
            "comms_boost",
            "context_notes",
            "high_speech_sensitivity",
            "initial_prompt",
            "condition_on_previous_text",
            "file_name",
        }
    }

    if normalized == "whisperx":
        if is_whisperx_available():
            return WhisperXProvider(**kwargs), "whisperx", None
        if not allow_fallback:
            raise WhisperXUnavailableError(
                "WhisperX is not installed in the backend environment."
            )
        from app.providers.transcription.faster_whisper import FasterWhisperTranscriber

        warning = (
            "WhisperX is not installed in the backend environment; "
            "automatically fell back to Faster-Whisper."
        )
        logger.warning(warning)
        return FasterWhisperTranscriber(**fw_kwargs), "faster_whisper", warning

    from app.providers.transcription.faster_whisper import FasterWhisperTranscriber

    return FasterWhisperTranscriber(**fw_kwargs), "faster_whisper", None


__all__ = [
    "BaseTranscriber",
    "TRANSCRIPTION_PROVIDERS",
    "WhisperXProvider",
    "WhisperXUnavailableError",
    "create_transcriber",
    "is_whisperx_available",
]
