"""Reusable fake translation and transcription providers for deterministic tests."""
import asyncio
from typing import List

from app.pipeline.context import PipelineCancelled, RawSegment
from app.providers.transcription.base import BaseTranscriber
from app.providers.translation.base import BaseTranslator


class FakeTranslator(BaseTranslator):
    """Deterministic in-memory translator with configurable failing indices and call recording."""

    def __init__(self, fail_indices=(), prefix: str = "תרגום", suffix: str = "?"):
        self.fail_indices = set(fail_indices)
        self.prefix = prefix
        self.suffix = suffix
        self.calls: List[List[int]] = []
        self.notes: List[str] = []

    async def build_glossary(
        self, segments, source_lang, target_lang, context_notes="", known_glossary=None
    ):
        return {"summary": "test", "characters": [], "terms": []}

    async def translate_segments(
        self,
        segments,
        source_lang,
        target_lang,
        context_notes="",
        apply_rtl=False,
        on_batch_complete=None,
        glossary=None,
        context_segments=None,
    ):
        self.calls.append([s.index for s in segments])
        self.notes.append(context_notes)
        for s in segments:
            if s.index in self.fail_indices:
                s.translated_text, s.translation_failed = None, True
            else:
                s.translated_text = f"{self.prefix} {s.index}{self.suffix}"
                s.translation_failed = False
        if on_batch_complete:
            await on_batch_complete(segments, 100.0)
        return segments


class SlowTranscriber(BaseTranscriber):
    """Emits segments slowly and honours the pipeline cancellation flag."""

    def __init__(self):
        self.stopped_early = False

    async def transcribe(self, audio_path, language=None, on_segment_callback=None, cancel_flag=None):
        for i in range(1, 200):
            if cancel_flag is not None and cancel_flag.is_set():
                self.stopped_early = True
                raise PipelineCancelled()
            await asyncio.sleep(0.01)
            if on_segment_callback:
                await on_segment_callback(
                    RawSegment(index=i, start_time=i, end_time=i + 0.5, text="hi"),
                    i / 2,
                )
        return []
