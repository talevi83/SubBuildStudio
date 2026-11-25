"""Regression tests for the defects found in job_320de88c99 (2h08m feature film, WhisperX + vocal
isolation + Claude Haiku translation). Each section maps to one numbered issue of the bug report."""
import sys
import types

import numpy as np
import pytest

from app.pipeline.context import RawSegment, TranscribedWord

MOVIE_DURATION = 7683.7


def _seg(idx, start, end, text, conf=0.9, speaker=None, words=True):
    toks = text.split()
    ws = []
    if words and toks:
        step = (end - start) / len(toks)
        ws = [
            TranscribedWord(word=f" {t}", start=round(start + i * step, 3), end=round(start + (i + 1) * step, 3),
                            probability=conf, speaker=speaker)
            for i, t in enumerate(toks)
        ]
    return RawSegment(index=idx, start_time=start, end_time=end, text=text, confidence=conf, words=ws, speaker=speaker)


# ---------------------------------------------------------------------------
# Issue 2: the watermark/outro filter deleted real mid-movie dialogue
# ---------------------------------------------------------------------------

class TestWatermarkFilter:
    def test_mid_movie_dialogue_with_ripped_from_is_not_filtered(self):
        from app.pipeline.gap_recovery import is_watermark_hallucination

        line = "The cord not yet cut, they ripped from me."
        assert is_watermark_hallucination(line, 5135.1, MOVIE_DURATION) is False
        # Position-free callers (gap recovery) must not drop it either
        assert is_watermark_hallucination(line) is False

    @pytest.mark.parametrize("line", [
        "Ripped from my arms.",
        "The letter was transcribed by a monk at Canterbury.",
        "This sword was encoded by no smith I know of.",
        "I watched it on TV 2021 nights in a row, every one of them.",
        "Copyright laws mean nothing to a starving man, you understand that?",
    ])
    def test_credit_verbs_inside_real_sentences_are_kept(self, line):
        from app.pipeline.gap_recovery import is_watermark_hallucination

        assert is_watermark_hallucination(line, 3000.0, MOVIE_DURATION) is False

    @pytest.mark.parametrize("line", [
        "© BF-WATCH TV 2021",
        "Subtitles by OpenSubtitles",
        "Subtitles by the Amara.org community",
        "Transcript by CastingWords",
        "Synced and corrected by explosiveskull",
        "Ripped by YIFY",
        "www.example.com",
        "TV 2021",
    ])
    def test_real_watermarks_are_filtered_anywhere(self, line):
        from app.pipeline.gap_recovery import is_watermark_hallucination

        assert is_watermark_hallucination(line, 100.0, MOVIE_DURATION) is True

    def test_outro_phrases_only_filtered_in_the_final_minutes(self):
        from app.pipeline.gap_recovery import in_outro_window, is_watermark_hallucination

        assert is_watermark_hallucination("Thanks for watching!", 3000.0, MOVIE_DURATION) is False
        assert is_watermark_hallucination("Thanks for watching!", MOVIE_DURATION - 60, MOVIE_DURATION) is True
        # Unknown position keeps the strict legacy behaviour for gap-recovery clips
        assert is_watermark_hallucination("Thanks for watching!") is True
        # Window: at least 3 minutes, 4% of the media, capped at 7 minutes
        assert in_outro_window(MOVIE_DURATION - 300, MOVIE_DURATION) is True   # 4% of 7683s = 307s
        assert in_outro_window(MOVIE_DURATION - 320, MOVIE_DURATION) is False
        assert in_outro_window(500.0, 600.0) is True                           # short clip: last 180s
        assert in_outro_window(15000.0, 20000.0) is False                      # 7-minute cap


# ---------------------------------------------------------------------------
# Issue 1: punctuation-only cues and breath hallucinations
# ---------------------------------------------------------------------------

class TestPunctuationOnlyCues:
    @pytest.mark.parametrize("text,expected", [
        (".", False), (". .", False), ("...", False), ("… …", False), ("♪", False), (" - ", False),
        ("Oh.", True), ("הו.", True), ("7", True), ("No!", True),
    ])
    def test_has_spoken_content(self, text, expected):
        from app.utils.text_rules import has_spoken_content

        assert has_spoken_content(text) is expected

    def test_refine_raw_segments_drops_punctuation_only_segments_and_tokens(self):
        from app.pipeline.segmentation import refine_raw_segments

        raw = [
            _seg(1, 150.0, 151.0, "We must continue our work."),
            _seg(2, 154.5, 155.5, "."),
            _seg(3, 159.8, 160.8, ". ."),
            _seg(4, 166.7, 167.7, "..."),
            _seg(5, 170.0, 172.0, "Burn it to the ground!"),
        ]
        out = refine_raw_segments(raw, max_chars_per_line=42)
        texts = [s["original_text"] for s in out]
        assert texts == ["We must continue our work.", "Burn it to the ground!"]

    def test_refine_raw_segments_all_punctuation_yields_nothing(self):
        from app.pipeline.segmentation import refine_raw_segments

        assert refine_raw_segments([_seg(1, 1.0, 2.0, "."), _seg(2, 3.0, 4.0, "...")]) == []

    def test_phantom_interjection_filter_is_conservative(self):
        from app.providers.transcription.whisperx_provider import filter_phantom_interjections

        segs = [
            _seg(1, 10.0, 12.0, "Where is the king?", conf=0.8),
            _seg(2, 12.2, 12.6, "Yeah.", conf=0.2),          # inside a conversation: kept
            _seg(3, 40.0, 41.0, "Thank you.", conf=0.22),    # alone in silence, unaligned: dropped
            _seg(4, 60.0, 60.5, "No.", conf=0.25),           # not a stock phantom phrase: kept
            _seg(5, 80.0, 81.0, "Thank you.", conf=0.85),    # well aligned: kept
            _seg(6, 100.0, 101.0, "Oh.", conf=0.1, words=False),  # no alignment info: kept
        ]
        kept, dropped = filter_phantom_interjections(segs)
        assert dropped == 1
        assert [s.text for s in kept] == ["Where is the king?", "Yeah.", "No.", "Thank you.", "Oh."]
        assert [s.index for s in kept] == [1, 2, 3, 4, 5]

    def test_vad_onset_is_not_lowered_on_isolated_vocals(self):
        from app.providers.transcription.whisperx_provider import WhisperXProvider

        mixed = WhisperXProvider(device="cpu", compute_type="int8", comms_boost=True)
        isolated = WhisperXProvider(device="cpu", compute_type="int8", comms_boost=True, isolated_vocals=True)
        assert mixed._build_vad_options()["vad_onset"] < 0.5
        assert isolated._build_vad_options() == {"vad_onset": 0.50, "vad_offset": 0.363}


@pytest.mark.asyncio
async def test_whisperx_run_drops_dots_keeps_mid_movie_dialogue(monkeypatch, tmp_path):
    """End-to-end through the WhisperX provider with a fake whisperx module."""
    from app.providers.transcription.whisperx_provider import WhisperXProvider

    aligned = [
        {"start": 154.5, "end": 155.5, "text": ".", "words": [{"word": ".", "start": 154.5, "end": 155.5, "score": 0.1}]},
        {"start": 194.27, "end": 196.51, "text": ". .", "words": []},
        {"start": 5135.1, "end": 5138.2, "text": "The cord not yet cut, they ripped from me.",
         "words": [{"word": w, "start": 5135.1 + i * 0.35, "end": 5135.4 + i * 0.35, "score": 0.8}
                   for i, w in enumerate("The cord not yet cut, they ripped from me.".split())]},
        {"start": 6000.0, "end": 6002.0, "text": "Subtitles by OpenSubtitles", "words": []},
    ]
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None, ipc_collect=lambda: None)

    class _ASR:
        def transcribe(self, _audio, batch_size=8, language=None):
            return {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "x"}]}

    fake_wx = types.ModuleType("whisperx")
    fake_wx.load_model = lambda *_a, **_k: _ASR()
    # 16 kHz samples: the provider derives the media duration from the array length
    fake_wx.load_audio = lambda _p: np.zeros(int(MOVIE_DURATION * 16000), dtype=np.int8)
    fake_wx.load_align_model = lambda language_code, device: (object(), {})
    fake_wx.align = lambda *_a, **_k: {"segments": aligned}
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisperx", fake_wx)
    monkeypatch.setattr("app.providers.transcription.whisperx_provider.is_whisperx_available", lambda: True)

    provider = WhisperXProvider(device="cpu", compute_type="int8")
    segs = await provider.transcribe(str(tmp_path / "a.wav"), language="en")
    assert [s.text for s in segs] == ["The cord not yet cut, they ripped from me."]


# ---------------------------------------------------------------------------
# Issue 5: translation output-token bloat (adaptive thinking), caching, batch size, payload
# ---------------------------------------------------------------------------

class TestAnthropicThinkingAndCaching:
    @pytest.mark.parametrize("model,expected", [
        ("claude-haiku-5-5", {"thinking": {"type": "disabled"}}),
        ("claude-haiku-5", {"thinking": {"type": "disabled"}}),
        ("claude-sonnet-5", {"thinking": {"type": "disabled"}}),
        ("claude-opus-5", {"thinking": {"type": "disabled"}}),
        ("claude-sonnet-5-5", {"thinking": {"type": "between_tools"}}),
        ("claude-opus-5-5", {"output_config": {"effort": "low"}}),
        ("claude-fable-5-2", {"output_config": {"effort": "low"}}),
        ("claude-haiku-4-5", {}),
        ("claude-3-5-haiku-20241022", {}),
        ("gpt-5-mini", {}),
    ])
    def test_default_thinking_is_the_lowest_the_model_accepts(self, model, expected):
        from app.providers.llm.client import anthropic_thinking_fields

        assert anthropic_thinking_fields(model, "default") == expected

    def test_explicit_effort_is_passed_through(self):
        from app.providers.llm.client import anthropic_thinking_fields

        assert anthropic_thinking_fields("claude-haiku-5-5", "high") == {"output_config": {"effort": "high"}}
        assert anthropic_thinking_fields("claude-haiku-4-5", "high") == {}

    @pytest.mark.asyncio
    async def test_anthropic_request_shape_usage_and_cache_cost(self, monkeypatch):
        from pydantic import BaseModel

        from app.core.model_catalog import UsageTracker
        from app.providers.llm import client as llm
        from app.providers.translation.gemini_provider import (
            PER_REQUEST_CONTEXT_HEADING,
            TranslationBatchResponseSchema,
        )

        sent = {}

        async def fake_post(url, headers, payload, timeout, provider_label, max_attempts=4):
            sent.update(payload)
            return {
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": '{"translations":[{"id":1,"text":"שלום"}]}'},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 3000, "output_tokens": 40},
            }

        monkeypatch.setattr(llm, "_post_json_with_retry", fake_post)
        spec = llm.ProviderSpec(provider="anthropic", model="claude-haiku-5-5", api_key="k")
        usage = UsageTracker()
        system = f"STATIC RULES\n{PER_REQUEST_CONTEXT_HEADING}\nGlossary: Plowman -> פלאומן"
        resp = await llm.call_text_json(spec, system_instruction=system, prompt="{}",
                                        schema_cls=TranslationBatchResponseSchema, usage=usage, stage="translation")

        assert sent["thinking"] == {"type": "disabled"}
        assert "temperature" not in sent
        blocks = sent["system"]
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert blocks[0]["text"].startswith("STATIC RULES") and "JSON Schema" in blocks[0]["text"]
        assert "Plowman" not in blocks[0]["text"]          # per-batch glossary is not in the cached prefix
        assert "cache_control" not in blocks[1] and "Plowman" in blocks[1]["text"]
        assert resp.input_tokens == 3200 and resp.output_tokens == 40

        st = usage.stages["translation"]
        assert st.input_tokens == 3200
        # Haiku 5.5: $0.10/M input, $0.50/M output; cache reads at 10% of the input rate
        expected = (200 + 3000 * 0.10) / 1e6 * 0.10 + 40 / 1e6 * 0.50
        assert st.cost_usd == pytest.approx(expected, rel=1e-6)


class TestLeanTranslationRequests:
    def test_anthropic_batches_are_large_and_others_unchanged(self):
        from app.providers.translation.gemini_provider import ANTHROPIC_BATCH_SIZE, default_batch_size_for_model

        assert ANTHROPIC_BATCH_SIZE >= 50
        assert default_batch_size_for_model("anthropic", "claude-haiku-5-5", 35) == ANTHROPIC_BATCH_SIZE
        assert default_batch_size_for_model("openai", "gpt-5-mini", 35) == 18
        assert default_batch_size_for_model("google", "gemini-3.5-flash-lite", 35) == 18

    def test_system_prompt_static_prefix_is_identical_across_batches(self):
        from app.providers.translation.gemini_provider import build_subtitle_system_prompt, split_cacheable_system_prompt

        g1 = {"characters": [{"name": "Plowman", "target_name": "פלאומן", "gender": "male"}], "terms": []}
        g2 = {"characters": [{"name": "Alice", "target_name": "אליס", "gender": "female"}], "terms": []}
        p1 = build_subtitle_system_prompt("English", "Hebrew", glossary=g1, report_addressee_form=True)
        p2 = build_subtitle_system_prompt("English", "Hebrew", glossary=g2, report_addressee_form=True)
        s1, d1 = split_cacheable_system_prompt(p1)
        s2, d2 = split_cacheable_system_prompt(p2)
        assert s1 == s2 and "פלאומן" in d1 and "אליס" in d2
        assert "פלאומן" not in s1

    def test_response_schema_is_minimal(self):
        from app.providers.translation.gemini_provider import (
            TranslationBatchResponseSchema,
            TranslationBatchWithFormResponseSchema,
        )

        item = TranslationBatchResponseSchema.model_json_schema()["$defs"]["TranslatedItemSchema"]
        assert set(item["properties"]) == {"id", "text"}
        form_item = TranslationBatchWithFormResponseSchema.model_json_schema()["$defs"]["TranslatedItemWithFormSchema"]
        assert set(form_item["properties"]) == {"id", "text", "addressee_form"}

    @pytest.mark.asyncio
    async def test_request_payload_is_compact_and_has_no_duplicate_ids(self, monkeypatch):
        import json
        from types import SimpleNamespace

        from app.providers.translation.base import TranslatableSegment
        from app.providers.translation.gemini_provider import GeminiTranslator

        seen = []

        async def fake_generate(self, contents, system_prompt, schema, temperature, **kwargs):
            seen.append(contents)
            batch = json.loads(contents)["target_batch"]
            return SimpleNamespace(text=json.dumps({"translations": [{"id": it["id"], "text": f"ת{it['id']}"} for it in batch]}))

        monkeypatch.setattr(GeminiTranslator, "_generate", fake_generate)
        tr = GeminiTranslator(api_key="test-key", batch_size=5)
        segs = [TranslatableSegment(index=i, start_time=float(i), end_time=i + 1.0, original_text=f"Line {i}") for i in range(1, 8)]
        await tr.translate_segments(segs, "English", "Hebrew")
        assert all(s.translated_text for s in segs)
        assert "\n" not in seen[0] and ": " not in seen[0]   # no pretty-printing
        first = json.loads(seen[0])
        assert all("index" not in it for part in first.values() for it in part)

    def test_translator_logger_is_provider_neutral(self):
        import app.providers.translation.gemini_provider as gp

        assert gp.logger.name == "app.providers.translation.translator"


# ---------------------------------------------------------------------------
# Issue 4a: diarization collapsed 14 voices into 3; untrusted speaker ids drove genders
# ---------------------------------------------------------------------------

def _synthetic_voices(n_speakers=6, per_speaker=30, shared=0.45, noise_var=0.6, seed=7):
    """Window embeddings whose centroids sit ~0.55 apart (distinct people with a shared 'room' tone)
    while windows of one person sit ~0.38 apart: the regime real film audio is in."""
    rng = np.random.default_rng(seed)
    dim = 512
    common = rng.normal(size=dim); common /= np.linalg.norm(common)
    embs, truth, windows = [], [], []
    t = 0.0
    for k in range(n_speakers):
        u = rng.normal(size=dim); u -= (u @ common) * common; u /= np.linalg.norm(u)
        voice = np.sqrt(shared) * common + np.sqrt(1 - shared) * u
        for _ in range(per_speaker):
            noise = rng.normal(size=dim); noise *= np.sqrt(noise_var) / np.linalg.norm(noise)
            v = voice + noise
            embs.append(v / np.linalg.norm(v)); truth.append(k)
            windows.append((t, t + 1.6)); t += 2.0
    return np.vstack(embs).astype(np.float32), truth, windows


class TestDiarizationClustering:
    def test_distinct_voices_are_not_merged_into_a_few_speakers(self):
        from app.pipeline.diarization import assign_window_speakers

        emb, truth, windows = _synthetic_voices()
        labels = assign_window_speakers(emb, windows)
        assert len({l for l in labels if l}) == 6
        # Every found speaker is one true voice (purity)
        for lbl in {l for l in labels if l}:
            assert len({t for t, l in zip(truth, labels) if l == lbl}) == 1

    def test_old_shared_threshold_collapses_the_same_voices(self):
        """Documents the bug: reusing the clustering threshold for centroid merging collapses them."""
        from app.pipeline.diarization import CLUSTER_THRESHOLD, assign_window_speakers

        emb, _, windows = _synthetic_voices()
        labels = assign_window_speakers(emb, windows, merge_threshold=CLUSTER_THRESHOLD)
        assert len({l for l in labels if l}) < 6

    def test_one_voice_stays_one_speaker(self):
        from app.pipeline.diarization import assign_window_speakers

        emb, _, windows = _synthetic_voices(n_speakers=1, per_speaker=60)
        assert len({l for l in assign_window_speakers(emb, windows) if l}) == 1


def _timeline(shares):
    out, t = [], 0.0
    for sid, sec in shares.items():
        out.append({"start": t, "end": t + sec, "speaker": sid}); t += sec
    return out


class TestDiarizationAssessment:
    CHARACTERS = [{"name": n} for n in ("Plowman", "King Richard", "Wat Tyler", "John Ball", "Alice",
                                        "The King's Mother", "Archbishop", "Treasurer", "Walworth", "Messenger", "Bishop")]

    def test_collapsed_clustering_is_unreliable(self):
        from app.pipeline.diarization import assess_diarization

        speakers = {f"SPEAKER_0{i}": {"gender": "unknown"} for i in (1, 2, 3)}
        q = assess_diarization(speakers, _timeline({"SPEAKER_01": 900, "SPEAKER_02": 700, "SPEAKER_03": 650}), self.CHARACTERS)
        assert q["reliable"] is False
        assert set(q["untrusted"]) == {"SPEAKER_01", "SPEAKER_02", "SPEAKER_03"}
        assert any("11 named characters" in r for r in q["reasons"])

    def test_mixed_voice_clusters_are_untrusted_individually(self):
        from app.pipeline.diarization import assess_diarization

        speakers = {f"SPEAKER_{i:02d}": {"gender": "male"} for i in range(1, 9)}
        speakers["SPEAKER_01"]["voice_mixed"] = True
        q = assess_diarization(speakers, _timeline({sid: 100 for sid in speakers}), self.CHARACTERS)
        assert q["reliable"] is True and q["untrusted"] == ["SPEAKER_01"]

    def test_two_person_podcast_is_reliable(self):
        from app.pipeline.diarization import assess_diarization

        speakers = {"SPEAKER_01": {"gender": "male"}, "SPEAKER_02": {"gender": "female"}}
        q = assess_diarization(speakers, _timeline({"SPEAKER_01": 1800, "SPEAKER_02": 1700}),
                               [{"name": "Host"}, {"name": "Guest"}])
        assert q == {"reliable": True, "untrusted": [], "reasons": [], "speakers": 2}

    def test_voice_gender_flags_a_cluster_pooling_men_and_women(self):
        from app.pipeline.voice_gender import apply_voice_genders

        speakers = {"SPEAKER_01": {"gender": "unknown"}, "SPEAKER_02": {"gender": "unknown"}}
        est = {
            "SPEAKER_01": {"gender": "unknown", "confidence": "low", "median_f0": 186.0, "voiced_seconds": 400.0, "iqr_rel": 1.05},
            "SPEAKER_02": {"gender": "male", "confidence": "high", "median_f0": 118.0, "voiced_seconds": 300.0, "iqr_rel": 0.30},
        }
        apply_voice_genders(speakers, est)
        assert speakers["SPEAKER_01"]["voice_mixed"] is True
        assert "voice_mixed" not in speakers["SPEAKER_02"]


# The Queen Mother scene (cues 404-418): with every line on one collapsed speaker id, "my lady"
# said BY Plowman was carried onto the Queen's own lines, which then addressed Plowman as a woman.
QUEEN_SCENE = [
    (404, 2009.49, 2012.72, "Dare you not show your face to the mother of the crown?"),
    (405, 2020.18, 2021.35, "Forgive me, my lady."),
    (406, 2022.46, 2023.77, "Who are you?"),
    (407, 2023.85, 2025.57, "I am Plowman from East Fen."),
    (409, 2033.32, 2035.67, "We come to petition the king, my lady."),
    (411, 2040.09, 2041.62, "You sullied the cathedral."),
    (412, 2043.04, 2046.92, "You think the king will hear your petition when you burn his lands"),
]


def _queen_scene(speaker="SPEAKER_02"):
    return [{"index": i, "start_time": s, "end_time": e, "original_text": t, "segment_type": "DIALOGUE",
             "speaker": speaker} for i, s, e, t in QUEEN_SCENE]


class TestDiarizationGuard:
    def test_collapsed_ids_carry_the_wrong_listener_gender(self):
        """Documents the bug the guard prevents."""
        from app.pipeline.addressee import resolve_addressees

        rec = resolve_addressees(_queen_scene(), {"SPEAKER_02": {"label": "Plowman", "gender": "male"}})
        assert rec[411].gender == "female"  # the Queen talking to Plowman, wrongly feminine

    def test_untrusted_ids_only_keep_line_local_evidence(self):
        from app.pipeline.addressee import resolve_addressees

        blanked = [dict(s, speaker=None) for s in _queen_scene()]
        rec = resolve_addressees(blanked, {"SPEAKER_02": {"label": "Plowman", "gender": "male"}})
        assert rec[405].gender == "female" and rec[405].source == "honorific"   # "my lady" itself
        assert 411 not in rec or rec[411].gender != "female"
        assert 406 not in rec or rec[406].gender != "female"

    def test_guard_quarantines_collapsed_speakers_before_glossary_sync(self):
        from types import SimpleNamespace

        from app.pipeline.stages.translate_stage import _apply_diarization_guard, _sync_glossary_to_speakers

        ctx = SimpleNamespace(
            job_id="job_test",
            warnings=[],
            diarization_quality=None,
            speakers={sid: {"label": f"Speaker {i}", "gender": "unknown"} for i, sid in enumerate(("SPEAKER_01", "SPEAKER_02", "SPEAKER_03"), 1)},
            diarization_timeline=_timeline({"SPEAKER_01": 900, "SPEAKER_02": 700, "SPEAKER_03": 650}),
            glossary={"characters": [dict(c, gender="female" if c["name"] in ("Alice", "The King's Mother") else "male",
                                          speaker_id="SPEAKER_02") for c in TestDiarizationAssessment.CHARACTERS]},
        )
        untrusted = _apply_diarization_guard(ctx)
        assert untrusted == {"SPEAKER_01", "SPEAKER_02", "SPEAKER_03"}
        assert ctx.diarization_quality["reliable"] is False
        assert all("speaker_id" not in c for c in ctx.glossary["characters"])
        _sync_glossary_to_speakers(ctx.glossary, ctx.speakers)
        assert all(p["gender"] == "unknown" for p in ctx.speakers.values())
        assert any("unreliable" in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# Issue 3: micro-fragmentation (false speaker flickers, dramatic pauses, duplicated boundary words)
# ---------------------------------------------------------------------------

# Word timings and per-word speaker labels exactly as stored for job_320de88c99
FRAGMENTED = {
    "cues 17-18": [("Our", 276.74, 277.74, "SPEAKER_01"), ("king", 277.80, 278.00, "SPEAKER_03"),
                   ("is", 278.06, 278.12, "SPEAKER_03"), ("indisposed.", 278.20, 279.00, "SPEAKER_03")],
    "cues 175-178": [("Every", 990.97, 991.12, "SPEAKER_02"), ("night", 991.17, 991.32, "SPEAKER_02"),
                     ("I", 991.37, 991.52, "SPEAKER_02"), ("see", 991.57, 991.77, "SPEAKER_01"),
                     ("a", 991.79, 991.81, "SPEAKER_01"), ("lot", 993.21, 993.31, "SPEAKER_01"),
                     ("of", 993.45, 993.49, "SPEAKER_01"), ("devils", 993.57, 994.15, "SPEAKER_01"),
                     ("coming", 995.91, 996.31, "SPEAKER_01"), ("for", 996.41, 996.65, "SPEAKER_01"),
                     ("us.", 996.77, 996.87, "SPEAKER_01")],
    "cues 339-341": [("We", 1742.33, 1742.48, "SPEAKER_02"), ("never", 1742.53, 1742.68, "SPEAKER_02"),
                     ("seek", 1742.73, 1742.81, "SPEAKER_01"), ("the", 1742.83, 1742.89, "SPEAKER_01"),
                     ("truth.", 1746.58, 1746.70, "SPEAKER_01")],
    "cues 626-628": [("They're", 2913.55, 2914.78, "SPEAKER_01"), ("sending", 2916.08, 2917.31, "SPEAKER_01"),
                     ("a", 2918.54, 2918.68, "SPEAKER_02"), ("boat.", 2918.74, 2919.04, "SPEAKER_02")],
}
EXPECTED = {
    "cues 17-18": "Our king is indisposed.",
    "cues 175-178": "Every night I see a lot of devils coming for us.",
    "cues 339-341": "We never seek the truth.",
    "cues 626-628": "They're sending a boat.",
}


def _diarized_raw(words):
    """One WhisperX-style raw segment per sentence, then the diarize stage's speaker split."""
    from app.pipeline.diarization import split_raw_segments_by_speaker

    timeline = [{"start": s, "end": e, "speaker": sp} for _, s, e, sp in words]
    raw = [RawSegment(index=1, start_time=words[0][1], end_time=words[-1][2], text=" ".join(w[0] for w in words),
                      confidence=0.9, words=[TranscribedWord(word=f" {w}", start=s, end=e, probability=0.9) for w, s, e, _ in words])]
    return split_raw_segments_by_speaker(raw, timeline)


class TestFragmentMerging:
    @pytest.mark.parametrize("case", list(FRAGMENTED))
    def test_report_examples_become_one_subtitle(self, case):
        from app.pipeline.segmentation import refine_raw_segments

        out = refine_raw_segments(_diarized_raw(FRAGMENTED[case]), max_chars_per_line=42)
        assert [" ".join(s["original_text"].split()) for s in out] == [EXPECTED[case]]

    def test_flicker_words_inherit_their_runs_speaker(self):
        segs = _diarized_raw(FRAGMENTED["cues 17-18"])
        assert len(segs) == 1
        assert {w.speaker for w in segs[0].words} == {segs[0].speaker}

    def test_unfinished_cue_merges_across_a_dramatic_pause_of_the_same_speaker(self):
        from app.pipeline.segmentation import refine_raw_segments

        raw = [_seg(1, 10.0, 11.0, "Every night I see", speaker="SPEAKER_01"),
               _seg(2, 13.5, 15.0, "devils coming for us.", speaker="SPEAKER_01")]
        out = refine_raw_segments(raw, max_chars_per_line=42)
        assert len(out) == 1 and out[0]["start_time"] <= 10.0 and out[0]["end_time"] >= 15.0

    @pytest.mark.parametrize("a,b", [
        ("Who are you?", "I am Plowman from East Fen."),       # finished sentence: a real turn
        ("Wait for me", "Plowman, stay with the carts."),      # capitalised new sentence after a pause
    ])
    def test_real_turns_between_speakers_stay_separate(self, a, b):
        from app.pipeline.segmentation import refine_raw_segments

        raw = [_seg(1, 10.0, 11.2, a, speaker="SPEAKER_01"), _seg(2, 12.6, 14.4, b, speaker="SPEAKER_02")]
        out = refine_raw_segments(raw, max_chars_per_line=42)
        assert len(out) == 2 and [s["speaker"] for s in out] == ["SPEAKER_01", "SPEAKER_02"]

    def test_continuation_never_breaks_line_or_duration_limits(self):
        from app.pipeline.segmentation import refine_raw_segments

        raw = [_seg(1, 10.0, 13.0, "We march to free our king from the traitors who", speaker="SPEAKER_01"),
               _seg(2, 14.0, 17.5, "hold him prisoner in the tower of London tonight.", speaker="SPEAKER_01")]
        for cue in refine_raw_segments(raw, max_chars_per_line=42):
            lines = cue["original_text"].split("\n")
            assert len(lines) <= 2 and all(len(l) <= 42 for l in lines)
            assert cue["end_time"] - cue["start_time"] <= 7.0 + 1e-6

    def test_overlapping_duplicate_boundary_word_is_dropped_but_stutter_kept(self):
        from app.pipeline.segmentation import refine_raw_segments

        dup = [RawSegment(index=1, start_time=1.0, end_time=2.0, text="They're sending", confidence=0.9,
                          words=[TranscribedWord(word=" They're", start=1.0, end=1.4, probability=0.9),
                                 TranscribedWord(word=" sending", start=1.5, end=2.0, probability=0.9)]),
               RawSegment(index=2, start_time=1.9, end_time=2.6, text="sending a boat.", confidence=0.9,
                          words=[TranscribedWord(word=" sending", start=1.9, end=2.1, probability=0.9),
                                 TranscribedWord(word=" a", start=2.15, end=2.2, probability=0.9),
                                 TranscribedWord(word=" boat.", start=2.25, end=2.6, probability=0.9)])]
        assert [" ".join(s["original_text"].split()) for s in refine_raw_segments(dup)] == ["They're sending a boat."]

        stutter = [_seg(1, 1.0, 2.4, "No, no, no!")]
        assert refine_raw_segments(stutter)[0]["original_text"] == "No, no, no!"


# ---------------------------------------------------------------------------
# Issue 4b: local qwen2.5:7b pre-tagging injected 198 ungrounded gender hints
# ---------------------------------------------------------------------------

GLOSSARY = {"characters": [
    {"name": "Plowman", "gender": "male"}, {"name": "Alice", "gender": "female"},
    {"name": "Wat Tyler", "gender": "male"}, {"name": "The King's Mother", "gender": "female"},
]}


class TestPretagGuardrails:
    def _patterns(self):
        from app.pipeline.gender_pretag import _character_name_patterns

        return _character_name_patterns(GLOSSARY)

    def test_listener_hint_needs_a_named_character_of_that_gender(self):
        from app.pipeline.gender_pretag import grounded_listener_hint

        pats = self._patterns()
        assert grounded_listener_hint("F", "Go to the shore.", ["As you said, my lady."], pats) is None
        assert grounded_listener_hint("F", "Alice, go to the shore.", [], pats) == "F"
        assert grounded_listener_hint("F", "Go to the shore.", ["Alice!"], pats) == "F"     # named a line before
        assert grounded_listener_hint("F", "Go to the shore, Plowman.", [], pats) is None   # named man, not a woman
        assert grounded_listener_hint("M", "Go to the shore, Plowman.", [], pats) == "M"
        assert grounded_listener_hint("G", "Go to the shore.", [], pats) is None
        assert grounded_listener_hint("G", "Listen up, everyone!", [], pats) == "G"
        assert grounded_listener_hint("?", "Alice, go.", [], pats) is None

    @pytest.mark.asyncio
    async def test_pretag_drops_ungrounded_guesses_and_the_models_speaker_gender(self, monkeypatch):
        import httpx

        from app.pipeline.gender_pretag import pretag_gender_hints
        from app.providers.translation.base import TranslatableSegment

        async def _get(self, url, **kwargs):
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})

        async def _post(self, url, json=None, **kwargs):
            # The messenger scene (cues 621-623): the model calls the male messenger "F"
            return httpx.Response(200, json={"response": '{"items": ['
                                             '{"index": 621, "speaker": "F", "to": "F"},'
                                             '{"index": 622, "speaker": "F", "to": "F"},'
                                             '{"index": 630, "speaker": "M", "to": "M"}]}'})

        monkeypatch.setattr(httpx.AsyncClient, "get", _get)
        monkeypatch.setattr(httpx.AsyncClient, "post", _post)
        segs = [
            TranslatableSegment(621, 2900.2, 2901.2, "Go to the shore."),
            TranslatableSegment(622, 2901.3, 2902.5, "Find out what they want."),
            TranslatableSegment(630, 2925.0, 2926.0, "Plowman, are you ready?"),
        ]
        hints = await pretag_gender_hints(segs, target_lang="Hebrew", glossary=GLOSSARY)
        assert hints == {630: "[Speaker: ?, To: M]"}

    def test_voice_genders_are_labelled_as_estimates_for_the_translator(self):
        from app.pipeline.speaker_match import format_speaker_identity

        speakers = {"SPEAKER_01": {"label": "Speaker 1", "gender": "female", "gender_source": "voice"},
                    "SPEAKER_02": {"label": "Plowman", "gender": "male"}}
        assert format_speaker_identity("SPEAKER_01", speakers) == "SPEAKER_01 (female by voice)"
        assert format_speaker_identity("SPEAKER_02", speakers) == "SPEAKER_02 (Plowman, male)"


# ---------------------------------------------------------------------------
# Issue 6: on-screen scan sent 488 frames at "high" to find one title card
# ---------------------------------------------------------------------------

def _app_settings(cost_saver=True, provider="google", model="gemini-3.5-flash-lite", res="high"):
    from types import SimpleNamespace

    return SimpleNamespace(
        cost_saver_mode=cost_saver,
        vision_media_resolution=res,
        vision=SimpleNamespace(provider=provider, model=model, thinking="default", flex=True, key_id=None),
    )


class TestOnscreenTwoPass:
    @pytest.mark.parametrize("kwargs,expected", [
        ({}, "medium"),
        ({"res": "medium"}, "low"),
        ({"res": "low"}, None),
        ({"cost_saver": False}, None),
        ({"provider": "anthropic", "model": "claude-haiku-5-5"}, None),
        ({"model": "gemini-2.5-flash"}, None),   # 2.5 ignores media_resolution
    ])
    def test_screening_resolution(self, kwargs, expected):
        from app.pipeline.stages.ocr_stage import screening_resolution

        assert screening_resolution(_app_settings(**kwargs)) == expected

    @pytest.mark.asyncio
    async def test_only_frames_with_text_are_read_again_at_full_resolution(self, monkeypatch):
        from types import SimpleNamespace

        from app.pipeline.frame_sampling import Frame
        from app.pipeline.onscreen_merge import FrameDetection
        from app.pipeline.stages.ocr_stage import OnScreenTextStage as Stage

        calls = []

        async def fake_detect(self, ctx, client, frames, instruction, stage, on_batch=None, sensitivity=2, resolution=None):
            calls.append((resolution, [f.time for f in frames]))
            if resolution == "medium":
                return [FrameDetection(time=6840.0, text="[ONE YEAR LATER]", until=6844.0)], 0
            return [FrameDetection(time=f.time, text="[ONE YEAR LATER]", until=f.time + 4) for f in frames], 0

        monkeypatch.setattr(Stage, "_detect_batches", fake_detect)
        frames = [Frame(time=float(t), path=f"f{t}.jpg", change=1.0) for t in range(0, 7000, 14)] + \
                 [Frame(time=6840.0, path="title.jpg", change=1.0)]
        dets, failed = await Stage()._scan_two_pass(SimpleNamespace(job_id="j"), None, frames, "instr", None, 2, _app_settings())
        assert failed == 0
        assert calls[0][0] == "medium" and len(calls[0][1]) == len(frames)
        assert calls[1] == ("high", [6840.0])                       # one frame re-read at full detail
        assert [d.text for d in dets] == ["[ONE YEAR LATER]"]

    @pytest.mark.asyncio
    async def test_failed_confirmation_keeps_the_screening_answer(self, monkeypatch):
        from types import SimpleNamespace

        from app.pipeline.frame_sampling import Frame
        from app.pipeline.onscreen_merge import FrameDetection
        from app.pipeline.stages.ocr_stage import OnScreenTextStage as Stage

        async def fake_detect(self, ctx, client, frames, instruction, stage, on_batch=None, sensitivity=2, resolution=None):
            if resolution == "medium":
                return [FrameDetection(time=10.0, text="[LONDON]", until=12.0)], 0
            return [], 1

        monkeypatch.setattr(Stage, "_detect_batches", fake_detect)
        frames = [Frame(time=10.0, path="a.jpg", change=1.0), Frame(time=20.0, path="b.jpg", change=1.0)]
        dets, _ = await Stage()._scan_two_pass(SimpleNamespace(job_id="j"), None, frames, "i", None, 2, _app_settings())
        assert [d.text for d in dets] == ["[LONDON]"]
