"""Unit and integration tests for transcription anti-hallucination, micro-segment merging,
minimum duration enforcement, and locked entity glossary translation upgrades."""
import json
import types

import pytest

import app.providers.transcription.faster_whisper as fw
import app.providers.translation.gemini_provider as gp
from app.core.glossary_profiles import combine_context_notes, parse_user_glossary
from app.pipeline.context import PipelineContext
from app.pipeline.segmentation import (
    enforce_min_duration,
    merge_micro_segments,
    postprocess_subtitle_segments,
)
from app.pipeline.stages.translate_stage import TranslateStage
from app.providers.translation.base import TranslatableSegment
from app.utils.srt_formatter import format_balanced_lines


# ---------------------------------------------------------------------------
# 1. Faster-Whisper anti-hallucination tuning & custom initial_prompt
# ---------------------------------------------------------------------------

class _RecordingWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        seg = types.SimpleNamespace(
            start=0.0,
            end=1.4,
            text=" Work fast, Ploughman.",
            avg_logprob=-0.15,
            words=[
                types.SimpleNamespace(word=" Work", start=0.0, end=0.35, probability=0.95),
                types.SimpleNamespace(word=" fast,", start=0.38, end=0.75, probability=0.93),
                types.SimpleNamespace(word=" Ploughman.", start=0.80, end=1.40, probability=0.91),
            ],
        )
        return iter([seg]), types.SimpleNamespace(duration=2.0, language="en")


@pytest.mark.asyncio
async def test_whisper_default_anti_hallucination_and_initial_prompt(monkeypatch):
    fake_model = _RecordingWhisperModel()
    monkeypatch.setattr(fw, "_load_model", lambda *a: fake_model)
    monkeypatch.setattr(fw, "get_hardware_info", lambda: {"cuda_available": False})

    glossary_input = (
        "1381 revolt drama. Characters: Wat Tyler, John Ball = ג׳ון בול, "
        "Ploughman -> החורש, Domesday, Corpus Christi, Beauchamp"
    )
    transcriber = fw.FasterWhisperTranscriber(
        "large-v3",
        device="cpu",
        vad_filter=True,
        initial_prompt=glossary_input,
    )
    segments = await transcriber.transcribe("movie.wav", language="en")
    assert len(segments) == 1

    call_kwargs = fake_model.calls[0]
    assert call_kwargs["condition_on_previous_text"] is False
    assert call_kwargs["vad_filter"] is True
    assert call_kwargs["vad_parameters"] == {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    }
    assert call_kwargs["compression_ratio_threshold"] == 2.4
    assert call_kwargs["log_prob_threshold"] == -1.0
    assert call_kwargs["no_speech_threshold"] == 0.6

    prompt = call_kwargs["initial_prompt"]
    for expected in ("Wat Tyler", "John Ball", "Ploughman", "Domesday", "Corpus Christi", "Beauchamp"):
        assert expected in prompt
    # Target-language Hebrew annotations must be stripped on English audio
    assert "ג׳ון" not in prompt and "החורש" not in prompt


def test_whisper_condition_on_previous_text_can_be_overridden_explicitly(monkeypatch):
    monkeypatch.setattr(fw, "get_hardware_info", lambda: {"cuda_available": False})
    t_default = fw.FasterWhisperTranscriber("tiny", device="cpu")
    assert t_default.condition_on_previous_text is False

    t_enabled = fw.FasterWhisperTranscriber("tiny", device="cpu", condition_on_previous_text=True)
    assert t_enabled.condition_on_previous_text is True


# ---------------------------------------------------------------------------
# 2. Micro-segment merging, minimum duration (1.0s / 50ms buffer), 2-line balance
# ---------------------------------------------------------------------------

def test_merge_micro_segments_joins_short_gaps_and_respects_punctuation_and_limits():
    raw_cues = [
        # Pair 1: gap 0.25s <= 0.4s, no terminal punctuation -> MERGE
        {"index": 1, "start_time": 1.00, "end_time": 1.55, "original_text": "Work fast,", "segment_type": "DIALOGUE"},
        {"index": 2, "start_time": 1.80, "end_time": 2.90, "original_text": "Ploughman, grab the jewels!", "segment_type": "DIALOGUE"},
        # Pair 2: gap 0.20s <= 0.4s, ends with '.' AND both < 20 chars -> MERGE
        {"index": 3, "start_time": 4.00, "end_time": 4.40, "original_text": "Easy.", "segment_type": "DIALOGUE"},
        {"index": 4, "start_time": 4.60, "end_time": 5.20, "original_text": "Easy!", "segment_type": "DIALOGUE"},
        # Pair 3: cut-off syllable with ellipsis -> MERGE even though next line >= 20 chars
        {"index": 5, "start_time": 6.00, "end_time": 6.30, "original_text": "ל...", "segment_type": "DIALOGUE"},
        {"index": 6, "start_time": 6.50, "end_time": 8.00, "original_text": "לאן אתם לוקחים את המגילות האלה?", "segment_type": "DIALOGUE"},
        # Pair 4: ends with '.' and >= 20 chars -> DO NOT MERGE
        {"index": 7, "start_time": 9.00, "end_time": 10.50, "original_text": "Hold the northern gate now.", "segment_type": "DIALOGUE"},
        {"index": 8, "start_time": 10.70, "end_time": 12.00, "original_text": "Nobody passes.", "segment_type": "DIALOGUE"},
        # Pair 5: gap 0.65s > 0.4s -> DO NOT MERGE
        {"index": 9, "start_time": 13.00, "end_time": 13.60, "original_text": "Wait here", "segment_type": "DIALOGUE"},
        {"index": 10, "start_time": 14.25, "end_time": 15.40, "original_text": "until dark.", "segment_type": "DIALOGUE"},
    ]

    merged = merge_micro_segments(raw_cues, max_gap=0.4, max_chars_per_line=42, short_sentence_chars=20)
    texts = [s["original_text"].replace("\n", " ") for s in merged]
    assert texts == [
        "Work fast, Ploughman, grab the jewels!",
        "Easy. Easy!",
        "ל... לאן אתם לוקחים את המגילות האלה?",
        "Hold the northern gate now.",
        "Nobody passes.",
        "Wait here",
        "until dark.",
    ]


def test_merge_micro_segments_never_exceeds_two_line_limit_or_merges_distinct_speakers():
    too_long_pair = [
        {
            "index": 1,
            "start_time": 0.0,
            "end_time": 2.0,
            "original_text": "We have marched all night from Canterbury across the river,",
            "segment_type": "DIALOGUE",
        },
        {
            "index": 2,
            "start_time": 2.2,
            "end_time": 4.2,
            "original_text": "and we demand to speak with the King before sunset!",
            "segment_type": "DIALOGUE",
        },
    ]
    assert len(merge_micro_segments(too_long_pair, max_chars_per_line=42)) == 2

    distinct_speakers = [
        {"index": 1, "start_time": 0.0, "end_time": 0.6, "original_text": "Who's there", "speaker": "SPEAKER_01"},
        {"index": 2, "start_time": 0.8, "end_time": 1.6, "original_text": "John Ball.", "speaker": "SPEAKER_02"},
    ]
    assert len(merge_micro_segments(distinct_speakers, max_chars_per_line=42)) == 2


def test_enforce_min_duration_extends_to_1s_and_clamps_50ms_before_next_cue():
    cues = [
        # Plenty of room before next cue at 3.0s -> extended from 0.35s to full 1.0s
        {"index": 1, "start_time": 1.00, "end_time": 1.35, "original_text": "Halt!", "segment_type": "DIALOGUE"},
        # Tight room before next cue at 3.70s -> extended up to 3.70 - 0.05 = 3.65s
        {"index": 2, "start_time": 3.00, "end_time": 3.30, "original_text": "Move back!", "segment_type": "DIALOGUE"},
        # Long single-line cue -> formatted cleanly into 2 balanced lines <= 42 chars
        {
            "index": 3,
            "start_time": 3.70,
            "end_time": 6.50,
            "original_text": "Every man and woman in this shire shall hear the charter read aloud today.",
            "segment_type": "DIALOGUE",
        },
    ]
    out = postprocess_subtitle_segments(cues, max_gap=0.4, min_duration=1.0, buffer_sec=0.05, max_chars_per_line=42)
    assert out[0]["start_time"] == 1.00
    assert out[0]["end_time"] == pytest.approx(2.00, abs=1e-3)
    assert out[1]["end_time"] == pytest.approx(3.65, abs=1e-3)
    assert out[2]["start_time"] - out[1]["end_time"] >= 0.049

    lines = out[2]["original_text"].split("\n")
    assert len(lines) == 2
    assert all(len(ln) <= 42 for ln in lines)

    # 3-line unhyphenated input is re-balanced into at most 2 lines
    three_lines = "The Domesday rolls\nare kept inside\nthe abbey vault."
    balanced = format_balanced_lines(three_lines, max_chars_per_line=42)
    assert len(balanced.split("\n")) <= 2


# ---------------------------------------------------------------------------
# 3. Pre-translation entity glossary, locked injection, sliding context & Hebrew rules
# ---------------------------------------------------------------------------

def test_parse_user_glossary_and_combine_notes():
    user_input = (
        "Historical drama set in 1381.\n"
        "John Ball = ג׳ון בול (male)\n"
        "Ploughman -> החורש (male)\n"
        "Domesday = ספר יום הדין, Corpus Christi -> קורפוס כריסטי"
    )
    parsed = parse_user_glossary(user_input)
    assert parsed is not None
    chars = {c["name"]: c for c in parsed["characters"]}
    terms = {t["source"]: t for t in parsed["terms"]}

    assert chars["John Ball"]["target_name"] == "ג׳ון בול"
    assert chars["John Ball"]["gender"] == "male"
    assert chars["John Ball"]["locked"] is True
    assert chars["Ploughman"]["target_name"] == "החורש"
    assert terms["Domesday"]["translation"] == "ספר יום הדין"
    assert terms["Corpus Christi"]["translation"] == "קורפוס כריסטי"

    # Identical initial_prompt & context_notes are deduplicated
    assert combine_context_notes("Same note", "Same note") == "Same note"


@pytest.mark.asyncio
async def test_locked_glossary_injected_into_all_batches_with_sliding_translated_context(monkeypatch):
    monkeypatch.setattr(gp.asyncio, "sleep", lambda _: None)
    recorded_calls = []

    class _FakeGenAIModels:
        def generate_content(self, model, contents, config):
            recorded_calls.append((contents, config.system_instruction))
            if not contents.strip().startswith("{"):
                # Pre-translation glossary extraction pass
                return types.SimpleNamespace(
                    text=json.dumps({
                        "summary": "1381 English Peasants' Revolt historical drama.",
                        "characters": [
                            {
                                "name": "Wat Tyler",
                                "target_name": "ווט טיילר",
                                "gender": "male",
                                "notes": "rebel leader",
                                "speaker_id": "",
                            },
                            {
                                "name": "John Ball",
                                "target_name": "ג׳ון בול",
                                "gender": "male",
                                "notes": "priest; sometimes misheard by ASR as John Bard",
                                "speaker_id": "",
                            },
                        ],
                        "terms": [
                            {"source": "Beauchamp", "translation": "בושאמפ", "notes": "noble estate"},
                        ],
                        "register_notes": "Elevated period drama; no modern slang.",
                    })
                )
            payload = json.loads(contents)
            return types.SimpleNamespace(
                text=json.dumps({
                    "translations": [
                        {
                            "index": item["id"],
                            "translated_text": f"שורה מתורגמת מספר {item['id']} בעברית מלאה",
                            "addressee_form": "",
                        }
                        for item in payload["target_batch"]
                    ]
                })
            )

    translator = gp.GeminiTranslator(
        api_key="test-key",
        batch_size=25,
        prev_context_size=4,
        lookahead_size=3,
    )
    translator.client = types.SimpleNamespace(models=_FakeGenAIModels())

    # 30 segments -> 2 batches (25 + 5); batch 2 has an ASR mishearing ("John Bard") and no verbatim "Domesday"
    refined_segments = [
        {
            "index": i,
            "start_time": float(i * 2),
            "end_time": float(i * 2 + 1.5),
            "original_text": (
                "John Bard speaks at Corpus Christi!" if i == 28 else f"Stand firm at gate {i}, men."
            ),
            "translated_text": None,
            "translation_failed": False,
            "segment_type": "DIALOGUE",
        }
        for i in range(1, 31)
    ]

    ctx = PipelineContext(
        job_id="job_uprising",
        video_path="uprising.mkv",
        file_name="uprising.mkv",
        enable_translation=True,
        source_language="en",
        target_language="Hebrew",
        initial_prompt="Ploughman = החורש (male)\nDomesday = ספר יום הדין",
        refined_segments=refined_segments,
    )

    stage = TranslateStage(translator=translator)
    await stage.execute(ctx)

    # 1 glossary call + 2 translation batch calls
    assert len(recorded_calls) == 3

    _, batch1_sys = recorded_calls[1]
    batch2_contents, batch2_sys = recorded_calls[2]

    # Full locked glossary (both user-supplied and LLM-extracted) is injected into EVERY batch
    for sys_prompt in (batch1_sys, batch2_sys):
        assert "LOCKED TERMINOLOGY & CHARACTER GLOSSARY" in sys_prompt
        assert "החורש" in sys_prompt
        assert "ספר יום הדין" in sys_prompt
        assert "ג׳ון בול" in sys_prompt
        assert "בושאמפ" in sys_prompt
        # Anti-calque Hebrew rules present
        assert "לאט, לאט" in sys_prompt
        assert "קל, קל" in sys_prompt

    # Batch 2 receives the last 4 translated lines from Batch 1 as read-only previous_context
    batch2_payload = json.loads(batch2_contents)
    prev_ctx = batch2_payload["previous_context"]
    assert [item["id"] for item in prev_ctx] == [22, 23, 24, 25]
    assert all(item["translated_text"].startswith("שורה מתורגמת מספר") for item in prev_ctx)


@pytest.mark.asyncio
async def test_openai_and_anthropic_glossary_and_batch_translation_pipeline(tmp_path, monkeypatch):
    """Verify Anthropic (glossary pass) and OpenAI (batch translation) receive the full locked glossary,
    18-line mini-model batching, strict JSON schema (id + text), sliding context, and anti-calque Hebrew rules."""
    import httpx
    from app.core import provider_keys as pk
    from app.core.config import settings

    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    oai_key = pk.add_key("openai", api_key="sk-test-openai-key", label="OAI")
    ant_key = pk.add_key("anthropic", api_key="sk-ant-test-key", label="Claude")

    http_requests = []

    async def fake_post(self, url, headers=None, json=None, **kwargs):
        http_requests.append((url, headers, json))
        if "/v1/messages" in url:
            # Anthropic glossary pass (even with brief preamble before JSON)
            glossary_obj = {
                "summary": "1381 historical drama.",
                "characters": [
                    {
                        "name": "John Ball",
                        "target_name": "ג׳ון בול",
                        "gender": "male",
                        "notes": "rebel priest",
                        "speaker_id": "",
                    }
                ],
                "terms": [
                    {"source": "Corpus Christi", "translation": "קורפוס כריסטי", "notes": "festival"}
                ],
                "register_notes": "Period drama register.",
            }
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "Here is the JSON:\n" + __import__("json").dumps(glossary_obj)}],
                    "usage": {"input_tokens": 500, "output_tokens": 180},
                },
            )
        # OpenAI chat/completions batch translation using explicit `id` and `text` fields
        user_payload = __import__("json").loads(json["messages"][1]["content"])
        out_items = [
            {
                "id": item["id"],
                "text": f"תרגום מלא לשורה {item['id']} בעברית תקנית",
                "addressee_form": "",
            }
            for item in user_payload["target_batch"]
        ]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": __import__("json").dumps({"translations": out_items})}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 400},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    translator = gp.GeminiTranslator(
        provider="openai",
        model_name="gpt-4.1-mini",
        key_id=oai_key["id"],
        glossary_provider="anthropic",
        glossary_model="claude-sonnet-4-5",
        glossary_key_id=ant_key["id"],
    )
    assert translator.batch_size == 18
    assert translator.prev_context_size == 5

    refined_segments = [
        {
            "index": i,
            "start_time": float(i * 2),
            "end_time": float(i * 2 + 1.5),
            "original_text": f"Easy, easy! Hold the line at gate {i}!",
            "translated_text": None,
            "translation_failed": False,
            "segment_type": "DIALOGUE",
        }
        for i in range(1, 41)
    ]
    ctx = PipelineContext(
        job_id="job_multi_prov",
        video_path="uprising.mkv",
        file_name="uprising.mkv",
        enable_translation=True,
        source_language="en",
        target_language="Hebrew",
        initial_prompt="Ploughman = החורש (male)",
        refined_segments=refined_segments,
    )

    stage = TranslateStage(translator=translator)
    await stage.execute(ctx)

    # 1 Anthropic glossary request + 3 OpenAI batch requests (18 + 18 + 4 lines)
    assert len(http_requests) == 4
    assert "/v1/messages" in http_requests[0][0]
    for _, _, oai_body in http_requests[1:]:
        rf = oai_body["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        item_schema = rf["json_schema"]["schema"]["properties"]["translations"]["items"]
        assert item_schema["additionalProperties"] is False
        assert "id" in item_schema["properties"] and "text" in item_schema["properties"]
        assert "id" in item_schema["required"] and "text" in item_schema["required"]

        sys_msg = oai_body["messages"][0]["content"]
        assert "LOCKED TERMINOLOGY & CHARACTER GLOSSARY" in sys_msg
        assert "החורש" in sys_msg
        assert "ג׳ון בול" in sys_msg
        assert "קורפוס כריסטי" in sys_msg
        assert "לאט, לאט" in sys_msg

    second_batch_user = json.loads(http_requests[2][2]["messages"][1]["content"])
    assert len(second_batch_user["target_batch"]) == 18
    assert [it["id"] for it in second_batch_user["previous_context"]] == [14, 15, 16, 17, 18]
    assert all(it["translated_text"].startswith("תרגום מלא לשורה") for it in second_batch_user["previous_context"])


# ---------------------------------------------------------------------------
# 4. Regression tests for the 4 bugs exposed in job_5d93f7cd9e (The Uprising)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bug1_voice_gender_orchestral_false_positives_and_contextual_override(monkeypatch):
    """Ensure 173.9 Hz & noisy orchestral audio are not classified as female, and contextual dialogue/glossary
    male cues override false acoustic `voice` gender estimates without triggering feminine 2nd-person rewrites."""
    import numpy as np
    from app.pipeline.voice_gender import classify_f0, estimate_speaker_genders

    sr = 16000
    # 1a. 173.9 Hz (< 195 Hz) and wide-IQR scattered pitch stay unknown
    assert classify_f0(173.9, 5.0)["gender"] == "unknown"
    assert classify_f0(194.9, 5.0)["gender"] == "unknown"
    assert classify_f0(225.0, 3.0, iqr_f0=135.0)["gender"] == "unknown"

    # 1b. Shouted / boom-EQ male voices (123.1 Hz & 111.1 Hz) with dominant 2nd harmonic (246.2 Hz & 222.2 Hz)
    # must NOT octave-double into female!
    t = np.arange(int(2.0 * sr)) / sr
    shouted_male_123 = (0.28 * np.sin(2 * np.pi * 123.1 * t) + 0.95 * np.sin(2 * np.pi * 246.2 * t)).astype(np.float32)
    est_male = estimate_speaker_genders(shouted_male_123, sr, [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"}])
    assert est_male["SPEAKER_01"]["gender"] == "male"
    assert abs(est_male["SPEAKER_01"]["median_f0"] - 123.1) < 5.0

    # Broadband crowd/orchestral noise has high spectral flatness -> unknown
    rng = np.random.default_rng(42)
    noisy_score = (0.35 * rng.standard_normal(3 * sr)).astype(np.float32)
    est = estimate_speaker_genders(noisy_score, sr, [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_01"}])
    assert est["SPEAKER_01"]["gender"] == "unknown"

    # 1c. Even if acoustic pitch previously mislabeled SPEAKER_01/02/03 as female (`gender_source='voice'`),
    # contextual titles ("My lord", "Father", "I am your husband") & glossary override them to male
    # and prevent `_fix_listener_mismatches` from rewriting masculine 2nd-person Hebrew to feminine.
    monkeypatch.setattr(gp.asyncio, "sleep", lambda _: None)
    calls = []

    class _FakeGenAI:
        def generate_content(self, model, contents, config):
            calls.append((contents, config.system_instruction))
            if not contents.strip().startswith("{"):
                return types.SimpleNamespace(
                    text=json.dumps({
                        "summary": "1381 revolt drama.",
                        "characters": [
                            {"name": "John Ball", "target_name": "ג׳ון בול", "gender": "male", "notes": "priest", "speaker_id": "SPEAKER_03"},
                        ],
                        "terms": [],
                        "register_notes": "",
                    })
                )
            payload = json.loads(contents)
            return types.SimpleNamespace(
                text=json.dumps({
                    "translations": [
                        {"id": it["id"], "text": "אדוני, אתה חייב לבוא עכשיו.", "addressee_form": "m"}
                        for it in payload["target_batch"]
                    ]
                })
            )

    translator = gp.GeminiTranslator(api_key="test-key", batch_size=18)
    translator.client = types.SimpleNamespace(models=_FakeGenAI())

    ctx = PipelineContext(
        job_id="job_5d93_bug1",
        video_path="The_Uprising_2026.mkv",
        file_name="The_Uprising_2026.mkv",
        enable_translation=True,
        source_language="en",
        target_language="Hebrew",
        speakers={
            "SPEAKER_01": {"label": "Speaker 1", "gender": "female", "gender_source": "voice", "voice_confidence": "high", "voice_f0": 246.2},
            "SPEAKER_02": {"label": "Speaker 2", "gender": "female", "gender_source": "voice", "voice_confidence": "high", "voice_f0": 222.2},
            "SPEAKER_03": {"label": "Speaker 3", "gender": "female", "gender_source": "voice", "voice_confidence": "medium", "voice_f0": 173.9},
        },
        refined_segments=[
            {"index": 1, "start_time": 1.0, "end_time": 2.8, "original_text": "My lord, you must come now.", "speaker": "SPEAKER_02", "segment_type": "DIALOGUE"},
            {"index": 2, "start_time": 3.2, "end_time": 5.0, "original_text": "I am your husband and your king, sir.", "speaker": "SPEAKER_01", "segment_type": "DIALOGUE"},
            {"index": 3, "start_time": 5.5, "end_time": 7.2, "original_text": "Stand ready, Father.", "speaker": "SPEAKER_01", "segment_type": "DIALOGUE"},
            {"index": 4, "start_time": 7.6, "end_time": 9.4, "original_text": "God be with you, my son.", "speaker": "SPEAKER_03", "segment_type": "DIALOGUE"},
        ],
    )
    await TranslateStage(translator=translator).execute(ctx)

    assert ctx.speakers["SPEAKER_01"]["gender"] == "male"
    assert ctx.speakers["SPEAKER_02"]["gender"] == "male"
    assert ctx.speakers["SPEAKER_03"]["gender"] == "male"
    assert not any("GENDER CHECK" in sys_p for _, sys_p in calls)


@pytest.mark.asyncio
async def test_bug2_gap_recovery_strict_thresholds_watermarks_and_initial_prompt_both_passes(tmp_path, monkeypatch):
    """Ensure gap recovery rejects watermarks (`© BF-WATCH TV 2021`), high no_speech (>=0.30), low logprob (<=-0.55),
    and low word-prob (<=0.65), and forwards `initial_prompt` (user glossary or title metadata) to both passes."""
    import wave
    import numpy as np
    from app.pipeline import gap_recovery as gr
    from app.pipeline.context import RawSegment

    # Watermarks & broadcast hallucinations caught
    for bad in ("© BF-WATCH TV 2021", "Subtitles by OpenSubtitles", "Transcript by CastingWords"):
        assert gr.is_watermark_hallucination(bad) is True
        assert gr.rejection_reason(bad, -0.2, 0.05, 1.0, [0.95] * 3, 1.5) is not None

    # Strict thresholds reject job_5d93f7cd9e's hallucinated noise lines
    assert gr.rejection_reason("You're a cunt, Buckethead!", -0.97, 0.20, 1.1, [0.80] * 4, 1.8).startswith("low confidence")
    assert gr.rejection_reason("And the vector shall pay for all", -0.40, 0.73, 1.1, [0.80] * 7, 2.2).startswith("probably not speech")
    assert gr.rejection_reason("Marcy, please!", -0.40, 0.15, 1.0, [0.60, 0.62], 1.2).startswith("unsure words")

    # Title metadata derivation when user leaves initial_prompt blank
    assert fw.derive_title_prompt("The.Uprising.2026.1080p.WEB-DL.x265.mkv") == "The Uprising (2026)"

    sr = 16000
    wav_path = tmp_path / "The.Uprising.2026.1080p.mkv.wav"
    audio = np.full(16 * sr, 0.05, dtype=np.float32)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())

    seen_calls = []

    class _PromptCheckModel:
        def transcribe(self, audio_arg, **kwargs):
            seen_calls.append(kwargs)
            if kwargs.get("vad_filter") is False:
                return iter([
                    types.SimpleNamespace(
                        start=6.0,
                        end=7.5,
                        text=" © BF-WATCH TV 2021",
                        avg_logprob=-0.25,
                        no_speech_prob=0.05,
                        compression_ratio=1.0,
                        words=[types.SimpleNamespace(word=" BF-WATCH", start=6.0, end=7.5, probability=0.9)],
                    ),
                    types.SimpleNamespace(
                        start=9.0,
                        end=10.8,
                        text=" Hold the bridge, Ploughman!",
                        avg_logprob=-0.30,
                        no_speech_prob=0.10,
                        compression_ratio=1.1,
                        words=[
                            types.SimpleNamespace(word=" Hold", start=9.0, end=9.4, probability=0.88),
                            types.SimpleNamespace(word=" the", start=9.4, end=9.6, probability=0.90),
                            types.SimpleNamespace(word=" bridge,", start=9.6, end=10.1, probability=0.86),
                            types.SimpleNamespace(word=" Ploughman!", start=10.1, end=10.8, probability=0.85),
                        ],
                    ),
                ]), None
            return iter([
                types.SimpleNamespace(
                    start=1.0,
                    end=2.5,
                    text=" © BF-WATCH TV 2021",
                    avg_logprob=-0.2,
                    words=[],
                ),
                types.SimpleNamespace(
                    start=3.0,
                    end=4.5,
                    text=" Stand fast!",
                    avg_logprob=-0.2,
                    words=[
                        types.SimpleNamespace(word=" Stand", start=3.0, end=3.7, probability=0.92),
                        types.SimpleNamespace(word=" fast!", start=3.7, end=4.5, probability=0.94),
                    ],
                ),
            ]), types.SimpleNamespace(duration=16.0, language="en")

    monkeypatch.setattr(fw, "_load_model", lambda *a: _PromptCheckModel())
    monkeypatch.setattr(fw, "get_hardware_info", lambda: {"cuda_available": False})

    transcriber = fw.FasterWhisperTranscriber(
        "large-v3",
        device="cpu",
        initial_prompt="John Ball, Ploughman, Corpus Christi",
        file_name="The.Uprising.2026.1080p.mkv",
    )
    main_segs = await transcriber.transcribe(str(wav_path), language="en")
    assert [s.text for s in main_segs] == ["Stand fast!"]  # watermark dropped in primary pass too!
    recovered = await transcriber.recover_missed_speech(str(wav_path), "en", main_segs)
    assert [r.text for r in recovered] == ["Hold the bridge, Ploughman!"]

    # Both primary pass and gap-recovery pass received `initial_prompt`
    assert len(seen_calls) == 2
    assert "Ploughman" in seen_calls[0]["initial_prompt"]
    assert "Ploughman" in seen_calls[1]["initial_prompt"]


@pytest.mark.asyncio
async def test_bug3_strict_batch_id_alignment_prevents_index_shift_and_preserves_concise_hebrew(monkeypatch):
    """When a model merges two lines in a batch and drops an ID, the misaligned batch is rejected as a whole
    and split cleanly into smaller halves rather than accepting off-by-one shifted translations."""
    monkeypatch.setattr(gp.asyncio, "sleep", lambda _: None)

    # Concise idiomatic Hebrew (e.g. 10-13 chars for 30-35 char English) must NOT be flagged as truncated
    assert gp._looks_truncated("Keep your eyes open and stay quiet.", "פקחו עיניים!") is False
    assert gp._looks_truncated("Stand by for exact coordinates. Nishioka, eyes on.", "המתן לנ") is True

    calls = []

    class _ShiftingOnLargeBatchModels:
        def generate_content(self, model, contents, config):
            payload = json.loads(contents)
            batch = payload["target_batch"]
            calls.append([it["id"] for it in batch])
            if len(batch) >= 10:
                # Simulate model merging first two lines and shifting every subsequent translation by 1!
                shifted = []
                for pos in range(len(batch) - 1):
                    shifted.append({
                        "id": batch[pos]["id"],
                        "text": f"WRONG_SHIFTED_FROM_{batch[pos + 1]['id']}",
                    })
                return types.SimpleNamespace(text=json.dumps({"translations": shifted}))
            return types.SimpleNamespace(
                text=json.dumps({
                    "translations": [
                        {"id": it["id"], "text": f"תרגום מדויק #{it['id']}"}
                        for it in batch
                    ]
                })
            )

    translator = gp.GeminiTranslator(api_key="test-key", batch_size=12)
    translator.client = types.SimpleNamespace(models=_ShiftingOnLargeBatchModels())

    segs = [
        TranslatableSegment(index=i, start_time=float(i * 2), end_time=float(i * 2 + 1.5), original_text=f"Dialogue line number {i}.")
        for i in range(1, 13)
    ]
    await translator.translate_segments(segs, "English", "Hebrew")

    # Misaligned 12-item response was rejected completely, split into 6 + 6, and every ID matches 1:1
    assert calls == [list(range(1, 13)), list(range(1, 7)), list(range(7, 13))]
    for s in segs:
        assert s.translated_text == f"תרגום מדויק #{s.index}"
        assert not s.translation_failed


@pytest.mark.asyncio
async def test_bug4_no_single_word_subsecond_splits_in_diarization_or_post_translation(monkeypatch):
    """Short 2-3 word lines must wrap with `\\n` inside one subtitle block rather than splitting into
    0.7s single-word cues (`'לך'` / `'אל אביך.'`, `'תסתכל'` / `'על'` / `'לי!'`), and sub-1.0s fragments
    with gap <= 0.35s merge into their neighbor with >= 1.0s duration."""
    from app.pipeline import diarization
    from app.pipeline.context import RawSegment, TranscribedWord
    from app.utils.speaker_split import split_dual_speaker_segments

    # 4a. Acoustic diarization flicker inside a short phrase ("Look at me!") does NOT chop into 1-word cues
    timeline = [
        {"start": 0.0, "end": 0.35, "speaker": "SPEAKER_01"},
        {"start": 0.35, "end": 0.70, "speaker": "SPEAKER_02"},
        {"start": 0.70, "end": 1.20, "speaker": "SPEAKER_01"},
    ]
    raw = [
        RawSegment(
            index=1,
            start_time=0.0,
            end_time=1.2,
            text="Look at me!",
            words=[
                TranscribedWord("Look", 0.0, 0.35),
                TranscribedWord(" at", 0.35, 0.70),
                TranscribedWord(" me!", 0.70, 1.20),
            ],
        )
    ]
    unsplit = diarization.split_raw_segments_by_speaker(raw, timeline)
    assert len(unsplit) == 1
    assert unsplit[0].text == "Look at me!"

    # 4b. Short single-sentence segment with bogus dashes stays ONE subtitle block instead of time-slicing
    short_dashed = [
        {
            "index": 1,
            "start_time": 10.0,
            "end_time": 11.4,
            "original_text": "Go to your father.",
            "translated_text": "- לך\n- אל אביך.",
            "segment_type": "DIALOGUE",
        }
    ]
    kept_single = split_dual_speaker_segments(short_dashed)
    assert len(kept_single) == 1
    assert kept_single[0]["translated_text"] == "לך אל אביך."

    # 4c. Pre-existing sub-1.0s single-word fragments ("תסתכל" / "על" / "לי!") with gap <= 0.35s merge in TranslateStage
    monkeypatch.setattr(gp.asyncio, "sleep", lambda _: None)
    from app.core.config import settings
    monkeypatch.setattr(settings, "gemini_glossary_pass", False)

    class _LiteralStub:
        def generate_content(self, model, contents, config):
            payload = json.loads(contents)
            mapping = {1: "תסתכל", 2: "על", 3: "לי!", 4: "לך", 5: "אל אביך."}
            return types.SimpleNamespace(
                text=json.dumps({
                    "translations": [
                        {"id": it["id"], "text": mapping[it["id"]], "addressee_form": "m"}
                        for it in payload["target_batch"]
                    ]
                })
            )

    translator = gp.GeminiTranslator(api_key="test-key", batch_size=18)
    translator.client = types.SimpleNamespace(models=_LiteralStub())

    ctx = PipelineContext(
        job_id="job_5d93_bug4",
        video_path="uprising.mkv",
        file_name="uprising.mkv",
        enable_translation=True,
        source_language="en",
        target_language="Hebrew",
        refined_segments=[
            {"index": 1, "start_time": 1.00, "end_time": 1.45, "original_text": "Look", "segment_type": "DIALOGUE"},
            {"index": 2, "start_time": 1.60, "end_time": 1.95, "original_text": "at", "segment_type": "DIALOGUE"},
            {"index": 3, "start_time": 2.10, "end_time": 2.65, "original_text": "me!", "segment_type": "DIALOGUE"},
            {"index": 4, "start_time": 5.00, "end_time": 5.55, "original_text": "Go", "segment_type": "DIALOGUE"},
            {"index": 5, "start_time": 5.75, "end_time": 6.60, "original_text": "to your father.", "segment_type": "DIALOGUE"},
        ],
    )
    await TranslateStage(translator=translator).execute(ctx)

    assert len(ctx.translated_segments) == 2
    assert ctx.translated_segments[0]["translated_text"] == "תסתכל על לי!"
    assert ctx.translated_segments[1]["translated_text"] == "לך אל אביך."
    assert all(seg["end_time"] - seg["start_time"] >= 1.0 - 1e-3 for seg in ctx.translated_segments)


