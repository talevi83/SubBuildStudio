"""Gender in translation, checked on a real video: tests/fixtures/media/conversation_2_genders.mp4.

Two people talk in an office kitchenette. Nothing in the words says who is a man or a woman, so
for Hebrew ("did you grab" -> to a woman; "your notebook" -> to a man) the app has to get it from
the voices: diarization must keep the two speakers apart, voice pitch must give each a gender,
and the listener resolver must turn that into who each line is spoken to.

The transcript comes from the scene's script (conversation_2_genders.json), so no Whisper model is
needed; the audio, the speaker split and the pitch all come from the video itself.
"""
import json
import shutil
import subprocess
import types
import uuid
from pathlib import Path

import pytest

from app.core.config import settings
from app.pipeline import diarization
from app.pipeline.addressee import HIGH, resolve_addressees
from app.pipeline.context import PipelineContext
from app.pipeline.stages.translate_stage import TranslateStage
from app.pipeline.voice_gender import estimate_speaker_genders, label_speakers_by_voice
from app.providers.translation import gemini_provider as gp

MEDIA = Path(__file__).resolve().parents[1] / "fixtures" / "media"
SCENE = json.loads((MEDIA / "conversation_2_genders.json").read_text(encoding="utf-8"))
VIDEO = MEDIA / SCENE["video"]
# The speaker model is not in the repo; use the copy the app downloaded, when there is one
CACHED_MODEL = Path(__file__).resolve().parents[2] / "storage" / "models" / diarization.MODEL_FILENAME

HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
needs_model = pytest.mark.skipif(
    not (CACHED_MODEL.is_file() and CACHED_MODEL.stat().st_size > 1_000_000),
    reason="speaker diarization model is not cached in backend/storage/models",
)

# Ground truth of the scene: speaker letter -> the id the app gives the first/second voice
SCRIPT_IDS = {"A": "SPEAKER_01", "B": "SPEAKER_02"}


@pytest.fixture(scope="module")
def scene_wav(tmp_path_factory) -> Path:
    """The video's audio as the pipeline uses it: 16 kHz mono 16-bit WAV."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    wav = tmp_path_factory.mktemp("scene") / "conversation_2_genders.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(VIDEO), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True, timeout=60,
    )
    return wav


def _script_timeline():
    return [{"start": ln["start"], "end": ln["end"], "speaker": SCRIPT_IDS[ln["speaker"]]} for ln in SCENE["lines"]]


def _script_segments(speaker_of_line):
    return [
        {
            "index": i,
            "start_time": ln["start"],
            "end_time": ln["end"],
            "original_text": ln["text"],
            "translated_text": None,
            "translation_failed": False,
            "segment_type": "DIALOGUE",
            "speaker": speaker_of_line(ln),
        }
        for i, ln in enumerate(SCENE["lines"], start=1)
    ]


def test_the_resource_matches_its_script():
    assert VIDEO.is_file() and VIDEO.stat().st_size > 100_000
    assert {ln["speaker"] for ln in SCENE["lines"]} == set(SCENE["speakers"])
    # The point of the scene: no name and no he/she to lean on
    for ln in SCENE["lines"]:
        words = set(ln["text"].lower().replace(",", " ").replace("?", " ").replace(".", " ").split())
        assert not words & {"he", "she", "him", "her", "his", "hers", "sir", "ma'am"}


@needs_ffmpeg
def test_voice_pitch_tells_the_man_from_the_woman(scene_wav):
    samples, sr = diarization._read_wav_float32(scene_wav)
    estimates = estimate_speaker_genders(samples, sr, _script_timeline())
    for letter, info in SCENE["speakers"].items():
        est = estimates[SCRIPT_IDS[letter]]
        assert est["gender"] == info["gender"], (letter, est)
        assert est["confidence"] == "high"
    assert 95 <= estimates["SPEAKER_01"]["median_f0"] <= 150     # low baritone
    assert 200 <= estimates["SPEAKER_02"]["median_f0"] <= 280    # high female voice


@needs_ffmpeg
@needs_model
def test_diarization_keeps_two_speakers_and_labels_them_by_voice(scene_wav):
    embedder = diarization.OnnxSpeakerEmbedder(onnx_path=CACHED_MODEL)
    spans = [(ln["start"], ln["end"]) for ln in SCENE["lines"]]
    timeline, speakers = diarization.diarize_audio_spans(scene_wav, spans, embedder=embedder)
    # Each voice has a single line here; neither may be swallowed by the other as "noise"
    assert len(speakers) == 2

    label_speakers_by_voice(scene_wav, timeline, speakers)
    segments = _script_segments(lambda ln: diarization._dominant_speaker(ln["start"], ln["end"], timeline))
    by_letter = {ln["speaker"]: seg["speaker"] for ln, seg in zip(SCENE["lines"], segments)}
    assert by_letter["A"] != by_letter["B"]
    for letter, info in SCENE["speakers"].items():
        profile = speakers[by_letter[letter]]
        assert profile["gender"] == info["gender"] and profile["gender_source"] == "voice"

    records = resolve_addressees(segments, speakers, [])
    for i, ln in enumerate(SCENE["lines"], start=1):
        listener = SCENE["speakers"][ln["addressee"]]["gender"]
        assert records[i].target == by_letter[ln["addressee"]] and records[i].gender == listener
        assert records[i].confidence >= HIGH


@needs_ffmpeg
@needs_model
@pytest.mark.asyncio
async def test_the_diarization_stage_hands_on_speakers_with_a_gender(scene_wav, monkeypatch):
    from app.pipeline.context import RawSegment
    from app.pipeline.stages.diarize_stage import DiarizeStage

    monkeypatch.setattr(diarization, "model_path", lambda: CACHED_MODEL)
    ctx = PipelineContext(
        job_id=f"job_{uuid.uuid4().hex[:6]}", video_path=str(VIDEO), file_name=VIDEO.name,
        audio_wav_path=str(scene_wav), enable_diarization=True,
        raw_segments=[RawSegment(index=i, start_time=ln["start"], end_time=ln["end"], text=ln["text"])
                      for i, ln in enumerate(SCENE["lines"], start=1)],
    )
    await DiarizeStage().execute(ctx)
    assert not ctx.warnings
    by_line = [seg.speaker for seg in ctx.raw_segments]
    assert len(set(by_line)) == 2
    for seg, ln in zip(ctx.raw_segments, SCENE["lines"]):
        assert ctx.speakers[seg.speaker]["gender"] == SCENE["speakers"][ln["speaker"]]["gender"]


def _translator(fake_generate, monkeypatch) -> gp.GeminiTranslator:
    monkeypatch.setattr(gp.asyncio, "sleep", lambda _: None)
    monkeypatch.setattr(settings, "gemini_glossary_pass", False)
    translator = gp.GeminiTranslator(api_key="test-key")
    translator.client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=fake_generate))
    return translator


@needs_ffmpeg
@pytest.mark.asyncio
async def test_translator_is_told_who_speaks_to_whom_and_a_wrong_you_is_fixed(scene_wav, monkeypatch):
    speakers = {sid: {"label": f"Speaker {n}", "gender": "unknown", "color": "#38BDF8"}
                for n, sid in enumerate(SCRIPT_IDS.values(), start=1)}
    label_speakers_by_voice(scene_wav, _script_timeline(), speakers)

    payloads, prompts = [], []

    def fake_generate(model, contents, config):
        payload = json.loads(contents)
        payloads.append(payload)
        prompts.append(config.system_instruction)
        fixing = "GENDER CHECK" in config.system_instruction
        out = []
        for item in payload["target_batch"]:
            if item["id"] == 1:
                # First try: a masculine "you" for a line said to the woman
                text = "היי, את לקחת את הקפה שלי מהשיש?" if fixing else "היי, אתה לקחת את הקפה שלי מהשיש?"
                out.append({"index": 1, "translated_text": text, "addressee_form": "f" if fixing else "m"})
            else:
                out.append({"index": item["id"], "translated_text": "כן, השארתי אותו ממש ליד המחברת שלך.",
                            "addressee_form": "m"})
        return types.SimpleNamespace(text=json.dumps({"translations": out}))

    ctx = PipelineContext(
        job_id=f"job_{uuid.uuid4().hex[:6]}", video_path=str(VIDEO), file_name=VIDEO.name,
        enable_translation=True, target_language="Hebrew", source_language=SCENE["source_language"],
        speakers=speakers, refined_segments=_script_segments(lambda ln: SCRIPT_IDS[ln["speaker"]]),
    )
    await TranslateStage(translator=_translator(fake_generate, monkeypatch)).execute(ctx)

    first = {item["id"]: item for item in payloads[0]["target_batch"]}
    # The man asks the woman; the woman answers the man — given as facts, not guesses
    # Genders come from the voice here, so the translator is told they are estimates
    assert first[1]["speaker"] == "SPEAKER_01 (male by voice)" and first[1]["addressee"] == "SPEAKER_02 (female)"
    assert first[2]["speaker"] == "SPEAKER_02 (female by voice)" and first[2]["addressee"] == "SPEAKER_01 (male)"
    assert "`addressee_form`" in prompts[0]

    # The masculine "you" said to the woman was caught and translated once more, for that line only
    assert len(payloads) == 2 and [it["id"] for it in payloads[1]["target_batch"]] == [1]
    rows = {s["index"]: s for s in ctx.refined_segments}
    assert rows[1]["translated_text"] == "היי, את לקחת את הקפה שלי מהשיש?"
    assert rows[1]["addressee"]["gender"] == "female" and rows[1]["addressee"]["used"] == "f"
    assert rows[2]["addressee"]["gender"] == "male" and rows[2]["addressee"]["used"] == "m"
