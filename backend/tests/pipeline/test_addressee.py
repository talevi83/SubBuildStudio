"""Who each line is spoken to: the resolver, how its confidence reaches the translator, the
re-check of translations that contradict the listener, and the tag stored on each row."""
import json
import time
import types
import uuid
from typing import List

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core import edit_history
from app.core.config import settings
from app.core.database import engine
from app.main import app
from app.models.db_models import Job, JobStatus, SubtitleSegmentRecord
from app.pipeline.addressee import (
    HIGH,
    AddresseeRecord,
    manual_record,
    merge_local_guess,
    needs_listener_gender,
    resolve_addressees,
    split_into_exchanges,
)
from app.pipeline.context import PipelineContext
from app.pipeline.gender_pretag import build_scene_batches, pretag_gender_hints
from app.pipeline.qa import check_segment, listener_form_mismatch, second_person_forms
from app.pipeline.stages.translate_stage import TranslateStage
from app.providers.translation import gemini_provider as gp
from app.providers.translation.base import BaseTranslator, TranslatableSegment

SPEAKERS = {
    "SPEAKER_01": {"label": "דנה", "gender": "female", "color": "#F472B6"},
    "SPEAKER_02": {"label": "אבי", "gender": "male", "color": "#38BDF8"},
    "SPEAKER_03": {"label": "שרה", "gender": "female", "color": "#4ADE80"},
}
CHARACTERS = [
    {"name": "Dana", "target_name": "דנה", "gender": "female", "speaker_id": "SPEAKER_01"},
    {"name": "Avi", "target_name": "אבי", "gender": "male", "speaker_id": "SPEAKER_02"},
    {"name": "Sarah", "target_name": "שרה", "gender": "female", "speaker_id": "SPEAKER_03"},
]


def _lines(*rows):
    """(speaker number or None, text[, gap before in seconds]) -> dialogue segments 1..N."""
    out, t = [], 0.0
    for i, row in enumerate(rows, start=1):
        sp, text = row[0], row[1]
        t += row[2] if len(row) > 2 else 0.4
        out.append({
            "index": i, "start_time": t, "end_time": t + 1.5, "original_text": text,
            "speaker": f"SPEAKER_{sp:02d}" if sp else None, "segment_type": "DIALOGUE",
        })
        t += 1.5
    return out


# ---------------------------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Are you coming?", True),
    ("Come here.", True),
    ("Sit down, please.", True),
    ("Look, I don't know.", True),
    ("Ready?", True),
    ("Let's go.", True),
    ("Please, just stop.", True),
    ("The ship is moving.", False),
    ("Thank you.", False),
    ("Do I know him?", False),
    ("I'm tired.", False),
    ("", False),
])
def test_lines_that_need_the_listeners_gender(text, expected):
    assert needs_listener_gender(text) is expected


def test_a_named_listener_stays_the_listener_on_the_speakers_next_lines():
    segments = _lines(
        (1, "Sarah, are you ready?"),
        (3, "Almost."),
        (1, "Take your time."),
        (1, "Call me when you are done."),
    )
    records = resolve_addressees(segments, SPEAKERS, CHARACTERS)
    assert records[1].source == "vocative" and records[1].target == "SPEAKER_03"
    # Sarah answers the one who called her by name
    assert records[2].target == "SPEAKER_01" and records[2].tier == "high"
    # ...and is still the listener two lines later, a little less surely each line
    assert records[3].target == "SPEAKER_03" and records[3].form == "f" and records[3].tier == "high"
    assert records[4].target == "SPEAKER_03" and records[4].tier == "high"
    assert records[4].confidence <= records[3].confidence <= records[1].confidence


def test_a_third_person_speaking_in_between_breaks_the_carry():
    segments = _lines(
        (1, "Sarah, wait for me."),
        (3, "Fine."),
        (2, "What about me?"),
        (1, "You stay here."),
    )
    records = resolve_addressees(segments, SPEAKERS, CHARACTERS)
    # Avi has just asked, so the answer is to him, not to Sarah who was named three lines ago
    assert records[4].target == "SPEAKER_02" and records[4].form == "m"


def test_group_address_carries_forward_only_and_never_onto_yourself():
    segments = _lines(
        (1, "Are you coming with me?"),
        (2, "In a minute."),
        (1, "Listen up, everyone!"),
        (1, "We leave at dawn, so get some sleep."),
        (1, "Avi, behave yourself."),
    )
    records = resolve_addressees(segments, SPEAKERS, CHARACTERS)
    assert records[1].number != "plural"            # said before the group was addressed
    assert records[3].form == "pl" and records[3].tier == "high"
    assert records[4].form == "pl" and records[4].tier == "medium"  # likely still everyone: a guess
    assert records[5].form == "m" and records[5].target == "SPEAKER_02"


def test_one_sentence_cut_over_two_subtitles_has_one_listener():
    segments = _lines(
        (None, "I already told you"),
        (None, "to leave it alone, sir.", 0.1),
        (None, "Nothing happened here.", 20.0),
    )
    records = resolve_addressees(segments, {}, [])
    assert records[2].form == "m" and records[2].source == "honorific"
    assert records[1].form == "m" and records[1].tier == "high"  # no speakers needed for this
    assert 3 not in records


def test_two_person_exchange_and_long_silences():
    segments = _lines(
        (1, "Where were you?"),
        (2, "Out."),
        (1, "Tell me the truth."),
        (3, "Is anybody home?", 30.0),
    )
    assert split_into_exchanges(segments) == [(0, 3), (3, 4)]
    records = resolve_addressees(segments, SPEAKERS, CHARACTERS)
    assert [records[i].target for i in (1, 2, 3)] == ["SPEAKER_02", "SPEAKER_01", "SPEAKER_02"]
    assert all(records[i].tier == "high" for i in (1, 2, 3))
    assert 4 not in records  # alone in her exchange: nobody to name


def test_an_unknown_neighbour_does_not_hand_the_line_to_the_other_one():
    speakers = {
        "SPEAKER_01": {"label": "Speaker 1", "gender": "male"},
        "SPEAKER_02": {"label": "Speaker 2", "gender": "unknown"},
        "SPEAKER_03": {"label": "Speaker 3", "gender": "female"},
    }
    segments = _lines(
        (2, "What do you want?"),
        (1, "I need you to listen."),       # the answer to speaker 2, whose gender nobody knows
        (3, "Leave him alone."),
        (1, "Stay out of this, Mrs. Cole."),
    )
    records = resolve_addressees(segments, speakers, [])
    assert 2 not in records                  # not "female" just because speaker 3 talks next
    assert records[4].form == "f" and records[4].source == "honorific" and records[4].tier == "high"


def test_weak_cues_are_guesses_and_endearments_are_not_even_that():
    segments = _lines((None, "Calm down, dude."), (None, "Come on, honey.", 20.0), (None, "Yes, ma'am.", 20.0))
    records = resolve_addressees(segments, {}, [])
    assert records[1].form == "m" and records[1].tier == "medium"
    assert records[2].tier == "low"
    assert records[3].form == "f" and records[3].tier == "high"


def test_a_listener_set_by_hand_wins_and_spreads_to_the_next_lines():
    segments = _lines((1, "Are you sure?"), (1, "Then go."), (2, "I am going."))
    segments[0]["addressee"] = manual_record("female").to_dict()
    records = resolve_addressees(segments, SPEAKERS, CHARACTERS)
    assert records[1].locked and records[1].form == "f" and records[1].confidence == 1.0
    assert records[2].form == "f" and records[2].source == "carry" and not records[2].locked
    # the same through the explicit argument
    again = resolve_addressees(_lines((1, "Are you sure?")), SPEAKERS, CHARACTERS, manual={1: manual_record("group")})
    assert again[1].form == "pl" and again[1].display() == "group (plural)"


def test_record_round_trip_and_display():
    rec = AddresseeRecord(gender="female", number="singular", target="SPEAKER_03", label="שרה",
                          confidence=0.9512, source="vocative", used="f")
    data = rec.to_dict()
    assert data == {"gender": "female", "number": "singular", "confidence": 0.95, "source": "vocative",
                    "target": "SPEAKER_03", "label": "שרה", "used": "f"}
    back = AddresseeRecord.from_dict(data)
    assert back.form == "f" and back.tier == "high" and back.display(SPEAKERS) == "SPEAKER_03 (שרה, female)"
    assert AddresseeRecord.from_dict({"gender": "nonsense", "confidence": "x"}).form is None
    assert AddresseeRecord.from_dict("oops") is None
    assert AddresseeRecord(gender="female", number="plural").display() == "group (plural, all female)"
    assert manual_record("nobody") is None


def test_local_model_is_a_second_opinion_not_a_fact():
    records = {
        1: AddresseeRecord(gender="male", number="singular", confidence=0.95, source="vocative"),
        2: AddresseeRecord(gender="male", number="singular", confidence=0.55, source="turn"),
        3: AddresseeRecord(gender="male", number="singular", confidence=0.55, source="turn"),
    }
    merge_local_guess(records, 1, "female")     # a confident record is left alone
    merge_local_guess(records, 2, "male")       # agreement makes a guess surer
    merge_local_guess(records, 3, "female")     # disagreement leaves a weaker guess
    merge_local_guess(records, 4, "group (plural)")
    merge_local_guess(records, 5, None)
    # a listener known by name but not by gender keeps the name; the gender is the model's guess
    records[6] = AddresseeRecord(number="singular", label="אלכס", confidence=0.9, source="vocative")
    merge_local_guess(records, 6, "female")
    assert records[6].label == "אלכס" and records[6].form == "f" and records[6].tier == "medium"
    assert records[1].form == "m" and records[1].confidence == 0.95
    assert records[2].form == "m" and records[2].confidence > 0.55
    assert records[3].form == "f" and records[3].confidence < HIGH
    assert records[4].form == "pl" and records[4].tier == "medium" and records[4].source == "local_ai"
    assert 5 not in records


# ---------------------------------------------------------------------------------------------
# The local model reads lines in their scene
# ---------------------------------------------------------------------------------------------

def test_scene_batches_carry_the_dialogue_around_each_asked_line():
    segs = [TranslatableSegment(i, i * 2.0, i * 2.0 + 1.5, f"line {i}", speaker="SPEAKER_01") for i in range(1, 21)]
    batches = build_scene_batches([10], segs, speakers=SPEAKERS, known_to={9: "F"}, context_lines=2)
    assert len(batches) == 1 and batches[0]["ask"] == [10]
    lines = batches[0]["lines"]
    assert [ln["index"] for ln in lines] == [8, 9, 10, 11, 12]
    assert [ln["index"] for ln in lines if ln.get("ask")] == [10]
    assert lines[1]["to"] == "F" and lines[0]["speaker"] == "דנה (female)"
    # far-apart lines do not blow one request up past its line budget
    many = build_scene_batches(list(range(1, 21, 4)), segs, context_lines=3, max_scene_lines=10)
    assert len(many) > 1 and all(len(b["lines"]) <= 10 for b in many)


@pytest.mark.asyncio
async def test_local_pretag_asks_only_about_open_lines_and_ignores_answers_it_did_not_ask_for(monkeypatch):
    posted = []

    async def _fake_get(self, url, **kwargs):
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})

    async def _fake_post(self, url, json=None, **kwargs):
        posted.append(json)
        return httpx.Response(200, json={"response": '{"items": ['
                                                     '{"index": 1, "speaker": "F", "to": "M"},'
                                                     '{"index": 2, "speaker": "?", "to": "F"},'
                                                     '{"index": 3, "speaker": "?", "to": "F"}]}'})

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    segs = [
        TranslatableSegment(1, 0, 2, "Sarah, are you ready?"),
        TranslatableSegment(2, 2, 4, "Come here."),
        TranslatableSegment(3, 4, 6, "The door is open."),
    ]
    known = {1: AddresseeRecord(gender="female", number="singular", confidence=0.95, source="vocative")}
    # Without a glossary nothing grounds the model's "F" for line 2: the guess is dropped
    assert await pretag_gender_hints(segs, target_lang="Hebrew", addressees=known) == {}
    posted.clear()
    glossary = {"characters": [{"name": "Sarah", "gender": "female"}]}
    hints = await pretag_gender_hints(segs, target_lang="Hebrew", addressees=known, glossary=glossary)
    # line 1 is settled, line 3 has nothing gendered: only the imperative was asked, with its scene;
    # its "F" is kept because Sarah (female) is named on the line before it
    assert hints == {2: "[Speaker: ?, To: F]"}
    assert len(posted) == 1
    sent = json.loads(posted[0]["prompt"].split("Lines: ", 1)[1])
    assert [ln["index"] for ln in sent] == [1, 2, 3]
    assert [ln["index"] for ln in sent if ln.get("ask")] == [2] and sent[0]["to"] == "F"

    posted.clear()
    all_known = {**known, 2: AddresseeRecord(gender="male", number="singular", confidence=0.9, source="honorific")}
    assert await pretag_gender_hints(segs, target_lang="Hebrew", addressees=all_known) == {}
    assert posted == []  # nothing left to ask: no request at all


# ---------------------------------------------------------------------------------------------
# How the translator is told, and the re-check of what it wrote
# ---------------------------------------------------------------------------------------------

def test_hebrew_forms_and_listener_mismatch():
    assert second_person_forms("אתה בא?") == {"m"}
    assert second_person_forms("בואי הנה.") == {"f"}
    assert second_person_forms("אתם מוכנים?") == {"pl"}
    assert second_person_forms("ראיתי את דני") == set()      # the object marker, not "you"
    assert second_person_forms("היא תגיד לו") == set()       # 3rd person feminine
    assert listener_form_mismatch("f", "אתה בא?") == "m"
    assert listener_form_mismatch("f", "בואי הנה.", "m") is None   # the text itself wins over the report
    assert listener_form_mismatch("f", "מה שלומך?", "m") == "m"    # unreadable text: the report counts
    assert listener_form_mismatch("f", "מה שלומך?") is None
    assert listener_form_mismatch(None, "אתה בא?") is None

    sure = {"gender": "female", "number": "singular", "confidence": 0.95, "source": "vocative", "label": "שרה"}
    issues = check_segment(1, 1, "Sarah, are you coming?", "שרה, אתה בא?", "DIALOGUE", False, None, addressee=sure)
    assert [i.kind for i in issues] == ["agreement"] and "שרה" in issues[0].message
    assert check_segment(1, 1, "Sarah, are you coming?", "שרה, את באה?", "DIALOGUE", False, None, addressee=sure) == []
    guess = json.dumps({**sure, "confidence": 0.6})
    assert check_segment(1, 1, "Are you coming?", "אתה בא?", "DIALOGUE", False, None, addressee=guess) == []


def _translator_with(fake_generate, monkeypatch, **kw) -> gp.GeminiTranslator:
    monkeypatch.setattr(gp.asyncio, "sleep", lambda _: None)
    monkeypatch.setattr(settings, "gemini_glossary_pass", False)
    translator = gp.GeminiTranslator(api_key="test-key", **kw)
    translator.client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=fake_generate))
    return translator


def _ctx(segments, **kw) -> PipelineContext:
    for s in segments:
        s.setdefault("translated_text", None)
        s.setdefault("translation_failed", False)
    return PipelineContext(
        job_id=f"job_{uuid.uuid4().hex[:6]}", video_path="clip.mp4", file_name="clip.mp4",
        enable_translation=True, target_language="Hebrew", source_language="en",
        glossary={"summary": "", "characters": CHARACTERS, "terms": [], "register_notes": ""},
        speakers={k: dict(v) for k, v in SPEAKERS.items()}, refined_segments=segments, **kw,
    )


@pytest.mark.asyncio
async def test_translator_gets_facts_guesses_or_nothing_and_rows_keep_their_record(monkeypatch):
    payloads, instructions = [], []

    def fake_generate(model, contents, config):
        instructions.append(config.system_instruction)
        payload = json.loads(contents)
        payloads.append(payload)
        return types.SimpleNamespace(text=json.dumps({"translations": [
            {"index": it["id"], "translated_text": f"תרגום {it['id']}", "addressee_form": "f" if it["id"] == 1 else ""}
            for it in payload["target_batch"]
        ]}))

    segments = _lines(
        (1, "Sarah, are you ready?"),            # named: a fact
        (3, "Almost."),
        (2, "Calm down, dude.", 30.0),           # weak cue, nobody around: a guess
        (None, "Come here.", 30.0),              # needs a listener, nothing known
        (None, "The door is open.", 30.0),       # nothing gendered at all
    )
    ctx = _ctx(segments)
    await TranslateStage(translator=_translator_with(fake_generate, monkeypatch)).execute(ctx)

    batch = {it["id"]: it for it in payloads[0]["target_batch"]}
    assert batch[1]["addressee"] == "SPEAKER_03 (שרה, female)" and "addressee_guess" not in batch[1]
    assert batch[3]["addressee_guess"] == "male" and "addressee" not in batch[3]
    assert "addressee" not in batch[4] and "addressee_guess" not in batch[4]
    assert "`addressee_guess`" in instructions[0] and "`addressee_form`" in instructions[0]

    stored = {s["index"]: s.get("addressee") for s in ctx.refined_segments}
    assert stored[1]["source"] == "vocative" and stored[1]["used"] == "f" and stored[1]["confidence"] >= HIGH
    assert stored[3]["gender"] == "male" and "used" not in stored[3]
    assert stored[4] == {"gender": "unknown", "number": "unknown", "confidence": 0.0, "source": "unresolved"}
    assert stored[5] is None


@pytest.mark.asyncio
async def test_a_translation_that_contradicts_its_listener_is_translated_once_more(monkeypatch):
    calls = []

    def fake_generate(model, contents, config):
        payload = json.loads(contents)
        calls.append((config.system_instruction, [it["id"] for it in payload["target_batch"]]))
        fixing = "GENDER CHECK" in config.system_instruction
        out = []
        for it in payload["target_batch"]:
            if it["id"] == 1:
                out.append({"index": 1, "translated_text": "שרה, את מוכנה? בואי." if fixing else "שרה, אתה מוכן?"})
            else:
                out.append({"index": it["id"], "translated_text": "כמעט סיימתי."})
        return types.SimpleNamespace(text=json.dumps({"translations": out}))

    persisted: List[List[int]] = []

    async def on_persist(batch):
        persisted.append([b.index for b in batch])

    ctx = _ctx(_lines((1, "Sarah, are you ready?"), (3, "Almost done.")))
    stage = TranslateStage(translator=_translator_with(fake_generate, monkeypatch), on_batch_persist=on_persist)
    await stage.execute(ctx)

    assert [idx for _, idx in calls] == [[1, 2], [1]]      # one extra request, for the wrong line only
    assert ctx.refined_segments[0]["translated_text"] == "שרה, את מוכנה? בואי."
    assert persisted == [[1, 2], [1]]

    # A retry that is still wrong does not replace the first translation, and the switch turns it off
    def always_wrong(model, contents, config):
        payload = json.loads(contents)
        calls.append((config.system_instruction, [it["id"] for it in payload["target_batch"]]))
        return types.SimpleNamespace(text=json.dumps({"translations": [
            {"index": it["id"], "translated_text": "אתה מוכן?" if it["id"] == 1 else "כמעט."}
            for it in payload["target_batch"]
        ]}))

    calls.clear()
    ctx = _ctx(_lines((1, "Sarah, are you ready?"), (3, "Almost done.")))
    await TranslateStage(translator=_translator_with(always_wrong, monkeypatch)).execute(ctx)
    assert len(calls) == 2 and ctx.refined_segments[0]["translated_text"] == "אתה מוכן?"

    calls.clear()
    monkeypatch.setattr(settings, "addressee_autofix", False)
    ctx = _ctx(_lines((1, "Sarah, are you ready?"), (3, "Almost done.")))
    await TranslateStage(translator=_translator_with(always_wrong, monkeypatch)).execute(ctx)
    assert len(calls) == 1


# ---------------------------------------------------------------------------------------------
# The tag stored on each row
# ---------------------------------------------------------------------------------------------

class _PlainTranslator(BaseTranslator):
    async def translate_segments(self, segments, source_lang, target_lang, context_notes="", apply_rtl=False,
                                 on_batch_complete=None, glossary=None, context_segments=None):
        for s in segments:
            s.translated_text, s.translation_failed = f"חדש {s.index}", False
        if on_batch_complete:
            await on_batch_complete(segments, 100.0)
        return segments


def _job(texts: List[str]) -> str:
    job_id = f"test_{uuid.uuid4().hex[:8]}"
    with Session(engine) as session:
        session.add(Job(id=job_id, video_path="x.mkv", file_name="x.mkv", status=JobStatus.COMPLETED,
                        target_language="Hebrew"))
        session.commit()
        for i, t in enumerate(texts, start=1):
            session.add(SubtitleSegmentRecord(job_id=job_id, index=i, start_time=i * 2.0, end_time=i * 2.0 + 1.5,
                                              original_text=t, translated_text=f"ישן {i}"))
        session.commit()
    return job_id


def _wait(client, job_id, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_listener_can_be_set_and_cleared_on_a_row_and_undone():
    with TestClient(app) as client:
        job_id = _job(["Are you coming?", "Yes."])
        seg = client.get(f"/api/jobs/{job_id}").json()["segments"][0]
        assert seg["addressee"] is None

        r = client.patch(f"/api/jobs/{job_id}/segments/{seg['id']}", json={"addressee_gender": "female"})
        assert r.status_code == 200
        assert r.json()["addressee"] == {
            "gender": "female", "number": "singular", "target": None, "label": None,
            "confidence": 1.0, "source": "manual", "locked": True, "used": None,
        }
        assert client.patch(f"/api/jobs/{job_id}/segments/{seg['id']}",
                            json={"addressee_gender": "robot"}).status_code == 422
        # a text edit leaves the tag alone; null clears it
        r = client.patch(f"/api/jobs/{job_id}/segments/{seg['id']}", json={"translated_text": "את באה?"})
        assert r.json()["addressee"]["locked"] is True
        r = client.patch(f"/api/jobs/{job_id}/segments/{seg['id']}", json={"addressee_gender": None})
        assert r.json()["addressee"] is None
        # the tag is part of the edit history
        assert client.post(f"/api/jobs/{job_id}/history/undo").status_code == 200
        assert client.get(f"/api/jobs/{job_id}").json()["segments"][0]["addressee"]["gender"] == "female"


def test_editing_the_translation_by_hand_drops_the_form_the_translator_reported():
    with TestClient(app) as client:
        job_id = _job(["Sarah, how are you?"])
        record = {"gender": "female", "number": "singular", "confidence": 0.95, "source": "vocative", "used": "m"}
        with Session(engine) as session:
            row = session.exec(
                select(SubtitleSegmentRecord).where(SubtitleSegmentRecord.job_id == job_id)
            ).one()
            row.addressee_json, row.translated_text = json.dumps(record), "שרה, מה שלומך?"
            session.add(row)
            session.commit()
            seg_id = row.id
        flagged = client.get(f"/api/jobs/{job_id}/qa").json()
        assert any(i["kind"] == "agreement" for i in flagged)

        r = client.patch(f"/api/jobs/{job_id}/segments/{seg_id}", json={"translated_text": "שרה, מה שלומך היום?"})
        assert r.json()["addressee"]["used"] is None and r.json()["addressee"]["gender"] == "female"
        assert not any(i["kind"] == "agreement" for i in client.get(f"/api/jobs/{job_id}/qa").json())


def test_retranslating_rows_as_spoken_to_a_woman_keeps_the_tag_and_guides_the_lines_after(monkeypatch):
    monkeypatch.setattr(TranslateStage, "_make_translator", lambda self, ctx: _PlainTranslator())
    with TestClient(app) as client:
        job_id = _job(["Are you coming?", "The door is open.", "Come here."])
        segs = client.get(f"/api/jobs/{job_id}").json()["segments"]
        r = client.post(f"/api/jobs/{job_id}/translate",
                        json={"segment_ids": [segs[0]["id"]], "addressee_gender": "female"})
        assert r.status_code == 200
        job = _wait(client, job_id)
        by_index = {s["index"]: s for s in job["segments"]}
        assert by_index[1]["translated_text"] == "חדש 1"
        assert by_index[1]["addressee"]["locked"] is True and by_index[1]["addressee"]["gender"] == "female"
        assert by_index[2]["addressee"] is None and by_index[2]["translated_text"] == "ישן 2"

        # A later full run keeps the user's tag, and marks the imperative nobody could resolve
        assert client.post(f"/api/jobs/{job_id}/translate", json={}).status_code == 200
        job = _wait(client, job_id)
        by_index = {s["index"]: s for s in job["segments"]}
        assert by_index[1]["addressee"]["locked"] is True and by_index[1]["addressee"]["source"] == "manual"
        assert by_index[2]["addressee"] is None
        assert by_index[3]["addressee"]["source"] == "unresolved" and by_index[3]["addressee"]["confidence"] == 0.0


def test_rows_without_a_tag_hash_like_snapshots_taken_before_tags_existed():
    plain = SubtitleSegmentRecord(job_id="j", index=1, start_time=0.0, end_time=1.0, original_text="Hi")
    blob, digest, count = edit_history._encode([plain])
    assert count == 1 and "addressee_json" not in edit_history._decode(blob)[0]
    tagged = SubtitleSegmentRecord(job_id="j", index=1, start_time=0.0, end_time=1.0, original_text="Hi",
                                   addressee_json=json.dumps(manual_record("male").to_dict()))
    blob2, digest2, _ = edit_history._encode([tagged])
    assert digest2 != digest and edit_history._decode(blob2)[0]["addressee_json"]
