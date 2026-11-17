from types import SimpleNamespace as NS

from app.pipeline.qa import check_rows

GLOSS = {
    "characters": [{"name": "Chandler", "target_name": "צ'נדלר"}],
    "terms": [{"source": "bridge", "translation": "גשר"}],
}


def row(i, orig, tr, typ="DIALOGUE", failed=False):
    return NS(id=i, index=i, original_text=orig, translated_text=tr, segment_type=typ, translation_failed=failed)


def kinds(rows, g=GLOSS):
    return [(i.segment_id, i.kind) for i in check_rows(rows, g)]


def test_latin_word_flagged_but_acronyms_allowed():
    assert kinds([row(1, "x", "זה somehow קרה")]) == [(1, "latin")]
    assert kinds([row(2, "x", "הם מ-FBI והכל OK")]) == []


def test_agreement_mismatch():
    assert kinds([row(1, "x", "את תהיה שם")]) == [(1, "agreement")]
    assert kinds([row(2, "x", "אתה תהיי שם")]) == [(2, "agreement")]
    assert kinds([row(3, "x", "את תהיי שם"), row(4, "x", "אתה תהיה שם"), row(5, "x", "תן את הכל")]) == []


def test_glossary_name_and_term():
    assert kinds([row(1, "Chandler is here", "צנדלר כאן")]) == [(1, "name")]
    assert kinds([row(2, "Chandler is here", "צ'נדלר כאן")]) == []
    assert kinds([row(3, "On the bridge now", "על הסיפון עכשיו")]) == [(3, "term")]


def test_missing_translation_and_on_screen():
    assert kinds([row(1, "hello", "", failed=True)]) == [(1, "missing")]
    assert kinds([row(2, "SIGN", None, typ="ON_SCREEN")]) == []


def test_extended_qa_flags_overlap_and_extreme_durations():
    rows = [
        NS(id=1, index=1, start_time=0.0, end_time=0.3, original_text="Hi", translated_text="היי",
           segment_type="DIALOGUE", translation_failed=False),
        NS(id=2, index=2, start_time=1.0, end_time=4.0, original_text="One", translated_text="אחת",
           segment_type="DIALOGUE", translation_failed=False),
        NS(id=3, index=3, start_time=3.5, end_time=12.0, original_text="Two", translated_text="שתיים",
           segment_type="DIALOGUE", translation_failed=False),
        NS(id=4, index=4, start_time=1.0, end_time=3.0, original_text="[TITLE]", translated_text="[כותרת]",
           segment_type="ON_SCREEN", translation_failed=False),
    ]
    issues = check_rows(rows, glossary=None)
    by_pair = {(i.segment_id, i.kind) for i in issues}
    assert (1, "timing") in by_pair
    assert (3, "overlap") in by_pair
    assert (3, "timing") in by_pair
    assert (4, "overlap") not in by_pair  # ON_SCREEN lane is separate from DIALOGUE lane
