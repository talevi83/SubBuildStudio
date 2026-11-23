"""Who is each dialogue line spoken TO?

Gendered target languages (Hebrew, Arabic) conjugate 2nd-person forms by the listener's gender and
number, which English "you" never tells. This module resolves a structured, confidence-scored
addressee per dialogue line from the text and the diarized speakers, with no model calls:

  1. Direct evidence in the line itself: a manual tag, plural address ("you guys"), a character
     addressed by name, an honorific ("sir", "ma'am").
  2. Conversation state: the dialogue is cut into exchanges (runs without a long silence). Inside
     one, a speaker keeps talking to the same listener until something says otherwise, so strong
     evidence is carried to the speaker's neighbouring lines with decaying confidence.
  3. Turn-taking: the other participant of a two-person exchange, the speaker who was just
     addressed by name and now replies, question/answer adjacency.

The confidence decides how the translator is told: a fact (`addressee`), a guess
(`addressee_guess`), or nothing at all, so a weak guess is never presented as certain.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.pipeline.speaker_match import (
    _DEFAULT_SPEAKER_LABEL_RE,
    _DUAL_SPEAKER_LINE_RE,
    _PLURAL_PHRASE_RE,
    _clean_text,
    _extract_character_aliases,
    _is_informative_speaker,
    _match_vocative_character,
    _seg_get,
)

# Confidence tiers: >= HIGH is handed to the translator as a fact, >= MEDIUM as a guess
HIGH = 0.75
MEDIUM = 0.5

CONF_MANUAL = 1.0
CONF_PLURAL = 0.95
CONF_NAMED = 0.95
CONF_NAMED_NO_GENDER = 0.9
CONF_HONORIFIC = 0.9
CONF_HONORIFIC_WEAK = 0.6
CONF_ENDEARMENT = 0.4
CONF_REPLY_TO_ADDRESS = 0.85
CONF_DYAD = 0.8
CONF_SANDWICH = 0.7
CONF_ANSWER = 0.7
CONF_QUESTION = 0.65
CONF_ONE_NEIGHBOUR = 0.55
CONF_NEAREST = 0.45
CONF_LOCAL_AI = 0.65
CONF_CAP = 0.97

# Carrying evidence along a speaker's own lines
CARRY_FORWARD = 0.9
CARRY_BACKWARD = 0.65     # "Are you deaf? Sarah, I'm talking to you": the name says less about the line before it
CARRY_PLURAL = 0.72       # a speaker often turns to one person right after addressing everyone
CARRY_INTERRUPTED = 0.6   # someone other than the listener spoke in between
CARRY_CONTINUATION = 0.97  # the same unfinished sentence, cut over two subtitles
CARRY_MAX_LINES = 6
SCENE_GAP_SECONDS = 8.0

_GENDERS = ("male", "female", "mixed", "unknown")
_NUMBERS = ("singular", "plural", "unknown")
_FORMS = ("m", "f", "pl")


@dataclass
class AddresseeRecord:
    """The listener of one dialogue line."""
    gender: str = "unknown"        # male | female | mixed | unknown
    number: str = "unknown"        # singular | plural | unknown
    target: Optional[str] = None   # SPEAKER_XX of the listener, when they are a diarized speaker
    label: Optional[str] = None    # the listener's name as written in the target language
    confidence: float = 0.0
    source: str = ""               # manual | plural | vocative | honorific | reply | dyad | turn | carry | local_ai | unresolved
    locked: bool = False           # set by the user: never recomputed
    used: Optional[str] = None     # 2nd-person form the translator reports having written: m | f | pl

    @property
    def form(self) -> Optional[str]:
        """The 2nd-person form this listener calls for: m, f, pl, or None when unknown."""
        if self.number == "plural":
            return "pl"
        if self.gender == "male":
            return "m"
        if self.gender == "female":
            return "f"
        return None

    @property
    def tier(self) -> str:
        if self.confidence >= HIGH:
            return "high"
        if self.confidence >= MEDIUM:
            return "medium"
        return "low"

    def agrees_with(self, other: "AddresseeRecord") -> bool:
        if self.target and other.target:
            return self.target == other.target
        return bool(self.form and self.form == other.form)

    def display(self, speakers: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """How the listener is named to the translator, e.g. `SPEAKER_02 (דוד, male)`."""
        if self.number == "plural":
            return "group (plural, all female)" if self.gender == "female" else "group (plural)"
        label = self.label
        gender = self.gender if self.gender in ("male", "female") else None
        info = (speakers or {}).get(self.target) if self.target else None
        if isinstance(info, dict):
            sp_label = str(info.get("label") or "").strip()
            if sp_label and not _DEFAULT_SPEAKER_LABEL_RE.match(sp_label):
                label = sp_label
            sp_gender = str(info.get("gender") or "").strip().lower()
            if gender is None and sp_gender in ("male", "female"):
                gender = sp_gender
        parts = [p for p in (label, gender) if p]
        if self.target:
            return f"{self.target} ({', '.join(parts)})" if parts else self.target
        if label:
            return f"{label} ({gender})" if gender else label
        return gender

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "gender": self.gender,
            "number": self.number,
            "confidence": round(float(self.confidence), 2),
            "source": self.source,
        }
        if self.target:
            out["target"] = self.target
        if self.label:
            out["label"] = self.label
        if self.locked:
            out["locked"] = True
        if self.used in _FORMS:
            out["used"] = self.used
        return out

    @classmethod
    def from_dict(cls, data: Any) -> Optional["AddresseeRecord"]:
        if not isinstance(data, dict):
            return None
        gender = str(data.get("gender") or "unknown").strip().lower()
        number = str(data.get("number") or "unknown").strip().lower()
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        used = str(data.get("used") or "").strip().lower()
        return cls(
            gender=gender if gender in _GENDERS else "unknown",
            number=number if number in _NUMBERS else "unknown",
            target=str(data["target"]).strip() or None if data.get("target") else None,
            label=str(data["label"]).strip() or None if data.get("label") else None,
            confidence=confidence,
            source=str(data.get("source") or ""),
            locked=bool(data.get("locked")),
            used=used if used in _FORMS else None,
        )


def manual_record(choice: Optional[str]) -> Optional[AddresseeRecord]:
    """The record for a listener the user picked by hand: male, female or group."""
    key = str(choice or "").strip().lower()
    if key == "male":
        return AddresseeRecord(gender="male", number="singular", confidence=CONF_MANUAL, source="manual", locked=True)
    if key == "female":
        return AddresseeRecord(gender="female", number="singular", confidence=CONF_MANUAL, source="manual", locked=True)
    if key in ("group", "plural", "mixed"):
        return AddresseeRecord(gender="mixed", number="plural", confidence=CONF_MANUAL, source="manual", locked=True)
    return None


# ----------------------------------------------------------------------------------------------
# Does a line need the listener's gender at all?
# ----------------------------------------------------------------------------------------------

_THANKS_RE = re.compile(r"\b(?:thank|thanks|bless|damn|screw|fuck)\s+you\b", re.IGNORECASE)
_SECOND_PERSON_RE = re.compile(
    r"\b(?:you|your|yours|yourself|yourselves|ya|y['’]?all)\b|\byou['’](?:re|ve|ll|d)\b|\blet['’]?s\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(r"\b(?:i am|i['’]m|i was|i['’]ve been)\b", re.IGNORECASE)
# "Mrs. Cole" is not the end of a sentence
_SENTENCE_SPLIT_RE = re.compile(r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<=[.!?…])\s+|\s*\n\s*")
_SENTENCE_LEAD_RE = re.compile(
    r"^(?:[-–—]\s*)?(?:(?:please|just|now|then|so|and|but|hey|okay|ok|well|oh|quick|quickly|here|go\s+ahead\s+and)[\s,]+)*",
    re.IGNORECASE,
)
_IMPERATIVE_VERBS = frozenset("""
answer ask back be believe breathe bring call calm catch check close come cut do don't dont drink drive
drop eat excuse find fire follow forget forgive freeze get give go grab guess hang help hide hold hurry
imagine jump keep kill leave let listen look lower meet move never open pardon pass pay pick play promise
pull push put raise read relax remember run save say send show shut sing sit sleep smile speak stand start
stay step stop swear take talk tell think throw trust try turn use wait wake walk watch write
""".split())
_NOT_IMPERATIVE_NEXT = frozenset({"i", "we", "he", "she", "it", "they", "is", "was", "are", "were", "has", "had", "will", "would", "of"})
_ADJECTIVE_QUESTION_RE = re.compile(
    r"^(?:[-–—]\s*)?(?:ready|sure|okay|ok|hungry|tired|scared|afraid|happy|done|finished|hurt|alone|awake|serious|"
    r"kidding|coming|going|leaving|thirsty|cold|sick|busy|married|lost|nervous)\?+$",
    re.IGNORECASE,
)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s and s.strip()]


def needs_listener_gender(text: Optional[str]) -> bool:
    """True when translating the line into a gendered language depends on who is being spoken to:
    a 2nd-person pronoun, an imperative ("Come here.") or a bare adjective question ("Ready?")."""
    raw = str(text or "")
    if not raw.strip():
        return False
    if _SECOND_PERSON_RE.search(_THANKS_RE.sub(" ", raw)):
        return True
    for sentence in _sentences(raw):
        if _ADJECTIVE_QUESTION_RE.match(sentence):
            return True
        body = _SENTENCE_LEAD_RE.sub("", sentence)
        words = re.findall(r"[A-Za-z]+(?:['’][a-z]+)?", body)
        if not words:
            continue
        first = words[0].lower().replace("’", "'")
        if first not in _IMPERATIVE_VERBS:
            continue
        after = body[body.find(words[0]) + len(words[0]):].lstrip()[:1]
        # "Look, I don't know." is an imperative; "Do I know him?" is not
        if after in ",!." or not (len(words) > 1 and words[1].lower() in _NOT_IMPERATIVE_NEXT):
            return True
    return False


def needs_speaker_gender(text: Optional[str]) -> bool:
    return bool(_FIRST_PERSON_RE.search(str(text or "")))


def is_gender_sensitive(text: Optional[str]) -> bool:
    return needs_listener_gender(text) or needs_speaker_gender(text)


# ----------------------------------------------------------------------------------------------
# Direct evidence inside a line
# ----------------------------------------------------------------------------------------------

_VOCATIVE_LEAD = (
    r"^(?:[-–—]\s*)?(?:(?:hey|hi|hello|listen|look|come\s+on|well|sorry|please|excuse\s+me|thank\s+you|thanks|"
    r"yes|no|yeah|yep|nope|aye|copy|roger|okay|ok|oh|wait|alright|all\s+right|goodbye|bye|good\s+(?:morning|night|evening))"
    r"[\s,!?\-]*)?"
)


def _vocative_re(words: str) -> "re.Pattern[str]":
    """`words` addressed to someone: at the start of a sentence, or after a comma."""
    return re.compile(
        rf"{_VOCATIVE_LEAD}\b(?:{words})\b(?:\s*[,!?.]|\s*$)|,\s*\b(?:{words})\b(?:\s*[.!?,]|\s*$)",
        re.IGNORECASE,
    )


_NAME = r"[A-Z][A-Za-z'\-]+"
_MASC_STRONG_RE = _vocative_re(
    r"sir|my\s+lord|my\s+king|(?-i:Mr\.?\s+" + _NAME + r")|dad|daddy|father|papa|grandpa|grandfather|son|brother|uncle|mister"
)
_FEM_STRONG_RE = _vocative_re(
    r"ma['’]?am|madam|my\s+lady|my\s+queen|(?-i:(?:Mrs|Ms|Miss)\.?\s+" + _NAME + r")|mom|mommy|mum|mother|mama|"
    r"grandma|grandmother|sister|aunt|auntie"
)
_MASC_WEAK_RE = re.compile(
    _vocative_re(r"bro|dude|pal|buddy|king|boy").pattern + r"|(?:^hey\s+|,\s*)\bman\b(?:\s*[,!?.]|$)",
    re.IGNORECASE,
)
_FEM_WEAK_RE = _vocative_re(r"sis|girl|queen|lady")
_ENDEARMENT_FEM_RE = _vocative_re(r"honey|sweetheart|darling")

_GROUP_WORDS = (
    r"guys|folks|everyone|everybody|kids|children|boys|girls|men|women|team|class|crew|people|gentlemen|ladies|"
    r"fellas|lads|troops|soldiers"
)
_PLURAL_VOCATIVE_RE = re.compile(
    _vocative_re(_GROUP_WORDS).pattern
    + rf"|\b(?:hey|hi|hello|listen\s+up|wake\s+up|come\s+on)[,\s]+\b(?:{_GROUP_WORDS})\b",
    re.IGNORECASE,
)
_FEMALE_GROUP_RE = _vocative_re(r"girls|ladies|women")
_LADIES_AND_GENTLEMEN_RE = re.compile(r"ladies\s+and\s+gentlemen", re.IGNORECASE)
_YOURSELVES_RE = re.compile(r"\byourselves\b", re.IGNORECASE)
_YOURSELF_RE = re.compile(r"\byourself\b", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"[.!?…:\]\)\"”״]\s*$")


def _any_sentence(pattern: "re.Pattern[str]", sentences: Sequence[str]) -> bool:
    return any(pattern.search(s) for s in sentences)


def _direct_evidence(
    text: str,
    speaker: str,
    alias_to_char: Sequence[Tuple[str, int]],
    char_by_idx: Dict[int, Dict[str, Any]],
    speakers: Optional[Dict[str, Any]],
) -> Optional[AddresseeRecord]:
    """What the line itself says about its listener (plural address, a name, an honorific)."""
    sentences = _sentences(text) or [text]

    if _PLURAL_PHRASE_RE.search(text) or _YOURSELVES_RE.search(text) or _any_sentence(_PLURAL_VOCATIVE_RE, sentences):
        all_female = _any_sentence(_FEMALE_GROUP_RE, sentences) and not _LADIES_AND_GENTLEMEN_RE.search(text)
        return AddresseeRecord(
            gender="female" if all_female else "mixed", number="plural", confidence=CONF_PLURAL, source="plural"
        )

    masc = _any_sentence(_MASC_STRONG_RE, sentences)
    fem = _any_sentence(_FEM_STRONG_RE, sentences)
    honorific = "male" if masc and not fem else "female" if fem and not masc else None

    voc_char = None
    if alias_to_char:
        voc_char = _match_vocative_character(text, alias_to_char, char_by_idx)
        if voc_char is None:
            for sentence in sentences:
                voc_char = _match_vocative_character(sentence, alias_to_char, char_by_idx)
                if voc_char is not None:
                    break
    if voc_char is not None:
        vc_sp = str(voc_char.get("speaker_id") or "").strip()
        if not (speaker and vc_sp and vc_sp == speaker):
            vc_gender = str(voc_char.get("gender") or "unknown").strip().lower()
            if vc_gender == "mixed group":
                return AddresseeRecord(gender="mixed", number="plural", confidence=CONF_PLURAL, source="plural")
            label = str(voc_char.get("target_name") or voc_char.get("name") or "").strip() or None
            target = vc_sp if vc_sp and vc_sp in (speakers or {}) else None
            if vc_gender in ("male", "female"):
                return AddresseeRecord(
                    gender=vc_gender, number="singular", target=target, label=label,
                    confidence=CONF_NAMED, source="vocative",
                )
            return AddresseeRecord(
                gender=honorific or "unknown", number="singular", target=target, label=label,
                confidence=CONF_NAMED_NO_GENDER, source="vocative",
            )

    if honorific:
        return AddresseeRecord(gender=honorific, number="singular", confidence=CONF_HONORIFIC, source="honorific")

    weak_m = _any_sentence(_MASC_WEAK_RE, sentences)
    weak_f = _any_sentence(_FEM_WEAK_RE, sentences)
    if weak_m != weak_f:
        return AddresseeRecord(
            gender="male" if weak_m else "female", number="singular",
            confidence=CONF_HONORIFIC_WEAK, source="honorific",
        )
    if not weak_m and _any_sentence(_ENDEARMENT_FEM_RE, sentences):
        return AddresseeRecord(gender="female", number="singular", confidence=CONF_ENDEARMENT, source="honorific")
    return None


# ----------------------------------------------------------------------------------------------
# Conversation structure
# ----------------------------------------------------------------------------------------------

def split_into_exchanges(dialogue: Sequence[Any], max_gap: float = SCENE_GAP_SECONDS) -> List[Tuple[int, int]]:
    """Cut time-ordered dialogue into exchanges: `(start, end)` position ranges (end exclusive)
    separated by more than `max_gap` seconds of silence."""
    ranges: List[Tuple[int, int]] = []
    start = 0
    for pos in range(1, len(dialogue)):
        gap = float(_seg_get(dialogue[pos], "start_time", 0.0)) - float(_seg_get(dialogue[pos - 1], "end_time", 0.0))
        if gap > max_gap:
            ranges.append((start, pos))
            start = pos
    if dialogue:
        ranges.append((start, len(dialogue)))
    return ranges


def _speaker_of(seg: Any) -> str:
    return str(_seg_get(seg, "speaker") or "").strip()


def _from_speaker(
    speaker_id: str,
    speakers: Optional[Dict[str, Any]],
    confidence: float,
    source: str,
) -> Optional[AddresseeRecord]:
    """A record naming a diarized speaker as the listener (None when nothing is known about them)."""
    if not speaker_id or not _is_informative_speaker(speaker_id, speakers):
        return None
    info = (speakers or {}).get(speaker_id) or {}
    gender = str(info.get("gender") or "").strip().lower()
    label = str(info.get("label") or "").strip()
    return AddresseeRecord(
        gender=gender if gender in ("male", "female") else "unknown",
        number="singular",
        target=speaker_id,
        label=label if label and not _DEFAULT_SPEAKER_LABEL_RE.match(label) else None,
        confidence=confidence,
        source=source,
    )


def _turn_candidate(
    dialogue: Sequence[Any],
    i: int,
    speakers: Optional[Dict[str, Any]],
    max_turn_gap: float,
) -> Optional[AddresseeRecord]:
    """The neighbouring speaker this line is most likely said to, from turn-taking alone."""
    seg = dialogue[i]
    sp = _speaker_of(seg)
    raw_text = str(_seg_get(seg, "original_text", "") or "")
    if not sp or _DUAL_SPEAKER_LINE_RE.search(raw_text):
        return None
    text = _clean_text(raw_text)
    n = len(dialogue)

    prev_sp, prev_gap = "", float("inf")
    j, steps = i - 1, 0
    while j >= 0 and steps < 4:
        cand_sp = _speaker_of(dialogue[j])
        gap = float(_seg_get(dialogue[j + 1], "start_time", 0.0)) - float(_seg_get(dialogue[j], "end_time", 0.0))
        if not cand_sp:
            break
        if cand_sp != sp:
            if gap <= max_turn_gap:
                prev_sp, prev_gap = cand_sp, max(0.0, gap)
            break
        if gap > 2.5:
            break
        j -= 1
        steps += 1

    nxt_sp, nxt_gap = "", float("inf")
    j, steps = i + 1, 0
    while j < n and steps < 4:
        cand_sp = _speaker_of(dialogue[j])
        gap = float(_seg_get(dialogue[j], "start_time", 0.0)) - float(_seg_get(dialogue[j - 1], "end_time", 0.0))
        if not cand_sp:
            break
        if cand_sp != sp:
            if gap <= max_turn_gap:
                nxt_sp, nxt_gap = cand_sp, max(0.0, gap)
            break
        if gap > 2.5:
            break
        j += 1
        steps += 1

    # The likeliest neighbour is chosen first and only then checked for a known gender or name: a
    # neighbour nothing is known about must not hand the line to the other one by default.
    answered = (
        bool(prev_sp) and i >= 1 and prev_gap <= 2.0
        and _clean_text(_seg_get(dialogue[i - 1], "original_text", "")).endswith("?")
    )
    if prev_sp and nxt_sp:
        if prev_sp == nxt_sp:
            return _from_speaker(nxt_sp, speakers, CONF_SANDWICH, "turn")
        if text.endswith("?"):
            return _from_speaker(nxt_sp, speakers, CONF_QUESTION, "turn")
        if answered:
            return _from_speaker(prev_sp, speakers, CONF_ANSWER, "turn")
        return _from_speaker(nxt_sp if nxt_gap <= prev_gap else prev_sp, speakers, CONF_NEAREST, "turn")
    if nxt_sp:
        return _from_speaker(nxt_sp, speakers, CONF_QUESTION if text.endswith("?") else CONF_ONE_NEIGHBOUR, "turn")
    if prev_sp:
        return _from_speaker(prev_sp, speakers, CONF_ANSWER if answered else CONF_ONE_NEIGHBOUR, "turn")
    return None


def _carried(
    dialogue: Sequence[Any],
    direct: Sequence[Optional[AddresseeRecord]],
    lo: int,
    hi: int,
    i: int,
) -> Optional[AddresseeRecord]:
    """Strong evidence from the speaker's own nearest line in the exchange `[lo, hi)`, decayed.

    Someone who has just been addressed by name is still the listener two lines later; a third
    person speaking in between makes that much less likely.
    """
    sp = _speaker_of(dialogue[i])
    if not sp:
        return None
    best: Optional[AddresseeRecord] = None
    for step, decay in ((-1, CARRY_FORWARD), (1, CARRY_BACKWARD)):
        own_lines = 0
        others: set = set()
        j = i + step
        while lo <= j < hi and own_lines < CARRY_MAX_LINES:
            other_sp = _speaker_of(dialogue[j])
            if other_sp != sp:
                if other_sp:
                    others.add(other_sp)
                j += step
                continue
            own_lines += 1
            ev = direct[j]
            if ev is not None and ev.confidence >= HIGH:
                if step > 0 and (ev.number == "plural" or others):
                    # Looking ahead is only safe inside the speaker's own uninterrupted run:
                    # "Listen up, everyone" or a name said after someone else spoke tells
                    # nothing about the lines before it
                    break
                if ev.number == "plural":
                    conf = ev.confidence * (CARRY_PLURAL ** own_lines)
                else:
                    conf = ev.confidence * (decay ** own_lines)
                    strangers = (others - {ev.target}) if ev.target else others
                    # an unnamed listener ("sir") is presumably the one other person talking
                    if len(strangers) > (0 if ev.target else 1):
                        conf *= CARRY_INTERRUPTED
                if best is None or conf > best.confidence:
                    best = AddresseeRecord(
                        gender=ev.gender, number=ev.number, target=ev.target, label=ev.label,
                        confidence=conf, source="carry",
                    )
                break  # nearer evidence always shadows farther evidence
            j += step
    return best


def _continuation_pairs(dialogue: Sequence[Any]) -> List[Tuple[int, int]]:
    """Positions `(a, a + 1)` where one unfinished sentence runs over two subtitles."""
    pairs: List[Tuple[int, int]] = []
    for a in range(len(dialogue) - 1):
        first, second = dialogue[a], dialogue[a + 1]
        if _speaker_of(first) != _speaker_of(second):
            continue
        text = _clean_text(_seg_get(first, "original_text", ""))
        nxt = str(_seg_get(second, "original_text", "") or "").lstrip()
        if not text or not nxt or _TERMINAL_RE.search(text) or nxt[:1] in "-–—[":
            continue
        gap = float(_seg_get(second, "start_time", 0.0)) - float(_seg_get(first, "end_time", 0.0))
        if gap <= 1.0:
            pairs.append((a, a + 1))
    return pairs


def _pick(candidates: Sequence[Optional[AddresseeRecord]]) -> Optional[AddresseeRecord]:
    """The most confident candidate, a little surer when an independent one agrees with it."""
    found = [c for c in candidates if c is not None]
    if not found:
        return None
    found.sort(key=lambda c: c.confidence, reverse=True)
    best = found[0]
    if best.locked:
        return best
    for other in found[1:]:
        if other.confidence >= MEDIUM and best.agrees_with(other):
            best.confidence = min(CONF_CAP, 1.0 - (1.0 - best.confidence) * (1.0 - other.confidence * 0.5))
            if best.gender == "unknown" and other.gender in ("male", "female"):
                best.gender = other.gender
            if not best.target and other.target:
                best.target, best.label = other.target, best.label or other.label
            break
    return best


def resolve_addressees(
    segments: Sequence[Any],
    speakers: Optional[Dict[str, Dict[str, Any]]] = None,
    characters: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    manual: Optional[Dict[int, Any]] = None,
    max_turn_gap: float = 4.5,
    scene_gap: float = SCENE_GAP_SECONDS,
) -> Dict[int, AddresseeRecord]:
    """Resolve the listener of every dialogue line: `{segment index: AddresseeRecord}`.

    `manual` holds the user's own tags by segment index; a segment's stored `addressee` that is
    `locked` counts as one too. Lines nothing can be said about are left out.
    """
    dialogue = sorted(
        (s for s in (segments or ()) if (_seg_get(s, "segment_type", "DIALOGUE") or "DIALOGUE") != "ON_SCREEN"),
        key=lambda s: (float(_seg_get(s, "start_time", 0.0)), int(_seg_get(s, "index", 0))),
    )
    if not dialogue:
        return {}

    # Characters plus named speaker profiles, for recognising who is called by name
    merged_chars: List[Dict[str, Any]] = [dict(c) for c in (characters or ()) if isinstance(c, dict)]
    known_names = {str(c.get("name") or "").strip().lower() for c in merged_chars}
    for sid, sp_info in (speakers or {}).items():
        if not isinstance(sp_info, dict):
            continue
        label = str(sp_info.get("label") or "").strip()
        cname = str(sp_info.get("character_name") or "").strip()
        cand_name = cname or (label if label and not _DEFAULT_SPEAKER_LABEL_RE.match(label) else "")
        if cand_name and cand_name.lower() not in known_names:
            merged_chars.append({
                "name": cand_name,
                "target_name": label or cand_name,
                "gender": str(sp_info.get("gender") or "unknown").strip().lower(),
                "speaker_id": sid,
            })
            known_names.add(cand_name.lower())
    alias_to_char: List[Tuple[str, int]] = []
    char_by_idx: Dict[int, Dict[str, Any]] = {}
    for idx, ch, aliases in _extract_character_aliases(merged_chars):
        char_by_idx[idx] = ch
        alias_to_char.extend((al, idx) for al in aliases)
    alias_to_char.sort(key=lambda item: len(item[0]), reverse=True)

    n = len(dialogue)
    texts = [_clean_text(str(_seg_get(s, "original_text", "") or "")) for s in dialogue]
    indices = [int(_seg_get(s, "index", 0)) for s in dialogue]

    # 1. What each line says about its own listener
    direct: List[Optional[AddresseeRecord]] = [None] * n
    for i, seg in enumerate(dialogue):
        tagged = (manual or {}).get(indices[i])
        tagged_rec = tagged if isinstance(tagged, AddresseeRecord) else AddresseeRecord.from_dict(tagged)
        if tagged_rec is None:
            stored = AddresseeRecord.from_dict(_seg_get(seg, "addressee"))
            tagged_rec = stored if stored is not None and stored.locked else None
        if tagged_rec is not None:
            tagged_rec.locked, tagged_rec.confidence, tagged_rec.source = True, CONF_MANUAL, "manual"
            tagged_rec.used = None
            direct[i] = tagged_rec
        elif texts[i]:
            direct[i] = _direct_evidence(texts[i], _speaker_of(seg), alias_to_char, char_by_idx, speakers)

    # One sentence cut over two subtitles has one listener
    for _ in range(2):
        for a, b in _continuation_pairs(dialogue):
            for src, dst in ((a, b), (b, a)):
                ev = direct[src]
                if ev is not None and ev.confidence >= HIGH and direct[dst] is None:
                    direct[dst] = AddresseeRecord(
                        gender=ev.gender, number=ev.number, target=ev.target, label=ev.label,
                        confidence=ev.confidence * CARRY_CONTINUATION, source="carry",
                    )

    # 2-3. Conversation state and turn-taking, one exchange at a time
    resolved: Dict[int, AddresseeRecord] = {}
    for lo, hi in split_into_exchanges(dialogue, scene_gap):
        ids = [_speaker_of(dialogue[p]) for p in range(lo, hi)]
        participants = {sid for sid in ids if sid}
        unlabeled = sum(1 for sid in ids if not sid)
        named_outsider = any(
            direct[p] is not None and direct[p].source == "vocative" and direct[p].target not in participants
            for p in range(lo, hi)
        )
        # Two voices are a two-person talk only when nobody else is named or addressed as a group
        group_present = any(direct[p] is not None and direct[p].number == "plural" for p in range(lo, hi))
        dyad = len(participants) == 2 and unlabeled * 5 <= (hi - lo) and not named_outsider and not group_present

        for i in range(lo, hi):
            if not texts[i]:
                continue
            sp = ids[i - lo]
            own = direct[i]
            if own is not None and own.locked:
                resolved[indices[i]] = own
                continue

            candidates: List[Optional[AddresseeRecord]] = [own]
            if sp:
                # Replying to whoever has just called this speaker by name
                if i - 1 >= lo:
                    prev_sp = ids[i - 1 - lo]
                    prev_ev = direct[i - 1]
                    gap = float(_seg_get(dialogue[i], "start_time", 0.0)) - float(_seg_get(dialogue[i - 1], "end_time", 0.0))
                    if prev_sp and prev_sp != sp and prev_ev is not None and prev_ev.target == sp and gap <= max_turn_gap:
                        candidates.append(_from_speaker(prev_sp, speakers, CONF_REPLY_TO_ADDRESS, "reply"))
                turn = _turn_candidate(dialogue, i, speakers, max_turn_gap)
                candidates.append(turn)
                if dyad and not _DUAL_SPEAKER_LINE_RE.search(str(_seg_get(dialogue[i], "original_text", "") or "")):
                    other = next(iter(participants - {sp}), "")
                    candidates.append(_from_speaker(other, speakers, CONF_DYAD, "dyad"))
                if own is None or own.confidence < HIGH:
                    candidates.append(_carried(dialogue, direct, lo, hi, i))

            # "yourself" is one person: a carried or guessed group does not apply to this line
            if _YOURSELF_RE.search(texts[i]):
                candidates = [c for c in candidates if c is not None and c.number != "plural"]
            best = _pick(candidates)
            if best is not None and (best.form or best.target or best.label):
                resolved[indices[i]] = best
    return resolved


def merge_local_guess(
    records: Dict[int, AddresseeRecord],
    seg_index: int,
    listener: Optional[str],
) -> Optional[AddresseeRecord]:
    """Fold a local model's answer for one line (`male`, `female` or `group…`) into `records`.

    A small local model is a second opinion, never a fact: on its own it is a guess, it confirms a
    guess that agrees with it, and it weakens one that does not. Confident and manual records stay.
    """
    key = str(listener or "").strip().lower()
    if "group" in key or "plural" in key:
        guess = AddresseeRecord(gender="mixed", number="plural", confidence=CONF_LOCAL_AI, source="local_ai")
    elif key in ("male", "female"):
        guess = AddresseeRecord(gender=key, number="singular", confidence=CONF_LOCAL_AI, source="local_ai")
    else:
        return records.get(seg_index)
    current = records.get(seg_index)
    if current is None or not (current.form or current.target or current.label):
        records[seg_index] = guess
    elif current.locked:
        pass
    elif current.form is None:
        # Known who is addressed but not their gender: the name stays, the gender is only a guess
        if guess.number != "plural":
            current.gender, current.number = guess.gender, "singular"
            current.confidence = min(current.confidence, guess.confidence)
    elif current.confidence >= HIGH:
        pass
    elif current.form == guess.form:
        current.confidence = min(CONF_CAP, 1.0 - (1.0 - current.confidence) * (1.0 - guess.confidence * 0.5))
    elif guess.confidence > current.confidence:
        guess.confidence *= 0.8
        records[seg_index] = guess
    else:
        current.confidence *= 0.8
    return records[seg_index]


def unresolved_record() -> AddresseeRecord:
    """Marks a line that needs the listener's gender when nothing could be established."""
    return AddresseeRecord(source="unresolved")
