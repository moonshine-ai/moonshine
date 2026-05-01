"""Alphanumeric listener for character-by-character speech-to-text input.

Designed for dictating passwords, serial numbers, codes, and other
alphanumeric strings where the user speaks one character at a time,
optionally preceded by modifiers like "upper case" or "capital".

Usage::

    from moonshine_voice import AlphanumericListener

    def handle(event):
        print(event)

    listener = AlphanumericListener(handle)
    transcriber.add_listener(listener)

    # User says: "capital H", "e", "l", "l", "o", "at sign",
    #            "one", "two", "three", "stop"
    # callback receives Character events for H, e, l, l, o, @, 1, 2, 3
    # then a Stopped event with text="Hello@123"
"""

import json
import os
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from moonshine_voice.moonshine_api import TranscriptLine
from moonshine_voice.transcriber import (
    TranscriptEvent,
    LineCompleted,
    LineTextChanged,
)
from moonshine_voice.utils import get_assets_path


# ---------------------------------------------------------------------------
# Callback event types
# ---------------------------------------------------------------------------

class AlphanumericEventType(Enum):
    CHARACTER = auto()
    UNDO = auto()
    CLEAR = auto()
    STOPPED = auto()
    NONE = auto()


@dataclass
class AlphanumericEvent:
    """Event delivered to the callback each time something happens.

    Attributes
    ----------
    type:
        The kind of event (CHARACTER, UNDO, CLEAR, or STOPPED).
    character:
        The resolved character for CHARACTER events, or the removed
        character for UNDO events. ``None`` for CLEAR / STOPPED.
    text:
        The full assembled string *after* this event has been applied.
    """

    type: AlphanumericEventType
    character: Optional[str]
    text: str


@dataclass
class AlphanumericMatch:
    """Pure classification result produced by :class:`AlphanumericMatcher`.

    Has a ``type`` (one of :class:`AlphanumericEventType`, with ``NONE``
    meaning "unrecognized") and an optional resolved ``character``.
    """

    type: AlphanumericEventType
    character: Optional[str] = None

    @property
    def is_character(self) -> bool:
        return self.type is AlphanumericEventType.CHARACTER

    @property
    def is_terminator(self) -> bool:
        return self.type is AlphanumericEventType.STOPPED

    @property
    def is_recognized(self) -> bool:
        return self.type is not AlphanumericEventType.NONE


# ---------------------------------------------------------------------------
# Vocabulary tables
# ---------------------------------------------------------------------------

_LETTER_WORDS: Dict[str, str] = {
    "a": "a", "ay": "a", "hey": "a", "aye": "a",
    "b": "b", "bee": "b",
    "c": "c", "see": "c", "sea": "c",
    "d": "d", "dee": "d",
    "e": "e",
    "f": "f", "ef": "f", "eff": "f",
    "g": "g", "gee": "g",
    "h": "h", "aitch": "h",
    "i": "i", "eye": "i",
    "j": "j", "jay": "j",
    "k": "k", "kay": "k", "okay": "k", "ok": "k",
    "l": "l", "el": "l", "ell": "l",
    "m": "m", "em": "m",
    "n": "n", "en": "n", "and": "n",
    "o": "o", "oh": "o",
    "p": "p", "pee": "p",
    "q": "q", "queue": "q", "cue": "q",
    "r": "r", "are": "r", "ar": "r", "ah": "r", "uh-huh": "r", "aww": "r", "awe": "r",
    "s": "s", "es": "s", "ess": "s",
    "t": "t", "tee": "t",
    "u": "u", "you": "u",
    "v": "v", "vee": "v",
    "w": "w", "double u": "w", "double you": "w",
    "x": "x", "ex": "x",
    "y": "y", "why": "y", "wye": "y",
    "z": "z", "zee": "z", "zed": "z", "zet": "z",
}

_DIGIT_WORDS: Dict[str, str] = {
    "zero": "0", "0": "0",
    "one": "1", "won": "1", "1": "1",
    "two": "2", "to": "2", "too": "2", "2": "2",
    "three": "3", "3": "3",
    "four": "4", "for": "4", "4": "4",
    "five": "5", "5": "5",
    "six": "6", "6": "6",
    "seven": "7", "7": "7",
    "eight": "8", "ate": "8", "8": "8",
    "nine": "9", "niner": "9", "9": "9",
}

_SPECIAL_CHAR_WORDS: Dict[str, str] = {
    # Punctuation
    "period": ".", "dot": ".", "full stop": ".", "point": ".",
    "comma": ",",
    "colon": ":",
    "semicolon": ";", "semi colon": ";",
    "exclamation mark": "!", "exclamation point": "!", "exclamation": "!", "bang": "!",
    "question mark": "?",

    # Brackets / parens
    "open parenthesis": "(", "left parenthesis": "(", "open paren": "(", "left paren": "(",
    "close parenthesis": ")", "right parenthesis": ")", "close paren": ")", "right paren": ")",
    "open bracket": "[", "left bracket": "[",
    "close bracket": "]", "right bracket": "]",
    "open brace": "{", "left brace": "{", "open curly": "{", "left curly": "{",
    "close brace": "}", "right brace": "}", "close curly": "}", "right curly": "}",

    # Common password / code characters
    "at sign": "@", "at": "@", "at symbol": "@",
    "hash": "#", "hashtag": "#", "pound sign": "#", "number sign": "#", "pound": "#",
    "dollar sign": "$", "dollar": "$",
    "percent": "%", "percent sign": "%", "per cent": "%",
    "caret": "^", "carrot": "^", "hat": "^",
    "ampersand": "&", "and sign": "&",
    "asterisk": "*", "star": "*",
    "hyphen": "-", "dash": "-", "minus": "-",
    "underscore": "_", "under score": "_",
    "plus": "+", "plus sign": "+",
    "equals": "=", "equal sign": "=", "equals sign": "=",
    "pipe": "|", "vertical bar": "|",
    "backslash": "\\", "back slash": "\\",
    "forward slash": "/", "slash": "/",
    "tilde": "~",
    "grave": "`", "backtick": "`", "back tick": "`",
    "apostrophe": "'", "single quote": "'",
    "quote": "\"", "double quote": "\"", "quotation mark": "\"",

    # Whitespace / control
    "space": " ",
}

_UPPER_MODIFIERS = frozenset({
    "upper case", "uppercase", "upper", "capital", "cap", "big",
    "shift",
})

_UNDO_WORDS = frozenset({
    "undo", "delete", "backspace", "back space", "erase",
    "scratch that", "remove",
})

_CLEAR_WORDS = frozenset({
    "clear", "clear all", "reset", "start over",
})

_STOP_WORDS = frozenset({
    "stop", "end", "finish", "finished", "done",
    "complete", "that's it", "submit", "confirm",
    "i'm done", "all done", "go", "enter",
})


# ---------------------------------------------------------------------------
# Spelling-CNN integration tables
#
# The SpellingCNN ONNX model emits class labels using *spoken* digit words
# ("zero".."nine") for the digit classes and single characters ("a".."z")
# for letters.  Callers comparing predictions across models, and the
# fusion logic in :class:`AlphanumericListener`, want a single canonical
# form so we collapse the digit words to their character equivalents.
# ---------------------------------------------------------------------------

_SPELL_CLASS_TO_CHAR: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


# ---------------------------------------------------------------------------
# Weak-homonym blocklist for fusion
#
# These are spoken phrases that the matcher would happily resolve to a
# single character in spelling mode (``"okay"`` -> ``k``, ``"you"`` -> ``u``)
# but that fire constantly as fillers in real conversational speech.  On
# the People's Speech eval set they hit 18%-21% precision (vs 95%+ for
# unambiguous matches), so when a SpellingCNN prediction is available the
# listener treats them as "weak" and lets the audio-side model overrule
# them.  ``"and" -> n`` was 56% precision on the same set -- noisy but
# net positive when its votes are kept.
# ---------------------------------------------------------------------------

_DEFAULT_WEAK_HOMONYMS: frozenset = frozenset({"okay", "ok", "you"})


# ---------------------------------------------------------------------------
# Spoken-form tables – character -> TTS-friendly phrase.
#
# These are the *output* side of the alphanumeric vocabulary: given a
# resolved character, what phrase should a TTS engine pronounce so the
# user hears it unambiguously?  We pick phonetic / full-word variants
# consistent with the spelling-alphabet names used for input recognition
# (``"aitch"``/``"haitch"`` for "h", ``"hash"`` for "#", ``"one"`` for
# "1", etc.).  Upper-case letters get a ``"capital "`` prefix applied at
# lookup time rather than duplicated in the table.
#
# Both :func:`spoken_form` (used by :class:`AlphanumericListener`'s TTS
# repeat-back) and :func:`moonshine_voice.dialog_flow.spell_out` consume
# these, so there's exactly one place the forms are defined.
# ---------------------------------------------------------------------------

_SPELL_OUT_LETTERS: Dict[str, str] = {
    "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee",
    "f": "eff", "g": "gee", "h": "haitch", "i": "eye", "j": "jay",
    "k": "kay", "l": "el", "m": "em", "n": "en", "o": "oh",
    "p": "pee", "q": "queue", "r": "ar", "s": "ess", "t": "tee",
    "u": "you", "v": "vee", "w": "double you", "x": "ex", "y": "why",
    "z": "zee",
}

_SPELL_OUT_DIGITS: Dict[str, str] = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

_SPELL_OUT_SYMBOLS: Dict[str, str] = {
    ".": "period", ",": "comma", ":": "colon", ";": "semicolon",
    "!": "exclamation mark", "?": "question mark",
    "(": "open parenthesis", ")": "close parenthesis",
    "[": "open bracket", "]": "close bracket",
    "{": "open brace", "}": "close brace",
    "@": "at sign", "#": "hash", "$": "dollar", "%": "percent",
    "^": "caret", "&": "ampersand", "*": "asterisk",
    "-": "hyphen", "_": "underscore", "+": "plus", "=": "equals",
    "|": "pipe", "\\": "backslash", "/": "slash",
    "~": "tilde", "`": "backtick",
    "'": "apostrophe", '"': "quote",
    " ": "space",
}


def spoken_form(char: str) -> str:
    """Return a TTS-friendly phrase for a single character.

    * Letters use their spelling-alphabet sounds – ``"h"`` → ``"haitch"``,
      ``"w"`` → ``"double you"``.
    * Upper-case letters are prefixed with ``"capital "`` –
      ``"H"`` → ``"capital haitch"``.
    * Digits become their word form – ``"1"`` → ``"one"``.
    * Common symbols use their spoken name – ``"#"`` → ``"hash"``,
      ``"@"`` → ``"at sign"``, ``" "`` → ``"space"``.
    * Anything else (unknown char, empty string, multi-char input) is
      returned unchanged so callers never lose information silently.
    """
    if not isinstance(char, str) or len(char) != 1:
        return char
    if char.isalpha():
        token = _SPELL_OUT_LETTERS.get(char.lower(), char.lower())
        if char.isupper():
            token = f"capital {token}"
        return token
    if char in _SPELL_OUT_DIGITS:
        return _SPELL_OUT_DIGITS[char]
    if char in _SPELL_OUT_SYMBOLS:
        return _SPELL_OUT_SYMBOLS[char]
    return char


# ---------------------------------------------------------------------------
# Input normalisation
#
# STT output is noisy: the same word can come back as ``"aww"``, ``"Aww."``,
# ``'"Aww,"'`` or ``"\u201cAww.\u201d"`` depending on the model and any
# downstream cleanup.  Before we look anything up we strip the punctuation
# and quote characters that aren't part of any vocabulary key, lower-case
# the result, and collapse runs of whitespace.  The same normalisation is
# applied to every lookup key / command word at module-init so apostrophe-
# containing entries like ``"that's it"`` (which becomes ``"thats it"``)
# still match user utterances such as ``"That's it."``.
# ---------------------------------------------------------------------------

_NORMALIZE_DROP_CHARS = (
    ".,!?"            # sentence-ending punctuation Whisper sprinkles in
    "\"'"             # straight quotes / apostrophes
    "\u2018\u2019"    # curly single quotes (\u2018 / \u2019)
    "\u201c\u201d"    # curly double quotes (\u201c / \u201d)
)
_NORMALIZE_TRANS = str.maketrans("", "", _NORMALIZE_DROP_CHARS)


def _normalize(text: str) -> str:
    """Return *text* lower-cased with quotes/punctuation removed.

    Characters in :data:`_NORMALIZE_DROP_CHARS` are deleted, the result is
    lower-cased, and internal runs of whitespace are collapsed to a single
    space.  Used on both inbound STT text and the vocabulary keys so
    matching is robust to noisy punctuation while still recognising
    multi-word phrases like ``"upper case"``.
    """
    if not text:
        return ""
    cleaned = text.lower().translate(_NORMALIZE_TRANS)
    return " ".join(cleaned.split())


def _build_lookup() -> Dict[str, str]:
    """Merge all character vocabularies into a single lookup table."""
    lookup: Dict[str, str] = {}
    for table in (_SPECIAL_CHAR_WORDS, _DIGIT_WORDS, _LETTER_WORDS):
        for spoken, char in table.items():
            lookup[_normalize(spoken)] = char
    return lookup


# Precomputed once at import time so every ``AlphanumericMatcher`` with the
# default vocabulary can reuse the same table instead of rebuilding it per
# instance.  ``AlphanumericMatcher`` only copies this dict when a caller
# supplies ``custom_words``.
_DEFAULT_LOOKUP: Dict[str, str] = _build_lookup()

# Normalised copies of the command vocabularies, used by ``classify`` so the
# user-facing source above stays readable (``"that's it"``, ``"i'm done"``)
# while runtime comparisons happen on the apostrophe-stripped forms.
_UPPER_MODIFIERS_NORM: frozenset = frozenset(
    _normalize(s) for s in _UPPER_MODIFIERS
)
_UNDO_WORDS_NORM: frozenset = frozenset(_normalize(s) for s in _UNDO_WORDS)
_CLEAR_WORDS_NORM: frozenset = frozenset(_normalize(s) for s in _CLEAR_WORDS)
_STOP_WORDS_NORM: frozenset = frozenset(_normalize(s) for s in _STOP_WORDS)

# ``classify()`` matches upper-case modifier phrases by longest-prefix first
# (so ``"upper case"`` wins over ``"upper"``).  Sorting on every call is pure
# waste; do it once at import.
_UPPER_MODIFIERS_BY_LEN: tuple = tuple(
    sorted(_UPPER_MODIFIERS_NORM, key=len, reverse=True)
)


# ---------------------------------------------------------------------------
# English number-word parser  (10 – 1000)
# ---------------------------------------------------------------------------

_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}

_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _parse_number_words(text: str) -> Optional[int]:
    """Parse English number words in the range 10 – 1 000.

    Handles forms like:
      - "ten" .. "nineteen"
      - "twenty" .. "ninety nine" (with or without hyphen)
      - "one hundred" .. "nine hundred and ninety nine"
      - "a hundred", "a thousand", "one thousand"

    Returns ``None`` when *text* is not a recognisable number phrase or
    is in the range 0 – 9 (those are handled by the single-digit lookup).
    """
    # Normalise: strip, lowercase, collapse hyphens to spaces
    s = text.replace("-", " ").strip()
    # Remove a filler "and" used in e.g. "one hundred and ten"
    words = s.split()
    words = [w for w in words if w != "and"]
    if not words:
        return None

    # "a hundred" / "a thousand"
    if words[0] == "a":
        words[0] = "one"

    result = 0
    i = 0

    # --- optional hundreds component ---
    if (
        i < len(words)
        and words[i] in _ONES
        and i + 1 < len(words)
        and words[i + 1] == "hundred"
    ):
        result += _ONES[words[i]] * 100
        i += 2

    # --- optional hundreds component (bare "hundred" = 100) ---
    if i == 0 and len(words) >= 1 and words[0] == "hundred":
        result += 100
        i += 1

    # --- "thousand" ---
    if (
        i < len(words)
        and words[i] in _ONES
        and i + 1 < len(words)
        and words[i + 1] == "thousand"
    ):
        val = _ONES[words[i]]
        if val == 1:
            result += 1000
            i += 2
            if i == len(words):
                return result
            return None  # we only support up to 1000
        return None

    if i == 0 and len(words) >= 1 and words[0] == "thousand":
        result += 1000
        i += 1
        if i == len(words):
            return result
        return None

    # --- tens / teens / ones remainder ---
    if i < len(words) and words[i] in _TEENS:
        result += _TEENS[words[i]]
        i += 1
    elif i < len(words) and words[i] in _TENS:
        result += _TENS[words[i]]
        i += 1
        if i < len(words) and words[i] in _ONES:
            result += _ONES[words[i]]
            i += 1
    elif i < len(words) and words[i] in _ONES:
        result += _ONES[words[i]]
        i += 1

    if i != len(words):
        return None

    # Only handle 10+ here; 0-9 are covered by the single-digit table
    if result < 10 or result > 1000:
        return None
    return result


class AlphanumericMatcher:
    """Stateless classifier for spelled letter / digit / command utterances.

    Exposes the "does this utterance match?" half of the alphanumeric
    vocabulary as a pure function, without any of the listener or
    buffer-management machinery.  This is useful for dialog flows that
    want to reuse the same vocabulary for ``ask(mode=SPELLED)`` or
    ``ask(mode=DIGITS)`` prompts without pulling in the whole event
    listener.

    The same recognition rules apply as for :class:`AlphanumericListener`:
    letters, digits, number words 10-1000, upper-case modifiers, special
    characters, and the undo / clear / stop command vocabularies.

    Parameters
    ----------
    custom_words:
        Optional dict mapping spoken forms to output characters.  Takes
        highest priority and overrides built-in mappings.
    accept_letters:
        If ``False``, utterances that resolve to an alphabetic character
        are reported as ``NONE`` (useful for digit-only modes).
    accept_digits:
        If ``False``, utterances that resolve to a digit character are
        reported as ``NONE`` (useful for letter-only modes).
    accept_specials:
        If ``False``, utterances that resolve to a special character
        (``@``, ``#``, …) are reported as ``NONE``.
    """

    def __init__(
        self,
        *,
        custom_words: Optional[Dict[str, str]] = None,
        accept_letters: bool = True,
        accept_digits: bool = True,
        accept_specials: bool = True,
    ):
        if custom_words:
            # Only pay the copy cost when the caller actually needs to
            # override the built-in vocabulary.
            lookup = dict(_DEFAULT_LOOKUP)
            for spoken, char in custom_words.items():
                key = _normalize(spoken)
                if key:
                    lookup[key] = char
            self._lookup = lookup
        else:
            # Share the module-level precomputed table – read-only in
            # practice, and lets every matcher (plus the DialogFlow
            # cache) use the same dict without rebuilding it.
            self._lookup = _DEFAULT_LOOKUP
        self._accept_letters = accept_letters
        self._accept_digits = accept_digits
        self._accept_specials = accept_specials

    # -- Public API --------------------------------------------------------

    def classify(self, raw_text: Optional[str]) -> AlphanumericMatch:
        """Classify a single utterance into an :class:`AlphanumericMatch`."""

        if raw_text is None:
            return AlphanumericMatch(AlphanumericEventType.NONE)
        text = _normalize(raw_text)
        if not text:
            return AlphanumericMatch(AlphanumericEventType.NONE)

        if text in _STOP_WORDS_NORM:
            return AlphanumericMatch(AlphanumericEventType.STOPPED)
        if text in _CLEAR_WORDS_NORM:
            return AlphanumericMatch(AlphanumericEventType.CLEAR)
        if text in _UNDO_WORDS_NORM:
            return AlphanumericMatch(AlphanumericEventType.UNDO)

        make_upper = False
        for mod in _UPPER_MODIFIERS_BY_LEN:
            if text.startswith(mod + " "):
                text = text[len(mod):].strip()
                make_upper = True
                break
            if text == mod:
                return AlphanumericMatch(AlphanumericEventType.NONE)

        char = self._resolve(text)
        if char is None:
            return AlphanumericMatch(AlphanumericEventType.NONE)

        if not self._char_accepted(char):
            return AlphanumericMatch(AlphanumericEventType.NONE)

        if make_upper:
            char = char.upper()
        return AlphanumericMatch(AlphanumericEventType.CHARACTER, character=char)

    def classify_sequence(self, raw_text: Optional[str]) -> List[AlphanumericMatch]:
        """Classify a potentially multi-token utterance.

        First tries to classify the utterance as a whole; if that returns
        ``NONE`` and the utterance contains multiple tokens, falls back to
        classifying each token individually (so ``"h o m e"`` resolves to
        four ``CHARACTER`` events).  The returned list preserves order and
        stops after the first ``STOPPED`` match.
        """

        whole = self.classify(raw_text)
        if whole.is_recognized:
            return [whole]

        if not raw_text:
            return [AlphanumericMatch(AlphanumericEventType.NONE)]

        tokens = raw_text.replace("-", " ").split()
        if len(tokens) <= 1:
            return [AlphanumericMatch(AlphanumericEventType.NONE)]

        results: List[AlphanumericMatch] = []
        for tok in tokens:
            m = self.classify(tok)
            results.append(m)
            if m.type is AlphanumericEventType.STOPPED:
                break
        return results

    # -- Internals ---------------------------------------------------------

    def _resolve(self, text: str) -> Optional[str]:
        if text in self._lookup:
            return self._lookup[text]
        num = _parse_number_words(text)
        if num is not None:
            return str(num)
        if text.isdigit():
            return text
        if len(text) == 1 and text.isprintable():
            return text
        return None

    def _char_accepted(self, char: str) -> bool:
        if not char:
            return False
        if char.isdigit():
            return self._accept_digits
        if char.isalpha():
            return self._accept_letters
        return self._accept_specials


# Convenience pre-configured matchers.
def letters_only_matcher(**kwargs) -> AlphanumericMatcher:
    return AlphanumericMatcher(accept_digits=False, accept_specials=False, **kwargs)


def digits_only_matcher(**kwargs) -> AlphanumericMatcher:
    return AlphanumericMatcher(accept_letters=False, accept_specials=False, **kwargs)


# ---------------------------------------------------------------------------
# Optional spelling-CNN ONNX predictor
#
# Companion to :class:`AlphanumericListener`: when the caller supplies a
# trained ``SpellingCNN`` exported to ONNX (see the ``moonshine-spelling``
# repo's ``scripts/export_spelling_cnn_onnx.py``), the listener can run the
# first second of every completed utterance through it and surface the
# resulting top-1 character prediction alongside whatever the STT-driven
# alphanumeric matcher produced.  Useful for A/B-ing the two predictors
# offline without having to plumb a separate model into the eval harness.
# ---------------------------------------------------------------------------


@dataclass
class SpellingPrediction:
    """Top-1 prediction returned by :class:`SpellingPredictor`.

    Attributes
    ----------
    character:
        The predicted class in single-character form -- letters stay as
        ``"a".."z"`` and digit-word classes from the model are
        canonicalised through :data:`_SPELL_CLASS_TO_CHAR` (``"zero"`` ->
        ``"0"``, ``"nine"`` -> ``"9"``).  This is the form fusion logic
        and downstream comparators want.
    probability:
        Softmax probability for the top class, in ``[0, 1]``.
    raw_class:
        Original class label as emitted by the ONNX model
        (``"zero"`` instead of ``"0"`` for digits).  Useful for debug
        logs and for callers that want to round-trip back to the model's
        own vocabulary.
    """

    character: str
    probability: float
    raw_class: str = ""

    def __post_init__(self) -> None:
        # Older callers may construct this dataclass with only the first
        # two positional args; default ``raw_class`` to ``character`` so
        # the field is never empty when the caller hasn't set it.
        if not self.raw_class:
            self.raw_class = self.character

    def __str__(self) -> str:
        return f"{self.character} ({self.probability * 100:.1f}%)"


# Environment variable name honoured by :func:`find_default_spelling_onnx_path`
# -- exported as a constant so callers can document / surface it consistently
# (e.g. error messages telling the user how to point at their own ONNX).
SPELLING_ONNX_ENV_VAR = "MOONSHINE_SPELLING_ONNX"

# Bundled-asset filename. ``find_default_spelling_onnx_path`` looks for this
# under the package's ``assets/`` directory; it isn't shipped in the repo by
# default, but a user (or a CI step) can drop a copy here to make the model
# discoverable without a sibling repo or environment variable.
_BUNDLED_SPELLING_ONNX_NAME = "spelling_cnn.onnx"

# Sibling-repo discovery: when this package is being imported from a source
# tree that lives next to the ``moonshine-spelling`` repo (the development
# workflow on the maintainer's machine), look for the most recent training
# run's ``.onnx`` so ``python -m moonshine_voice.alphanumeric_listener``
# works out of the box without any setup.
_SIBLING_REPO_DIRNAME = "moonshine-spelling"
_SIBLING_REPO_ONNX_GLOB = "checkpoints/run_*/spelling_cnn.onnx"


def find_default_spelling_onnx_path() -> Optional[str]:
    """Locate a default :class:`SpellingPredictor` ONNX file.

    Resolution order (first hit wins):

    1. ``$MOONSHINE_SPELLING_ONNX`` environment variable -- explicit user
       override, takes precedence over everything else.  Pointed-to file
       must exist; empty/whitespace values are ignored.
    2. ``<package>/assets/spelling_cnn.onnx`` -- a bundled copy of the
       exported SpellingCNN.  Not committed by default (the model lives
       in the ``moonshine-spelling`` repo) but copying / symlinking it
       here is the simplest way to ship the predictor with the package.
    3. Sibling ``moonshine-spelling`` repo when this module is being
       imported from a source checkout: ``../moonshine-spelling/checkpoints/
       run_*/spelling_cnn.onnx`` resolved relative to the repo root,
       newest run wins.  This is the maintainer-machine fallback so the
       CLI works without any configuration.

    Returns the resolved absolute path as a string, or ``None`` when no
    candidate exists.  This function never raises -- callers can treat a
    ``None`` return as "fall back to ASR-only".
    """
    env_path = os.environ.get(SPELLING_ONNX_ENV_VAR, "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        # An explicit override that doesn't exist is almost always a typo;
        # tell the user via stderr so they don't silently fall back to a
        # different model than they asked for.
        print(
            f"[moonshine_voice] {SPELLING_ONNX_ENV_VAR}={env_path!r} does not "
            "exist; ignoring and continuing the search.",
            file=sys.stderr,
        )

    bundled = get_assets_path() / _BUNDLED_SPELLING_ONNX_NAME
    if bundled.is_file():
        return str(bundled.resolve())

    sibling = _find_sibling_spelling_onnx()
    if sibling is not None:
        return sibling

    return None


def _find_sibling_spelling_onnx() -> Optional[str]:
    """Best-effort lookup of a sibling ``moonshine-spelling`` checkout.

    Walks up from this file until we find a directory that has a sibling
    named ``moonshine-spelling`` (the layout the maintainer's dev machine
    uses).  Returns the path of the most recently modified ``.onnx``
    matching :data:`_SIBLING_REPO_ONNX_GLOB`, or ``None``.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        sibling_repo = ancestor.parent / _SIBLING_REPO_DIRNAME
        if sibling_repo.is_dir():
            candidates = sorted(
                sibling_repo.glob(_SIBLING_REPO_ONNX_GLOB),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return str(candidates[0])
            return None
        # Stop walking once we leave a plausible workspace root -- there's
        # no point scanning all the way up to ``/``.
        if ancestor.parent == ancestor:
            break
    return None


class SpellingPredictor:
    """Runs a SpellingCNN ONNX model on raw 16 kHz waveform clips.

    Loads an ONNX file exported with ``--mode waveform`` from the
    ``moonshine-spelling`` repo and exposes a single :meth:`predict`
    method that takes a numpy waveform and returns the top-1 class plus
    its probability.  All audio config (sample rate, clip length, class
    list, input/output tensor names) is read from the ``.onnx``'s embedded
    custom metadata via ``onnxruntime``'s ``get_modelmeta()`` API – there's
    no need for the ``onnx`` Python package or a sibling ``.pt`` file.

    The dependency on ``onnxruntime`` is lazy: importing this module never
    requires ``onnxruntime``; only constructing a predictor does.

    Parameters
    ----------
    onnx_path:
        Filesystem path to a ``.onnx`` exported in waveform mode.
    providers:
        Optional list of ONNX Runtime execution providers.  Defaults to
        ``["CPUExecutionProvider"]`` so behaviour is deterministic across
        machines.
    """

    def __init__(
        self,
        onnx_path: str,
        *,
        providers: Optional[List[str]] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "SpellingPredictor requires onnxruntime. Install with "
                "`pip install onnxruntime` (or `onnxruntime-gpu` on CUDA hosts)."
            ) from exc

        self._onnx_path = onnx_path
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # warnings only
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers or ["CPUExecutionProvider"],
        )

        meta_map = self._session.get_modelmeta().custom_metadata_map or {}
        self._classes: List[str] = self._parse_classes(meta_map.get("classes"))
        self._sample_rate: int = self._safe_int(meta_map.get("sample_rate"), 16000)
        self._clip_seconds: float = self._safe_float(
            meta_map.get("clip_seconds"), 1.0
        )
        self._target_samples: int = max(
            1, int(round(self._sample_rate * self._clip_seconds))
        )
        self._mode: str = meta_map.get("mode", "waveform")

        sess_inputs = {i.name: i.shape for i in self._session.get_inputs()}
        sess_outputs = {o.name: o.shape for o in self._session.get_outputs()}
        self._input_name = meta_map.get("input_name") or next(iter(sess_inputs))
        if self._input_name not in sess_inputs:
            self._input_name = next(iter(sess_inputs))
        self._output_name = meta_map.get("output_name") or next(iter(sess_outputs))
        if self._output_name not in sess_outputs:
            self._output_name = next(iter(sess_outputs))

        if self._mode != "waveform":
            raise ValueError(
                f"SpellingPredictor only supports waveform-mode ONNX exports; "
                f"got mode={self._mode!r}. Re-export the model with "
                "`--mode waveform`."
            )

    # -- Public API --------------------------------------------------------

    @property
    def onnx_path(self) -> str:
        return self._onnx_path

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def clip_seconds(self) -> float:
        return self._clip_seconds

    def predict(
        self,
        audio: Any,
        sample_rate: Optional[int] = None,
    ) -> Optional[SpellingPrediction]:
        """Predict the spelled character from the first ``clip_seconds`` of ``audio``.

        ``audio`` can be any 1-D float-like buffer (``list``, ``tuple``,
        ``np.ndarray``).  We slice the first ``sample_rate * clip_seconds``
        samples (right-padding with zeros when shorter) and run a single
        forward pass.  Returns ``None`` when the input is empty or numpy
        isn't installed.

        ``sample_rate`` is used for a strict equality check against the
        rate the model was trained at; we deliberately don't ship a
        resampler so callers don't silently get wrong predictions on a
        mismatched rate.
        """
        try:
            import numpy as np
        except ImportError:
            return None

        if audio is None:
            return None
        if sample_rate is not None and sample_rate != self._sample_rate:
            raise ValueError(
                f"SpellingPredictor expects sample_rate={self._sample_rate}, "
                f"got {sample_rate}. Resample the audio before calling predict()."
            )

        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return None

        if wav.size >= self._target_samples:
            clip = wav[: self._target_samples]
        else:
            clip = np.zeros(self._target_samples, dtype=np.float32)
            clip[: wav.size] = wav

        x = clip[np.newaxis, :].astype(np.float32, copy=False)
        logits = self._session.run([self._output_name], {self._input_name: x})[0]
        row = np.asarray(logits[0], dtype=np.float32)

        # Numerically stable softmax for the top-1 probability; full softmax
        # over a 36-way head is cheap so we don't bother optimising further.
        row -= float(row.max())
        probs = np.exp(row)
        probs /= float(probs.sum())
        idx = int(np.argmax(probs))

        if 0 <= idx < len(self._classes):
            raw_class = self._classes[idx]
        else:
            raw_class = str(idx)
        canonical = _SPELL_CLASS_TO_CHAR.get(raw_class, raw_class)
        return SpellingPrediction(
            character=canonical,
            probability=float(probs[idx]),
            raw_class=raw_class,
        )

    # -- Internals ---------------------------------------------------------

    @staticmethod
    def _parse_classes(raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(data, list) or not all(isinstance(c, str) for c in data):
            return []
        return data

    @staticmethod
    def _safe_int(value: Optional[str], default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Optional[str], default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


class FusionStrategy(str, Enum):
    """Selects how :class:`AlphanumericListener` combines the two predictors.

    * ``ASR_ONLY``        -- only the matcher's STT-derived character is used.
                             Spelling predictions are still computed and
                             exposed via
                             :attr:`AlphanumericListener.last_spelling_prediction`,
                             but they never influence the emitted character.
    * ``SPELLING_ONLY``   -- the matcher is used solely for command words
                             (``stop`` / ``clear`` / ``undo``); every other
                             utterance is decided by the SpellingCNN.
    * ``SMART_ROUTER``    -- the data-driven fusion proven on the
                             ``moonshine-spelling`` People's Speech eval:
                             trust agreements, route ASR-vs-spelling
                             disagreements by whether the prediction is a
                             digit (ASR specialist) or a letter (spelling
                             specialist), and break same-class ties using
                             the spelling probability against
                             ``spelling_disagree_threshold``.
    * ``AUTO``            -- pick :attr:`SMART_ROUTER` when a spelling
                             predictor is configured, otherwise
                             :attr:`ASR_ONLY`.  This is the default and
                             matches the historical behaviour of callers
                             that don't load a SpellingCNN.
    """

    AUTO = "auto"
    ASR_ONLY = "asr_only"
    SPELLING_ONLY = "spelling_only"
    SMART_ROUTER = "smart_router"


class AlphanumericListener:
    """Listens for single-character dictation and assembles the result.

    This is a callable listener — pass it directly to
    ``transcriber.add_listener()`` or ``stream.add_listener()``.  It
    receives raw :class:`TranscriptEvent` objects and internally filters
    for the events it cares about.  All utterance-level recognition is
    delegated to :class:`AlphanumericMatcher`; this class only adds the
    listener plumbing (buffer, undo/clear/stop state, event dispatch).

    A single *callback* is invoked every time something meaningful
    happens.  The callback receives an :class:`AlphanumericEvent` whose
    ``type`` field describes what occurred:

    * ``CHARACTER`` — a letter, digit, or special character was recognised.
    * ``UNDO``      — the last character was removed.
    * ``CLEAR``     — the buffer was wiped.
    * ``STOPPED``   — the user said "stop" / "done" / "finish" / etc.

    See :class:`AlphanumericMatcher` for the full vocabulary.

    Parameters
    ----------
    callback:
        Called on every character, undo, clear, or stop event.
        Signature: ``(event: AlphanumericEvent) -> None``.
    use_line_completed:
        If ``True`` (default), characters are committed on
        ``LineCompleted`` events (high confidence).  If ``False``,
        characters are committed on ``LineTextChanged`` for lower
        latency.
    custom_words:
        Optional dict mapping spoken forms to output characters.
        Takes highest priority and overrides built-in mappings.
    matcher:
        Optional pre-built :class:`AlphanumericMatcher`.  When provided,
        ``custom_words`` is ignored and the matcher is used as-is.
    tts:
        Optional text-to-speech backend.  When provided:

        * Each ``CHARACTER`` event triggers a ``tts.say(phrase)`` call
          where ``phrase`` is the output of :func:`spoken_form` (e.g.
          ``"haitch"`` for ``"h"``, ``"capital ay"`` for ``"A"``,
          ``"one"`` for ``"1"``, ``"hash"`` for ``"#"``).
        * Each utterance the matcher *doesn't* recognise triggers a
          ``tts.play_error()`` call (a short two-tone beep), so the
          user hears audible feedback that their last word was
          ignored instead of waiting in silence.

        The backend needs to expose ``say(text: str) -> None`` for the
        per-character repeat-back; ``play_error()`` is called only if
        the backend actually defines it (so simple stubs stay usable).
        Exceptions from either call are swallowed (and logged under
        ``debug=True``) so a flaky TTS can't break the listener.
    spelling_predictor:
        Optional pre-built :class:`SpellingPredictor` to run on each
        completed utterance.  When set (or when ``spelling_onnx_path`` is
        provided), every ``LineCompleted`` event whose ``line.audio_data``
        is populated triggers a forward pass over the first
        ``predictor.clip_seconds`` of audio; the resulting top-1 prediction
        is exposed as :attr:`last_spelling_prediction` for callers (e.g.
        ``scripts/eval-alphanumeric.py``) to log alongside the
        STT-driven prediction.  Sharing one predictor across many short-
        lived listeners avoids reloading the ONNX session per utterance.
    spelling_onnx_path:
        Convenience: filesystem path to a SpellingCNN ONNX file.  When set
        and ``spelling_predictor`` is ``None``, a :class:`SpellingPredictor`
        is constructed from this path.  Useful for one-off scripts; for
        long-running apps prefer building one :class:`SpellingPredictor`
        and passing it via ``spelling_predictor``.
    fusion_strategy:
        How to combine the matcher's character with the SpellingCNN's
        prediction.  See :class:`FusionStrategy`.  Defaults to
        :attr:`FusionStrategy.AUTO`, which picks ``SMART_ROUTER`` when a
        spelling predictor is wired and ``ASR_ONLY`` otherwise -- so
        existing callers that don't use the SpellingCNN see no behaviour
        change, and callers that *do* wire one in get the data-driven
        fusion by default.
    spelling_disagree_threshold:
        Used by :attr:`FusionStrategy.SMART_ROUTER`.  When the matcher
        and the spelling model disagree *within the same class*
        (digit-vs-digit or letter-vs-letter), the spelling model's
        prediction is taken iff its probability is at least this value;
        otherwise the matcher wins.  ``0.5`` is the calibration
        sweet-spot from the People's Speech eval.
    weak_homonyms:
        Iterable of spoken phrases (case-insensitive) that the matcher
        resolves to a character but which fire constantly as fillers in
        non-spelling speech.  Defaults to :data:`_DEFAULT_WEAK_HOMONYMS`
        (``"okay"``, ``"ok"``, ``"you"``).  When a matcher hit's
        normalised raw text is in this set *and* a spelling prediction
        with probability >= ``weak_homonym_override_threshold`` is
        available, the matcher's character is treated as a miss so the
        spelling model takes over.  Pass an explicit empty iterable to
        disable.
    weak_homonym_override_threshold:
        Minimum spelling probability required to demote a weak-homonym
        match (see ``weak_homonyms``).  ``0.3`` is conservative -- well
        above the ~18-21% precision the demoted homonyms had on the
        People's Speech eval, so the demotion only fires when the
        spelling model has at least *some* opinion.
    debug:
        If ``True``, unrecognised utterances and TTS errors are logged
        to stderr.
    """

    def __init__(
        self,
        callback: Callable[[AlphanumericEvent], None],
        *,
        use_line_completed: bool = True,
        custom_words: Optional[Dict[str, str]] = None,
        matcher: Optional[AlphanumericMatcher] = None,
        tts: Optional[Any] = None,
        spelling_predictor: Optional[SpellingPredictor] = None,
        spelling_onnx_path: Optional[str] = None,
        fusion_strategy: Any = FusionStrategy.AUTO,
        spelling_disagree_threshold: float = 0.5,
        weak_homonyms: Optional[Any] = None,
        weak_homonym_override_threshold: float = 0.3,
        debug: bool = False,
    ):
        self._callback = callback
        self._use_line_completed = use_line_completed
        self._debug = debug
        self._tts = tts
        self._buffer: List[str] = []
        self._processed_line_ids: set = set()
        self._stopped = False
        self._matcher = matcher or AlphanumericMatcher(custom_words=custom_words)

        if spelling_predictor is None and spelling_onnx_path is not None:
            spelling_predictor = SpellingPredictor(spelling_onnx_path)
        self._spelling_predictor: Optional[SpellingPredictor] = spelling_predictor
        self._last_spelling_prediction: Optional[SpellingPrediction] = None

        # Accept the enum or a plain string spelling. Resolve AUTO based
        # on whether a spelling predictor is actually configured -- once,
        # at construction, so the per-utterance hot path is branch-free.
        try:
            strategy = FusionStrategy(fusion_strategy)
        except ValueError as exc:
            raise ValueError(
                f"Unknown fusion_strategy={fusion_strategy!r}; expected one of "
                f"{[s.value for s in FusionStrategy]}"
            ) from exc
        if strategy is FusionStrategy.AUTO:
            strategy = (
                FusionStrategy.SMART_ROUTER
                if self._spelling_predictor is not None
                else FusionStrategy.ASR_ONLY
            )
        self._fusion_strategy: FusionStrategy = strategy
        self._spelling_disagree_threshold = float(spelling_disagree_threshold)
        self._weak_homonym_override_threshold = float(weak_homonym_override_threshold)

        # Weak-homonym set is normalised once so the per-utterance check
        # is a constant-time membership test against the matcher's own
        # ``_normalize`` output.
        if weak_homonyms is None:
            phrases = _DEFAULT_WEAK_HOMONYMS
        else:
            phrases = weak_homonyms
        self._weak_homonyms: frozenset = frozenset(
            _normalize(p) for p in phrases if p
        )

    # -- Callable interface (receives TranscriptEvent from Stream._emit) -----

    def __call__(self, event: TranscriptEvent) -> None:
        if self._stopped:
            return
        if self._use_line_completed and isinstance(event, LineCompleted):
            self._process_utterance(event.line)
        elif not self._use_line_completed and isinstance(event, LineTextChanged):
            self._process_utterance(event.line)

    # -- Public API ----------------------------------------------------------

    @property
    def text(self) -> str:
        """The currently assembled text."""
        return "".join(self._buffer)

    @property
    def stopped(self) -> bool:
        """Whether a stop command has been received."""
        return self._stopped

    @property
    def matcher(self) -> AlphanumericMatcher:
        """The underlying :class:`AlphanumericMatcher`."""
        return self._matcher

    @property
    def spelling_predictor(self) -> Optional[SpellingPredictor]:
        """The optional :class:`SpellingPredictor`, or ``None`` if disabled."""
        return self._spelling_predictor

    @property
    def fusion_strategy(self) -> "FusionStrategy":
        """Resolved :class:`FusionStrategy` actually in use.

        ``AUTO`` is collapsed at construction time, so this never returns
        :attr:`FusionStrategy.AUTO`.
        """
        return self._fusion_strategy

    @property
    def last_spelling_prediction(self) -> Optional[SpellingPrediction]:
        """Most recent :class:`SpellingPrediction`, or ``None``.

        Set by :meth:`_process_utterance` whenever a ``LineCompleted``
        event arrives with non-empty ``line.audio_data`` *and* a
        :class:`SpellingPredictor` is configured.  Reset to ``None`` for
        utterances that don't carry audio so callers can tell "no audio"
        apart from "previous prediction".
        """
        return self._last_spelling_prediction

    def clear(self) -> None:
        """Programmatically clear the buffer."""
        self._buffer.clear()
        self._processed_line_ids.clear()
        self._stopped = False
        self._callback(AlphanumericEvent(
            type=AlphanumericEventType.CLEAR,
            character=None,
            text=self.text,
        ))

    def undo(self) -> Optional[str]:
        """Remove and return the last character, or ``None`` if empty."""
        if not self._buffer:
            return None
        removed = self._buffer.pop()
        self._callback(AlphanumericEvent(
            type=AlphanumericEventType.UNDO,
            character=removed,
            text=self.text,
        ))
        return removed

    # -- Internals -----------------------------------------------------------

    def _process_utterance(self, line: TranscriptLine) -> None:
        line_id = line.line_id
        raw_text = line.text
        if line_id in self._processed_line_ids:
            return
        self._processed_line_ids.add(line_id)

        self._run_spelling_predictor(line)
        spell_pred = self._last_spelling_prediction

        match = self._matcher.classify(raw_text)

        # Command words (stop / clear / undo) are owned by the matcher
        # and are never overridden by the spelling model -- the audio
        # classifier doesn't have a vocabulary for them and we don't
        # want a stray spelling prediction to silently consume an
        # explicit "stop".
        if match.type is AlphanumericEventType.STOPPED:
            self._stopped = True
            self._callback(AlphanumericEvent(
                type=AlphanumericEventType.STOPPED,
                character=None,
                text=self.text,
            ))
            return
        if match.type is AlphanumericEventType.CLEAR:
            self.clear()
            return
        if match.type is AlphanumericEventType.UNDO:
            self.undo()
            return

        asr_char: Optional[str] = (
            match.character
            if match.type is AlphanumericEventType.CHARACTER
            else None
        )

        # (B) Demote weak homonyms when the spelling model is confident
        # enough to say something. Phrases like "okay" and "you" only had
        # ~20% precision against the People's Speech labels, so a real
        # audio prediction is almost always the better source of truth.
        if (
            asr_char is not None
            and self._fusion_strategy is not FusionStrategy.ASR_ONLY
            and self._is_weak_homonym(raw_text)
            and spell_pred is not None
            and spell_pred.probability >= self._weak_homonym_override_threshold
        ):
            asr_char = None

        # (A) Smart-router fusion. ``_fuse_character`` is responsible for
        # all of the "trust agreements / route by class / break ties on
        # threshold" logic; the listener just dispatches the result.
        final_char = self._fuse_character(asr_char, spell_pred)

        if final_char is not None:
            self._buffer.append(final_char)
            self._callback(AlphanumericEvent(
                type=AlphanumericEventType.CHARACTER,
                character=final_char,
                text=self.text,
            ))
            self._speak_character(final_char)
            return

        if self._debug:
            print(f"[debug] unrecognised: {raw_text!r}", file=sys.stderr)
        self._play_error_feedback()

    def _is_weak_homonym(self, raw_text: str) -> bool:
        """Return True if ``raw_text`` normalises to a weak-homonym phrase.

        Uses the same :func:`_normalize` rule the matcher applies, so a
        noisy STT output like ``'"Okay,"`` still matches the bare
        ``"okay"`` entry in :attr:`_weak_homonyms`.
        """
        if not self._weak_homonyms:
            return False
        return _normalize(raw_text) in self._weak_homonyms

    def _fuse_character(
        self,
        asr_char: Optional[str],
        spell_pred: Optional[SpellingPrediction],
    ) -> Optional[str]:
        """Combine the matcher's character with the SpellingCNN's prediction.

        The behaviour is determined by :attr:`fusion_strategy`:

        * ``ASR_ONLY`` -- ignore ``spell_pred`` entirely.
        * ``SPELLING_ONLY`` -- prefer ``spell_pred``, fall back to
          ``asr_char`` only if no prediction is available.
        * ``SMART_ROUTER`` -- the data-driven policy proven on the
          People's Speech eval (see the module docstring): trust
          agreements, route by class on cross-class disagreements
          (digits go to ASR, letters go to spelling), and break
          same-class ties using ``spell_pred.probability`` against
          ``self._spelling_disagree_threshold``.

        Case is preserved through the matcher: if the matcher returned
        an upper-case letter (because the user spoke a "capital"
        modifier) and the spelling model wins the tiebreak, the spelling
        character is upper-cased before being returned so the user's
        case intent isn't lost.
        """
        if self._fusion_strategy is FusionStrategy.ASR_ONLY:
            return asr_char

        if self._fusion_strategy is FusionStrategy.SPELLING_ONLY:
            if spell_pred is not None:
                return self._apply_case(spell_pred.character, asr_char)
            return asr_char

        # SMART_ROUTER from here on.
        if spell_pred is None:
            return asr_char
        if asr_char is None:
            return spell_pred.character

        asr_lower = asr_char.lower()
        spell_lower = spell_pred.character.lower()
        if asr_lower == spell_lower:
            # Agreement on identity -- use the matcher's casing because
            # only the matcher can see the "capital" modifier.
            return asr_char

        asr_is_digit = asr_lower.isdigit()
        spell_is_digit = spell_lower.isdigit()
        if asr_is_digit and not spell_is_digit:
            # Cross-class disagreement: ASR's digit-word vocabulary
            # ("one", "six", "nine") is far more reliable than the
            # spelling model on real conversational speech (>97% vs
            # ~75%).  Trust ASR.
            return asr_char
        if spell_is_digit and not asr_is_digit:
            # The mirror case: ASR returned a letter but spelling thinks
            # this was a spoken digit. Spelling is the audio-side
            # specialist for that decision.
            return spell_pred.character

        # Same class, both confident enough to fire -- break the tie on
        # the spelling probability. The threshold is calibrated against
        # the spelling model's own per-bucket accuracy.
        if spell_pred.probability >= self._spelling_disagree_threshold:
            return self._apply_case(spell_pred.character, asr_char)
        return asr_char

    @staticmethod
    def _apply_case(char: str, hint: Optional[str]) -> str:
        """Up-case ``char`` iff the matcher's ``hint`` was upper-case.

        Used when the spelling model wins a tiebreak: the spelling head
        is case-blind (it's trained on phonemes), so we lean on the
        matcher to know whether the user said "capital".
        """
        if (
            hint is not None
            and hint.isalpha()
            and hint.isupper()
            and char.isalpha()
        ):
            return char.upper()
        return char

    def _speak_character(self, char: str) -> None:
        """Speak the TTS phrase for ``char`` if a TTS backend is wired."""
        if self._tts is None:
            return
        phrase = spoken_form(char)
        try:
            self._tts.say(phrase)
        except Exception as e:
            # A broken TTS must not break character recognition – the
            # callback has already committed ``char`` to the buffer.
            if self._debug:
                print(
                    f"[debug] tts.say({phrase!r}) failed: {e!r}",
                    file=sys.stderr,
                )

    def _play_error_feedback(self) -> None:
        """Ask the TTS backend to play its error beep, if it supports one.

        Called from the unrecognised-utterance branch so the user hears
        a short audible cue that their last word was ignored.  Tolerates
        backends that don't implement ``play_error`` (e.g. plain
        duck-typed stubs that only have ``say``) by checking for the
        method first.
        """
        if self._tts is None:
            return
        play_error = getattr(self._tts, "play_error", None)
        if play_error is None:
            return
        try:
            play_error()
        except Exception as e:
            if self._debug:
                print(
                    f"[debug] tts.play_error() failed: {e!r}",
                    file=sys.stderr,
                )

    def _run_spelling_predictor(self, line: TranscriptLine) -> None:
        """Run the SpellingCNN on the line's audio, stash the top-1 result.

        We always reset :attr:`last_spelling_prediction` first so callers
        can distinguish "this utterance had no audio" from "we still hold
        the previous utterance's prediction".  Failures (missing predictor,
        empty audio, ONNX runtime error) are swallowed and logged under
        ``debug=True`` -- a flaky ONNX session must not break character
        recognition, which is the listener's primary job.
        """
        self._last_spelling_prediction = None
        if self._spelling_predictor is None:
            return
        audio = line.audio_data
        if not audio:
            return
        try:
            self._last_spelling_prediction = self._spelling_predictor.predict(audio)
        except Exception as e:
            if self._debug:
                print(
                    f"[debug] spelling predictor failed: {e!r}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    import argparse
    import sys
    import threading
    import time

    from moonshine_voice import get_model_for_language
    from moonshine_voice.mic_transcriber import MicTranscriber

    parser = argparse.ArgumentParser(
        description="Alphanumeric dictation — speak letters, digits, and symbols one at a time."
    )
    parser.add_argument(
        "--language", type=str, default="en",
        help="Language to use for transcription",
    )
    parser.add_argument(
        "--model-arch", type=int, default=None,
        help="Model architecture to use for transcription",
    )
    parser.add_argument(
        "--spelling-onnx-path", type=str, default=None,
        help=(
            "Path to a SpellingCNN ONNX model. Defaults to "
            "find_default_spelling_onnx_path() (env var "
            f"${SPELLING_ONNX_ENV_VAR}, then bundled asset, then sibling "
            "'moonshine-spelling' checkout). Pass a path to override."
        ),
    )
    parser.add_argument(
        "--no-spelling-model", action="store_true",
        help=(
            "Skip the SpellingCNN entirely and run pure ASR-only fusion "
            "(useful for A/B comparisons or when the model isn't present)."
        ),
    )
    parser.add_argument(
        "--fusion-strategy",
        type=str,
        choices=[s.value for s in FusionStrategy],
        default=FusionStrategy.AUTO.value,
        help=(
            "How to combine matcher + spelling predictions. 'auto' picks "
            "smart_router when a SpellingCNN is loaded, asr_only otherwise."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Log unrecognised utterances to stderr",
    )
    args = parser.parse_args()

    model_path, model_arch = get_model_for_language(
        wanted_language=args.language, wanted_model_arch=args.model_arch,
    )

    # Resolve the spelling model once up-front so the user gets clear
    # feedback about what was loaded (or why nothing was) before the
    # mic starts and dictation output takes over the terminal.
    spelling_predictor: Optional[SpellingPredictor] = None
    if args.no_spelling_model:
        print(
            "Spelling model: disabled (--no-spelling-model). "
            "Falling back to ASR-only fusion.",
            file=sys.stderr,
        )
    else:
        onnx_path = args.spelling_onnx_path or find_default_spelling_onnx_path()
        if onnx_path is None:
            print(
                "Spelling model: none found. Set "
                f"${SPELLING_ONNX_ENV_VAR}, drop a copy at "
                f"<assets>/{_BUNDLED_SPELLING_ONNX_NAME}, or pass "
                "--spelling-onnx-path. Falling back to ASR-only.",
                file=sys.stderr,
            )
        else:
            try:
                spelling_predictor = SpellingPredictor(onnx_path)
                print(
                    f"Spelling model: loaded {onnx_path} "
                    f"({len(spelling_predictor.classes)} classes, "
                    f"sr={spelling_predictor.sample_rate}, "
                    f"clip={spelling_predictor.clip_seconds}s).",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"Spelling model: failed to load {onnx_path!r}: {e!r}. "
                    "Falling back to ASR-only.",
                    file=sys.stderr,
                )

    done = threading.Event()

    def on_event(event: AlphanumericEvent) -> None:
        if event.type == AlphanumericEventType.CHARACTER:
            print(event.character, end="", flush=True)
        elif event.type == AlphanumericEventType.UNDO:
            # Reprint the whole line after removing a character
            print(f"\r{event.text}", end="", flush=True)
        elif event.type == AlphanumericEventType.CLEAR:
            print(f"\r{' ' * 80}\r", end="", flush=True)
        elif event.type == AlphanumericEventType.STOPPED:
            print(flush=True)
            done.set()

    mic = MicTranscriber(model_path=model_path, model_arch=model_arch)
    listener = AlphanumericListener(
        on_event,
        spelling_predictor=spelling_predictor,
        fusion_strategy=args.fusion_strategy,
        debug=args.debug,
    )
    mic.add_listener(listener)

    print(
        f"Fusion strategy: {listener.fusion_strategy.value}. "
        "Listening — speak letters, digits, or symbols. "
        "Say \"stop\" or \"done\" to finish.",
        file=sys.stderr,
    )
    mic.start()
    try:
        while not done.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        mic.close()
        print(listener.text)
