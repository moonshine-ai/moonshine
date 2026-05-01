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

import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from moonshine_voice.transcriber import (
    TranscriptEvent,
    LineCompleted,
    LineTextChanged,
)


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

    # -- Callable interface (receives TranscriptEvent from Stream._emit) -----

    def __call__(self, event: TranscriptEvent) -> None:
        if self._stopped:
            return
        if self._use_line_completed and isinstance(event, LineCompleted):
            self._process_utterance(event.line.text, event.line.line_id)
        elif not self._use_line_completed and isinstance(event, LineTextChanged):
            self._process_utterance(event.line.text, event.line.line_id)

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

    def _process_utterance(self, raw_text: str, line_id: int) -> None:
        if line_id in self._processed_line_ids:
            return
        self._processed_line_ids.add(line_id)

        match = self._matcher.classify(raw_text)
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

        if match.type is AlphanumericEventType.CHARACTER and match.character is not None:
            self._buffer.append(match.character)
            self._callback(AlphanumericEvent(
                type=AlphanumericEventType.CHARACTER,
                character=match.character,
                text=self.text,
            ))
            self._speak_character(match.character)
            return

        if self._debug:
            print(f"[debug] unrecognised: {raw_text!r}", file=sys.stderr)
        self._play_error_feedback()

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
        "--debug", action="store_true",
        help="Log unrecognised utterances to stderr",
    )
    args = parser.parse_args()

    model_path, model_arch = get_model_for_language(
        wanted_language=args.language, wanted_model_arch=args.model_arch,
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
    listener = AlphanumericListener(on_event, debug=args.debug)
    mic.add_listener(listener)

    print("Listening — speak letters, digits, or symbols. "
          "Say \"stop\" or \"done\" to finish.", file=sys.stderr)
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
