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
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Union

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


# ---------------------------------------------------------------------------
# Vocabulary tables
# ---------------------------------------------------------------------------

_LETTER_WORDS: Dict[str, str] = {
    "a": "a", "ay": "a",
    "b": "b", "bee": "b",
    "c": "c", "see": "c", "sea": "c",
    "d": "d", "dee": "d",
    "e": "e",
    "f": "f", "ef": "f", "eff": "f",
    "g": "g", "gee": "g",
    "h": "h", "aitch": "h",
    "i": "i", "eye": "i",
    "j": "j", "jay": "j",
    "k": "k", "kay": "k",
    "l": "l", "el": "l", "ell": "l",
    "m": "m", "em": "m",
    "n": "n", "en": "n",
    "o": "o", "oh": "o",
    "p": "p", "pee": "p",
    "q": "q", "queue": "q", "cue": "q",
    "r": "r", "are": "r", "ar": "r",
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


def _build_lookup() -> Dict[str, str]:
    """Merge all character vocabularies into a single lookup table."""
    lookup: Dict[str, str] = {}
    lookup.update(_SPECIAL_CHAR_WORDS)
    lookup.update(_DIGIT_WORDS)
    lookup.update(_LETTER_WORDS)
    return lookup


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


class AlphanumericListener:
    """Listens for single-character dictation and assembles the result.

    This is a callable listener — pass it directly to
    ``transcriber.add_listener()`` or ``stream.add_listener()``.  It
    receives raw :class:`TranscriptEvent` objects and internally filters
    for the events it cares about.

    A single *callback* is invoked every time something meaningful
    happens.  The callback receives an :class:`AlphanumericEvent` whose
    ``type`` field describes what occurred:

    * ``CHARACTER`` — a letter, digit, or special character was recognised.
    * ``UNDO``      — the last character was removed.
    * ``CLEAR``     — the buffer was wiped.
    * ``STOPPED``   — the user said "stop" / "done" / "finish" / etc.

    Recognised spoken forms include:

    * **Letters** — ``"a"``–``"z"`` plus phonetic alternatives
      (``"ay"``, ``"bee"``, ``"see"``, …).
    * **Digits** — ``"zero"``–``"nine"`` (and homophones like
      ``"won"``/``"to"``/``"for"``/``"ate"``).
    * **Numbers 10–1000** — ``"ten"``, ``"twenty one"``,
      ``"three hundred and forty five"``, ``"one thousand"``, etc.
    * **Upper-case modifiers** — prefix a letter with ``"upper case"`` /
      ``"capital"`` / ``"cap"`` / ``"shift"`` to get upper-case.
    * **Special characters** — ``"dollar sign"``, ``"hash"``,
      ``"asterisk"``, ``"at sign"``, ``"exclamation mark"``, etc.
    * **Undo** — ``"undo"`` / ``"delete"`` / ``"backspace"``.
    * **Clear** — ``"clear"`` / ``"reset"`` / ``"start over"``.
    * **Stop** — ``"stop"`` / ``"end"`` / ``"finish"`` / ``"done"`` /
      ``"complete"`` / ``"submit"`` / ``"confirm"`` / ``"enter"``.

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
    debug:
        If ``True``, unrecognised utterances are logged to stderr.
    """

    def __init__(
        self,
        callback: Callable[[AlphanumericEvent], None],
        *,
        use_line_completed: bool = True,
        custom_words: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        self._callback = callback
        self._use_line_completed = use_line_completed
        self._debug = debug
        self._buffer: List[str] = []
        self._processed_line_ids: set = set()
        self._stopped = False

        self._lookup = _build_lookup()
        if custom_words:
            for spoken, char in custom_words.items():
                self._lookup[spoken.lower().strip()] = char

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

        if raw_text is None:
            return
        text = raw_text.strip().lower()
        if not text:
            return

        text = self._strip_trailing_punctuation(text)

        if text in _STOP_WORDS:
            self._stopped = True
            self._callback(AlphanumericEvent(
                type=AlphanumericEventType.STOPPED,
                character=None,
                text=self.text,
            ))
            return

        if text in _CLEAR_WORDS:
            self.clear()
            return

        if text in _UNDO_WORDS:
            self.undo()
            return

        make_upper = False
        for mod in sorted(_UPPER_MODIFIERS, key=len, reverse=True):
            if text.startswith(mod + " "):
                text = text[len(mod):].strip()
                make_upper = True
                break
            if text == mod:
                if self._debug:
                    print(f"[debug] bare modifier ignored: {raw_text!r}",
                          file=sys.stderr)
                return

        char = self._resolve(text)
        if char is None:
            if self._debug:
                print(f"[debug] unrecognised: {raw_text!r}",
                      file=sys.stderr)
            return

        if make_upper:
            char = char.upper()

        self._buffer.append(char)
        self._callback(AlphanumericEvent(
            type=AlphanumericEventType.CHARACTER,
            character=char,
            text=self.text,
        ))

    def _resolve(self, text: str) -> Optional[str]:
        """Try to resolve spoken text to a character or number string."""
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

    @staticmethod
    def _strip_trailing_punctuation(text: str) -> str:
        """Remove trailing periods/commas the STT may append."""
        while text and text[-1] in (".", ",", "!", "?"):
            text = text[:-1]
        return text.strip()


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
