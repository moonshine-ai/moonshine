"""Generator-based dialog flow runner for Moonshine Voice.

A *flow* is an ordinary Python generator function that yields prompts to the
runner and resumes with the user's answer:

    from moonshine_voice import dialog_flow as df

    def setup_wifi(d):
        ssid = yield d.ask("What's the name of your wifi network?")

        if not (yield d.confirm(f"I heard, {ssid}. Is that right?")):
            yield d.say("No problem, let's start over.")
            return

        password = yield d.ask(
            "Please spell the wifi password.",
            mode=df.SPELLED,
        )

        if (yield d.confirm("Would you like to hear it read back?")):
            yield d.say(f"I heard: {df.spell_out(password)}")

        if (yield d.confirm("Apply these changes?")):
            apply_wifi_config(ssid, password)
            yield d.say("Done. Your wifi is set up.")
        else:
            yield d.say("Okay, nothing changed.")

Register the flow against a trigger phrase and let the runner do the rest:

    dialog = (
        DialogFlow()
        .language("en")
        .listen_for("set up wifi", setup_wifi)
        .always("cancel", lambda d: d.cancel())
    )
    dialog.load()
    dialog.start_listening()

:meth:`DialogFlow.load` downloads and opens the speech recognition,
speech synthesis and phrase-matching models, and
:meth:`DialogFlow.start_listening` opens the microphone, so a voice
interface needs no other objects.  Supply your own with
:meth:`DialogFlow.use_mic_transcriber` / :meth:`DialogFlow.use_text_to_speech`
when you already have them, or drive the runner from text with
:meth:`DialogFlow.handle_utterance`.

There is no asyncio dependency – flows are driven synchronously from
whatever thread delivers transcript events, so flows can be unit-tested
without any audio, TTS, or event loop.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from moonshine_voice.alphanumeric_listener import (
    AlphanumericEventType,
    AlphanumericMatcher,
    digits_only_matcher,
    spoken_form,
)
from moonshine_voice.cached_embeddings import CachedEmbeddings
from moonshine_voice.download import (
    get_embedding_model,
    get_model_for_language,
    get_spelling_model_path,
)
from moonshine_voice.errors import MoonshineError
from moonshine_voice.mic_transcriber import MicTranscriber
from moonshine_voice.transcriber import (
    MOONSHINE_FLAG_SPELLING_MODE,
    Error,
    LineCompleted,
    LineStarted,
    ModelArch,
    TranscriptEventListener,
)
from moonshine_voice.tts import TextToSpeech, _parse_options_cli

# ---------------------------------------------------------------------------
# Input modes
# ---------------------------------------------------------------------------

FREE = "free"
SPELLED = "spelled"
DIGITS = "digits"
PHRASE = "phrase"


# ---------------------------------------------------------------------------
# Prompt objects – what a flow yields to the runner
# ---------------------------------------------------------------------------


@dataclass
class Prompt:
    """Base class for values a flow function may yield to the runner."""


@dataclass
class Say(Prompt):
    """Speak ``text`` and resume the generator once playback has finished."""

    text: str
    barge_in: bool = False


@dataclass
class Ask(Prompt):
    """Speak ``prompt`` and resume with the user's next utterance as a string."""

    prompt: str
    mode: str = FREE
    bias_terms: Optional[List[str]] = None
    timeout: Optional[float] = 8.0
    no_input_reprompt: Optional[str] = "Sorry, I didn't catch that. {prompt}"
    max_retries: int = 2


_DEFAULT_YES_PHRASES: Tuple[str, ...] = (
    "yes",
    "yeah",
    "yep",
    "correct",
    "that's right",
    "sure",
    "affirmative",
    "okay",
    "please do",
    "do it",
)

_DEFAULT_NO_PHRASES: Tuple[str, ...] = (
    "no",
    "nope",
    "incorrect",
    "that's wrong",
    "negative",
    "cancel",
    "don't do it",
    "stop",
)


@dataclass
class Confirm(Prompt):
    """Speak ``prompt`` and resume with a bool (yes / no)."""

    prompt: str
    timeout: Optional[float] = 6.0
    max_retries: int = 1
    threshold: float = 0.55
    no_input_reprompt: Optional[str] = (
        "Sorry, I didn't catch that. Was that a yes or a no? {prompt}"
    )
    yes_phrases: Sequence[str] = field(
        default_factory=lambda: _DEFAULT_YES_PHRASES
    )
    no_phrases: Sequence[str] = field(
        default_factory=lambda: _DEFAULT_NO_PHRASES
    )


@dataclass
class Choose(Prompt):
    """Speak ``prompt`` and resume with the key of the matched option.

    ``options`` maps option keys to canonical phrases.  Matching is done
    against the union of the key and its phrases, using the embedding
    model when available and falling back to substring matching.
    """

    prompt: str
    options: Mapping[str, Sequence[str]] = field(default_factory=dict)
    timeout: Optional[float] = 8.0
    max_retries: int = 2
    threshold: float = 0.55
    no_input_reprompt: Optional[str] = "Sorry, I didn't catch that. {prompt}"


# ---------------------------------------------------------------------------
# Exceptions thrown into the generator
# ---------------------------------------------------------------------------


class DialogError(Exception):
    """Base class for dialog-flow exceptions."""


class DialogCancelled(DialogError):
    """Raised into / from a flow to abandon it entirely."""


class DialogRestart(DialogError):
    """Raised into / from a flow to restart it from the beginning."""


class NoInputError(DialogError):
    """No utterance was received within the prompt's retry budget."""


class NoMatchError(DialogError):
    """Received an utterance but could not interpret it for this prompt."""


# ---------------------------------------------------------------------------
# Phrase matching via embeddings
# ---------------------------------------------------------------------------


class EmbeddingBackend(Protocol):
    """Minimal interface the phrase matcher needs from an embedding source.

    The internal embedding model satisfies this protocol via its
    :meth:`calculate_embedding` and :meth:`distance` methods – the
    latter is a thin wrapper around the native
    ``moonshine_calculate_embedding_distance`` C API so scoring happens
    in C rather than Python.
    """

    def calculate_embedding(self, sentence: str) -> Sequence[float]: ...

    def distance(
        self, embedding_a: Sequence[float], embedding_b: Sequence[float]
    ) -> float: ...


class PhraseMatcher:
    """Match an utterance to one of several key→phrases groups via embeddings.

    This is a tiny wrapper around an :class:`EmbeddingBackend`.  At
    construction time, the backend is
    used to compute an embedding for every phrase in every group; at
    match time, the utterance is embedded once and compared against
    every phrase using cosine similarity.  The key of the best-scoring
    phrase (above ``threshold``) is returned, or *None* if nothing
    clears the threshold.

    Use this to replace string / substring matching on user utterances
    with fuzzy, semantics-aware matching.

    Example::

        yes_no = PhraseMatcher(
            embedding_backend,
            {"yes": ["yes", "sure", "please"],
             "no":  ["no", "nope", "cancel"]},
            threshold=0.6,
        )
        assert yes_no.match("please go ahead") == "yes"
        assert yes_no.match("don't do that")    == "no"
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        phrases_by_key: Mapping[str, Sequence[str]],
        *,
        threshold: float = 0.55,
    ):
        if backend is None:
            raise ValueError("PhraseMatcher requires an embedding backend")
        self._backend = backend
        self._threshold = float(threshold)
        self._phrase_embeddings: Dict[str, List[Sequence[float]]] = {}
        for key, phrases in phrases_by_key.items():
            embeddings: List[Sequence[float]] = []
            for phrase in phrases:
                if not phrase:
                    continue
                try:
                    embeddings.append(backend.calculate_embedding(phrase))
                except Exception as e:
                    print(
                        f"PhraseMatcher: failed to embed {phrase!r}: {e}",
                        file=sys.stderr,
                    )
            self._phrase_embeddings[key] = embeddings

    @property
    def threshold(self) -> float:
        return self._threshold

    def match(self, utterance: str) -> Optional[str]:
        """Return the best-matching key, or *None* if below threshold."""
        key, _score = self.match_with_score(utterance)
        return key

    def match_with_score(
        self, utterance: str
    ) -> Tuple[Optional[str], float]:
        """Return ``(key, similarity)`` of the best match above threshold.

        When nothing clears ``threshold`` returns ``(None, best_sim)`` –
        callers can inspect the score for diagnostics / reprompts.
        """
        if not utterance:
            return None, 0.0
        try:
            u_emb = self._backend.calculate_embedding(utterance)
        except Exception as e:
            print(f"PhraseMatcher: failed to embed utterance: {e}", file=sys.stderr)
            return None, 0.0
        best_key: Optional[str] = None
        best_sim: float = -1.0
        for key, embeddings in self._phrase_embeddings.items():
            for e in embeddings:
                try:
                    sim = self._backend.distance(u_emb, e)
                except Exception as exc:
                    print(
                        f"PhraseMatcher: distance() failed: {exc}",
                        file=sys.stderr,
                    )
                    return None, 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_key = key
        if best_key is not None and best_sim >= self._threshold:
            return best_key, best_sim
        return None, max(best_sim, 0.0)


class SubstringMatcher:
    """Match an utterance by case-insensitive substring, with no model.

    Interchangeable with :class:`PhraseMatcher`, and used in its place
    when :meth:`DialogFlow.use_embeddings` is off.  It only recognises
    what the user literally said, so it's meant for tests and offline
    smoke checks rather than for real speech, where the wording never
    matches the trigger phrase exactly.

    A phrase matches when it appears in the utterance or the utterance
    appears in it; the longest matching phrase wins, so "turn off the
    lights" beats "lights". The score is the matched phrase's share of
    the utterance, which lets ``threshold`` behave roughly as it does
    for embeddings.
    """

    def __init__(
        self,
        phrases_by_key: Mapping[str, Sequence[str]],
        *,
        threshold: float = 0.55,
    ):
        self._threshold = float(threshold)
        self._phrases_by_key: Dict[str, List[str]] = {
            key: [p.strip().lower() for p in phrases if p and p.strip()]
            for key, phrases in phrases_by_key.items()
        }

    @property
    def threshold(self) -> float:
        return self._threshold

    def match(self, utterance: str) -> Optional[str]:
        key, _score = self.match_with_score(utterance)
        return key

    def match_with_score(self, utterance: str) -> Tuple[Optional[str], float]:
        text = (utterance or "").strip().lower()
        if not text:
            return None, 0.0
        best_key: Optional[str] = None
        best_len = 0
        for key, phrases in self._phrases_by_key.items():
            for phrase in phrases:
                if phrase in text or text in phrase:
                    if len(phrase) > best_len:
                        best_len = len(phrase)
                        best_key = key
        if best_key is None:
            return None, 0.0
        score = min(1.0, best_len / max(len(text), 1))
        return best_key, score


PhraseMatcherFactory = Callable[
    [Mapping[str, Sequence[str]], float], Optional[PhraseMatcher]
]


# ---------------------------------------------------------------------------
# Dialog – the context object passed to every flow function
# ---------------------------------------------------------------------------


class Dialog:
    """Context object handed to a flow as its first argument.

    Each method returns a :class:`Prompt` that the flow yields; the runner
    carries out the prompt and sends the result (if any) back into the
    generator.  ``Dialog`` itself performs no I/O, which keeps flows easy
    to unit-test: pass a ``Dialog`` to the flow, iterate the generator, and
    drive it with ``.send()``.
    """

    def __init__(self, trigger_phrase: str = "", *, state: Optional[Dict[str, Any]] = None):
        self.trigger_phrase = trigger_phrase
        self.state: Dict[str, Any] = dict(state) if state else {}
        self._last_spoken_prompt: Optional[str] = None

    def say(self, text: str, *, barge_in: bool = False) -> Say:
        self._last_spoken_prompt = text
        return Say(text=text, barge_in=barge_in)

    def ask(
        self,
        prompt: str,
        *,
        mode: str = FREE,
        bias_terms: Optional[Sequence[str]] = None,
        timeout: Optional[float] = 8.0,
        no_input_reprompt: Optional[str] = "Sorry, I didn't catch that. {prompt}",
        max_retries: int = 2,
    ) -> Ask:
        self._last_spoken_prompt = prompt
        return Ask(
            prompt=prompt,
            mode=mode,
            bias_terms=list(bias_terms) if bias_terms is not None else None,
            timeout=timeout,
            no_input_reprompt=no_input_reprompt,
            max_retries=max_retries,
        )

    def confirm(
        self,
        prompt: str,
        *,
        timeout: Optional[float] = 6.0,
        max_retries: int = 1,
    ) -> Confirm:
        self._last_spoken_prompt = prompt
        return Confirm(prompt=prompt, timeout=timeout, max_retries=max_retries)

    def choose(
        self,
        prompt: str,
        options: Mapping[str, Sequence[str]],
        *,
        timeout: Optional[float] = 8.0,
        max_retries: int = 2,
    ) -> Choose:
        self._last_spoken_prompt = prompt
        return Choose(
            prompt=prompt,
            options={k: list(v) for k, v in options.items()},
            timeout=timeout,
            max_retries=max_retries,
        )

    # -- flow control – these raise into the generator ---------------------

    def cancel(self) -> NoReturn:
        raise DialogCancelled()

    def restart(self) -> NoReturn:
        raise DialogRestart()

    def replay_last_prompt(self) -> Optional[Say]:
        """Return a :class:`Say` that re-speaks the most recent prompt.

        Intended for global "repeat" handlers; returns *None* if nothing has
        been spoken yet.
        """
        if self._last_spoken_prompt is None:
            return None
        return Say(text=self._last_spoken_prompt)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FlowFn = Callable[[Dialog], Iterator[Prompt]]
GlobalHandler = Callable[[Dialog], Optional[Prompt]]


# ---------------------------------------------------------------------------
# DialogFlow – the runner / listener
# ---------------------------------------------------------------------------


class _AlphaSession:
    """In-progress spelled / digit input buffered across utterances."""

    def __init__(self, matcher: AlphanumericMatcher):
        self.matcher = matcher
        self.buffer: List[str] = []


class _TranscriptBridge(TranscriptEventListener):
    """Feeds a transcriber's events into a :class:`DialogFlow`.

    Kept separate from the runner so that ``DialogFlow.on_error`` can be
    the public "tell me when something went wrong" setter rather than the
    transcript-event callback of the same name.
    """

    def __init__(self, runner: DialogFlow):
        self._runner = runner

    def on_line_started(self, event: LineStarted) -> None:
        self._runner._on_line_started(event)

    def on_line_completed(self, event: LineCompleted) -> None:
        self._runner._on_line_completed(event)

    def on_error(self, event: Error) -> None:
        self._runner._on_transcriber_error(event)


class _ActiveFlow:
    """Per-session state for a running flow."""

    def __init__(self, flow_fn: FlowFn, trigger_phrase: str):
        self.flow_fn = flow_fn
        self.trigger_phrase = trigger_phrase
        self.dialog = Dialog(trigger_phrase)
        self.generator: Iterator[Prompt] = flow_fn(self.dialog)
        self.current_prompt: Optional[Prompt] = None
        self.retry_count: int = 0
        self.alpha_session: Optional[_AlphaSession] = None


class DialogFlow:
    """Runner that drives generator-based conversational flows.

    Configure with the chainable setters, register flows with
    :meth:`listen_for` and :meth:`always`, then call :meth:`load` to open
    the models and :meth:`start_listening` to go live::

        dialog = (
            DialogFlow()
            .language("en")
            .listen_for("set up wifi", setup_wifi)
            .always("cancel", lambda d: d.cancel())
        )
        dialog.load()
        dialog.start_listening()

    Completed transcript lines are routed either to a matching trigger
    phrase (when no flow is active) or to the currently suspended
    generator (when one is).

    The runner is synchronous: when a flow yields a :class:`Say`, the
    runner speaks and blocks until the utterance has been played, then
    resumes the generator.  When a flow yields an input-expecting prompt
    (:class:`Ask` / :class:`Confirm` / :class:`Choose`), the runner speaks
    the prompt and returns control to the caller; the next completed
    transcript line resumes the generator.

    The runner mutes its microphone while the assistant is talking and
    flips the C++ spelling-CNN fusion path on for the duration of a
    ``SPELLED`` / ``DIGITS`` prompt, so neither needs wiring up by hand.

    """

    def __init__(self) -> None:
        # -- model configuration, applied by :meth:`load` -------------------
        self._language = "en"
        self._model_arch: Optional[ModelArch] = None
        self._voice: Optional[str] = None
        self._model_root: Optional[Path] = None
        self._wants_microphone = True
        self._wants_speech = True
        self._output_device: Optional[Any] = None
        self._tts_options: Optional[Dict[str, Any]] = None

        # -- engines --------------------------------------------------------
        self._tts: Optional[Any] = None
        self._mic: Optional[Any] = None
        self._owns_tts = False
        self._owns_mic = False
        self._listening = False
        self._bridge = _TranscriptBridge(self)
        self._bridged: List[Any] = []

        # -- observer callbacks ---------------------------------------------
        self._progress_fn: Optional[Callable[[float, str], None]] = None
        self._heard_fn: Optional[Callable[[str], None]] = None
        self._said_fn: Optional[Callable[[str], None]] = None
        self._error_fn: Optional[Callable[[BaseException], None]] = None

        self._speak_fn: Optional[Callable[[str], None]] = None
        self._mute_fn: Optional[Callable[[bool], None]] = None
        self._spelling_mode_fn: Optional[Callable[[bool], None]] = None
        self._success_beep_fn: Optional[Callable[[], None]] = None
        self._error_beep_fn: Optional[Callable[[], None]] = None
        self._beeps_enabled = True

        self._use_embeddings = True
        self._spelling_mode_active = False
        self._trigger_threshold = 0.7
        self._spell_feedback = True
        self._log_io = False
        self._ignore_stt_during_tts = True
        self._speaking = False
        # Transcript line IDs whose ``LineStarted`` event fired while
        # the assistant was talking; populated in :meth:`on_line_started`
        # and consumed in :meth:`on_line_completed` to drop self-capture
        # without making the user wait for an arbitrary post-TTS grace
        # window.  Protected by ``_lock`` since the listener thread that
        # delivers transcript events is independent of the thread driving
        # the flow.
        self._suspect_line_ids: set = set()
        self._debug = False
        self._log_start: Optional[float] = None
        self._log_last: Optional[float] = None

        # The default :class:`PhraseMatcher` factory runs on an
        # :class:`EmbeddingBackend`.  Library-level constants (e.g. the
        # default yes/no phrases) have their embeddings shipped via
        # ``assets/cached_embeddings.tsv`` and loaded by
        # :class:`CachedEmbeddings`; cache misses (typically user
        # utterances) fall through to the embedding model, which
        # :meth:`_embedding_backend` loads on first use so that merely
        # constructing a runner never downloads anything.
        self._cached_embeddings: Optional[CachedEmbeddings] = None
        self._owned_model: Optional[Any] = None
        self._backend: Optional[Any] = None
        self._phrase_matcher_factory: Optional[PhraseMatcherFactory] = (
            self._default_phrase_matcher
        )

        self._flows: Dict[str, FlowFn] = {}
        self._globals: Dict[str, GlobalHandler] = {}

        self._active: Optional[_ActiveFlow] = None
        self._lock = threading.RLock()

        self._matcher_cache: Dict[Any, Optional[PhraseMatcher]] = {}
        self._trigger_matcher: Optional[PhraseMatcher] = None

        # Cached alphanumeric matchers.  These are stateless (only the
        # per-prompt ``_AlphaSession`` holds buffer state), so one
        # instance per mode is enough.  Created on demand the first
        # time a ``SPELLED`` / ``DIGITS`` prompt is entered.
        self._spelled_matcher: Optional[AlphanumericMatcher] = None
        self._digits_matcher: Optional[AlphanumericMatcher] = None

    def _default_phrase_matcher(
        self,
        phrases_by_key: Mapping[str, Sequence[str]],
        threshold: float,
    ) -> Optional[PhraseMatcher]:
        backend = self._embedding_backend()
        if backend is None:
            return SubstringMatcher(phrases_by_key, threshold=threshold)
        return PhraseMatcher(backend, phrases_by_key, threshold=threshold)

    # -- configuration -------------------------------------------------------
    #
    # Every setter returns ``self`` so a runner can be built in one
    # expression, and all of them must be called before :meth:`load`.

    def language(self, code: str) -> DialogFlow:
        """Set the language for both recognition and speech (default ``"en"``)."""
        self._language = code
        return self

    def model_arch(self, arch: ModelArch) -> DialogFlow:
        """Pick a specific speech recognition model size."""
        self._model_arch = arch
        return self

    def voice(self, voice_id: str) -> DialogFlow:
        """Choose the synthesis voice used to speak prompts."""
        self._voice = voice_id
        return self

    def speech_options(self, options: Mapping[str, Any]) -> DialogFlow:
        """Pass advanced options straight through to the speech synthesizer."""
        self._tts_options = dict(options)
        return self

    def models_from(self, directory: Union[str, Path]) -> DialogFlow:
        """Read and cache model files under ``directory`` instead of the default cache."""
        self._model_root = Path(directory)
        return self

    def microphone(self, enabled: bool = True) -> DialogFlow:
        """Whether :meth:`load` should open a microphone (default ``True``).

        Turn this off to drive the runner from text with
        :meth:`handle_utterance`, or when you supply your own transcriber
        via :meth:`use_mic_transcriber`.
        """
        self._wants_microphone = bool(enabled)
        return self

    def speech(self, enabled: bool = True) -> DialogFlow:
        """Whether :meth:`load` should open a synthesizer (default ``True``).

        Turn this off for a silent runner: prompts are still logged and
        flows still advance, they just aren't spoken aloud.
        """
        self._wants_speech = bool(enabled)
        return self

    def output_device(self, device: Union[int, str]) -> DialogFlow:
        """Pin speech playback to a specific audio output device.

        Needed on machines where the host default isn't the speaker you
        want — a Raspberry Pi that defaults to HDMI while the speakers
        are on the 3.5 mm jack, for example.
        """
        self._output_device = device
        return self

    def trigger_threshold(self, threshold: float) -> DialogFlow:
        """Set the similarity a phrase must reach to fire (default ``0.7``).

        Raise it towards ``1.0`` to demand a closer match when triggers
        are firing on unrelated speech; lower it when they aren't firing
        on genuine attempts.
        """
        self._trigger_threshold = float(threshold)
        self._invalidate_trigger_matcher()
        return self

    def on_progress(self, callback: Callable[[float, str], None]) -> DialogFlow:
        """Report model download and load progress as ``(fraction, filename)``."""
        self._progress_fn = callback
        return self

    def on_heard(self, callback: Callable[[str], None]) -> DialogFlow:
        """Report every utterance the runner receives from the microphone."""
        self._heard_fn = callback
        return self

    def on_said(self, callback: Callable[[str], None]) -> DialogFlow:
        """Report every prompt the runner speaks."""
        self._said_fn = callback
        return self

    def on_error(
        self, callback: Callable[[BaseException], None]
    ) -> DialogFlow:
        """Report errors raised by a flow or by the audio pipeline.

        Without a handler the runner prints the error to stderr and
        carries on; a flow that raises is always torn down either way, so
        one bad turn can't wedge the runner.
        """
        self._error_fn = callback
        return self

    def speak_with(self, speak: Callable[[str], None]) -> DialogFlow:
        """Speak prompts with your own callable instead of the built-in synthesizer.

        ``speak(text)`` must block until playback has finished, since the
        runner resumes the flow as soon as it returns.  Setting this stops
        :meth:`load` from creating a synthesizer of its own.
        """
        self._speak_fn = speak
        return self

    def beeps(self, enabled: bool = True) -> DialogFlow:
        """Whether to play the recognition cue tones (default ``True``).

        The runner plays a short "got it" tone the moment an utterance
        matches a trigger or answers a prompt, and a distinct "didn't get
        that" tone when nothing matched, so a misrecognition never ends in
        silence.
        """
        self._beeps_enabled = bool(enabled)
        return self

    def spell_feedback(self, enabled: bool = True) -> DialogFlow:
        """Whether to echo each character during spelled input (default ``True``).

        Every character recognised during a ``SPELLED`` / ``DIGITS``
        prompt is spoken back using :func:`spoken_form` (``"haitch"`` for
        ``"h"``, ``"capital ay"`` for ``"A"``, ``"hash"`` for ``"#"``),
        and a "delete" / "scratch that" is echoed as ``"deleting
        <character>"`` so the user hears that the right letter came off
        the end.  Turn it off when there's no audio output and the echo
        would just be log spam.
        """
        self._spell_feedback = bool(enabled)
        return self

    def log_io(self, enabled: bool = True) -> DialogFlow:
        """Log the dialogue to stderr as ``user: …`` / ``assistant: …`` lines.

        This is the user-facing transcript of inputs and outputs; use
        :meth:`debug` for the verbose internal trace.  Off by default so
        callers that already format their own transcript don't end up
        with duplicate lines.
        """
        self._log_io = bool(enabled)
        return self

    def barge_in(self, enabled: bool = True) -> DialogFlow:
        """Allow the user to interrupt the assistant mid-prompt (default off).

        By default every utterance that arrives while the assistant is
        talking is dropped, because it's usually the microphone hearing
        the speakers.  That's a software guard on top of muting the mic:
        muting minimises self-capture, but on devices with weak echo
        cancellation the recognizer can still latch onto audio captured
        just before the mute, or onto speaker bleed the cancellation
        didn't suppress.  Enable barge-in only when you have reliable
        echo cancellation.
        """
        self._ignore_stt_during_tts = not bool(enabled)
        return self

    def debug(self, enabled: bool = True) -> DialogFlow:
        """Trace every internal stage transition, with timings, to stderr."""
        self._debug = bool(enabled)
        return self

    def use_embeddings(self, enabled: bool = True) -> DialogFlow:
        """Whether to match phrases semantically (default ``True``).

        With embeddings on, the runner downloads a small language model
        on :meth:`load` and matches what the user said against trigger
        phrases by meaning, so "set up wifi" also fires on "I need to get
        online".  Turn it off to fall back to case-insensitive substring
        matching and load no model, which is what offline tests usually
        want.
        """
        self._use_embeddings = bool(enabled)
        return self

    def use_cached_embeddings(
        self, cache: CachedEmbeddings
    ) -> DialogFlow:
        """Supply pre-computed phrase embeddings, bypassing the model for hits."""
        self._cached_embeddings = cache
        self._backend = cache
        return self

    def use_phrase_matcher(
        self, factory: PhraseMatcherFactory
    ) -> DialogFlow:
        """Replace the built-in phrase matching with your own implementation."""
        self._phrase_matcher_factory = factory
        return self

    def use_text_to_speech(self, tts: Any) -> DialogFlow:
        """Speak with an existing :class:`TextToSpeech` instead of creating one.

        The runner won't close a synthesizer it didn't create.
        """
        self._tts = tts
        self._owns_tts = False
        return self

    def use_mic_transcriber(self, transcriber: Any) -> DialogFlow:
        """Listen to an existing transcriber instead of opening a microphone.

        Accepts a :class:`MicTranscriber` or any object with the same
        ``add_listener`` / ``start`` / ``stop`` shape — a plain
        :class:`Transcriber` fed from a file works, which is handy for
        testing a flow against recorded audio.  The runner won't close a
        transcriber it didn't create.
        """
        self._mic = transcriber
        self._owns_mic = False
        self._attach_bridge(transcriber)
        return self

    # -- embedding backend ---------------------------------------------------

    def _embedding_backend(self) -> Optional[Any]:
        """The embedding backend, loading the phrase model on first use.

        :meth:`load` normally warms this up front, but it stays lazy so a
        runner driven purely by :meth:`handle_utterance` still works
        without an explicit load.  Returns *None* when embeddings are
        turned off, which leaves matching to the substring fallback.
        """
        with self._lock:
            if self._backend is not None:
                return self._backend
            if not self._use_embeddings:
                return None
            from moonshine_voice.embedding_model import EmbeddingModel

            self._report_progress(0.0, "embedding model")
            model_path, model_arch = get_embedding_model(
                cache_root=self._model_root
            )
            self._owned_model = EmbeddingModel(
                model_path=model_path, model_arch=model_arch
            )
            self._backend = CachedEmbeddings(fallback=self._owned_model)
            self._report_progress(1.0, "embedding model")
            return self._backend

    def _report_progress(self, fraction: float, name: str) -> None:
        if self._progress_fn is None:
            return
        try:
            self._progress_fn(fraction, name)
        except Exception as e:
            self._log(f"progress callback failed: {e!r}")

    # -- lifecycle ------------------------------------------------------------

    def load(self) -> DialogFlow:
        """Download and open everything the runner needs, and return self.

        Opens the phrase-matching model, a speech synthesizer, and a
        microphone transcriber, skipping any of them you've already
        supplied or turned off.  Blocking, since the first call may have
        to download models; report progress with :meth:`on_progress`.

        Call :meth:`start_listening` afterwards to begin listening.
        """
        if self._wants_speech and self._tts is None and self._speak_fn is None:
            self._report_progress(0.0, "speech synthesis")
            self._tts = (
                TextToSpeech()
                .language(self._language)
                .debug(self._debug)
                .output_device(self._output_device)
            )
            if self._voice is not None:
                self._tts.voice(self._voice)
            if self._tts_options:
                self._tts.options(self._tts_options)
            if self._progress_fn is not None:
                self._tts.on_progress(
                    lambda fraction, name: self._report_progress(fraction, name)
                )
            self._tts.load()
            self._owns_tts = True
            self._report_progress(1.0, "speech synthesis")

        if self._wants_microphone and self._mic is None:
            self._report_progress(0.0, "speech recognition")
            # Resolved here rather than left to MicTranscriber.load() so the
            # download lands under this runner's cache root.
            model_path, model_arch = get_model_for_language(
                self._language,
                self._model_arch,
                cache_root=self._model_root,
                on_progress=(
                    None
                    if self._progress_fn is None
                    else lambda fraction, name: self._report_progress(fraction, name)
                ),
            )
            # The spelling CNN is what makes dictated passwords and codes
            # accurate, but it isn't published for every language, and its
            # absence only costs accuracy inside SPELLED / DIGITS prompts.
            spelling_model_path: Optional[str] = None
            try:
                spelling_model_path = get_spelling_model_path(
                    self._language, cache_root=self._model_root
                )
            except Exception as e:
                self._log(f"load: spelling model lookup failed: {e!r}")
            self._mic = (
                MicTranscriber()
                .models_from(model_path)
                .model_arch(model_arch)
                .spelling_model(spelling_model_path)
            )
            self._mic.load()
            self._owns_mic = True
            self._attach_bridge(self._mic)
            self._report_progress(1.0, "speech recognition")

        self._wire_transcriber_hooks()
        self._embedding_backend()
        return self

    def _attach_bridge(self, transcriber: Any) -> None:
        """Route a transcriber's completed lines into this runner."""
        add_listener = getattr(transcriber, "add_listener", None)
        if not callable(add_listener):
            raise TypeError(
                "transcriber must have an add_listener method; got "
                f"{type(transcriber).__name__}"
            )
        if any(existing is transcriber for existing in self._bridged):
            return
        add_listener(self._bridge)
        self._bridged.append(transcriber)
        self._wire_transcriber_hooks()

    def _wire_transcriber_hooks(self) -> None:
        """Point the mute and spelling-mode hooks at the current transcriber.

        Both are duck-typed: a plain :class:`Transcriber` driven from a
        file has no microphone to mute, and a transcriber built without a
        spelling model ignores the flag, so a missing hook just means the
        corresponding behavior is a no-op.
        """
        mic = self._mic
        if mic is None:
            return
        if self._mute_fn is None and hasattr(mic, "_should_listen"):
            def mute(should_mute: bool) -> None:
                mic._should_listen = not should_mute

            self._mute_fn = mute
        if self._spelling_mode_fn is None:
            set_flags = getattr(mic, "set_transcribe_flags", None)
            if callable(set_flags):
                def set_spelling_mode(active: bool) -> None:
                    set_flags(MOONSHINE_FLAG_SPELLING_MODE if active else 0)

                self._spelling_mode_fn = set_spelling_mode

    def start_listening(self) -> None:
        """Start listening on the microphone.

        Calls :meth:`load` first if you haven't.  Returns as soon as the
        microphone is live — transcript lines arrive on the audio thread
        and drive your flows from there, so the caller is free to sleep,
        run a UI, or do anything else.
        """
        if self._mic is None:
            if not self._wants_microphone:
                raise MoonshineError(
                    "start_listening() needs a microphone, but this runner was "
                    "built with microphone(False). Either enable it, supply "
                    "one with use_mic_transcriber(), or drive the runner with "
                    "handle_utterance()."
                )
            self.load()
        if self._listening:
            return
        self._mic.start()
        self._listening = True

    def stop_listening(self) -> None:
        """Stop listening.  Safe to call when already stopped."""
        if self._mic is None or not self._listening:
            return
        self._mic.stop()
        self._listening = False

    def close(self) -> None:
        """Release everything this runner opened.

        Only closes what it created itself: a synthesizer or transcriber
        you passed in stays yours to close.
        """
        self.stop_listening()
        with self._lock:
            embedder, self._owned_model = self._owned_model, None
            self._backend = self._cached_embeddings
            mic, self._mic = self._mic, None
            owns_mic, self._owns_mic = self._owns_mic, False
            tts, self._tts = self._tts, None
            owns_tts, self._owns_tts = self._owns_tts, False
            bridged, self._bridged = self._bridged, []
        # Detach before closing, so a transcriber the caller still owns
        # isn't left delivering lines to a runner that's shut down.
        for transcriber in bridged:
            try:
                transcriber.remove_listener(self._bridge)
            except Exception as e:
                self._log(f"close: remove_listener failed: {e!r}")
        for resource, owned in ((embedder, True), (mic, owns_mic),
                                (tts, owns_tts)):
            if resource is None or not owned:
                continue
            try:
                resource.close()
            except Exception as e:
                self._log(f"close: {type(resource).__name__} failed: {e!r}")

    # -- registration -------------------------------------------------------

    def listen_for(self, trigger_phrase: str, flow: FlowFn) -> DialogFlow:
        """Start ``flow`` whenever the user says something like ``trigger_phrase``.

        Matching is by meaning rather than by wording: the phrase is
        embedded once here and compared against each utterance's embedding
        by cosine similarity, so "set up wifi" also fires on "I need to
        get online".
        """
        if not callable(flow):
            raise TypeError("flow must be callable")
        self._flows[trigger_phrase] = flow
        self._invalidate_trigger_matcher()
        return self

    def unregister_flow(self, trigger_phrase: str) -> bool:
        """Remove a flow registered with :meth:`listen_for`."""
        removed = self._flows.pop(trigger_phrase, None) is not None
        if removed:
            self._invalidate_trigger_matcher()
        return removed

    def always(self, trigger_phrase: str, handler: GlobalHandler) -> DialogFlow:
        """Register a phrase that stays live even while a flow is running.

        ``handler`` receives the current :class:`Dialog` (a fresh one when
        no flow is active) and may return a :class:`Prompt` to speak, or
        *None*.  It may also raise :class:`DialogCancelled` or
        :class:`DialogRestart` to influence the active flow, which is how
        "cancel" and "start over" are usually wired up.
        """
        self._globals[trigger_phrase] = handler
        self._invalidate_trigger_matcher()
        return self

    def _invalidate_trigger_matcher(self) -> None:
        self._trigger_matcher = None

    # -- inspection ---------------------------------------------------------

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def active_trigger(self) -> Optional[str]:
        with self._lock:
            return self._active.trigger_phrase if self._active else None

    @property
    def registered_flows(self) -> List[str]:
        return list(self._flows.keys())

    # -- transcript events, delivered by _TranscriptBridge ------------------

    def _on_line_started(self, event: LineStarted) -> None:
        """Tag any transcript line that opens while we're talking.

        Streaming ASRs commonly finalise a ``LineCompleted`` event
        several hundred milliseconds *after* the audio that produced
        it arrived, so a transcript started while the assistant was
        speaking often only completes once ``_speak`` has already
        returned.  Decide self-capture status at line-birth instead
        of completion: if a line was opened during TTS playback we
        record its ID here, then drop it on completion in
        :meth:`on_line_completed` regardless of how long the ASR took
        to finalise it.  Lines that open *after* TTS ends are accepted
        with no added latency – the user can talk the moment the
        assistant stops, no grace window required.

        Only takes effect when ``ignore_stt_during_tts`` is on; when
        it's off (true barge-in mode) we leave the set empty so every
        line completes through to :meth:`handle_utterance`.
        """
        if not self._ignore_stt_during_tts or not self._speaking:
            return
        line = getattr(event, "line", None)
        if line is None:
            return
        line_id = getattr(line, "line_id", None)
        if line_id is None:
            return
        with self._lock:
            self._suspect_line_ids.add(line_id)
        self._log(
            f"on_line_started: tagging line id={line_id} as self-capture "
            "(opened during TTS playback)"
        )

    def _on_line_completed(self, event: LineCompleted) -> None:
        if not event.line or not event.line.text:
            return
        utterance = event.line.text.strip()
        if not utterance:
            return
        line_id = getattr(event.line, "line_id", None)
        if line_id is not None:
            with self._lock:
                suspect = line_id in self._suspect_line_ids
                if suspect:
                    self._suspect_line_ids.discard(line_id)
            if suspect:
                self._log(
                    f"on_line_completed: dropping line id={line_id} "
                    f"utterance={_summarise(utterance)!r} (started during TTS)"
                )
                if self._log_io:
                    print(
                        f"user (ignored, self-capture): {utterance}",
                        file=sys.stderr,
                        flush=True,
                    )
                return
        self.handle_utterance(utterance)

    def _on_transcriber_error(self, event: Error) -> None:
        message = getattr(event, "message", None) or str(event)
        self._report_error(MoonshineError(message))

    def _report_error(self, error: BaseException) -> None:
        """Hand an error to the caller's handler, or print it if there isn't one."""
        if self._error_fn is None:
            print(f"DialogFlow: {error}", file=sys.stderr, flush=True)
            return
        try:
            self._error_fn(error)
        except Exception as e:
            print(f"DialogFlow: on_error handler failed: {e!r}", file=sys.stderr)

    # -- core dispatch ------------------------------------------------------

    def handle_utterance(self, utterance: str) -> bool:
        """Route an utterance.

        Returns ``True`` if it was consumed by a flow or a global handler,
        ``False`` otherwise.  All matching against registered triggers is
        done via embedding similarity – no string matching.

        When an active flow is waiting on a :class:`Ask` in ``SPELLED`` /
        ``DIGITS`` mode and the utterance is recognised by the
        alphanumeric matcher (a letter, digit, undo, clear, or stop
        word), that takes priority over fuzzy global trigger matching.
        This avoids accidents like ``"delete"`` embedding-matching
        ``"cancel"`` and tearing down the dictation flow.
        """
        self._log(
            f"handle_utterance: begin utterance={_summarise(utterance)!r} "
            f"active={'yes' if self._active is not None else 'no'}"
        )
        if self._heard_fn is not None:
            try:
                self._heard_fn(utterance)
            except Exception as e:
                self._log(f"on_heard handler failed: {e!r}")
        # Drop self-capture / TTS bleed-through.  On devices with weak
        # echo cancellation the STT can hand us a transcript of our
        # own speech (or of audio captured a beat before ``mute_fn``
        # silenced the mic); routing that into the live flow would
        # reliably trigger bogus retries, false confirmations, and
        # spurious global triggers.  When ``ignore_stt_during_tts`` is
        # on we discard the utterance with a debug log line so it can't
        # advance the flow or match a global trigger.
        if self._ignore_stt_during_tts and self._speaking:
            self._log(
                f"handle_utterance: dropping {_summarise(utterance)!r} "
                "(TTS in progress)"
            )
            if self._log_io:
                print(
                    f"user (ignored, TTS speaking): {utterance}",
                    file=sys.stderr,
                    flush=True,
                )
            return False
        if self._log_io:
            print(f"user: {utterance}", file=sys.stderr, flush=True)
        with self._lock:
            active = self._active

        if active is not None and self._should_short_circuit_to_alpha(
            active, utterance
        ):
            self._log("handle_utterance: alpha short-circuit → deliver")
            # The success/error cue is played by ``_deliver_to_active``
            # once interpretation has decided whether the line was
            # recognized — partial spelled input keeps quiet here so
            # the spell-back acts as feedback instead.
            self._deliver_to_active(active, utterance)
            return True

        self._log("handle_utterance: calling trigger matcher")
        trigger_kind, trigger_phrase = self._match_trigger(utterance)
        self._log(
            f"handle_utterance: trigger match → kind={trigger_kind} "
            f"phrase={trigger_phrase!r}"
        )
        if trigger_kind == "global":
            # A global handler matched — the line was recognized.  Cue
            # the user before the handler's potential ``Say`` reply
            # begins so the beep doesn't pile up on top of the TTS.
            self._play_success_beep()
            self._invoke_global(trigger_phrase)
            return True

        if active is not None:
            self._deliver_to_active(active, utterance)
            return True

        if trigger_kind == "flow":
            self._play_success_beep()
            self._start_flow(trigger_phrase)
            return True

        # Nothing in DialogFlow's domain matched, so play the "didn't get
        # that" cue: the line wasn't a flow trigger, a global, or an
        # answer to the active prompt, and silence here is a bad
        # experience.
        self._log("handle_utterance: no handler matched")
        self._play_error_beep()
        return False

    def _should_short_circuit_to_alpha(
        self, active: _ActiveFlow, utterance: str
    ) -> bool:
        """Return True if an alphanumeric prompt should consume ``utterance``
        ahead of global-trigger matching."""
        session = active.alpha_session
        if session is None:
            return False
        prompt = active.current_prompt
        if not isinstance(prompt, Ask) or prompt.mode not in (SPELLED, DIGITS):
            return False
        matches = session.matcher.classify_sequence(utterance)
        return any(
            m.type is not AlphanumericEventType.NONE for m in matches
        )

    # -- matching -----------------------------------------------------------

    def _match_trigger(
        self, utterance: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(kind, phrase)`` where ``kind`` is ``"global"`` / ``"flow"`` / ``None``.

        Globals outrank flows when both would match.
        """
        matcher = self._get_trigger_matcher()
        if matcher is None:
            self._log("match_trigger: no trigger matcher available")
            return None, None
        self._log("match_trigger: matcher.match begin")
        phrase = matcher.match(utterance)
        self._log(f"match_trigger: matcher.match end → phrase={phrase!r}")
        if phrase is None:
            return None, None
        if phrase in self._globals:
            return "global", phrase
        if phrase in self._flows:
            return "flow", phrase
        return None, None

    def _get_trigger_matcher(self) -> Optional[PhraseMatcher]:
        if self._trigger_matcher is not None:
            return self._trigger_matcher
        if self._phrase_matcher_factory is None:
            return None
        phrases_by_key: Dict[str, List[str]] = {}
        for p in self._globals.keys():
            phrases_by_key[p] = [p]
        for p in self._flows.keys():
            if p not in phrases_by_key:
                phrases_by_key[p] = [p]
        if not phrases_by_key:
            return None
        try:
            self._trigger_matcher = self._phrase_matcher_factory(
                phrases_by_key, self._trigger_threshold
            )
        except Exception as e:
            print(f"DialogFlow: trigger matcher creation failed: {e}", file=sys.stderr)
            self._trigger_matcher = None
        return self._trigger_matcher

    # -- flow lifecycle -----------------------------------------------------

    def _start_flow(self, trigger_phrase: str) -> None:
        flow_fn = self._flows.get(trigger_phrase)
        if flow_fn is None:
            return
        self._log(f"start_flow: trigger={trigger_phrase!r}")
        active = _ActiveFlow(flow_fn=flow_fn, trigger_phrase=trigger_phrase)
        with self._lock:
            self._active = active
        self._advance(active, value=None)

    def _deliver_to_active(self, active: _ActiveFlow, utterance: str) -> None:
        prompt = active.current_prompt
        if prompt is None:
            self._log("deliver_to_active: no current prompt; dropping")
            return
        prompt_kind = type(prompt).__name__
        self._log(
            f"deliver_to_active: begin prompt={prompt_kind} "
            f"utterance={_summarise(utterance)!r}"
        )
        try:
            value = self._interpret_answer(prompt, utterance, active)
        except _PartialInput:
            self._log("deliver_to_active: partial input; awaiting more")
            # Still gathering input (e.g. spelled letter-by-letter).
            # Don't beep or advance the generator yet — the per-character
            # spell-back from ``_speak_character_feedback`` is itself
            # the "I heard that letter" cue, and stacking a success
            # beep on top would be noisy at every keystroke.
            return
        except _Reprompt as r:
            # Recognized a line but couldn't interpret it for the
            # current prompt.  Beep first so the user notices their
            # last answer was rejected, then speak the reprompt.
            self._log(f"deliver_to_active: reprompt → {_summarise(r.text)!r}")
            self._play_error_beep()
            self._speak(r.text)
            return
        except _AbandonPrompt as a:
            # Out of retries — the prompt is being torn down.  Same
            # rationale as the reprompt path: cue the user that the
            # last utterance was the one that failed before the
            # generator's exception handler runs.
            self._log(f"deliver_to_active: abandon → {a.exc!r}")
            self._play_error_beep()
            self._throw(active, a.exc)
            return
        self._log(
            f"deliver_to_active: interpreted {prompt_kind} → "
            f"{_summarise(repr(value))}; advancing flow"
        )
        # Successful interpretation — play the recognition cue before
        # advancing the generator so the beep lands ahead of any
        # ``Say`` the next yield produces.
        self._play_success_beep()
        self._advance(active, value=value)

    def _advance(self, active: _ActiveFlow, value: Any) -> None:
        """Drive the generator until it blocks on user input or finishes."""
        while True:
            self._log(
                f"advance: generator.send({_summarise(repr(value))}) "
                f"flow={active.trigger_phrase!r}"
            )
            try:
                prompt = active.generator.send(value)
            except StopIteration:
                self._log("advance: generator finished (StopIteration)")
                self._finish_flow(active)
                return
            except DialogCancelled:
                self._log("advance: DialogCancelled raised")
                self._finish_flow(active)
                return
            except DialogRestart:
                self._log("advance: DialogRestart raised")
                active = self._restart_flow(active)
                value = None
                continue
            except Exception as e:
                self._report_error(
                    MoonshineError(
                        f"flow '{active.trigger_phrase}' raised {e!r}"
                    )
                )
                self._finish_flow(active)
                return

            self._log(
                f"advance: generator yielded {type(prompt).__name__}"
            )

            if isinstance(prompt, Say):
                self._speak(prompt.text)
                value = None
                continue

            if isinstance(prompt, (Ask, Confirm, Choose)):
                active.current_prompt = prompt
                active.retry_count = 0
                active.alpha_session = self._alpha_session_for(prompt)
                self._set_spelling_mode(self._spelling_mode_for_prompt(prompt))
                text = getattr(prompt, "prompt", "")
                if text:
                    self._speak(text)
                self._log(
                    f"advance: awaiting user input for "
                    f"{type(prompt).__name__}"
                )
                return

            if prompt is None:
                value = None
                continue

            print(
                f"DialogFlow: unknown prompt {prompt!r} yielded from "
                f"'{active.trigger_phrase}'; ignoring",
                file=sys.stderr,
            )
            value = None

    def _throw(self, active: _ActiveFlow, exc: BaseException) -> None:
        """Raise ``exc`` into the generator and process whatever it yields next."""
        try:
            prompt = active.generator.throw(type(exc), exc)
        except StopIteration:
            self._finish_flow(active)
            return
        except DialogCancelled:
            self._finish_flow(active)
            return
        except DialogRestart:
            active = self._restart_flow(active)
            self._advance(active, value=None)
            return
        except Exception as e:
            self._report_error(
                MoonshineError(f"flow '{active.trigger_phrase}' raised {e!r}")
            )
            self._finish_flow(active)
            return

        if isinstance(prompt, Say):
            self._speak(prompt.text)
            self._advance(active, value=None)
        elif isinstance(prompt, (Ask, Confirm, Choose)):
            active.current_prompt = prompt
            active.retry_count = 0
            active.alpha_session = self._alpha_session_for(prompt)
            self._set_spelling_mode(self._spelling_mode_for_prompt(prompt))
            text = getattr(prompt, "prompt", "")
            if text:
                self._speak(text)
        else:
            self._advance(active, value=None)

    def _set_spelling_mode(self, active: bool) -> None:
        """Toggle ``MOONSHINE_FLAG_SPELLING_MODE`` on the underlying transcriber.

        No-op when no ``spelling_mode_fn`` was provided or the state
        already matches.  We never let an exception from the user's
        callback abort the flow – the spelling fuser is an
        accuracy-only enhancement, not correctness.
        """
        if self._spelling_mode_fn is None:
            return
        if bool(active) == self._spelling_mode_active:
            return
        try:
            self._spelling_mode_fn(bool(active))
        except Exception as e:
            print(
                f"DialogFlow: spelling_mode_fn({active!r}) failed: {e}",
                file=sys.stderr,
            )
            return
        self._spelling_mode_active = bool(active)
        self._log(f"spelling_mode: {'on' if active else 'off'}")

    def _spelling_mode_for_prompt(self, prompt: Prompt) -> bool:
        """Whether ``prompt`` expects alphanumeric (spelled) input."""
        return isinstance(prompt, Ask) and prompt.mode in (SPELLED, DIGITS)

    def _alpha_session_for(self, prompt: Prompt) -> Optional[_AlphaSession]:
        if not isinstance(prompt, Ask):
            return None
        if prompt.mode == SPELLED:
            return _AlphaSession(matcher=self._get_spelled_matcher())
        if prompt.mode == DIGITS:
            return _AlphaSession(matcher=self._get_digits_matcher())
        return None

    def _get_spelled_matcher(self) -> AlphanumericMatcher:
        if self._spelled_matcher is None:
            self._spelled_matcher = AlphanumericMatcher()
        return self._spelled_matcher

    def _get_digits_matcher(self) -> AlphanumericMatcher:
        if self._digits_matcher is None:
            self._digits_matcher = digits_only_matcher()
        return self._digits_matcher

    def _restart_flow(self, active: _ActiveFlow) -> _ActiveFlow:
        trigger = active.trigger_phrase
        flow_fn = active.flow_fn
        self._finish_flow(active)
        new_active = _ActiveFlow(flow_fn=flow_fn, trigger_phrase=trigger)
        with self._lock:
            self._active = new_active
        return new_active

    def _finish_flow(self, active: _ActiveFlow) -> None:
        self._log(f"finish_flow: trigger={active.trigger_phrase!r}")
        with self._lock:
            if self._active is active:
                self._active = None
        # Leave the transcriber in its default (non-spelling) state so
        # subsequent free-form recognition / trigger matching isn't
        # perturbed by the spelling-CNN fuser.
        self._set_spelling_mode(False)

    def cancel(self) -> bool:
        """Abandon any currently running flow.  Returns ``True`` if there was one."""
        with self._lock:
            active = self._active
        if active is None:
            return False
        try:
            active.generator.close()
        except Exception:
            pass
        self._finish_flow(active)
        return True

    def say(self, text: str) -> None:
        """Speak ``text`` through the configured TTS, outside any flow.

        Public counterpart to flows yielding ``d.say(...)``: lets the
        application deliver welcome messages, status announcements,
        error notifications, and the like without first registering a
        single-shot flow.  Safe to call concurrently with an active
        flow – it just routes through the same path the runner uses for
        in-flow prompts, so:

        * the microphone is muted across playback so the assistant
          doesn't hear itself.
        * lines the recognizer opens during playback are tagged as
          self-capture and dropped on completion, unless
          :meth:`barge_in` is on.
        * :meth:`log_io` emits a clean ``assistant: ...`` line, matching
          the format used for in-flow speech.

        Blocks until playback finishes – mirrors the behaviour of an
        in-flow ``Say``.  Empty / ``None`` ``text`` is a no-op.  When
        nothing is configured to speak, the text is printed to stdout.
        """
        if not text:
            return
        self._speak(text)

    # -- global handler invocation ------------------------------------------

    def _invoke_global(self, trigger_phrase: str) -> None:
        handler = self._globals.get(trigger_phrase)
        if handler is None:
            return
        self._log(f"invoke_global: trigger={trigger_phrase!r}")
        with self._lock:
            active = self._active
        dialog = active.dialog if active is not None else Dialog(trigger_phrase)
        try:
            prompt = handler(dialog)
        except DialogCancelled:
            if active is not None:
                self._finish_flow(active)
            return
        except DialogRestart:
            if active is not None:
                new_active = self._restart_flow(active)
                self._advance(new_active, value=None)
            return
        except Exception as e:
            self._report_error(
                MoonshineError(f"handler for '{trigger_phrase}' raised {e!r}")
            )
            return

        if isinstance(prompt, Say):
            self._speak(prompt.text)
        elif isinstance(prompt, (Ask, Confirm, Choose)) and active is not None:
            active.current_prompt = prompt
            self._set_spelling_mode(self._spelling_mode_for_prompt(prompt))
            text = getattr(prompt, "prompt", "")
            if text:
                self._speak(text)

    # -- answer interpretation ---------------------------------------------

    def _interpret_answer(
        self, prompt: Prompt, utterance: str, active: _ActiveFlow
    ) -> Any:
        if isinstance(prompt, Ask):
            return self._interpret_ask(prompt, utterance, active)
        if isinstance(prompt, Confirm):
            return self._interpret_confirm(prompt, utterance, active)
        if isinstance(prompt, Choose):
            return self._interpret_choose(prompt, utterance, active)
        return utterance

    def _interpret_ask(
        self, prompt: Ask, utterance: str, active: _ActiveFlow
    ) -> str:
        text = utterance.strip()
        if not text:
            self._reprompt_or_abandon(prompt, active, NoInputError())

        if prompt.mode in (SPELLED, DIGITS):
            return self._interpret_alphanumeric(prompt, text, active)

        return text

    def _interpret_alphanumeric(
        self, prompt: Ask, utterance: str, active: _ActiveFlow
    ) -> str:
        """Drive the AlphanumericMatcher session for SPELLED / DIGITS.

        The user is expected to dictate one character per utterance (e.g.
        spelling a password over a microphone), so each completed
        utterance only updates an in-progress buffer.  The prompt does
        not advance until the user issues an explicit terminator
        ("done", "stop", "finish", "submit", "enter", …); at that point
        the assembled string is returned to the generator.

        If the entire utterance consists of multiple recognised tokens
        terminated by a stop word (e.g. ``"h e l l o done"``) the prompt
        also completes — everything before the terminator is kept.
        """

        session = active.alpha_session
        if session is None:
            session = self._alpha_session_for(prompt) or _AlphaSession(
                matcher=self._get_spelled_matcher()
            )
            active.alpha_session = session

        self._log("interpret_alphanumeric: classify_sequence begin")
        matches = session.matcher.classify_sequence(utterance)
        self._log(
            f"interpret_alphanumeric: classify_sequence end "
            f"({len(matches)} tokens)"
        )
        applied = False
        for m in matches:
            if m.type is AlphanumericEventType.STOPPED:
                result = "".join(session.buffer)
                session.buffer.clear()
                self._log(
                    f"interpret_alphanumeric: STOPPED → buffer={result!r}"
                )
                return result
            if m.type is AlphanumericEventType.CLEAR:
                session.buffer.clear()
                applied = True
            elif m.type is AlphanumericEventType.UNDO:
                removed: Optional[str] = (
                    session.buffer.pop() if session.buffer else None
                )
                applied = True
                if self._spell_feedback and removed is not None:
                    self._speak_undo_feedback(removed)
            elif m.type is AlphanumericEventType.CHARACTER and m.character is not None:
                session.buffer.append(m.character)
                applied = True
                if self._spell_feedback:
                    self._speak_character_feedback(m.character)

        if applied:
            # Characters accumulated – stay on this prompt and wait for the
            # user to keep spelling or say "done".  Also reset the retry
            # counter so earlier stray utterances don't count against us.
            active.retry_count = 0
            raise _PartialInput()

        # Nothing recognised.  If we have *no* characters yet the user is
        # probably just starting, so reprompt normally.  But once the user
        # is mid-spelling we silently drop unrecognised utterances (ASR
        # often picks up stray words like "and" / "uh" / background
        # speech); reprompting after every glitch would make them think
        # the whole prompt was restarted and their buffer was lost.
        if not session.buffer:
            self._reprompt_or_abandon(prompt, active, NoMatchError())

        if self._debug:
            print(
                f"DialogFlow: ignoring unrecognised utterance {utterance!r} "
                f"during spelled input (buffer={''.join(session.buffer)!r})",
                file=sys.stderr,
            )
        raise _PartialInput()

    def _interpret_confirm(
        self, prompt: Confirm, utterance: str, active: _ActiveFlow
    ) -> bool:
        self._log("interpret_confirm: fetching matcher")
        matcher = self._get_confirm_matcher(prompt)
        if matcher is None:
            self._log("interpret_confirm: no matcher available")
            self._reprompt_or_abandon(prompt, active, NoMatchError())
        self._log("interpret_confirm: matcher.match begin")
        key = matcher.match(utterance)
        self._log(f"interpret_confirm: matcher.match end → key={key!r}")
        if key == "yes":
            return True
        if key == "no":
            return False
        self._reprompt_or_abandon(prompt, active, NoMatchError())

    def _interpret_choose(
        self, prompt: Choose, utterance: str, active: _ActiveFlow
    ) -> str:
        self._log("interpret_choose: fetching matcher")
        matcher = self._get_choose_matcher(prompt)
        if matcher is None:
            self._log("interpret_choose: no matcher available")
            self._reprompt_or_abandon(prompt, active, NoMatchError())
        self._log("interpret_choose: matcher.match begin")
        key = matcher.match(utterance)
        self._log(f"interpret_choose: matcher.match end → key={key!r}")
        if key is not None:
            return key
        self._reprompt_or_abandon(prompt, active, NoMatchError())

    # -- PhraseMatcher caching ---------------------------------------------

    def _get_confirm_matcher(self, prompt: Confirm) -> Optional[PhraseMatcher]:
        cache_key = (
            "confirm",
            tuple(prompt.yes_phrases),
            tuple(prompt.no_phrases),
            float(prompt.threshold),
        )
        if cache_key in self._matcher_cache:
            return self._matcher_cache[cache_key]
        matcher = self._build_matcher(
            {"yes": list(prompt.yes_phrases), "no": list(prompt.no_phrases)},
            prompt.threshold,
        )
        self._matcher_cache[cache_key] = matcher
        return matcher

    def _get_choose_matcher(self, prompt: Choose) -> Optional[PhraseMatcher]:
        phrases_by_key: Dict[str, List[str]] = {}
        for key, phrases in prompt.options.items():
            collected: List[str] = [key]
            for p in phrases:
                if p and p not in collected:
                    collected.append(p)
            phrases_by_key[key] = collected
        cache_key = (
            "choose",
            tuple((k, tuple(v)) for k, v in phrases_by_key.items()),
            float(prompt.threshold),
        )
        if cache_key in self._matcher_cache:
            return self._matcher_cache[cache_key]
        matcher = self._build_matcher(phrases_by_key, prompt.threshold)
        self._matcher_cache[cache_key] = matcher
        return matcher

    def _build_matcher(
        self, phrases_by_key: Mapping[str, Sequence[str]], threshold: float
    ) -> Optional[PhraseMatcher]:
        if self._phrase_matcher_factory is None:
            return None
        self._log(
            f"build_matcher: building for {len(phrases_by_key)} keys "
            f"threshold={threshold}"
        )
        try:
            matcher = self._phrase_matcher_factory(phrases_by_key, float(threshold))
        except Exception as e:
            print(f"DialogFlow: failed to build phrase matcher: {e}", file=sys.stderr)
            return None
        self._log("build_matcher: done")
        return matcher

    def _reprompt_or_abandon(
        self, prompt: Prompt, active: _ActiveFlow, exc: BaseException
    ) -> NoReturn:
        max_retries = getattr(prompt, "max_retries", 0) or 0
        if active.retry_count >= max_retries:
            raise _AbandonPrompt(exc)
        active.retry_count += 1
        template = getattr(prompt, "no_input_reprompt", None) or "{prompt}"
        try:
            text = template.format(prompt=getattr(prompt, "prompt", ""))
        except Exception:
            text = getattr(prompt, "prompt", "")
        raise _Reprompt(text)

    # -- TTS ----------------------------------------------------------------

    def _speak(self, text: str) -> None:
        if not text:
            return
        self._log(f"speak: begin text={_summarise(text)!r}")
        if self._log_io:
            print(f"assistant: {text}", file=sys.stderr, flush=True)
        if self._said_fn is not None:
            try:
                self._said_fn(text)
            except Exception as e:
                self._log(f"on_said handler failed: {e!r}")
        muted = False
        if self._mute_fn is not None:
            try:
                self._mute_fn(True)
                muted = True
                self._log("speak: mic muted")
            except Exception as e:
                self._log(f"speak: mute_fn failed: {e!r}")
                muted = False
        # Flip the software-side speaking flag before we hand off to
        # the TTS so any utterance that races in from the STT
        # listener thread is dropped by ``handle_utterance``.  The
        # ``finally`` clears it even if the TTS raises so we don't
        # wedge the runner deaf to subsequent input.
        self._speaking = True
        try:
            if self._speak_fn is not None:
                self._speak_fn(text)
                self._log("speak: speak_fn returned")
            elif self._tts is not None:
                self._tts.say(text)
                self._log("speak: tts.say queued")
                try:
                    self._tts.wait()
                    self._log("speak: tts.wait returned")
                except Exception as e:
                    self._log(f"speak: tts.wait failed: {e!r}")
            else:
                print(f"[DialogFlow say] {text}")
        finally:
            self._speaking = False
            if muted and self._mute_fn is not None:
                try:
                    self._mute_fn(False)
                    self._log("speak: mic unmuted")
                except Exception as e:
                    self._log(f"speak: unmute failed: {e!r}")
        self._log("speak: done")

    def _speak_character_feedback(self, character: str) -> None:
        """Speak ``spoken_form(character)`` as mid-prompt spell-back.

        Invoked from :meth:`_interpret_alphanumeric` for each recognised
        character when ``spell_feedback=True``.  Failures are swallowed
        so a broken TTS can't derail an in-progress spelled input –
        the character has already been appended to the buffer.
        """
        phrase = spoken_form(character)
        self._log(
            f"spell_feedback: say {phrase!r} for character {character!r}"
        )
        try:
            self._speak(phrase)
        except Exception as e:
            self._log(f"spell_feedback: speak failed: {e!r}")

    # -- Success / error beeps ---------------------------------------------

    def _play_beep(self, kind: str) -> None:
        """Play the recognition cue identified by ``kind``.

        ``kind`` is ``"success"`` or ``"error"``.  The cue comes from the
        synthesizer's ``play_success`` / ``play_error``, which is
        duck-typed so backends without beep support (test stubs that only
        implement ``say``, say) stay silent rather than raising.
        Exceptions from the callback are always reported on stderr, even
        without :meth:`debug`: a broken beep is rare and surprising
        enough to be worth surfacing rather than swallowing.
        """
        if not self._beeps_enabled:
            return
        fn = self._success_beep_fn if kind == "success" else self._error_beep_fn
        if fn is None and self._tts is not None:
            method = getattr(self._tts, f"play_{kind}", None)
            if callable(method):
                fn = method
        if fn is None:
            return
        self._log(f"{kind}_beep: invoking {fn!r}")
        try:
            fn()
        except Exception as e:
            print(
                f"DialogFlow: {kind}_beep_fn raised {e!r}; "
                "the beep will be silent.",
                file=sys.stderr,
                flush=True,
            )
            return
        self._log(f"{kind}_beep: callback returned")

    def _play_success_beep(self) -> None:
        """Play the "recognized" cue, if a beep callback is wired.

        Fired before the TTS reply on every recognized utterance:
        trigger matches, completed alphanumeric input, and matched
        confirm / choose / free-form answers.
        """
        self._play_beep("success")

    def _play_error_beep(self) -> None:
        """Play the "not recognized" cue, if a beep callback is wired.

        Fired when an utterance can't be routed: no trigger matched,
        an active flow couldn't interpret it (reprompt / abandon),
        or the runner is unmounted entirely.
        """
        self._play_beep("error")

    def _speak_undo_feedback(self, character: str) -> None:
        """Speak ``"deleting <spoken_form(character)>"`` after an UNDO.

        Invoked from :meth:`_interpret_alphanumeric` when a "delete" /
        "backspace" / "undo" / "scratch that" command pops a character
        off the in-progress buffer (and ``spell_feedback=True``).  No-op
        when the buffer was already empty.  Failures are swallowed for
        the same reason as :meth:`_speak_character_feedback`.
        """
        phrase = f"deleting {spoken_form(character)}"
        self._log(
            f"spell_feedback: say {phrase!r} for undo of {character!r}"
        )
        try:
            self._speak(phrase)
        except Exception as e:
            self._log(f"spell_feedback: speak failed: {e!r}")

    # -- Debug logging ------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Emit a timestamped trace line to stderr when ``debug=True``.

        Each line shows:
          * ``+<delta>ms``  – wall time since the previous log line
          * ``<total>ms``   – wall time since the first log line emitted by
                              this DialogFlow instance
        so you can see both per-step cost and cumulative progress.
        """
        if not self._debug:
            return
        now = time.perf_counter()
        if self._log_start is None:
            self._log_start = now
            self._log_last = now
        delta_ms = (now - (self._log_last or now)) * 1000.0
        total_ms = (now - (self._log_start or now)) * 1000.0
        self._log_last = now
        print(
            f"[DialogFlow +{delta_ms:7.1f}ms / {total_ms:8.1f}ms] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _Reprompt(Exception):
    def __init__(self, text: str):
        self.text = text


class _AbandonPrompt(Exception):
    def __init__(self, exc: BaseException):
        self.exc = exc


class _PartialInput(Exception):
    """Signal from an interpreter that more input is needed before the
    current prompt can advance (used for multi-utterance spelled / digit
    dictation)."""


class _SpelledPhrase(str):
    """Space-joined spoken-form phrase that also iterates token-by-token.

    A ``str`` so ``f"I heard: {spell_out(password)}"`` interpolates as
    the natural joined phrase (``"ess ee ay bee"``), but with
    ``__iter__`` overridden to yield ``["ess", "ee", "ay", "bee"]``.
    That keeps backwards-compat with the older list-returning
    :func:`spell_out` API: a caller that wrote
    ``" ".join(spell_out(password))`` still gets the joined phrase
    back instead of the character-by-character ``"e s s   e e   a y
    b e e"`` you'd get from plain ``str`` iteration.
    """

    __slots__ = ()

    def __iter__(self):  # type: ignore[override]
        if not self:
            return iter(())
        return iter(self.split(" "))


def spell_out(s: str) -> _SpelledPhrase:
    """Return ``s`` as a TTS-friendly spoken-form phrase.

    Each character is rendered as a phrase the TTS engine can pronounce
    unambiguously (letters use spelling-alphabet sounds like ``"haitch"``
    for ``"h"``, upper-case letters are prefixed with ``"capital "``,
    digits become word form, common symbols use their spoken name) and
    the per-character phrases are joined with a single space.  The
    return value is a ``str`` subclass (:class:`_SpelledPhrase`) so it
    drops cleanly into both an ``f"I heard: {spell_out(password)}"``
    interpolation *and* a ``" ".join(spell_out(password))`` call —
    both produce the same joined phrase.  The per-character mapping
    lives in :func:`spoken_form` in ``alphanumeric_listener.py`` so
    the :class:`AlphanumericListener`'s TTS repeat-back and this
    function can't drift apart.

    ``spell_out("Hi#1")`` renders as ``"capital haitch eye hash
    one"`` (and iterates as
    ``["capital", "haitch", "eye", "hash", "one"]``).  Empty strings
    produce ``""``.  This is for *speaking* strings back at the user,
    not for matching their input (that's
    :class:`AlphanumericMatcher`'s job); callers that need the
    strict per-character list (e.g. for custom pacing between
    tokens) can use ``[spoken_form(c) for c in s]`` directly.
    """
    return _SpelledPhrase(" ".join(spoken_form(c) for c in s))


def _summarise(text: str, max_len: int = 60) -> str:
    """Truncate ``text`` for debug logs so we don't spam the terminal."""
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# CLI demo: `python -m moonshine_voice.dialog_flow`
#
# Live microphone + TTS demo of the wifi-setup flow.  Input comes from a
# :class:`MicTranscriber`, prompts are spoken through :class:`TextToSpeech`,
# and every trigger / confirmation / choice goes through the embedding model.
# The first run may download the transcription and embedding models.
# ---------------------------------------------------------------------------


def _run_beep_diagnostic(
    *,
    language: str,
    output_device: Optional[Any],
    tts_options: Optional[Dict[str, Any]],
    debug: bool,
) -> None:
    """Play the success / error beeps and a reference tone, then return.

    Used by ``--play-test-beeps`` to isolate "speech audible, beeps
    not" issues without going through mic + flow.  Plays in order:

    1. Success beep (ascending two-tone, ~0.41 s incl. pad).
    2. One second of silence.
    3. Error beep (descending two-tone, ~0.48 s incl. pad).
    4. One second of silence.
    5. A 500 ms 440 Hz reference tone at ~0.5 amplitude — twice as
       loud as the beeps' 0.25 amplitude — so a "reference audible
       but beeps not" outcome points at the beep waveform (raise
       amplitude or extend the lead-silence pad) rather than at
       the audio stack.

    Routes through the same :class:`TextToSpeech` instance that the
    main flow would use, so the diagnostic exercises the exact same
    stream-open / play / wait code path (and thus the same
    PortAudio / PipeWire / ALSA shim) that the real beeps go
    through.
    """
    print(
        "Playing test cue sequence: success beep, silence, error beep, "
        "silence, 440 Hz reference tone.",
        file=sys.stderr,
    )
    tts = (
        TextToSpeech()
        .language(language)
        .debug(debug)
        .output_device(output_device)
    )
    if tts_options:
        tts.options(tts_options)
    tts.load()
    try:
        print("--- success beep ---", file=sys.stderr)
        tts.play_success()
        tts.wait()
        time.sleep(1.0)

        print("--- error beep ---", file=sys.stderr)
        tts.play_error()
        tts.wait()
        time.sleep(1.0)

        # Tear down the TTS playback stream *before* the reference
        # tone tries to open its own.  Some USB DACs (e.g. the user's
        # Pi setup) reserve the device exclusively and PortAudio
        # can't re-open it while the previous stream is still
        # being closed by ALSA.  ``tts.close()`` joins both worker
        # threads and tells PortAudio to release the device, plus
        # a short grace nap covers any kernel-side teardown delay.
        try:
            tts.close()
        except Exception:
            pass
        time.sleep(0.3)

        print("--- 440 Hz reference tone (0.5 amplitude, 500 ms) ---", file=sys.stderr)
        try:
            import numpy as np
            import sounddevice as sd

            # Pick a sample rate the device will accept.  Some USB
            # DACs only expose a single rate (e.g. 48 kHz) and
            # PortAudio refuses to open the stream at any other
            # rate; fall back through 48 kHz / 44.1 kHz /
            # device-default until one sticks.
            candidate_rates: List[int] = [48000, 44100]
            try:
                info = sd.query_devices(output_device) if output_device is not None \
                    else sd.query_devices(kind="output")
                default_sr = int(info.get("default_samplerate", 0) or 0)
                if default_sr and default_sr not in candidate_rates:
                    candidate_rates.insert(0, default_sr)
            except Exception:
                pass
            sr_chosen: Optional[int] = None
            for sr in candidate_rates:
                try:
                    sd.check_output_settings(
                        samplerate=sr, channels=1, dtype="float32",
                        device=output_device,
                    )
                    sr_chosen = sr
                    break
                except Exception:
                    continue
            if sr_chosen is None:
                raise RuntimeError(
                    f"None of {candidate_rates} Hz worked for the "
                    f"reference tone on this device (it may be "
                    f"reserved exclusively by the previous TTS "
                    f"stream — that's harmless; rely on the beep "
                    f"audibility for the diagnosis)."
                )
            t = np.arange(int(sr_chosen * 0.5), dtype=np.float32) / float(sr_chosen)
            tone = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
            ramp_n = int(sr_chosen * 0.010)
            if ramp_n > 0:
                ramp = np.linspace(0.0, 1.0, ramp_n, dtype=np.float32)
                tone[:ramp_n] *= ramp
                tone[-ramp_n:] *= ramp[::-1]
            sd.play(tone, sr_chosen, device=output_device)
            sd.wait()
        except Exception as e:
            print(
                f"Reference tone playback skipped: {e!r}",
                file=sys.stderr,
            )

        print(
            "Done.  Tell the assistant which of the three cues you heard "
            "(success / error / reference).",
            file=sys.stderr,
        )
    finally:
        try:
            tts.close()
        except Exception:
            pass


if __name__ == "__main__":
    # No explicit prog: argparse reads argv[0], which the console script sets to
    # "moonshine-voice dialog" so the usage line matches how it was invoked.
    parser = argparse.ArgumentParser(
        description=(
            "Live microphone + TTS demo of DialogFlow, wired up to a "
            "wifi-setup flow.  Say 'set up wifi' to start, 'cancel' to "
            "abandon, or 'start over' to reset.  All trigger / "
            "confirmation / choice matching goes through the embedding "
            "model; the first run may download it."
        ),
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language for mic transcription and TTS (default: en).",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Print prompts instead of speaking them.",
    )
    parser.add_argument(
        "--tts-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra option forwarded to TextToSpeech; repeat for multiple "
            "(e.g. --tts-option speed=1.1 --tts-option voice=kokoro_af_heart)."
        ),
    )
    parser.add_argument(
        "--list-output-devices",
        action="store_true",
        help=(
            "List the PortAudio output devices the runner can pick "
            "from and exit.  Useful when the assistant is silent: the "
            "host default may not be the device with speakers (HDMI vs "
            "3.5 mm jack vs USB DAC)."
        ),
    )
    parser.add_argument(
        "--output-device",
        default=None,
        metavar="INDEX_OR_NAME",
        help=(
            "Pin TTS playback to a specific PortAudio output device.  "
            "Accepts an integer index, a decimal-string index, or a "
            "case-insensitive substring of the device name as shown "
            "by --list-output-devices.  Defaults to the host default."
        ),
    )
    parser.add_argument(
        "--play-test-beeps",
        action="store_true",
        help=(
            "Play the success and error beep cues followed by a "
            "louder reference 440 Hz tone, then exit.  Useful for "
            "diagnosing 'speech audible, beeps not' issues without "
            "going through the full mic + flow loop: if you hear "
            "the reference tone but not the beeps, raise the beep "
            "amplitude or report it; if you hear nothing, it's a "
            "system audio routing / volume / mute problem rather "
            "than a runner bug."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print DialogFlow stage-transition traces to stderr with "
            "per-step and cumulative timings, plus per-stage timing in "
            "the TextToSpeech synth and play workers (handy for "
            "diagnosing missing-beep issues)."
        ),
    )
    parser.add_argument(
        "--log-io",
        action="store_true",
        help=(
            "Print every utterance the runner receives from the STT "
            "and every prompt the assistant speaks as plain "
            "``user: ...`` / ``assistant: ...`` lines on stderr.  "
            "Distinct from --debug: this is the user-facing dialogue "
            "transcript without the verbose internal stage-transition "
            "trace."
        ),
    )
    args = parser.parse_args()

    # Handle the device-listing diagnostic first so the user can run it
    # without needing the embedding / spelling models downloaded.
    if args.list_output_devices:
        from moonshine_voice.tts import list_output_devices

        print(
            "PortAudio output devices (asterisk = current host default):",
            file=sys.stderr,
        )
        for line in list_output_devices():
            print(f"  {line}", file=sys.stderr)
        sys.exit(0)

    # Coerce purely numeric --output-device strings into ints so the TTS
    # device-resolver treats them as indices rather than name substrings.
    output_device: Optional[Any] = args.output_device
    if isinstance(output_device, str) and output_device.strip().isdigit():
        output_device = int(output_device.strip())

    tts_options: Dict[str, Any] = {}
    if args.tts_option:
        try:
            tts_options = dict(_parse_options_cli(args.tts_option))
        except ValueError as e:
            parser.error(str(e))

    # The beep diagnostic skips the embedding / spelling / mic
    # plumbing entirely — it's the fastest way to confirm whether the
    # success / error beeps are even audible on the chosen output
    # device.  Plays each beep with a one-second pause, then a louder
    # reference tone (~0.5 amplitude vs. the beeps' 0.25 amplitude)
    # so a missing beep against an audible reference points at the
    # beep waveform rather than the audio stack.
    if args.play_test_beeps:
        _run_beep_diagnostic(
            language=args.language,
            output_device=output_device,
            tts_options=tts_options,
            debug=args.debug,
        )
        sys.exit(0)

    # ---- Wifi-setup flow (inlined for the demo) --------------------------
    #
    # Gathers an SSID, confirms it, gathers a password one character at a
    # time via spelled input (each character is repeated back as it's
    # recognised, via DialogFlow's ``spell_feedback``), and asks for
    # confirmation before "applying" it.  ``apply_wifi_config`` here just
    # prints what it would have done.

    def wifi_setup(d):
        ssid = yield d.ask("What's the name of your wifi network?")
        if not (yield d.confirm(f"I heard, {ssid}. Is that right?")):
            yield d.say("No problem, let's start over.")
            return
        password = yield d.ask(
            "Please spell the wifi password, one letter at a time, "
            "and say 'done' when finished.",
            mode=SPELLED,
        )
        if (yield d.confirm("Apply these changes?")):
            print(
                f"\n[dialog_flow] apply_wifi_config("
                f"ssid={ssid!r}, password={password!r})",
                file=sys.stderr,
            )
            yield d.say("Done. Your wifi is set up.")
        else:
            yield d.say("Okay, nothing changed.")

    # ---- Wire up DialogFlow ----------------------------------------------

    runner = (
        DialogFlow()
        .language(args.language)
        .speech(not args.no_tts)
        .log_io(args.log_io)
        .debug(args.debug)
        .on_progress(
            lambda fraction, name: print(
                f"Loading {name}... {fraction:.0%}", file=sys.stderr
            )
        )
        .listen_for("set up wifi", wifi_setup)
        .always("cancel", lambda d: d.cancel())
        .always("start over", lambda d: d.restart())
    )
    if output_device is not None:
        runner.output_device(output_device)
    if tts_options:
        runner.speech_options(dict(tts_options))
    if not args.log_io:
        # The runner's own log_io transcript already covers both sides, so
        # only echo when it's off.
        runner.on_heard(lambda text: print(f"user: {text}", flush=True))
        runner.on_said(lambda text: print(f"assistant: {text}", flush=True))

    runner.load()

    print(
        "\n🎤 Ready. Say 'set up wifi' or something similar to start.",
        file=sys.stderr,
    )
    print("Press Ctrl+C to stop.\n", file=sys.stderr)

    runner.start_listening()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
    finally:
        runner.close()
