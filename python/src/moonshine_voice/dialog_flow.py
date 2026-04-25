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

The :class:`DialogFlow` runner is a :class:`TranscriptEventListener`, so it
composes with :class:`MicTranscriber` the same way :class:`IntentRecognizer`
does.  It has no asyncio dependency – flows are driven synchronously from
whatever thread delivers transcript events, so flows can be unit-tested
without any audio, TTS, or event loop.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
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
)

from moonshine_voice.alphanumeric_listener import (
    AlphanumericEventType,
    AlphanumericMatcher,
    digits_only_matcher,
    spoken_form,
)
from moonshine_voice.cached_embeddings import CachedEmbeddings
from moonshine_voice.download import get_embedding_model, get_model_for_language
from moonshine_voice.intent_recognizer import IntentRecognizer
from moonshine_voice.mic_transcriber import MicTranscriber
from moonshine_voice.transcriber import (
    Error,
    LineCompleted,
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
    against the union of the key and its phrases, using the intent
    recognizer when available and falling back to substring matching.
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

    :class:`moonshine_voice.IntentRecognizer` satisfies this protocol via
    its :meth:`calculate_embedding` and :meth:`distance` methods – the
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

    This is a tiny wrapper around an :class:`EmbeddingBackend` (typically
    an :class:`IntentRecognizer`).  At construction time, the backend is
    used to compute an embedding for every phrase in every group; at
    match time, the utterance is embedded once and compared against
    every phrase using cosine similarity.  The key of the best-scoring
    phrase (above ``threshold``) is returned, or *None* if nothing
    clears the threshold.

    Use this to replace string / substring matching on user utterances
    with fuzzy, semantics-aware matching.

    Example::

        yes_no = PhraseMatcher(
            intent_recognizer,
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


class DialogFlow(TranscriptEventListener):
    """Runner that drives generator-based conversational flows.

    Register flow functions against trigger phrases; completed transcript
    lines are routed either to the configured intent recognizer (when no
    flow is active) or to the currently suspended generator (when one is).

    The runner is synchronous: when a flow yields a :class:`Say`, the
    runner speaks and blocks until the utterance has been played, then
    resumes the generator.  When a flow yields an input-expecting prompt
    (:class:`Ask` / :class:`Confirm` / :class:`Choose`), the runner speaks
    the prompt and returns control to the caller; the next completed
    transcript line resumes the generator.

    Args:
        tts: Optional :class:`TextToSpeech` used to speak prompts.  If
            set, prompts are spoken via ``tts.say(text)`` and the runner
            blocks on ``tts.wait()`` before advancing the flow.
        intent_recognizer: Optional :class:`IntentRecognizer` used for
            matching trigger phrases to registered flows and for routing
            utterances when no flow is active.  When omitted, trigger
            matching falls back to case-insensitive substring matching.
        transcriber: Reserved for future per-prompt biasing hooks.
        speak_fn: Optional callable ``(text) -> None`` that speaks the text
            and blocks until playback finishes.  Overrides ``tts``.  Useful
            for tests and alternative TTS backends.
        mute_fn: Optional callable ``(should_mute: bool) -> None`` invoked
            before and after each spoken prompt so callers can silence the
            microphone while the assistant is talking.
        spell_feedback: If ``True``, every character recognised during a
            ``SPELLED`` / ``DIGITS`` prompt is spoken back to the user
            using :func:`spoken_form` (``"haitch"`` for ``"h"``,
            ``"capital ay"`` for ``"A"``, ``"hash"`` for ``"#"``, etc.).
            Matches the behaviour of ``AlphanumericListener(tts=…)`` but
            routes through this runner's own ``speak_fn``/``tts`` so the
            same mic-mute and logging apply.  Off by default so existing
            callers without a TTS aren't surprised by audible output.
    """

    def __init__(
        self,
        *,
        tts: Optional[Any] = None,
        intent_recognizer: Optional[Any] = None,
        transcriber: Optional[Any] = None,
        speak_fn: Optional[Callable[[str], None]] = None,
        mute_fn: Optional[Callable[[bool], None]] = None,
        phrase_matcher_factory: Optional[PhraseMatcherFactory] = None,
        cached_embeddings: Optional[CachedEmbeddings] = None,
        trigger_threshold: float = 0.7,
        spell_feedback: bool = False,
        debug: bool = False,
    ):
        self._tts = tts
        self._intent_recognizer = intent_recognizer
        self._transcriber = transcriber
        self._speak_fn = speak_fn
        self._mute_fn = mute_fn
        self._trigger_threshold = float(trigger_threshold)
        self._spell_feedback = bool(spell_feedback)
        self._debug = bool(debug)
        self._log_start: Optional[float] = None
        self._log_last: Optional[float] = None

        # Wire up an :class:`EmbeddingBackend` that the default
        # :class:`PhraseMatcher` factory will use.  Library-level
        # constants (e.g. the default yes/no phrases) have their
        # embeddings shipped via ``assets/cached_embeddings.tsv`` and
        # loaded by :class:`CachedEmbeddings`; cache misses (typically
        # user utterances) fall through to ``intent_recognizer``.  A
        # caller who wants to skip the cache entirely can pass
        # ``cached_embeddings=CachedEmbeddings(path="/dev/null")`` or
        # provide their own ``phrase_matcher_factory``.
        if cached_embeddings is None and intent_recognizer is not None:
            cached_embeddings = CachedEmbeddings(fallback=intent_recognizer)
        self._cached_embeddings = cached_embeddings

        if phrase_matcher_factory is None:
            backend = cached_embeddings or intent_recognizer
            if backend is not None:
                def _default_factory(
                    phrases_by_key: Mapping[str, Sequence[str]],
                    threshold: float,
                ) -> Optional[PhraseMatcher]:
                    return PhraseMatcher(
                        backend, phrases_by_key, threshold=threshold
                    )

                phrase_matcher_factory = _default_factory
        self._phrase_matcher_factory = phrase_matcher_factory

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

    # -- registration -------------------------------------------------------

    def register_flow(self, trigger_phrase: str, flow: FlowFn) -> None:
        """Register a flow function to be started when ``trigger_phrase`` fires.

        Trigger matching is always embedding-based: the phrase is embedded
        once at registration time and compared against the utterance's
        embedding at match time via cosine similarity.
        """
        if not callable(flow):
            raise TypeError("flow must be callable")
        self._flows[trigger_phrase] = flow
        self._invalidate_trigger_matcher()

    def unregister_flow(self, trigger_phrase: str) -> bool:
        removed = self._flows.pop(trigger_phrase, None) is not None
        if removed:
            self._invalidate_trigger_matcher()
        return removed

    def register_global(self, trigger_phrase: str, handler: GlobalHandler) -> None:
        """Register a phrase that is always live, even while a flow runs.

        ``handler`` receives the current :class:`Dialog` (a fresh one when
        no flow is active) and may return a :class:`Prompt` to speak, or
        *None*.  It may also raise :class:`DialogCancelled` or
        :class:`DialogRestart` to influence the active flow.
        """
        self._globals[trigger_phrase] = handler
        self._invalidate_trigger_matcher()

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

    # -- TranscriptEventListener implementation -----------------------------

    def on_line_completed(self, event: LineCompleted) -> None:
        if not event.line or not event.line.text:
            return
        utterance = event.line.text.strip()
        if not utterance:
            return
        self.process_utterance(utterance)

    def on_error(self, event: Error) -> None:
        pass

    # -- core dispatch ------------------------------------------------------

    def process_utterance(self, utterance: str) -> bool:
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
            f"process_utterance: begin utterance={_summarise(utterance)!r} "
            f"active={'yes' if self._active is not None else 'no'}"
        )
        with self._lock:
            active = self._active

        if active is not None and self._should_short_circuit_to_alpha(
            active, utterance
        ):
            self._log("process_utterance: alpha short-circuit → deliver")
            self._deliver_to_active(active, utterance)
            return True

        self._log("process_utterance: calling trigger matcher")
        trigger_kind, trigger_phrase = self._match_trigger(utterance)
        self._log(
            f"process_utterance: trigger match → kind={trigger_kind} "
            f"phrase={trigger_phrase!r}"
        )
        if trigger_kind == "global":
            self._invoke_global(trigger_phrase)
            return True

        if active is not None:
            self._deliver_to_active(active, utterance)
            return True

        if trigger_kind == "flow":
            self._start_flow(trigger_phrase)
            return True

        if self._intent_recognizer is not None:
            self._log("process_utterance: forwarding to intent recognizer")
            try:
                self._intent_recognizer.process_utterance(utterance)
            except Exception as e:
                print(f"DialogFlow: intent recognizer error: {e}", file=sys.stderr)
        self._log("process_utterance: no handler matched")
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
            # Still gathering input (e.g. spelled letter-by-letter).  Don't
            # advance the generator yet; wait for the next utterance.
            return
        except _Reprompt as r:
            self._log(f"deliver_to_active: reprompt → {_summarise(r.text)!r}")
            self._speak(r.text)
            return
        except _AbandonPrompt as a:
            self._log(f"deliver_to_active: abandon → {a.exc!r}")
            self._throw(active, a.exc)
            return
        self._log(
            f"deliver_to_active: interpreted {prompt_kind} → "
            f"{_summarise(repr(value))}; advancing flow"
        )
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
                print(
                    f"DialogFlow: flow '{active.trigger_phrase}' raised {e!r}",
                    file=sys.stderr,
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
            print(
                f"DialogFlow: flow '{active.trigger_phrase}' raised {e!r}",
                file=sys.stderr,
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
            text = getattr(prompt, "prompt", "")
            if text:
                self._speak(text)
        else:
            self._advance(active, value=None)

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

    def cancel_active(self) -> bool:
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
            print(
                f"DialogFlow: global handler '{trigger_phrase}' raised {e!r}",
                file=sys.stderr,
            )
            return

        if isinstance(prompt, Say):
            self._speak(prompt.text)
        elif isinstance(prompt, (Ask, Confirm, Choose)) and active is not None:
            active.current_prompt = prompt
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
                if session.buffer:
                    session.buffer.pop()
                applied = True
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
        muted = False
        if self._mute_fn is not None:
            try:
                self._mute_fn(True)
                muted = True
                self._log("speak: mic muted")
            except Exception as e:
                self._log(f"speak: mute_fn failed: {e!r}")
                muted = False
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


def spell_out(s: str) -> List[str]:
    """Return ``s`` as a list of TTS-friendly tokens, one per character.

    Each character is rendered as a phrase the TTS engine can pronounce
    unambiguously (letters use spelling-alphabet sounds like ``"haitch"``
    for ``"h"``, upper-case letters are prefixed with ``"capital "``,
    digits become word form, common symbols use their spoken name).
    The per-character mapping lives in :func:`spoken_form` in
    ``alphanumeric_listener.py`` so the :class:`AlphanumericListener`'s
    TTS repeat-back and this function can't drift apart.

    ``spell_out("Hi#1")`` →
    ``["capital haitch", "eye", "hash", "one"]``.  Empty strings produce
    ``[]``.  This is for *speaking* strings back at the user, not for
    matching their input (that's :class:`AlphanumericMatcher`'s job).
    """
    return [spoken_form(c) for c in s]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m moonshine_voice.dialog_flow",
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
        "--quantization",
        default="q4",
        help="Embedding model variant (default: q4).",
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
        "--debug",
        action="store_true",
        help=(
            "Print DialogFlow stage-transition traces to stderr with "
            "per-step and cumulative timings."
        ),
    )
    args = parser.parse_args()

    tts_options: Dict[str, Any] = {}
    if args.tts_option:
        try:
            tts_options = dict(_parse_options_cli(args.tts_option))
        except ValueError as e:
            parser.error(str(e))

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

    # ---- Model and hardware setup ----------------------------------------

    print("Loading transcription model...", file=sys.stderr)
    _model_path, _model_arch = get_model_for_language(args.language)

    print(
        f"Loading embedding model (variant={args.quantization}) – "
        "first run may download...",
        file=sys.stderr,
    )
    _embed_path, _embed_arch = get_embedding_model(variant=args.quantization)
    intent_recognizer = IntentRecognizer(
        model_path=_embed_path,
        model_arch=_embed_arch,
        model_variant=args.quantization,
    )

    print("Creating microphone transcriber...", file=sys.stderr)
    mic = MicTranscriber(model_path=_model_path, model_arch=_model_arch)

    tts: Optional[Any] = None
    if not args.no_tts:
        print("Creating TTS...", file=sys.stderr)
        tts_kwargs: Dict[str, Any] = {}
        if tts_options:
            tts_kwargs["options"] = dict(tts_options)
        tts = TextToSpeech(language=args.language, **tts_kwargs)

    def mute(should_mute: bool) -> None:
        # Stop the mic from recording our own speech while we're talking.
        mic._should_listen = not should_mute

    def speak(text: str) -> None:
        """Log every spoken prompt and (optionally) pass it through TTS."""
        print(f"assistant: {text}", flush=True)
        if tts is not None:
            tts.say(text)
            try:
                tts.wait()
            except Exception:
                pass

    class _CompletedLinePrinter(TranscriptEventListener):
        """Logs every completed mic line as ``user: <text>``."""

        def on_line_completed(self, event):
            print(f"user: {event.line.text}", flush=True)

    # ---- Wire up DialogFlow ----------------------------------------------

    runner = DialogFlow(
        speak_fn=speak,
        intent_recognizer=intent_recognizer,
        mute_fn=mute,
        spell_feedback=True,
        debug=args.debug,
    )
    runner.register_flow("set up wifi", wifi_setup)
    runner.register_global("cancel", lambda d: d.cancel())
    runner.register_global("start over", lambda d: d.restart())

    mic.add_listener(_CompletedLinePrinter())
    mic.add_listener(runner)

    print(
        "\n🎤 Ready. Say 'set up wifi' or something similar to start.",
        file=sys.stderr,
    )
    print("Press Ctrl+C to stop.\n", file=sys.stderr)

    mic.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
    finally:
        mic.stop()
        mic.close()
        intent_recognizer.close()
