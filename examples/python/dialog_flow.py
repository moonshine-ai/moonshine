"""Multi-step dialog flow example using Moonshine Voice.

This example demonstrates the :class:`DialogFlow` runner driving a
generator-based conversational flow that sets up a wifi network.  The flow
reads top-to-bottom like a script – branching is ``if`` / ``else``, retries
are ``while``, and sub-dialogs are ``yield from``.

Three ways to run it:

* Default – interactive keyboard mode: prompts are printed to stdout,
  you type replies on stdin.  Fast, no audio hardware required.
* ``--mic`` – live microphone + TTS mode: the :class:`MicTranscriber`
  gathers user input and the assistant speaks through
  :class:`TextToSpeech`.
* ``--scripted`` – canned-answer mode: drives the same flow from a
  pre-defined list of utterances, useful for smoke tests.
"""

import argparse
import sys
import time
from typing import Iterable

from typing import Optional

from moonshine_voice import (
    DialogFlow,
    IntentRecognizer,
    MicTranscriber,
    SPELLED,
    TextToSpeech,
    TranscriptEventListener,
    get_embedding_model,
    get_model_for_language,
    get_spelling_model_path,
    spell_out,
)
from moonshine_voice.transcriber import MOONSHINE_FLAG_SPELLING_MODE


# ---------------------------------------------------------------------------
# Flow definitions
# ---------------------------------------------------------------------------


def setup_wifi(d):
    """Classic slot-filling flow: network name, password, confirm, apply."""

    ssid = yield d.ask("What's the name of your wifi network?")

    if not (yield d.confirm(f"I heard, {ssid}. Is that right?")):
        yield d.say("No problem, let's start over.")
        return

    password = yield d.ask(
        "Please spell the wifi password, one letter at a time, and say 'done' when finished.",
        mode=SPELLED,
    )

    if (yield d.confirm("Would you like to hear it read back?")):
        yield d.say(f"I heard: {spell_out(password)}")

    if (yield d.confirm("Apply these changes?")):
        _apply_wifi_config(ssid, password)
        yield d.say("Done. Your wifi is set up.")
    else:
        yield d.say("Okay, nothing changed.")


def set_timezone(d):
    """Sub-flow that can be composed with ``yield from``."""

    tz = yield d.ask("Which timezone should I use?")
    if (yield d.confirm(f"I heard, {tz}. Use that?")):
        yield d.say(f"Timezone set to {tz}.")
    else:
        yield d.say("Leaving the timezone as it is.")


def full_onboarding(d):
    """Compose sub-flows with ``yield from``."""

    yield d.say("Let's get you set up.")
    yield from setup_wifi(d)
    yield from set_timezone(d)
    yield d.say("All done.")


def _apply_wifi_config(ssid: str, password: str) -> None:
    # This is where you'd integrate with whatever wifi backend you have.
    print(
        f"\n[dialog_flow] apply_wifi_config(ssid={ssid!r}, password={password!r})",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Terminal transcript printer
# ---------------------------------------------------------------------------


class TranscriptPrinter(TranscriptEventListener):
    """Echoes in-progress transcripts and logs every completed line."""

    def __init__(self):
        self.last_len = 0

    def _overwrite(self, text: str) -> None:
        print(f"\r{text}", end="", flush=True)
        if len(text) < self.last_len:
            print(" " * (self.last_len - len(text)), end="", flush=True)
        self.last_len = len(text)

    def on_line_started(self, event):
        self.last_len = 0

    def on_line_text_changed(self, event):
        self._overwrite(f"user (partial): {event.line.text}")

    def on_line_completed(self, event):
        if self.last_len:
            # Wipe the in-progress line so the finalized log starts clean.
            print(f"\r{' ' * self.last_len}\r", end="", flush=True)
            self.last_len = 0
        print(f"user: {event.line.text}", flush=True)


# ---------------------------------------------------------------------------
# Live microphone mode
# ---------------------------------------------------------------------------


def run_live(args: argparse.Namespace) -> None:
    print("Loading transcription model...", file=sys.stderr)
    model_path, model_arch = get_model_for_language(args.language)

    print("Loading embedding model...", file=sys.stderr)
    embedding_model_path, embedding_model_arch = get_embedding_model(
        args.embedding_model, args.quantization
    )

    print("Creating intent recognizer...", file=sys.stderr)
    intent_recognizer = IntentRecognizer(
        model_path=embedding_model_path,
        model_arch=embedding_model_arch,
        model_variant=args.quantization,
        threshold=args.threshold,
    )

    # Pre-fetch the alphanumeric spelling-CNN if one is published for
    # this language; DialogFlow flips MOONSHINE_FLAG_SPELLING_MODE on
    # only while the active prompt is in SPELLED / DIGITS mode (so
    # password / code dictation gets the C++ spelling-fusion path
    # without perturbing free-form recognition or trigger matching).
    print("Creating microphone transcriber...", file=sys.stderr)
    spelling_model_path: Optional[str] = None
    try:
        spelling_model_path = get_spelling_model_path(args.language)
    except Exception as e:
        print(
            f"Spelling model: lookup failed ({e!r}); SPELLED mode will "
            "fall back to matcher-only classification.",
            file=sys.stderr,
        )
    if spelling_model_path is not None:
        print(f"Spelling model: loaded {spelling_model_path}.", file=sys.stderr)

    mic = MicTranscriber(
        model_path=model_path,
        model_arch=model_arch,
        spelling_model_path=spelling_model_path,
    )

    tts = None
    if not args.no_tts:
        print("Creating TTS (this can take a moment on first run)...", file=sys.stderr)
        tts_kwargs = {}
        if getattr(args, "tts_options", None):
            tts_kwargs["options"] = dict(args.tts_options)
        tts = TextToSpeech(language=args.language, **tts_kwargs)

    def mute(should_mute: bool) -> None:
        # Stop the mic from recording our own speech while we're talking.
        mic._should_listen = not should_mute

    def set_spelling_mode(active: bool) -> None:
        """Toggle the C++ spelling-CNN fusion path on the live mic stream."""
        mic.set_transcribe_flags(MOONSHINE_FLAG_SPELLING_MODE if active else 0)

    def speak(text: str) -> None:
        """Log every spoken prompt and (optionally) pass it through TTS."""
        print(f"assistant: {text}", flush=True)
        if tts is not None:
            tts.say(text)
            try:
                tts.wait()
            except Exception:
                pass

    runner = DialogFlow(
        speak_fn=speak,
        intent_recognizer=intent_recognizer,
        mute_fn=mute,
        spelling_mode_fn=(
            set_spelling_mode if spelling_model_path is not None else None
        ),
        spell_feedback=True,
        debug=getattr(args, "debug", False),
    )
    runner.register_flow("set up wifi", setup_wifi)
    runner.register_flow("configure wifi", setup_wifi)
    runner.register_flow("onboard me", full_onboarding)
    runner.register_flow("set the timezone", set_timezone)

    runner.register_global("cancel", lambda d: d.cancel())
    runner.register_global("start over", lambda d: d.restart())

    mic.add_listener(TranscriptPrinter())
    mic.add_listener(runner)

    print(
        "\n🎤 Ready. Try saying 'set up wifi' or 'onboard me' "
        "(or 'cancel' to abandon the current flow).",
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


# ---------------------------------------------------------------------------
# Interactive keyboard mode (default)
# ---------------------------------------------------------------------------


def run_interactive(
    flow_name: str,
    *,
    embedding_model: str = "embeddinggemma-300m",
    quantization: str = "q4",
    debug: bool = False,
) -> None:
    """Keyboard-driven demo – prompts go to stdout, replies come from stdin.

    Useful when you want to exercise the flow without any audio hardware
    or TTS latency.  All trigger / confirmation / choice matching still
    goes through the embedding model (first run may download it).
    """

    def speak(text: str) -> None:
        print(f"assistant: {text}", flush=True)

    print(
        f"Loading embedding model (variant={quantization}) – first run may download...",
        file=sys.stderr,
    )
    embedding_model_path, embedding_model_arch = get_embedding_model(
        embedding_model, quantization
    )
    intent_recognizer = IntentRecognizer(
        model_path=embedding_model_path,
        model_arch=embedding_model_arch,
        model_variant=quantization,
    )

    runner = DialogFlow(
        speak_fn=speak,
        intent_recognizer=intent_recognizer,
        debug=debug,
    )
    runner.register_flow("set up wifi", setup_wifi)
    runner.register_flow("configure wifi", setup_wifi)
    runner.register_flow("onboard me", full_onboarding)
    runner.register_flow("set the timezone", set_timezone)
    runner.register_global("cancel", lambda d: d.cancel())
    runner.register_global("start over", lambda d: d.restart())

    trigger_map = {
        "wifi": "set up wifi",
        "onboard": "onboard me",
        "timezone": "set the timezone",
    }
    trigger = trigger_map.get(flow_name, flow_name)

    print(f"user:      {trigger}")
    runner.process_utterance(trigger)
    while runner.is_active:
        try:
            answer = input("you>      ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            runner.cancel_active()
            break
        if not answer:
            continue
        runner.process_utterance(answer)


# ---------------------------------------------------------------------------
# Scripted (no-audio) mode
# ---------------------------------------------------------------------------


def run_scripted(
    flow_name: str,
    answers: Iterable[str],
    *,
    embedding_model: str = "embeddinggemma-300m",
    quantization: str = "q4",
    debug: bool = False,
) -> None:
    """Drive a flow from a pre-canned list of utterances.

    Useful for smoke tests or running the example on machines without a
    microphone.  Speaks prompts to stdout instead of a TTS.  All
    trigger / confirmation / choice matching goes through the embedding
    model (first run may download it).
    """

    def speak(text: str) -> None:
        print(f"assistant: {text}")

    print(
        f"Loading embedding model (variant={quantization}) – first run may download...",
        file=sys.stderr,
    )
    embedding_model_path, embedding_model_arch = get_embedding_model(
        embedding_model, quantization
    )
    intent_recognizer = IntentRecognizer(
        model_path=embedding_model_path,
        model_arch=embedding_model_arch,
        model_variant=quantization,
    )

    runner = DialogFlow(
        speak_fn=speak,
        intent_recognizer=intent_recognizer,
        debug=debug,
    )
    runner.register_flow("set up wifi", setup_wifi)
    runner.register_flow("onboard me", full_onboarding)
    runner.register_flow("set the timezone", set_timezone)
    runner.register_global("cancel", lambda d: d.cancel())
    runner.register_global("start over", lambda d: d.restart())

    trigger_map = {
        "wifi": "set up wifi",
        "onboard": "onboard me",
        "timezone": "set the timezone",
    }
    trigger = trigger_map.get(flow_name, flow_name)

    print(f"user:      {trigger}")
    runner.process_utterance(trigger)
    for answer in answers:
        print(f"user:      {answer}")
        runner.process_utterance(answer)
        if not runner.is_active:
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", default="en")
    parser.add_argument("--embedding-model", default="embeddinggemma-300m")
    parser.add_argument("--quantization", default="q4")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Under --mic, print prompts instead of speaking them.",
    )
    parser.add_argument(
        "--tts-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra option forwarded to TextToSpeech (only meaningful under "
            "--mic); repeat for multiple (e.g. --tts-option speed=1.1 "
            "--tts-option voice=kokoro_af_heart)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mic",
        action="store_true",
        help=(
            "Use the microphone for input and TTS for output instead of the "
            "keyboard.  Off by default."
        ),
    )
    mode.add_argument(
        "--scripted",
        action="store_true",
        help="Run the flow with a canned list of utterances (no input needed).",
    )
    parser.add_argument(
        "--flow",
        default="wifi",
        choices=("wifi", "onboard", "timezone"),
        help="Which flow to run (used by --scripted and the default keyboard mode).",
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

    if args.tts_option:
        from moonshine_voice.tts import _parse_options_cli

        try:
            args.tts_options = dict(_parse_options_cli(args.tts_option))
        except ValueError as e:
            parser.error(str(e))
    else:
        args.tts_options = {}

    if args.mic:
        run_live(args)
    elif args.scripted:
        canned = {
            "wifi": [
                "HomeWifi", "yes", "s e c r e t 1 2 3", "done", "yes", "yes",
            ],
            "onboard": [
                "HomeWifi", "yes", "s w o r d f i s h", "done", "no", "yes",
                "America Los Angeles", "yes",
            ],
            "timezone": ["America New York", "yes"],
        }
        run_scripted(
            args.flow,
            canned[args.flow],
            embedding_model=args.embedding_model,
            quantization=args.quantization,
            debug=args.debug,
        )
    else:
        run_interactive(
            args.flow,
            embedding_model=args.embedding_model,
            quantization=args.quantization,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
