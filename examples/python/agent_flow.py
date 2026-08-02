"""Multi-step agent flow example using Moonshine Voice.

This example demonstrates the :class:`AgentFlow` runner driving a
generator-based conversational flow that sets up a wifi network.  The flow
reads top-to-bottom like a script: branching is ``if`` / ``else``, retries
are ``while``, and sub-dialogs are ``yield from``.

Three ways to run it:

* Default, interactive keyboard mode: prompts are printed to stdout,
  you type replies on stdin.  Fast, no audio hardware required.
* ``--mic``, live microphone mode: :class:`AgentFlow` opens the
  microphone and speaks its prompts aloud.
* ``--scripted``, canned-answer mode: drives the same flow from a
  pre-defined list of utterances, useful for smoke tests.
"""

import argparse
import sys
import time
from collections.abc import Iterable

from moonshine_voice import SPELLED, AgentFlow, spell_out

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
        f"\n[agent_flow] apply_wifi_config(ssid={ssid!r}, password={password!r})",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Live microphone mode
# ---------------------------------------------------------------------------


def run_live(args: argparse.Namespace) -> None:
    log_io = getattr(args, "log_io", False)

    # AgentFlow opens the recognizer, the synthesizer and the microphone
    # itself, so all this needs to describe is the conversation.
    runner = (
        AgentFlow()
        .language(args.language)
        .speech(not args.no_tts)
        .trigger_threshold(args.threshold)
        .log_io(log_io)
        .debug(getattr(args, "debug", False))
        .on_progress(
            lambda fraction, name: print(
                f"Loading {name}... {fraction:.0%}", file=sys.stderr
            )
        )
        .listen_for("set up wifi", setup_wifi)
        .listen_for("configure wifi", setup_wifi)
        .listen_for("onboard me", full_onboarding)
        .listen_for("set the timezone", set_timezone)
    )
    if getattr(args, "output_device", None) is not None:
        runner.output_device(args.output_device)
    if getattr(args, "tts_options", None):
        runner.speech_options(dict(args.tts_options))
    if not log_io:
        # With --log-io the runner prints both sides of the conversation
        # itself, so only echo when it's off.
        runner.on_heard(lambda text: print(f"user: {text}", flush=True))
        runner.on_said(lambda text: print(f"assistant: {text}", flush=True))

    runner.load()

    print(
        "\n🎤 Ready. Try saying 'set up wifi' or 'onboard me' "
        "(or 'cancel' to abandon the current flow).",
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


# ---------------------------------------------------------------------------
# Text-driven modes (no audio hardware)
# ---------------------------------------------------------------------------


def _build_text_runner(*, debug: bool = False) -> AgentFlow:
    """A runner with no microphone or synthesizer, printing prompts to stdout.

    Trigger, confirmation and choice matching still goes through the
    embedding model, so the first run may download it.
    """
    return (
        AgentFlow()
        .microphone(False)
        .speak_with(lambda text: print(f"assistant: {text}", flush=True))
        .debug(debug)
        .on_progress(
            lambda fraction, name: print(
                f"Loading {name}... {fraction:.0%}", file=sys.stderr
            )
        )
        .listen_for("set up wifi", setup_wifi)
        .listen_for("configure wifi", setup_wifi)
        .listen_for("onboard me", full_onboarding)
        .listen_for("set the timezone", set_timezone)
        .load()
    )


def _resolve_trigger(flow_name: str) -> str:
    return {
        "wifi": "set up wifi",
        "onboard": "onboard me",
        "timezone": "set the timezone",
    }.get(flow_name, flow_name)


# ---------------------------------------------------------------------------
# Interactive keyboard mode (default)
# ---------------------------------------------------------------------------


def run_interactive(
    flow_name: str,
    *,
    debug: bool = False,
) -> None:
    """Keyboard-driven demo: prompts go to stdout, replies come from stdin.

    Useful when you want to exercise the flow without any audio hardware
    or TTS latency.  All trigger / confirmation / choice matching still
    goes through the embedding model (first run may download it).
    """

    runner = _build_text_runner(debug=debug)
    trigger = _resolve_trigger(flow_name)

    print(f"user:      {trigger}")
    runner.handle_utterance(trigger)
    while runner.is_active:
        try:
            answer = input("you>      ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            runner.cancel()
            break
        if not answer:
            continue
        runner.handle_utterance(answer)
    runner.close()


# ---------------------------------------------------------------------------
# Scripted (no-audio) mode
# ---------------------------------------------------------------------------


def run_scripted(
    flow_name: str,
    answers: Iterable[str],
    *,
    debug: bool = False,
) -> None:
    """Drive a flow from a pre-canned list of utterances.

    Useful for smoke tests or running the example on machines without a
    microphone.  Speaks prompts to stdout instead of a TTS.  All
    trigger / confirmation / choice matching goes through the embedding
    model (first run may download it).
    """

    runner = _build_text_runner(debug=debug)
    trigger = _resolve_trigger(flow_name)

    print(f"user:      {trigger}")
    runner.handle_utterance(trigger)
    for answer in answers:
        print(f"user:      {answer}")
        runner.handle_utterance(answer)
        if not runner.is_active:
            break
    runner.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", default="en")
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
    parser.add_argument(
        "--list-output-devices",
        action="store_true",
        help=(
            "List PortAudio output devices and exit.  Useful when the "
            "assistant is silent under --mic, since the host default may "
            "not be the device with speakers."
        ),
    )
    parser.add_argument(
        "--output-device",
        default=None,
        metavar="INDEX_OR_NAME",
        help=(
            "Pin --mic TTS playback to a specific PortAudio output "
            "device (integer index or case-insensitive name "
            "substring).  Defaults to the host default."
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
            "Print AgentFlow stage-transition traces to stderr with "
            "per-step and cumulative timings, plus per-stage timing in "
            "the TextToSpeech synth and play workers (handy for "
            "diagnosing missing-beep / missing-audio issues)."
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

    if args.list_output_devices:
        from moonshine_voice.tts import list_output_devices

        print("PortAudio output devices (asterisk = current host default):")
        for line in list_output_devices():
            print(f"  {line}")
        sys.exit(0)

    # Coerce numeric strings to ints so the resolver picks them up as
    # indices instead of name substrings.
    if isinstance(args.output_device, str) and args.output_device.strip().isdigit():
        args.output_device = int(args.output_device.strip())

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
            debug=args.debug,
        )
    else:
        run_interactive(
            args.flow,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
