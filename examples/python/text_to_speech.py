"""Speaks text out loud using TextToSpeech, optionally in a cloned voice."""

import argparse
import sys

from moonshine_voice import TextToSpeech


class ProgressBar:
    """Draws a single-line download bar, replacing itself as it fills."""

    WIDTH = 30

    def __init__(self):
        self._done = False

    def update(self, fraction, filename):
        if self._done:
            return
        filled = int(self.WIDTH * fraction)
        bar = "#" * filled + "-" * (self.WIDTH - filled)
        print(f"\r[{bar}] {fraction:5.1%} {filename[-32:]:<32}", end="",
              file=sys.stderr, flush=True)
        if fraction >= 1.0:
            self._done = True
            print(file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Text to speech example")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello from Moonshine.",
        help="Text to speak",
    )
    parser.add_argument(
        "--language", type=str, default="en_us", help="Language to speak in"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice id, for example kokoro_af_heart (default: the engine's own)",
    )
    parser.add_argument(
        "--clone",
        type=str,
        default=None,
        metavar="WAV_PATH",
        help="Clone the voice in this recording instead of using --voice",
    )
    parser.add_argument(
        "--clone-from-mic",
        action="store_true",
        help="Clone the voice of whoever speaks into the microphone",
    )
    args = parser.parse_args()

    if args.voice is not None and (args.clone or args.clone_from_mic):
        parser.error("a cloned voice comes from the recording, so --voice cannot be set")

    cloning = bool(args.clone or args.clone_from_mic)
    tts = TextToSpeech().language(args.language)
    if args.voice is not None:
        tts.voice(args.voice)
    elif cloning:
        tts.cloning()
    if sys.stderr.isatty():
        tts.on_progress(ProgressBar().update)

    tts.load()

    with tts:
        if args.clone:
            tts.clone_from(args.clone)
        elif args.clone_from_mic:
            clone = tts.start_cloning()
            clone.on_ready(lambda: print("Got it, you can stop talking.", file=sys.stderr))
            print("Say something, so I can learn your voice...", file=sys.stderr)
            clone.from_microphone()
            tts.clone_from(clone)

        tts.say(args.text)
        tts.wait()


if __name__ == "__main__":
    main()
