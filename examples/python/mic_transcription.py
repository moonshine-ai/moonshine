"""Transcribes speech from a microphone using MicTranscriber."""

import argparse
import sys
import time

from moonshine_voice import MicTranscriber, ModelArch


class LineRewriter:
    """Rewrites the current terminal line in place as the text firms up."""

    def __init__(self):
        self._width = 0

    def partial(self, text):
        # Pad to the previous width so a shorter guess doesn't leave debris.
        padding = " " * max(self._width - len(text), 0)
        print(f"\r{text}{padding}", end="", flush=True)
        self._width = len(text)

    def final(self, line):
        self.partial(line.text)
        self._width = 0
        print()


def main():
    parser = argparse.ArgumentParser(description="Microphone transcription example")
    parser.add_argument(
        "--language", type=str, default="en", help="Language to use for transcription"
    )
    parser.add_argument(
        "--model-arch",
        type=int,
        default=None,
        help="Model size to use, instead of the default for the language",
    )
    args = parser.parse_args()

    mic = MicTranscriber().language(args.language)
    if args.model_arch is not None:
        mic.model_arch(ModelArch(args.model_arch))

    if sys.stdout.isatty():
        rewriter = LineRewriter()
        mic.on_text(rewriter.partial).on_line(rewriter.final)
    else:
        # Piped output can't be rewritten, so print each line once it is final.
        mic.on_line(lambda line: print(line.text, flush=True))

    mic.load()

    print("Listening to the microphone, press Ctrl+C to stop...", file=sys.stderr)
    with mic:
        mic.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            mic.stop()


if __name__ == "__main__":
    main()
