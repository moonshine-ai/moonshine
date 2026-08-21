#!/usr/bin/env python3
"""Measures what sub-sentence chunking would cost ZipVoice.

ZipVoice conditions every generation on a reference clip, and the flow-matching
solve runs `num_step` times over the whole sequence. A rolling acoustic prompt
would let it emit audio before a sentence is finished, by feeding the tail of
what it just generated back in as the reference for the next piece. That is
only worth building if the arithmetic works out, and the arithmetic is unusual
here: the reference is *prepended* to the sequence being solved, so a short
chunk pays for the reference frames as well as its own.

This measures the real thing rather than the model of it: synthesize a
paragraph whole, then in progressively smaller pieces, and compare total time.
A multiplier near one means chunking is close to free; a large multiplier means
the rolling prompt cannot pay for itself no matter how good it sounds.

Measured on an M-series Mac with a four-second reference, four sentences of
text: 1.00x whole, 1.12x in two pieces, 1.45x in four. The reference is roughly
160 frames and is prepended to every solve, so the shorter the piece the more
of each solve is spent regenerating context. Half-second pieces would be about
a dozen frames of new audio behind those 160, which is why sub-sentence
chunking is not worth building here: the only way to make it cheap is a shorter
prompt, and a shorter prompt is what costs speaker similarity.

Usage:
    python scripts/zipvoice-chunk-cost.py --reference sample.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Bright vixens jump, dozy fowl quack."
)


def pieces(text: str, count: int) -> list[str]:
    """Splits into roughly equal groups of sentences."""
    import re

    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+", text) if u.strip()]
    if count >= len(units):
        return units
    per = -(-len(units) // count)
    return [" ".join(units[i: i + per]) for i in range(0, len(units), per)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="WAV of the voice to clone")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--text", default=PARAGRAPH)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    from moonshine_voice import TextToSpeech

    tts = (
        TextToSpeech()
        .language("en_us")
        .cloning()
        .load()
    )
    tts.clone_from(args.reference, transcript=args.transcript or None)

    print(f"{'pieces':>7} {'audio s':>8} {'wall s':>8} {'RTF':>6} {'vs whole':>9}")
    baseline = None
    for count in (1, 2, 4, 8):
        parts = pieces(args.text, count)
        if count > 1 and len(parts) < 2:
            break
        started = time.perf_counter()
        samples = 0
        rate = 24000
        for part in parts:
            pcm, rate = tts.synthesize(part)
            samples += len(pcm)
        elapsed = time.perf_counter() - started
        seconds = samples / rate
        baseline = baseline or elapsed
        print(f"{len(parts):7d} {seconds:8.2f} {elapsed:8.2f} "
              f"{elapsed / seconds:6.2f} {elapsed / baseline:8.2f}x")
    tts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
