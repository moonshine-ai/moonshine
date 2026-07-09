#!/usr/bin/env python3
# Copyright 2026 Moonshine AI (MIT License)
"""Synthesize a two-speaker diarization test clip with ZipVoice TTS.

Alternates short lines from Samuel Beckett's *Endgame* between Nagg
(``zipvoice_english_male``) and Nell (``zipvoice_english_female``), inserts
brief pauses, and writes a mono PCM16 WAV suitable for transcription tests.

Example:
  python3 scripts/generate-diarization-test-audio.py \\
      --out test-assets/endgame_nagg_nell.wav
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

from moonshine_voice.tts import TextToSpeech, _write_wav_mono_pcm16  # noqa: E402

# Short alternating exchange from the Nagg/Nell bin scene in *Endgame*.
# Public-domain text; trimmed to land near thirty seconds when spoken.
DIALOGUE: List[Tuple[str, str]] = [
    ("zipvoice_english_male", "Do you hear me?"),
    ("zipvoice_english_female", "Yes."),
    ("zipvoice_english_male", "Do you hear me?"),
    ("zipvoice_english_female", "Yes."),
    (
        "zipvoice_english_male",
        "We laugh less heartily, so we have still that resource.",
    ),
    (
        "zipvoice_english_female",
        "We laugh less heartily, so we have still that resource?",
    ),
    ("zipvoice_english_male", "No, not heartily, just less heartily."),
    ("zipvoice_english_female", "Ah that's a good story, ah that's a good story."),
    ("zipvoice_english_male", "What are you talking about?"),
    ("zipvoice_english_female", "Laughing."),
    ("zipvoice_english_male", "I thought you were in pain."),
    ("zipvoice_english_female", "Not now."),
    (
        "zipvoice_english_male",
        "Do you remember the day I went and fetched you from that ditch?",
    ),
    ("zipvoice_english_female", "No."),
]

TARGET_DURATION_SECONDS = 30.0
PAUSE_SECONDS = 0.55
SYNTHESIS_SPEED = 0.92


def _synthesize_utterance(
    voice: str,
    text: str,
    asset_root: Path,
) -> Tuple[List[float], int]:
    tts = TextToSpeech(
        "en_us",
        voice=voice,
        asset_root=asset_root,
        download=False,
    )
    try:
        samples, sample_rate = tts.synthesize(text, speed=SYNTHESIS_SPEED)
        return list(samples), int(sample_rate)
    finally:
        tts.close()


def _append_silence(samples: List[float], sample_rate: int, seconds: float) -> None:
    if seconds <= 0.0:
        return
    samples.extend([0.0] * int(round(sample_rate * seconds)))


def generate_dialogue_wav(asset_root: Path) -> Tuple[List[float], int]:
    combined: List[float] = []
    sample_rate = 0

    for index, (voice, text) in enumerate(DIALOGUE):
        utterance, sample_rate = _synthesize_utterance(voice, text, asset_root)
        combined.extend(utterance)
        if index + 1 < len(DIALOGUE):
            _append_silence(combined, sample_rate, PAUSE_SECONDS)

    return combined, sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "test-assets" / "endgame_nagg_nell.wav",
        help="Output WAV path (default: test-assets/endgame_nagg_nell.wav)",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=REPO_ROOT / "core" / "moonshine-tts" / "data",
        help="Moonshine TTS g2p_root (default: core/moonshine-tts/data)",
    )
    parser.add_argument(
        "--also-copy-to-python-assets",
        action="store_true",
        default=True,
        help="Also write python/src/moonshine_voice/assets/<basename> (default: on)",
    )
    parser.add_argument(
        "--no-also-copy-to-python-assets",
        dest="also_copy_to_python_assets",
        action="store_false",
    )
    args = parser.parse_args()

    asset_root = args.asset_root.resolve()
    if not asset_root.is_dir():
        raise SystemExit(f"TTS asset root not found: {asset_root}")

    samples, sample_rate = generate_dialogue_wav(asset_root)
    duration = len(samples) / float(sample_rate)
    print(
        f"Synthesized {len(DIALOGUE)} lines at {sample_rate} Hz, "
        f"duration {duration:.1f}s"
    )

    out_path = args.out.resolve()
    _write_wav_mono_pcm16(out_path, samples, sample_rate)
    print(f"Wrote {out_path}")

    if args.also_copy_to_python_assets:
        py_path = (
            REPO_ROOT / "python" / "src" / "moonshine_voice" / "assets" / out_path.name
        )
        _write_wav_mono_pcm16(py_path, samples, sample_rate)
        print(f"Wrote {py_path}")

    if duration < 20.0 or duration > 40.0:
        print(
            f"Warning: duration {duration:.1f}s is outside the ~30s target "
            f"({TARGET_DURATION_SECONDS}s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
