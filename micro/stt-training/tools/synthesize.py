"""Synthesize the command vocabulary with Moonshine Voice ZipVoice TTS.

Every word in ``words.txt`` is spoken by each built-in ZipVoice speaker (and,
optionally, at a few speeds for prosody variety), resampled to 16 kHz, and
written in the Speech-Commands layout the trainer expects::

    data/tts/<voice>/<word>/<voice>_nohash_<rep>.wav

The first run downloads the ZipVoice models from the Moonshine CDN (cached
under MOONSHINE_VOICE_CACHE). Existing clips are skipped, so runs resume safely.

Requires: ``pip install moonshine-voice librosa soundfile``.

Example::

    python tools/synthesize.py --words-file words.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from stt_training.words import folder_for_word, load_words  # noqa: E402

TARGET_SR = 16000

# The 15 built-in ZipVoice speakers (one male + one female per accent where
# available). Discover the current set at runtime with
# ``moonshine_voice.download.list_tts_voices("en_us")``.
DEFAULT_VOICES = [
    "zipvoice_american_female",
    "zipvoice_american_male",
    "zipvoice_australian_male",
    "zipvoice_canadian_female",
    "zipvoice_canadian_male",
    "zipvoice_english_female",
    "zipvoice_english_male",
    "zipvoice_indian_female",
    "zipvoice_indian_male",
    "zipvoice_irish_female",
    "zipvoice_irish_male",
    "zipvoice_new_zealand_female",
    "zipvoice_northern_irish_female",
    "zipvoice_south_african_female",
    "zipvoice_south_african_male",
]


def discover_voices(language: str) -> list[str]:
    """Return the installed ZipVoice speakers, falling back to the known list.

    Queries moonshine-voice so the voice set tracks whatever the installed
    package version ships, rather than a hardcoded list that can drift.
    """
    try:
        from moonshine_voice import download as d

        present = d.list_tts_voices(language).get("present", [])
        voices = [v for v in present if v.startswith("zipvoice_")]
        if voices:
            return voices
    except Exception:  # noqa: BLE001 - fall back to the baked-in list
        pass
    return DEFAULT_VOICES


def _resample_to_16k(samples, sr: int) -> np.ndarray:
    y = np.asarray(samples, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--words-file", default="words.txt")
    ap.add_argument("--output-dir", default="data/tts")
    ap.add_argument("--language", default="en_us")
    ap.add_argument(
        "--voices",
        nargs="*",
        default=None,
        help="ZipVoice voices to use (default: all 15 built-in speakers).",
    )
    ap.add_argument(
        "--speeds",
        default="0.9,1.0,1.1",
        help="Comma-separated speaking speeds; one clip per (voice, word, speed).",
    )
    ap.add_argument("--dry-run", action="store_true", help="List planned clips, no synthesis.")
    args = ap.parse_args()

    words = load_words(args.words_file)
    voices = args.voices or discover_voices(args.language)
    speeds = [float(s) for s in str(args.speeds).split(",") if s.strip()]
    out_base = Path(args.output_dir)

    planned = [
        (v, w, i, sp)
        for v in voices
        for w in words
        for i, sp in enumerate(speeds)
    ]
    todo = [
        job for job in planned
        if not (out_base / job[0] / folder_for_word(job[1]) / f"{job[0]}_nohash_{job[2]}.wav").exists()
    ]
    print(
        f"{len(words)} words x {len(voices)} voices x {len(speeds)} speeds = "
        f"{len(planned)} clips; {len(todo)} to synthesize, "
        f"{len(planned) - len(todo)} already present."
    )
    if args.dry_run:
        for v, w, i, sp in todo:
            print(f"  {v}/{folder_for_word(w)}/{v}_nohash_{i}.wav (speed={sp})")
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    try:
        from moonshine_voice.tts import TextToSpeech
    except ImportError as exc:
        print(f"moonshine-voice not installed ({exc}). Run: pip install moonshine-voice")
        return 2

    made = 0
    # One engine per voice; iterate voices outer so the model loads once each.
    for voice in voices:
        voice_jobs = [job for job in todo if job[0] == voice]
        if not voice_jobs:
            continue
        print(f"[{voice}] synthesizing {len(voice_jobs)} clips...")
        tts = TextToSpeech(args.language, voice=voice)
        try:
            for _, word, rep, speed in voice_jobs:
                out_path = out_base / voice / folder_for_word(word) / f"{voice}_nohash_{rep}.wav"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if abs(speed - 1.0) > 1e-3:
                        samples, sr = tts.synthesize(word, speed=speed)
                    else:
                        samples, sr = tts.synthesize(word)
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARN failed {voice}/{word} (speed={speed}): {exc}")
                    continue
                y = _resample_to_16k(samples, sr)
                sf.write(str(out_path), y, TARGET_SR, subtype="PCM_16")
                made += 1
        finally:
            close = getattr(tts, "close", None)
            if callable(close):
                close()
        print(f"[{voice}] done ({made} total so far)")

    print(f"Synthesized {made} clips under {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
