"""Text-to-speech example using Moonshine Voice.

This example demonstrates how to use the TextToSpeech class to synthesize
speech from text. It uses the ``misaki`` library for grapheme-to-phoneme
conversion and the Moonshine TTS C engine for waveform generation.

Requirements:
    pip install misaki soundfile

Usage:
    python text_to_speech.py "Hello, world!"
    python text_to_speech.py --language en --output hello.wav "Hello, world!"
    python text_to_speech.py --language ja "こんにちは世界"
"""

import argparse
import sys

import numpy as np

from misaki import en, ja

from moonshine_voice import TextToSpeech, TTSModelArch


def create_g2p(language: str):
    """Create a grapheme-to-phoneme converter for the given language."""
    if language == "en":
        return en.G2P(british=False)
    elif language == "ja":
        return ja.JAG2P(version="pyopenjtalk")
    else:
        raise ValueError(f"Unsupported language: {language}")


def phonemize(g2p, text: str) -> str:
    """Convert text to IPA phonemes using the given G2P converter."""
    phonemes = " ".join(
        t.phonemes for t in g2p(text)[1] if t.phonemes is not None
    )
    return phonemes


def save_wav(path: str, audio_data: np.ndarray, sample_rate: int):
    """Save audio data to a WAV file."""
    try:
        import soundfile as sf

        sf.write(path, audio_data, sample_rate)
    except ImportError:
        # Fallback to the wave module from the standard library.
        import wave

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-speech example using Moonshine Voice"
    )
    parser.add_argument(
        "text",
        type=str,
        help="The text to synthesize into speech",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the TTS model directory (containing model.onnx and vocab.json)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "ja"],
        help="Language for grapheme-to-phoneme conversion (default: en)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    args = parser.parse_args()

    # 1. Set up the G2P front-end.
    print(f"Loading G2P for language '{args.language}'...", file=sys.stderr)
    g2p = create_g2p(args.language)

    # 2. Convert text → phonemes.
    phonemes = phonemize(g2p, args.text)
    print(f"Phonemes: {phonemes}", file=sys.stderr)

    # 3. Load the TTS model.
    print(f"Loading TTS model from '{args.model_path}'...", file=sys.stderr)
    with TextToSpeech(args.model_path, TTSModelArch.TSUKI) as tts:
        # 4. Generate audio.
        result = tts.generate(phonemes)
        print(
            f"Generated {len(result.audio_data)} samples at {result.sample_rate} Hz "
            f"({len(result.audio_data) / result.sample_rate:.2f}s)",
            file=sys.stderr,
        )

        # 5. Save to WAV.
        save_wav(args.output, result.audio_data, result.sample_rate)
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
