"""Text-to-speech example using Moonshine Voice.

This example demonstrates how to use the TextToSpeech class to synthesize
speech from text. It uses the ``phonemizer`` library (backed by espeak) for
grapheme-to-phoneme conversion and the Moonshine TTS C engine for waveform
generation.

Requirements:
    pip install phonemizer soundfile
    # espeak-ng must also be installed on the system:
    #   macOS:  brew install espeak-ng
    #   Ubuntu: sudo apt install espeak-ng

Usage:
    python text_to_speech.py "Hello, world!"
    python text_to_speech.py --model-name tsuki-max-en "Hello, world!"
    python text_to_speech.py --model-path /path/to/model "Hello, world!"
    python text_to_speech.py --language en-us "Hello, world!"
"""

import argparse
import sys

import numpy as np

from phonemizer import phonemize

from moonshine_voice import TextToSpeech, TTSModelArch, get_tts_model


def text_to_phonemes(text: str, language: str = "en-us") -> str:
    """Convert text to IPA phonemes using espeak via phonemizer."""
    return phonemize(
        text,
        language=language,
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        strip=True,
    )


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
        default=None,
        help="Path to a local TTS model directory (containing model.onnx and vocab.json). "
        "If not provided, the model is downloaded automatically.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="tsuki-max-en",
        help="Name of the TTS model to download (default: tsuki-max-en). "
        "Ignored when --model-path is given.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en-us",
        help="espeak language code for phonemization (default: en-us)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    args = parser.parse_args()

    # 1. Convert text → phonemes.
    phonemes = text_to_phonemes(args.text, language=args.language)
    print(f"Phonemes: {phonemes}", file=sys.stderr)

    # 2. Resolve model path (download if needed).
    if args.model_path:
        model_path = args.model_path
        model_arch = TTSModelArch.TSUKI
    else:
        print(f"Downloading TTS model '{args.model_name}'...", file=sys.stderr)
        model_path, model_arch = get_tts_model(args.model_name)

    # 3. Load the TTS model.
    print(f"Loading TTS model from '{model_path}'...", file=sys.stderr)
    with TextToSpeech(model_path, model_arch) as tts:
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
