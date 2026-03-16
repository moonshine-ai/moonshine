"""Text-to-speech example using Moonshine Voice.

This example demonstrates how to use the TextToSpeech class to synthesize
speech from text. Grapheme-to-phoneme conversion is handled internally by
the Moonshine TTS engine.

Requirements:
    pip install soundfile  (optional, for high-quality WAV output)

Usage:
    python text_to_speech.py "Hello, world!"
    python text_to_speech.py --model-name tsuki-max-en "Hello, world!"
    python text_to_speech.py --model-path /path/to/model "Hello, world!"
"""

import argparse
import sys

import numpy as np

from moonshine_voice import TextToSpeech, TTSModelArch, get_tts_model


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
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    args = parser.parse_args()

    # 1. Resolve model path (download if needed).
    if args.model_path:
        model_path = args.model_path
        model_arch = TTSModelArch.TSUKI
    else:
        print(f"Downloading TTS model '{args.model_name}'...", file=sys.stderr)
        model_path, model_arch = get_tts_model(args.model_name)

    # 2. Load the TTS model.
    print(f"Loading TTS model from '{model_path}'...", file=sys.stderr)
    with TextToSpeech(model_path, model_arch) as tts:
        # 3. Generate audio (G2P is handled internally by the C library).
        print(f"Generating speech for: {args.text!r}", file=sys.stderr)
        result = tts.generate(args.text)
        print(
            f"Generated {len(result.audio_data)} samples at {result.sample_rate} Hz "
            f"({len(result.audio_data) / result.sample_rate:.2f}s)",
            file=sys.stderr,
        )

        # 4. Save to WAV.
        save_wav(args.output, result.audio_data, result.sample_rate)
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
