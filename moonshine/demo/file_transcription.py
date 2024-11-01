"""WAV file long-form transcription with Moonshine ONNX models."""

import argparse
import os
import sys
import time
import wave

import numpy as np
import tokenizers

from silero_vad import get_speech_timestamps, load_silero_vad

MOONSHINE_DEMO_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(MOONSHINE_DEMO_DIR, ".."))

from onnx_model import MoonshineOnnxModel


def main(model_name, wav_path):
    model = MoonshineOnnxModel(model_name=model_name)

    tokenizer = tokenizers.Tokenizer.from_file(
        os.path.join(MOONSHINE_DEMO_DIR, "..", "assets", "tokenizer.json")
    )

    with wave.open(wav_path) as f:
        params = f.getparams()
        assert (
            params.nchannels == 1
            and params.framerate == 16000
            and params.sampwidth == 2
        ), f"WAV file must have 1 channel, 16KHz rate, and int16 precision."
        audio = f.readframes(params.nframes)
    audio = np.frombuffer(audio, np.int16) / np.iinfo(np.int16).max
    audio = audio.astype(np.float32)

    vad_model = load_silero_vad()
    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        max_speech_duration_s=30,
        min_silence_duration_ms=2000,
        min_speech_duration_ms=250,
        speech_pad_ms=400,
    )
    chunks = [audio[ts["start"] : ts["end"]] for ts in speech_timestamps]

    chunks_length = 0
    transcription = ""

    start_time = time.time()

    for chunk in chunks:
        tokens = model.generate(chunk[None, ...])
        transcription += tokenizer.decode_batch(tokens)[0] + " "

        chunks_length += len(chunk)

    time_took = time.time() - start_time

    print(f"""
{transcription}

  model realtime factor:  {((chunks_length / 16000) / time_took):.2f}x
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="file_transcription.py",
        description="Standalone file transcription with Moonshine ONNX models.",
    )
    parser.add_argument(
        "--model_name",
        help="Model to run the demo with.",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
    )
    parser.add_argument(
        "--wav_path",
        help="Path to speech WAV file.",
        default=os.path.join(
            MOONSHINE_DEMO_DIR, "..", "assets", "a_tale_of_two_cities.wav"
        ),
    )
    args = parser.parse_args()
    main(**vars(args))
