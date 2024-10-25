import os
import sys
import argparse
import wave
import numpy as np
import tokenizers

MOONSHINE_DEMO_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(MOONSHINE_DEMO_DIR, ".."))

from onnx_model import MoonshineOnnxModel


def main(models_dir, wav_file):
    m = MoonshineOnnxModel(models_dir=models_dir)
    with wave.open(wav_file) as f:
        params = f.getparams()
        assert (
            params.nchannels == 1
            and params.framerate == 16_000
            and params.sampwidth == 2
        ), f"wave file should have 1 channel, 16KHz, and int16"
        audio = f.readframes(params.nframes)
    audio = np.frombuffer(audio, np.int16) / 32768.0
    audio = audio.astype(np.float32)[None, ...]
    tokens = m.generate(audio)
    tokenizer = tokenizers.Tokenizer.from_file(
        os.path.join(MOONSHINE_DEMO_DIR, "..", "assets", "tokenizer.json")
    )
    text = tokenizer.decode_batch(tokens)
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="onnx_standalone",
        description="Standalone ONNX demo of Moonshine models",
    )
    parser.add_argument(
        "--models_dir", help="Directory containing ONNX files", required=True
    )
    parser.add_argument("--wav_file", help="Speech WAV file", required=True)
    args = parser.parse_args()
    main(**vars(args))
