import os
from pathlib import Path
import tokenizers
from model import MoonshineTFLiteModel

ASSETS_DIR = os.path.join(Path(__file__).parents[1], "moonshine", "assets")

def load_audio(audio):
    if isinstance(audio, (str, Path)):
        import librosa

        audio, _ = librosa.load(audio, sr=16_000)
        return audio[None, ...]
    else:
        return audio

def assert_audio_size(audio):
    assert len(audio.shape) == 2, "audio should be of shape [batch, samples]"
    num_seconds = audio.size / 16000
    assert 0.1 < num_seconds < 64, (
        "Moonshine models support audio segments that are between 0.1s and 64s in a single transcribe call. For transcribing longer segments, pre-segment your audio and provide shorter segments."
    )
    return num_seconds


def transcribe(audio, model_dir=None, model_name="moonshine/base", model_precision="float"):
    model = MoonshineTFLiteModel(model_dir=model_dir, model_name=model_name, model_precision=model_precision)
    audio = load_audio(audio)
    assert_audio_size(audio)

    tokens = model.generate(audio)
    return load_tokenizer().decode_batch(tokens)[0]

def load_tokenizer():
    tokenizer_file = os.path.join(ASSETS_DIR, "tokenizer.json")
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    return tokenizer

def benchmark(audio, model_dir=None, model_name="moonshine/base", model_precision="float"):
    import time

    model = MoonshineTFLiteModel(model_dir=model_dir, model_name=model_name, model_precision=model_precision)
    audio = load_audio(audio)
    num_seconds = assert_audio_size(audio)

    print("Warming up...")
    for _ in range(4):
        _ = model.generate(audio)

    print("Benchmarking...")
    N = 8
    start_time = time.time_ns()
    for _ in range(N):
        _ = model.generate(audio)
    end_time = time.time_ns()

    elapsed_time = (end_time - start_time) / N
    elapsed_time /= 1e6

    print(f"Time to transcribe {num_seconds:.2f}s of speech is {elapsed_time:.2f}ms")

if __name__ == '__main__':

    import tokenizers
    import sys

    if len(sys.argv) < 2:
        audio_filename = os.path.join(ASSETS_DIR, "beckett.wav")
    else:
        audio_filename = sys.argv[1]

    if len(sys.argv) < 3:
        model_name = "moonshine/base"
    else:
        model_name = sys.argv[2]

    if len(sys.argv) < 4:
        model_dir = None
    else:
        model_dir = sys.argv[3]

    audio = load_audio(audio_filename)

    text = transcribe(audio, model_name=model_name, model_dir=model_dir, model_precision="float")
    print(text)
    
    benchmark(audio, model_name=model_name, model_dir=model_dir, model_precision="float")
    