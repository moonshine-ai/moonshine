from pathlib import Path
import tokenizers
from .model import MoonshineOnnxModel

from . import ASSETS_DIR


def load_audio(audio):
    if isinstance(audio, (str, Path)):
        import librosa

        audio, _ = librosa.load(audio, sr=16_000)
    return audio[None, ...]


def assert_audio_size(audio):
    assert len(audio.shape) == 2, "audio should be of shape [batch, samples]"
    num_seconds = audio.size / 16000
    assert 0.1 < num_seconds < 64, (
        "Moonshine models support audio segments that are between 0.1s and 64s in a single transcribe call. For transcribing longer segments, pre-segment your audio and provide shorter segments."
    )
    return num_seconds


def transcribe(audio, model="moonshine/base"):
    if isinstance(model, str):
        model = MoonshineOnnxModel(model_name=model)
    assert isinstance(model, MoonshineOnnxModel), (
        f"Expected a MoonshineOnnxModel model or a model name, not a {type(model)}"
    )
    audio = load_audio(audio)
    assert_audio_size(audio)

    tokens = model.generate(audio)
    return load_tokenizer().decode_batch(tokens)


def load_tokenizer():
    tokenizer_file = ASSETS_DIR / "tokenizer.json"
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    return tokenizer


def benchmark(audio, model="moonshine/base"):
    import time

    if isinstance(model, str):
        model = MoonshineOnnxModel(model_name=model)
    assert isinstance(model, MoonshineOnnxModel), (
        f"Expected a MoonshineOnnxModel model or a model name, not a {type(model)}"
    )

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
