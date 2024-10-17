import os
import tokenizers
import keras
from .model import load_model, Moonshine


def transcribe(audio, model="moonshine/base"):
    if isinstance(model, str):
        model = load_model(model)
    assert isinstance(
        model, Moonshine
    ), f"Expected a Moonshine model or a model name, not a {type(model)}"

    if isinstance(audio, str):
        import librosa

        audio, _ = librosa.load(audio, sr=16_000)
        audio = keras.ops.expand_dims(keras.ops.convert_to_tensor(audio), 0)
    assert len(keras.ops.shape(audio)) == 2, "audio should be of shape [batch, samples]"

    tokens = model.generate(audio)
    tokenizer_file = os.path.join(os.path.dirname(__file__), "assets", "tokenizer.json")
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
    return tokenizer.decode_batch(tokens)


def benchmark(audio, model="moonshine/base"):
    import time

    if isinstance(model, str):
        model = load_model(model)
    assert isinstance(
        model, Moonshine
    ), f"Expected a Moonshine model or a model name, not a {type(model)}"

    if isinstance(audio, str):
        import librosa

        audio, _ = librosa.load(audio, sr=16_000)
        audio = keras.ops.expand_dims(keras.ops.convert_to_tensor(audio), 0)
    assert len(keras.ops.shape(audio)) == 2, "audio should be of shape [batch, samples]"

    num_seconds = keras.ops.convert_to_numpy(keras.ops.size(audio) / 16_000)

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
