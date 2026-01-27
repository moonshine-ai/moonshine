from moonshine_voice.download import (
    get_model_for_language,
    load_wav_file,
    Transcriber,
    ModelArch,
)
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wav_path", type=str, default="test-assets/two_cities.wav")
parser.add_argument("--chunk_duration", type=float, default=0.48)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--options", type=str, default=None)
args = parser.parse_args()

options = {}
if args.options is not None:
    for option in args.options.split(","):
        key, value = option.split("=")
        options[key] = value

tiny_path, tiny_arch = get_model_for_language("en", ModelArch.TINY)
base_path, base_arch = get_model_for_language("en", ModelArch.BASE)
tiny_streaming_path, tiny_streaming_arch = get_model_for_language(
    "en", ModelArch.TINY_STREAMING
)
small_streaming_path, small_streaming_arch = get_model_for_language("en", ModelArch.SMALL_STREAMING)
medium_streaming_path, medium_streaming_arch = get_model_for_language("en", ModelArch.MEDIUM_STREAMING)

models = [
    (tiny_path, tiny_arch),
    (base_path, base_arch),
    (tiny_streaming_path, tiny_streaming_arch),
    (small_streaming_path, small_streaming_arch),
    (medium_streaming_path, medium_streaming_arch),
]

audio_data, sample_rate = load_wav_file(args.wav_path)
audio_duration = len(audio_data) / sample_rate

for model in models:
    path, arch = model
    transcriber = Transcriber(path, arch, options=options)
    transcriber.start()

    start_time = time.time()
    chunk_duration = args.chunk_duration
    chunk_size = int(chunk_duration * sample_rate)
    alignment_samples = (sample_rate * 80) // 1000
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        if i + chunk_size >= len(audio_data) and alignment_samples > 0:
            remainder = len(chunk) % alignment_samples
            if remainder:
                chunk = chunk + [0.0] * (alignment_samples - remainder)
        transcriber.add_audio(chunk, sample_rate)
    end_time = time.time()
    duration = end_time - start_time

    transcript = transcriber.stop()

    print(f"Model: {path} ({arch})")
    total_latency_ms = 0
    for line in transcript.lines:
        total_latency_ms += line.last_transcription_latency_ms
        if args.verbose:
            print(
                f"Line: {line.text}, Latency: {line.last_transcription_latency_ms:.0f}ms"
            )

    print(
        f"Transcription took {duration} seconds ({(duration / audio_duration) * 100:.2f}% of audio duration)"
    )
    if transcript.lines:
        average_latency_ms = total_latency_ms / len(transcript.lines)
        print(f"Average latency: {average_latency_ms:.0f}ms")
    else:
        print("Average latency: n/a (no transcript lines)")

    transcriber.close()
