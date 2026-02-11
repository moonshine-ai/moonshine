# Benchmark to compare Moonshine and Whisper model latency in live speech scenarios.
#
# This is an opinionated benchmark that looks at the latency and total compute cost
# of the two families of models in a situation that is representative of many common
# real-time voice applications' requirements:
#
#  - Speech needs to be responded to as quickly as possible once a user completes a phrase.
#  - The phrases are of durations between a range of one to ten seconds.
#
# These are very different requirements from bulk offline processing scenarios, where the
# overall throughput of the system is more important, and so the latency on a single
# segment of speech is less important than the overall throughput of the system. This
# allows optimizations like batch processing.
#
# We are not claiming that Whisper is not a great model for offline processing, but we
# do want to highlight the advantages we that Moonshine offers for live speech
# applications with real-time latency requirements.
#
# The experimental setup is as follows:
#
#  - We use the two_cities.wav audio file as a test case, since it has a mix of short
# and long phrases. You can vary this by passing in your own audio file with the
# --wav_path argument.
#  - We use the Moonshine Tiny, Base, Tiny Streaming, Small Streaming, and Medium
# Streaming models.
#  - We compare these to the Whisper Tiny, Base, Small, and Large v3 models. Since the
# Moonshine Medium Streaming model achieves lower WER than Whisper Large v3 we compare
# those two, otherwise we compare each with their namesake.
#  - We use the Moonshine VAD segmenter to split the audio into phrases, and feed each
# phrase to Whisper for transcription.
#  - Response latency for both models is measured as the time between a phrase being
# identified as complete by the VAD segmenter and the transcribed text being returned.
# For Whisper this means the full transcription time, but since the Moonshine models
# are streaming we can do a lot of the work while speech is still happening, so the
# latency is much lower.
#  - We measure the total compute cost of the models by totalling the duration of the
# audio processing times for each model, and then expressing that as a percentage of the
# total audio duration. This is the inverse of the commonly used real-time factor (RTF)
# metric, but it reflects the compute load required for a real-time application.
# - We're using faster-whisper for Whisper, since that seems to provide the best
# cross-platform performance. We're also sticking with the CPU, since most applications
# can't rely on GPU or NPU acceleration being present on all the platforms they target.
# We know there are a lot of great GPU/NPU-accelerated Whisper implementations out there,
# but these aren't portable enough to be useful for the applications we care about.
#
from moonshine_voice import (
    get_model_for_language,
    load_wav_file,
    Transcriber,
    ModelArch,
    TranscriptEventListener,
)
from faster_whisper import WhisperModel

import time
import argparse
import numpy as np

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
small_streaming_path, small_streaming_arch = get_model_for_language(
    "en", ModelArch.SMALL_STREAMING
)
medium_streaming_path, medium_streaming_arch = get_model_for_language(
    "en", ModelArch.MEDIUM_STREAMING
)

if args.verbose:
    print("Loading Whisper models...")
whisper_model_sizes = ["tiny", "base", "small", "medium", "large-v3"]
whisper_models = [
    (WhisperModel(model_size, "cpu", compute_type="int8"))
    for model_size in whisper_model_sizes
]
if args.verbose:
    print("Whisper models loaded")

models = [
    (tiny_path, tiny_arch, "tiny-en", whisper_models[0], whisper_model_sizes[0]),
    (base_path, base_arch, "base-en", whisper_models[1], whisper_model_sizes[1]),
    (
        tiny_streaming_path,
        tiny_streaming_arch,
        "tiny-streaming-en",
        whisper_models[0],
        whisper_model_sizes[0],
    ),
    (
        small_streaming_path,
        small_streaming_arch,
        "small-streaming-en",
        whisper_models[2],
        whisper_model_sizes[2],
    ),
    (
        medium_streaming_path,
        medium_streaming_arch,
        "medium-streaming-en",
        whisper_models[4],
        whisper_model_sizes[4],
    ),
]

audio_data, sample_rate = load_wav_file(args.wav_path)
audio_duration = len(audio_data) / sample_rate

for model in models:
    path, arch, model_name, whisper_model, whisper_model_size = model

    for which_pass in ["moonshine", "whisper"]:
        if which_pass == "moonshine":
            transcriber = Transcriber(path, arch, options=options)
        else:
            transcriber = Transcriber(path, arch, options={"skip_transcription": True})

            class WhisperListener(TranscriptEventListener):
                def __init__(self):
                    self.total_latency_ms = 0.0
                    self.total_processing_duration = 0

                def on_line_completed(self, event):
                    audio_data = event.line.audio_data
                    start_time = time.time()
                    segments, info = whisper_model.transcribe(
                        audio=np.array(audio_data)
                    )
                    whisper_text = " ".join([segment.text for segment in segments])
                    end_time = time.time()
                    processing_duration = end_time - start_time
                    self.total_latency_ms += processing_duration * 1000.0
                    self.total_processing_duration += processing_duration
                    if args.verbose:
                        print(
                            f"Whisper: {whisper_text} ({(processing_duration / audio_duration) * 100:.2f}% of audio duration)"
                        )

            listener = WhisperListener()
            transcriber.add_listener(listener)

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

        total_latency_ms = 0
        for line in transcript.lines:
            total_latency_ms += line.last_transcription_latency_ms
            if args.verbose and which_pass == "moonshine":
                print(
                    f"Line: {line.text}, Latency: {line.last_transcription_latency_ms:.0f}ms"
                )

        if which_pass == "moonshine":
            compute_load_ratio = duration / audio_duration
            average_latency_ms = total_latency_ms / len(transcript.lines)
            print(
                f"Moonshine {model_name}: latency={average_latency_ms:.0f}ms, compute load={compute_load_ratio:.2%}"
            )
        else:
            compute_load_ratio = listener.total_processing_duration / audio_duration
            average_latency_ms = listener.total_latency_ms / len(transcript.lines)
            print(
                f"Whisper {whisper_model_size}: latency={average_latency_ms:.0f}ms, compute load={compute_load_ratio:.2%}"
            )

        transcriber.close()
