"""Basic usage example for Moonshine Voice."""

import argparse
import os
from typing import List

from moonshine_voice import (
    Transcriber,
    get_assets_path,
    load_wav_file,
    TranscriptEventListener,
    get_model_for_language,
)


def transcribe_without_streaming(
    transcriber: Transcriber, audio_data: List[float], sample_rate: int
):
    """Transcribe audio data offline without streaming."""
    transcript = transcriber.transcribe_without_streaming(
        audio_data, sample_rate=sample_rate, flags=0
    )
    for line in transcript.lines:
        print(
            f"Transcript: [{line.start_time:.2f}s - {line.start_time + line.duration:.2f}s] {line.text}"
        )


# Example: Streaming transcription
def transcribe_with_streaming(
    transcriber: Transcriber, audio_data: List[float], sample_rate: int
):
    """Example of streaming transcription."""

    transcriber.start()

    class TestListener(TranscriptEventListener):
        def on_line_started(self, event):
            print(f"{event.line.start_time:.2f}s: Line started: {event.line.text}")

        def on_line_text_changed(self, event):
            print(f"{event.line.start_time:.2f}s: Line text changed: {event.line.text}")

        def on_line_completed(self, event):
            print(f"{event.line.start_time:.2f}s: Line completed: {event.line.text}")

    listener = TestListener()
    transcriber.remove_all_listeners()
    transcriber.add_listener(listener)

    chunk_duration = 0.1
    chunk_size = int(chunk_duration * sample_rate)
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        transcriber.add_audio(chunk, sample_rate)

    transcriber.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic transcription example")
    parser.add_argument(
        "--language", type=str, default="en", help="Language to use for transcription"
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default=None,
        help="Model architecture to use for transcription",
    )
    # or nargs='+' to require at least one
    parser.add_argument("input_files", nargs="*")

    args = parser.parse_args()
    if len(args.input_files) == 0:
        input_files = [os.path.join(get_assets_path(), "two_cities.wav")]
    else:
        input_files = args.input_files

    model_path, model_arch = get_model_for_language(args.language, args.model_arch)
    transcriber = Transcriber(model_path=model_path, model_arch=model_arch)

    for input_file in input_files:
        audio_data, sample_rate = load_wav_file(input_file)
        print("*" * 80)
        print(f"Transcribing {input_file} offline without streaming...")
        transcribe_without_streaming(transcriber, audio_data, sample_rate)
        print("*" * 80)
        print(f"Transcribing {input_file} with streaming...")
        transcribe_with_streaming(transcriber, audio_data, sample_rate)
