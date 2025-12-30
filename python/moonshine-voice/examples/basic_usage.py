"""Basic usage example for Moonshine Voice."""

import os

from moonshine_voice import (
    Transcriber,
    ModelArch,
    get_model_path,
    get_assets_path,
    load_wav_file,
    TranscriptEventListener,
)


# Example: Transcribe audio from a file
def transcribe_audio_file(model_path: str, model_arch: ModelArch, audio_file: str):
    """Transcribe audio from a file."""
    # Initialize transcriber
    transcriber = Transcriber(model_path=model_path, model_arch=model_arch)

    try:
        # Load audio file using the built-in WAV loader
        audio_data, sample_rate = load_wav_file(audio_file)
        print(f"Loaded {len(audio_data)} samples at {sample_rate} Hz")

        # Transcribe
        transcript = transcriber.transcribe_without_streaming(
            audio_data, sample_rate=sample_rate
        )

        # Print results
        for line in transcript.lines:
            print(
                f"[{line.start_time:.2f}s - {line.start_time + line.duration:.2f}s] {line.text}"
            )
    finally:
        transcriber.close()


# Example: Streaming transcription
def streaming_transcription_example(
    model_path: str, model_arch: ModelArch, audio_file: str
):
    """Example of streaming transcription."""
    transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
    # Create a stream. The update interval is the interval in seconds between full transcriptions being generated.
    stream = transcriber.create_stream(update_interval=0.5)
    stream.start()

    class TestListener(TranscriptEventListener):
        def on_line_started(self, event):
            print(f"Line started: {event.line.text}")

        def on_line_text_changed(self, event):
            print(f"Line text changed: {event.line.text}")

        def on_line_completed(self, event):
            print(f"Line completed: {event.line.text}")

    listener = TestListener()
    stream.add_listener(listener)

    audio_data, sample_rate = load_wav_file(audio_file)
    chunk_duration = 0.1
    chunk_size = int(chunk_duration * sample_rate)
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        stream.add_audio(chunk, sample_rate)

    stream.stop()
    stream.close()

    transcriber.close()


if __name__ == "__main__":
    model_path = str(get_model_path("tiny-en"))
    beckett_wav_path = os.path.join(get_assets_path(), "beckett.wav")
    transcribe_audio_file(model_path, ModelArch.TINY, beckett_wav_path)
    
    two_cities_wav_path = os.path.join(get_assets_path(), "two_cities.wav")
    streaming_transcription_example(model_path, ModelArch.TINY, two_cities_wav_path)
