"""Basic usage example for Moonshine Voice."""

import os

from moonshine_voice import Transcriber, ModelArch, get_model_path, get_assets_path, load_wav_file

# Example: Transcribe audio from a file
def transcribe_audio_file(model_path: str, model_arch: ModelArch, audio_file: str):
    """Transcribe audio from a file."""
    # Initialize transcriber
    transcriber = Transcriber(
        model_path=model_path,
        model_arch=model_arch
    )

    try:
        # Load audio file using the built-in WAV loader
        audio_data, sample_rate = load_wav_file(audio_file)
        print(f"Loaded {len(audio_data)} samples at {sample_rate} Hz")
        
        # Transcribe
        transcript = transcriber.transcribe(audio_data, sample_rate=sample_rate)
        
        # Print results
        for line in transcript.lines:
            print(f"[{line.start_time:.2f}s - {line.start_time + line.duration:.2f}s] {line.text}")
    finally:
        transcriber.close()


# Example: Streaming transcription
def streaming_transcription_example(model_path: str, model_arch: ModelArch):
    """Example of streaming transcription."""
    transcriber = Transcriber(
        model_path=model_path,
        model_arch=model_arch
    )

    try:
        # Create a stream
        stream = transcriber.create_stream()
        stream.start()

        try:
            # In a real application, you would capture audio chunks here
            # while True:
            #     audio_chunk = capture_audio_chunk()  # Your audio capture function
            #     stream.add_audio(audio_chunk, sample_rate=16000)
            #     
            #     # Get updated transcript periodically
            #     transcript = stream.transcribe()
            #     for line in transcript.lines:
            #         if line.is_updated:
            #             print(f"Updated: {line.text}")
            
            print("Streaming transcription example (requires audio capture)")
        finally:
            stream.stop()
            stream.close()
    finally:
        transcriber.close()


if __name__ == "__main__":
    model_path = str(get_model_path("tiny-en"))
    beckett_wav_path = os.path.join(get_assets_path(), "beckett.wav")
    transcribe_audio_file(model_path, ModelArch.TINY, beckett_wav_path)
    streaming_transcription_example(model_path, ModelArch.TINY)

