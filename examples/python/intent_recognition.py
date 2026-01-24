"""Intent recognition example using Moonshine Voice.

This example demonstrates how to use the IntentRecognizer to recognize
intents from transcribed speech. The IntentRecognizer acts as a 
TranscriptEventListener, automatically processing completed transcript
lines to detect registered intents.

Requirements:
    - Moonshine Voice library built with intent recognition support
    - Gemma embedding model downloaded to embeddinggemma-300m-ONNX/
    - A microphone for live transcription (or use with audio files)
"""

import argparse
import sys
import time

from moonshine_voice import (
    MicTranscriber,
    IntentRecognizer,
    EmbeddingModelArch,
    TranscriptEventListener,
    get_model_for_language,
)


# Define intent handlers
def on_lights_on(trigger: str, utterance: str, similarity: float):
    """Handler for turning lights on."""
    print(f"\n💡 LIGHTS ON! (matched '{trigger}' with {similarity:.0%} confidence)")


def on_lights_off(trigger: str, utterance: str, similarity: float):
    """Handler for turning lights off."""
    print(f"\n🌑 LIGHTS OFF! (matched '{trigger}' with {similarity:.0%} confidence)")


def on_weather(trigger: str, utterance: str, similarity: float):
    """Handler for weather queries."""
    print(f"\n🌤️  WEATHER: It's sunny and 72°F! (matched '{trigger}' with {similarity:.0%} confidence)")


def on_timer(trigger: str, utterance: str, similarity: float):
    """Handler for timer requests."""
    print(f"\n⏰ TIMER: Setting a timer! (matched '{trigger}' with {similarity:.0%} confidence)")


def on_music_play(trigger: str, utterance: str, similarity: float):
    """Handler for playing music."""
    print(f"\n🎵 PLAYING MUSIC! (matched '{trigger}' with {similarity:.0%} confidence)")


def on_music_stop(trigger: str, utterance: str, similarity: float):
    """Handler for stopping music."""
    print(f"\n🔇 STOPPING MUSIC! (matched '{trigger}' with {similarity:.0%} confidence)")


class TranscriptPrinter(TranscriptEventListener):
    """Listener that prints transcript updates to the terminal."""

    def __init__(self):
        self.last_line_text_length = 0

    def update_last_terminal_line(self, new_text: str):
        print(f"\r{new_text}", end="", flush=True)
        if len(new_text) < self.last_line_text_length:
            diff = self.last_line_text_length - len(new_text)
            print(f"{' ' * diff}", end="", flush=True)
        self.last_line_text_length = len(new_text)

    def on_line_started(self, event):
        self.last_line_text_length = 0

    def on_line_text_changed(self, event):
        self.update_last_terminal_line(f"📝 {event.line.text}")

    def on_line_completed(self, event):
        self.update_last_terminal_line(f"📝 {event.line.text}")
        print()  # New line after completion


def main():
    parser = argparse.ArgumentParser(
        description="Intent recognition example with Moonshine Voice"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language to use for transcription (default: en)",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default=None,
        help="Model architecture to use for transcription",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma-300m-ONNX",
        help="Path to the embedding model directory (default: embeddinggemma-300m-ONNX)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for intent matching (default: 0.6)",
    )
    args = parser.parse_args()

    # Load the transcription model
    print("Loading transcription model...", file=sys.stderr)
    model_path, model_arch = get_model_for_language(args.language, args.model_arch)

    # Create the intent recognizer (implements TranscriptEventListener)
    # The embedding model is loaded internally based on the path and architecture
    print(f"Loading embedding model and creating intent recognizer...", file=sys.stderr)
    print(f"  Embedding model: {args.embedding_model}", file=sys.stderr)
    print(f"  Threshold: {args.threshold}", file=sys.stderr)
    intent_recognizer = IntentRecognizer(
        model_path=args.embedding_model,
        model_arch=EmbeddingModelArch.GEMMA_300M,
        threshold=args.threshold,
    )

    # Register intents with their trigger phrases and handlers
    intent_recognizer.register_intent("turn on the lights", on_lights_on)
    intent_recognizer.register_intent("turn off the lights", on_lights_off)
    intent_recognizer.register_intent("what is the weather", on_weather)
    intent_recognizer.register_intent("set a timer", on_timer)
    intent_recognizer.register_intent("play some music", on_music_play)
    intent_recognizer.register_intent("stop the music", on_music_stop)

    print(f"Registered {intent_recognizer.intent_count} intents", file=sys.stderr)

    # Create the microphone transcriber
    mic_transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch, device=3)

    # Add both the transcript printer and intent recognizer as listeners
    # The intent recognizer will process completed lines and trigger handlers
    transcript_printer = TranscriptPrinter()
    mic_transcriber.add_listener(intent_recognizer)
    mic_transcriber.add_listener(transcript_printer)

    print("\n" + "=" * 60, file=sys.stderr)
    print("🎤 Listening for voice commands...", file=sys.stderr)
    print("Try saying:", file=sys.stderr)
    print("  - 'Turn on the lights' or 'Switch on the lights'", file=sys.stderr)
    print("  - 'Turn off the lights' or 'Lights off'", file=sys.stderr)
    print("  - 'What's the weather like?' or 'Weather forecast'", file=sys.stderr)
    print("  - 'Set a timer' or 'Start a timer for 5 minutes'", file=sys.stderr)
    print("  - 'Play some music' or 'Play my playlist'", file=sys.stderr)
    print("  - 'Stop the music' or 'Pause playback'", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("Press Ctrl+C to stop.\n", file=sys.stderr)

    mic_transcriber.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...", file=sys.stderr)
    finally:
        intent_recognizer.close()
        mic_transcriber.stop()
        mic_transcriber.close()


if __name__ == "__main__":
    main()
