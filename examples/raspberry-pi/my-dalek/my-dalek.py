import argparse
import sys
import time

from moonshine_voice import (
    MicTranscriber,
    Transcriber,
    load_wav_file,
    TranscriptEventListener,
    IntentRecognizer,
    get_model_for_language,
    get_embedding_model,
)

parser = argparse.ArgumentParser(
    description="Control a robot from your Raspberry Pi using voice commands"
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
    default="embeddinggemma-300m",
    help="Embedding model name (default: embeddinggemma-300m)",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.6,
    help="Similarity threshold for intent matching (default: 0.6)",
)
args = parser.parse_args()

def on_intent_triggered_on(trigger: str, utterance: str, similarity: float):
    """Handler for when an intent is triggered."""
    print(
        f"\n'{trigger.upper()}' triggered by '{utterance}' with {similarity:.0%} confidence"
    )

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
        self.update_last_terminal_line(f"{event.line.text}")

    def on_line_completed(self, event):
        self.update_last_terminal_line(f"{event.line.text}")
        print()  # New line after completion

# Load the transcription model
print("Loading transcription model...", file=sys.stderr)
model_path, model_arch = get_model_for_language("en", args.model_arch)

# Download and load the embedding model for intent recognition
quantization = "q4"
print(
    f"Loading embedding model ({args.embedding_model}, variant={quantization})...",
    file=sys.stderr,
)
embedding_model_path, embedding_model_arch = get_embedding_model(
    args.embedding_model, quantization
)

# Create the intent recognizer (implements TranscriptEventListener)
print(
    f"Creating intent recognizer (threshold={args.threshold})...", file=sys.stderr
)
intent_recognizer = IntentRecognizer(
    model_path=embedding_model_path,
    model_arch=embedding_model_arch,
    model_variant=quantization,
    threshold=args.threshold,
)

def on_move_forward(trigger: str, utterance: str, similarity: float):
    print(f"Moving forward with {similarity:.0%} confidence")
def on_move_backward(trigger: str, utterance: str, similarity: float):
    print(f"Moving backward with {similarity:.0%} confidence")
def on_turn_left(trigger: str, utterance: str, similarity: float):
    print(f"Turning left with {similarity:.0%} confidence")
def on_turn_right(trigger: str, utterance: str, similarity: float):
    print(f"Turning right with {similarity:.0%} confidence")
def on_exterminate(trigger: str, utterance: str, similarity: float):
    print(f"EXTERMINATE! with {similarity:.0%} confidence")

# Register intents with their trigger phrases and handlers
intents = {
    "move forward": on_move_forward,
    "move backward": on_move_backward,
    "turn left": on_turn_left,
    "turn right": on_turn_right,
    "kill all humans": on_exterminate,
    "exterminate": on_exterminate,
}
for intent, handler in intents.items():
    intent_recognizer.register_intent(intent, handler)

print(f"Registered {intent_recognizer.intent_count} intents", file=sys.stderr)

transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch)

# Add both the transcript printer and intent recognizer as listeners
# The intent recognizer will process completed lines and trigger handlers
transcript_printer = TranscriptPrinter()
transcriber.add_listener(transcript_printer)
transcriber.add_listener(intent_recognizer)

print("\n" + "=" * 60, file=sys.stderr)
print("🎤 Listening for voice commands...", file=sys.stderr)
print("Try saying phrases with the same meaning as these actions:", file=sys.stderr)
for intent in intents.keys():
    print(f"  - '{intent}'", file=sys.stderr)
print(
    "We're doing fuzzy matching of natural language, so phrases like 'Go forward' or 'Move ahead' or 'Advance' will trigger the 'move forward' action, for example."
)
print("=" * 60, file=sys.stderr)
print("Press Ctrl+C to stop.\n", file=sys.stderr)

transcriber.start()
try:
    # Loop forever, listening for voice commands.
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\nStopping...", file=sys.stderr)
finally:
    intent_recognizer.close()
    transcriber.stop()
    transcriber.close()
