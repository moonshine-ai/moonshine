import argparse
import sys
import time

from moonshine_voice import (
    DialogFlow,
    MicTranscriber,
    TranscriptEventListener,
    get_model_for_language,
)

parser = argparse.ArgumentParser(
    description="Control a robot from your Raspberry Pi using voice commands"
)
parser.add_argument(
    "--model-arch",
    type=int,
    default=None,
    help="Model architecture to use for transcription",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.7,
    help="Similarity threshold for command matching (default: 0.7)",
)
args = parser.parse_args()

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

def on_move_forward(d):
    print("Moving forward")
def on_move_backward(d):
    print("Moving backward")
def on_turn_left(d):
    print("Turning left")
def on_turn_right(d):
    print("Turning right")
def on_exterminate(d):
    print("EXTERMINATE!")

# DialogFlow matches what it hears against these phrases semantically, using an
# embedding model it downloads and loads the first time a command comes in.
# These are "globals": single-shot commands that stay live at all times, as
# opposed to the multi-turn conversations you'd register with register_flow.
commands = {
    "move forward": on_move_forward,
    "move backward": on_move_backward,
    "turn left": on_turn_left,
    "turn right": on_turn_right,
    "kill all humans": on_exterminate,
    "exterminate": on_exterminate,
}
# The beeps are the runner's audio cues for matched / unmatched speech; this
# example reports what it heard in text instead, so silence them.
dalek = DialogFlow(
    trigger_threshold=args.threshold,
    success_beep_fn=lambda: None,
    error_beep_fn=lambda: None,
)
for phrase, handler in commands.items():
    dalek.register_global(phrase, handler)

transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch)

# Add both the transcript printer and the command runner as listeners. The
# runner processes completed lines and calls the matching handler.
transcript_printer = TranscriptPrinter()
transcriber.add_listener(transcript_printer)
transcriber.add_listener(dalek)

print("\n" + "=" * 60, file=sys.stderr)
print("🎤 Listening for voice commands...", file=sys.stderr)
print("Try saying phrases with the same meaning as these actions:", file=sys.stderr)
for phrase in commands.keys():
    print(f"  - '{phrase}'", file=sys.stderr)
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
    transcriber.stop()
    transcriber.close()
    dalek.close()
