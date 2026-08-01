import argparse
import sys
import time

from moonshine_voice import AgentFlow

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

# AgentFlow matches what it hears against these phrases semantically, using an
# embedding model it downloads and loads for us. These are "globals": one-shot
# commands that stay live at all times, as opposed to the multi-turn
# conversations you'd register with listen_for.
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
dalek = (
    AgentFlow()
    .language("en")
    .trigger_threshold(args.threshold)
    .speech(False)
    .beeps(False)
    .on_heard(lambda text: print(text))
    .on_progress(
        lambda fraction, name: print(
            f"Loading {name}... {fraction:.0%}", file=sys.stderr
        )
    )
)
if args.model_arch is not None:
    dalek.model_arch(args.model_arch)
for phrase, handler in commands.items():
    dalek.always(phrase, handler)

# Downloads and opens the transcription and phrase-matching models.
dalek.load()

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

dalek.start_listening()
try:
    # Loop forever, listening for voice commands.
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\nStopping...", file=sys.stderr)
finally:
    dalek.close()
