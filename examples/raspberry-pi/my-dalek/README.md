# My Dalek

This is an example of using the [Moonshine Voice](https://github.com/moonshine-ai/moonshine) library to build a voice interface that could control a robot from your Raspberry Pi (Dalek not included).

To run it, first `cd` into this directory and install the Moonshine Voice pip package:

```bash
pip install moonshine-voice
```

If you see a warning about system packages, you can either override that with an uneccessarily scary-sounding flag:

```bash
pip install --break-system-packages moonshine-voice
```

Or use a virtual environment:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install moonshine-voice
```

If you're using `uv`, you'll need to run `source .venv/bin/activate` every time you log back in before you can run the script.

Then make sure you have a USB microphone plugged into your Pi so the script can hear you.

After that's working, run the script:

```bash
python my-dalek.py
```

You should see output like this:

```bash
Registered 6 intents

============================================================
🎤 Listening for voice commands...
Try saying phrases with the same meaning as these actions:
  - 'move forward'
  - 'move backward'
  - 'turn left'
  - 'turn right'
  - 'kill all humans'
  - 'exterminate'
We're doing fuzzy matching of natural language, so phrases like 'Go forward' or 'Move ahead' or 'Advance' will trigger the 'move forward' action, for example.
============================================================
Press Ctrl+C to stop.
```

As the instructions suggest, try saying some phrases you might use to control a robot's movement or actions, like "Go ahead" or "Murder everyone".

If you look at the code, you will see that we have a set of functions that we match to the phrases that should trigger them:

```python
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

intents = {
    "move forward": on_move_forward,
    "move backward": on_move_backward,
    "turn left": on_turn_left,
    "turn right": on_turn_right,
    "kill all humans": on_exterminate,
    "exterminate": on_exterminate,
}
```

In a real project each function would have code that controls the robot's wheels, or activates its sink-plunger death-ray, but since I've had trouble understanding Davros's documentation, I've left acquiring the hardware and implementing those as an exercise for the reader.

You can also change the phrases to whatever you need for your application, and the same kind of semantic matching (recognizing sentences that have similar meanings to the target) will work for them too. We've seen people use this for everything from controlling industrial machinery to answering frequently-asked questions, so [let us know](mailto:contact@moonshine.ai) how you get on!