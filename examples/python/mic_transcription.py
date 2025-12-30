"""Uses the MicTranscriber class to transcribe audio from a microphone."""

import sys
import time

from moonshine_voice import (
    MicTranscriber,
    ModelArch,
    get_model_path,
    TranscriptEventListener,
)

# If we're on an interactive terminal, show the transcription in real time
# as it's being generated.
class TerminalListener(TranscriptEventListener):
    def __init__(self):
        self.last_line_text_length = 0

    # Assume we're on a terminal, and so we can use a carriage return to 
    # overwrite the last line with the latest text.
    def update_last_terminal_line(self, new_text: str):
        print(f"\r{new_text}", end="", flush=True)
        if len(new_text) < self.last_line_text_length:
            # If the new text is shorter than the last line, we need to 
            # overwrite the last line with spaces.
            diff = self.last_line_text_length - len(new_text)
            print(f"{' ' * diff}", end="", flush=True)
        # Update the length of the last line text.
        self.last_line_text_length = len(new_text)

    def on_line_started(self, event):
        self.last_line_text_length = 0

    def on_line_text_changed(self, event):
        self.update_last_terminal_line(event.line.text)

    def on_line_completed(self, event):
        self.update_last_terminal_line(event.line.text)
        print("\n", end="", flush=True)

# If we're not on an interactive terminal, print each line as it's completed.
class FileListener(TranscriptEventListener):
    def on_line_completed(self, event):
        print(event.line.text)

if __name__ == "__main__":
    model_path = str(get_model_path("tiny-en"))
    model_arch = ModelArch.TINY

    mic_transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch)

    if sys.stdout.isatty():
        listener = TerminalListener()
    else:
        listener = FileListener()

    print("Listening to the microphone, press Ctrl+C to stop...", file=sys.stderr)        
    mic_transcriber.add_listener(listener)
    mic_transcriber.start()
    try:
        while True:
            time.sleep(0.1)
    finally:
        mic_transcriber.stop()
        mic_transcriber.close()
