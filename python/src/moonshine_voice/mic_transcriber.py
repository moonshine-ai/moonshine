from moonshine_voice.transcriber import (
    Transcriber,
    TranscriptEvent,
    TranscriptEventListener,
    TranscriptLine,
    ModelArch,
)
from moonshine_voice.utils import get_model_path

import numpy as np
import sounddevice as sd
import time
from typing import Callable


class MicTranscriber:
    """MicTranscriber is a class that transcribes audio from a microphone."""

    def __init__(
        self,
        model_path: str,
        model_arch: ModelArch = ModelArch.TINY,
        update_interval: float = 0.5,
        device: int = None,
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 1024,
    ):
        self.transcriber = Transcriber(model_path, model_arch)
        self.mic_stream = self.transcriber.create_stream(update_interval)
        self._should_listen = False
        self._sd_stream = None
        self._device = device
        self._samplerate = samplerate
        self._channels = channels
        self._blocksize = blocksize

    def _start_listening(self):
        """
        Start listening to the microphone (or specified audio device).
        Incoming audio blocks are automatically fed to self.mic_stream.add_audio().
        """

        def audio_callback(in_data, frames, time, status):
            if not self._should_listen:
                return
            if status:
                print(f"MicTranscriber: {status}")
            if in_data is not None:
                # Flatten and convert to float32 if needed
                audio_data = in_data.astype(np.float32).flatten()
                # Call add_audio on the stream
                self.mic_stream.add_audio(audio_data, self._samplerate)

        self._sd_stream = sd.InputStream(
            samplerate=self._samplerate,
            blocksize=self._blocksize,
            device=self._device,
            channels=self._channels,
            dtype="float32",
            callback=audio_callback,
        )
        self._sd_stream.start()

    def start(self):
        self.mic_stream.start()
        if self._sd_stream is None:
            self._start_listening()
        self._should_listen = True

    def stop(self):
        self._should_listen = False
        self.mic_stream.stop()

    def close(self):
        self.mic_stream.close()
        self.transcriber.close()

    def add_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        self.mic_stream.add_listener(listener)

    def remove_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        self.mic_stream.remove_listener(listener)

    def remove_all_listeners(self):
        self.mic_stream.remove_all_listeners()


if __name__ == "__main__":
    import argparse
    import sys
    from moonshine_voice import get_model_for_language

    parser = argparse.ArgumentParser(description="MicTranscriber example")
    parser.add_argument(
        "--language", type=str, default="en", help="Language to use for transcription"
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default="4",
        help="Model architecture to use for transcription",
    )
    args = parser.parse_args()
    model_path, model_arch = get_model_for_language(
        wanted_language=args.language, wanted_model_arch=int(args.model_arch)
    )

    mic_transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch)

    class TerminalListener(TranscriptEventListener):
        def __init__(self):
            self.last_line_text_length = 0

        # Assume we're on a terminal, and so we can use a carriage return to
        # overwrite the last line with the latest text.
        def update_last_terminal_line(self, line: TranscriptLine):
            if line.has_speaker_id:
                speaker_prefix = f"Speaker #{line.speaker_index}: "
            else:
                speaker_prefix = ""
            new_text = f"{speaker_prefix}{line.text}"
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
            self.update_last_terminal_line(event.line)

        def on_line_completed(self, event):
            self.update_last_terminal_line(event.line)
            print("\n", end="", flush=True)

    # If we're not on an interactive terminal, print each line as it's completed.
    class FileListener(TranscriptEventListener):
        def on_line_completed(self, event):
            print(event.line.text)

    if sys.stdout.isatty():
        listener = TerminalListener()
    else:
        listener = FileListener()

    mic_transcriber.add_listener(listener)

    print(f"Listening to the microphone, press Ctrl+C to stop...", file=sys.stderr)
    mic_transcriber.start()
    try:
        while True:
            time.sleep(0.1)
    finally:
        mic_transcriber.stop()
        mic_transcriber.close()
