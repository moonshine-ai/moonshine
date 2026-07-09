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
import sys
import time
from typing import Callable, Optional


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
        options: dict = None,
        spelling_model_path: str = None,
        transcribe_flags: int = 0,
    ):
        # Pass-through convenience: callers that only want spelling-mode
        # don't need to construct an ``options`` dict themselves.
        if spelling_model_path is not None:
            options = dict(options) if options else {}
            options.setdefault("spelling_model_path", spelling_model_path)
        self.transcriber = Transcriber(model_path, model_arch, options=options)
        self.mic_stream = self.transcriber.create_stream(
            update_interval, transcribe_flags=transcribe_flags,
        )
        self._should_listen = False
        self._sd_stream = None
        self._device = device
        self._samplerate = samplerate
        self._channels = channels
        self._blocksize = blocksize

    def _query_device_default_samplerate(self) -> Optional[int]:
        """Return the input device's native default sample rate, or None on failure.

        Used as a fallback when the requested rate isn't natively supported by
        the capture device (common on USB mics that only do 44100/48000 Hz).
        """
        try:
            info = sd.query_devices(self._device, "input")
        except (sd.PortAudioError, OSError, ValueError) as e:
            print(f"MicTranscriber: could not query device info: {e}", file=sys.stderr)
            return None
        rate = info.get("default_samplerate") if isinstance(info, dict) else None
        try:
            rate = int(rate) if rate else None
        except (TypeError, ValueError):
            rate = None
        return rate if rate and rate > 0 else None

    def _open_input_stream(self, samplerate: int, callback) -> sd.InputStream:
        stream = sd.InputStream(
            samplerate=samplerate,
            blocksize=self._blocksize,
            device=self._device,
            channels=self._channels,
            dtype="float32",
            callback=callback,
        )
        return stream

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
                # The Moonshine C API resamples to its internal 16 kHz, so we
                # pass whatever rate the device is actually capturing at.
                self.mic_stream.add_audio(audio_data, self._samplerate)

        try:
            self._sd_stream = self._open_input_stream(self._samplerate, audio_callback)
        except sd.PortAudioError as e:
            # Most commonly PaErrorCode -9997 (Invalid sample rate) when the
            # capture device doesn't natively support our requested rate.
            # Fall back to the device's default rate; the C API will resample.
            fallback = self._query_device_default_samplerate()
            if fallback is None or fallback == self._samplerate:
                raise
            print(
                f"MicTranscriber: device does not support {self._samplerate} Hz "
                f"({e}); falling back to {fallback} Hz.",
                file=sys.stderr,
            )
            self._samplerate = fallback
            self._sd_stream = self._open_input_stream(self._samplerate, audio_callback)
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

    @property
    def transcribe_flags(self) -> int:
        """Flags currently applied to streamed ``update_transcription`` calls."""
        return self.mic_stream.transcribe_flags

    def set_transcribe_flags(self, flags: int) -> None:
        """Update the per-update flags on the underlying mic stream.

        Convenience wrapper around :meth:`Stream.set_transcribe_flags`.
        Lets DialogFlow flip ``MOONSHINE_FLAG_SPELLING_MODE`` on only
        while a ``SPELLED`` / ``DIGITS`` prompt is in progress.
        """
        self.mic_stream.set_transcribe_flags(flags)

    def add_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        self.mic_stream.add_listener(listener)

    def remove_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        self.mic_stream.remove_listener(listener)

    def remove_all_listeners(self):
        self.mic_stream.remove_all_listeners()

    def push_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        """Push a temporary listener, saving the current listeners on a stack."""
        self.mic_stream.push_listener(listener)

    def pop_listener(self) -> None:
        """Restore the listeners that were active before the last push."""
        self.mic_stream.pop_listener()

    def pop_all_listeners(self) -> None:
        """Unwind the entire listener stack, restoring the original listeners."""
        self.mic_stream.pop_all_listeners()


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
        type=int,
        default=None,
        help="Model architecture to use for transcription",
    )
    args = parser.parse_args()
    model_path, model_arch = get_model_for_language(
        wanted_language=args.language, wanted_model_arch=args.model_arch
    )

    mic_transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch)

    class TerminalListener(TranscriptEventListener):
        def __init__(self):
            self.last_line_text_length = 0

        # Assume we're on a terminal, and so we can use a carriage return to
        # overwrite the last line with the latest text.
        def update_last_terminal_line(self, line: TranscriptLine):
            if line.speaker_spans:
                # Use the speaker with the most speech time within the line.
                durations = {}
                for span in line.speaker_spans:
                    durations[span.speaker_index] = (
                        durations.get(span.speaker_index, 0.0) + span.duration
                    )
                dominant = max(durations, key=durations.get)
                speaker_prefix = f"Speaker #{dominant}: "
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
