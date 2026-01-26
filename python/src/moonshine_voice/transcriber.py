"""Transcriber module for Moonshine Voice."""

import ctypes
from abc import ABC
from dataclasses import dataclass
import os
from typing import Callable, List, Optional

from moonshine_voice.moonshine_api import (
    _MoonshineLib,
    ModelArch,
    Transcript,
    TranscriptLine,
    TranscriptC,
    TranscriberOptionC,
)
from moonshine_voice.errors import MoonshineError, check_error
from moonshine_voice.utils import get_model_path, get_assets_path, load_wav_file


# Event classes
@dataclass
class TranscriptEvent:
    """Base class for transcript events."""

    line: TranscriptLine
    stream_handle: int


@dataclass
class LineStarted(TranscriptEvent):
    """Event emitted when a new transcription line starts."""

    pass


@dataclass
class LineUpdated(TranscriptEvent):
    """Event emitted when an existing transcription line is updated."""

    pass


@dataclass
class LineTextChanged(TranscriptEvent):
    """Event emitted when the text of a transcription line changes."""

    pass


@dataclass
class LineCompleted(TranscriptEvent):
    """Event emitted when a transcription line is completed."""

    pass


@dataclass
class Error:
    """Event emitted when an error occurs."""

    error: Exception
    stream_handle: int
    line: Optional[TranscriptLine] = None


class Transcriber:
    """Main transcriber class for Moonshine Voice."""

    MOONSHINE_HEADER_VERSION = 20000
    MOONSHINE_FLAG_FORCE_UPDATE = 1 << 0

    def __init__(
        self,
        model_path: str,
        model_arch: ModelArch = ModelArch.BASE,
        update_interval: float = 0.5,
        options: dict = None,
    ):
        """
        Initialize a transcriber.

        Args:
            model_path: Path to the directory containing model files
            model_arch: Model architecture to use
        """
        self._lib_wrapper = _MoonshineLib()
        self._lib = self._lib_wrapper.lib
        self._handle = None
        self._model_path = model_path
        self._model_arch = model_arch
        self._update_interval = update_interval if update_interval is not None else 0.5
        self._default_stream = None

        # Load the transcriber
        model_path_bytes = model_path.encode("utf-8")
        if options is not None:
            options_count = len(options)
            # Create a ctypes array from the options list
            options_array = (TranscriberOptionC * options_count)(
                *[
                    TranscriberOptionC(
                        name=name.encode("utf-8"), value=str(value).encode("utf-8")
                    )
                    for name, value in options.items()
                ]
            )
        else:
            options_array = None
            options_count = 0
        handle = self._lib.moonshine_load_transcriber_from_files(
            model_path_bytes,
            model_arch.value,
            options_array,
            options_count,
            self.MOONSHINE_HEADER_VERSION,
        )

        if handle < 0:
            error_str = self._lib.moonshine_error_to_string(handle)
            raise MoonshineError(
                f"Failed to load transcriber: {error_str.decode('utf-8') if error_str else 'Unknown error'}"
            )

        self._handle = handle

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Free the transcriber resources."""
        if self._handle is not None:
            self._lib.moonshine_free_transcriber(self._handle)
            self._handle = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def transcribe_without_streaming(
        self,
        audio_data: List[float],
        sample_rate: int = 16000,
        flags: int = 0,
    ) -> Transcript:
        """
        Transcribe audio data without streaming.

        Args:
            audio_data: List of audio samples (PCM float, -1.0 to 1.0)
            sample_rate: Sample rate in Hz (default: 16000)
            flags: Flags for transcription (default: 0)

        Returns:
            Transcript object containing the transcription lines
        """
        if self._handle is None:
            raise MoonshineError("Transcriber is not initialized")

        # Convert audio data to ctypes array
        audio_array = (ctypes.c_float * len(audio_data))(*audio_data)
        audio_length = len(audio_data)

        # Prepare output transcript pointer
        out_transcript = ctypes.POINTER(TranscriptC)()

        error = self._lib.moonshine_transcribe_without_streaming(
            self._handle,
            audio_array,
            audio_length,
            sample_rate,
            flags,
            ctypes.byref(out_transcript),
        )

        check_error(error)

        # Parse the transcript structure
        return self._parse_transcript(out_transcript)

    def _parse_transcript(
        self, transcript_ptr: ctypes.POINTER(TranscriptC)
    ) -> Transcript:
        """
        Parse the C transcript structure into a Python Transcript object.
        """
        if not transcript_ptr:
            return Transcript(lines=[])

        transcript = transcript_ptr.contents
        lines = []

        for i in range(transcript.line_count):
            line_c = transcript.lines[i]

            # Extract text
            text = ""
            if line_c.text:
                text = ctypes.string_at(line_c.text).decode("utf-8", errors="ignore")

            # Extract audio data if available
            audio_data = None
            if line_c.audio_data and line_c.audio_data_count > 0:
                audio_array = ctypes.cast(
                    line_c.audio_data,
                    ctypes.POINTER(ctypes.c_float * line_c.audio_data_count),
                ).contents
                audio_data = list(audio_array)

            line = TranscriptLine(
                text=text,
                start_time=line_c.start_time,
                duration=line_c.duration,
                line_id=line_c.id,
                is_complete=bool(line_c.is_complete),
                is_updated=bool(line_c.is_updated),
                is_new=bool(line_c.is_new),
                has_text_changed=bool(line_c.has_text_changed),
                audio_data=audio_data,
                last_transcription_latency_ms=line_c.last_transcription_latency_ms,
            )
            lines.append(line)

        return Transcript(lines=lines)

    def get_version(self) -> int:
        """Get the version of the loaded Moonshine library."""
        return self._lib.moonshine_get_version()

    def create_stream(self, update_interval: float = None, flags: int = 0) -> "Stream":
        """
        Create a new stream for real-time transcription.

        Args:
            flags: Flags for stream creation (default: 0)
            update_interval: Interval in seconds between updates (default: 0.5)

        Returns:
            Stream object for real-time transcription
        """
        if update_interval is None:
            update_interval = self._update_interval
        return Stream(self, update_interval, flags)

    def get_default_stream(self) -> "Stream":
        """Get the default stream."""
        if self._default_stream is None:
            self._default_stream = self.create_stream()
        return self._default_stream

    def start(self):
        """Start the default stream."""
        self.get_default_stream().start()

    def stop(self):
        """Stop the default stream."""
        return self.get_default_stream().stop()

    def add_audio(self, audio_data: List[float], sample_rate: int = 16000):
        """Add audio data to the default stream."""
        self.get_default_stream().add_audio(audio_data, sample_rate)

    def update_transcription(self, flags: int = 0) -> Transcript:
        """Update the transcription from the default stream."""
        return self.get_default_stream().update_transcription(flags)

    def add_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        """Add a listener to the default stream."""
        self.get_default_stream().add_listener(listener)

    def remove_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        """Remove a listener from the default stream."""
        self.get_default_stream().remove_listener(listener)

    def remove_all_listeners(self) -> None:
        """Remove all listeners from the default stream."""
        self.get_default_stream().remove_all_listeners()


# Event listener interface
class TranscriptEventListener(ABC):
    """
    Abstract base class for transcript event listeners.

    Subclass this and override the methods you want to handle.
    All methods have default no-op implementations, so you only need
    to override the ones you care about.
    """

    def on_line_started(self, event: LineStarted) -> None:
        """Called when a new transcription line starts."""
        pass

    def on_line_updated(self, event: LineUpdated) -> None:
        """Called when an existing transcription line is updated."""
        pass

    def on_line_text_changed(self, event: LineTextChanged) -> None:
        """Called when the text of a transcription line changes."""
        pass

    def on_line_completed(self, event: LineCompleted) -> None:
        """Called when a transcription line is completed."""
        pass

    def on_error(self, event: Error) -> None:
        """Called when an error occurs."""
        pass


# Streaming functionality
class Stream:
    """Stream for real-time transcription."""

    def __init__(
        self, transcriber: Transcriber, update_interval: float = 0.5, flags: int = 0
    ):
        """Initialize a stream."""
        self._transcriber = transcriber
        self._lib = transcriber._lib
        self._handle = None
        self._listeners: List[Callable[[TranscriptEvent], None]] = []
        self._update_interval = update_interval
        self._stream_time = 0.0
        self._last_update_time = 0.0
        handle = self._lib.moonshine_create_stream(transcriber._handle, flags)
        check_error(handle)
        self._handle = handle

    def start(self):
        """Start the stream."""
        error = self._lib.moonshine_start_stream(
            self._transcriber._handle, self._handle
        )
        check_error(error)

    def stop(self):
        """Stop the stream."""
        error = self._lib.moonshine_stop_stream(self._transcriber._handle, self._handle)
        check_error(error)
        # There may be some audio left in the stream, so we need to transcribe it to
        # get the final transcript and emit events.
        try:
            # transcribe() already calls _notify_from_transcript(), so we just call it
            result = self.update_transcription(0)
            return result
        except Exception as e:
            self._emit_error(e)

    def add_audio(self, audio_data: List[float], sample_rate: int = 16000):
        """Add audio data to the stream."""
        audio_array = (ctypes.c_float * len(audio_data))(*audio_data)
        error = self._lib.moonshine_transcribe_add_audio_to_stream(
            self._transcriber._handle,
            self._handle,
            audio_array,
            len(audio_data),
            sample_rate,
            0,
        )
        check_error(error)
        self._stream_time += len(audio_data) / sample_rate
        if self._stream_time - self._last_update_time >= self._update_interval:
            self.update_transcription(0)
            self._last_update_time = self._stream_time

    def update_transcription(self, flags: int = 0) -> Transcript:
        """Update the transcription from the stream."""
        out_transcript = ctypes.POINTER(TranscriptC)()
        error = self._lib.moonshine_transcribe_stream(
            self._transcriber._handle, self._handle, flags, ctypes.byref(out_transcript)
        )
        check_error(error)
        transcript = self._transcriber._parse_transcript(out_transcript)
        self._notify_from_transcript(transcript)
        return transcript

    def add_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        """
        Add an event listener to the stream.

        Args:
            listener: A callable that takes a TranscriptEvent and returns None.
                      Can be a function, lambda, or TranscriptEventListener instance.
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        """
        Remove an event listener from the stream.

        Args:
            listener: The listener to remove.
        """
        if listener in self._listeners:
            self._listeners.remove(listener)

    def remove_all_listeners(self) -> None:
        """Remove all event listeners from the stream."""
        self._listeners.clear()

    def _notify_from_transcript(self, transcript: Transcript) -> None:
        """Emit events based on transcript line properties."""
        for line in transcript.lines:
            if line.is_new:
                self._emit(LineStarted(line=line, stream_handle=self._handle))
            if line.is_updated and not line.is_new and not line.is_complete:
                self._emit(LineUpdated(line=line, stream_handle=self._handle))
            if line.has_text_changed:
                self._emit(LineTextChanged(line=line, stream_handle=self._handle))
            if line.is_complete and line.is_updated:
                self._emit(LineCompleted(line=line, stream_handle=self._handle))

    def _emit(self, event: TranscriptEvent) -> None:
        """Emit an event to all registered listeners."""
        for listener in self._listeners:
            try:
                # If it's a TranscriptEventListener instance, call the appropriate method
                if isinstance(listener, TranscriptEventListener):
                    if isinstance(event, LineStarted):
                        listener.on_line_started(event)
                    elif isinstance(event, LineUpdated):
                        listener.on_line_updated(event)
                    elif isinstance(event, LineTextChanged):
                        listener.on_line_text_changed(event)
                    elif isinstance(event, LineCompleted):
                        listener.on_line_completed(event)
                    elif isinstance(event, Error):
                        listener.on_error(event)
                else:
                    # Otherwise, treat it as a callable that takes the event
                    listener(event)
            except Exception as e:
                # Don't let listener errors break the stream
                # Emit an error event if possible, but don't recurse
                try:
                    error_event = Error(line=None, stream_handle=self._handle, error=e)
                    # Only emit to other listeners to avoid recursion
                    for other_listener in self._listeners:
                        if other_listener != listener:
                            try:
                                if isinstance(other_listener, TranscriptEventListener):
                                    other_listener.on_error(error_event)
                                else:
                                    other_listener(error_event)
                            except Exception:
                                pass  # Ignore errors in error handlers
                except Exception:
                    pass  # If we can't even emit the error, just continue

    def _emit_error(self, error: Exception) -> None:
        """Emit an error event."""
        error_event = Error(line=None, stream_handle=self._handle, error=error)
        self._emit(error_event)

    def close(self):
        """Free the stream resources."""
        if self._handle is not None:
            self._lib.moonshine_free_stream(self._transcriber._handle, self._handle)
            self._handle = None
        self.remove_all_listeners()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    import argparse
    from moonshine_voice import get_model_for_language

    parser = argparse.ArgumentParser(description="Model info example")
    parser.add_argument(
        "--language", type=str, default=None, help="Language to use for transcription"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the model directory (overrides language if set)",
    )
    parser.add_argument(
        "--model-arch",
        type=int,
        default=None,
        help="Model architecture to use for transcription",
    )
    parser.add_argument(
        "--wav-path", type=str, default=None, help="Path to the WAV file to transcribe"
    )
    parser.add_argument(
        "--options",
        type=str,
        default=None,
        help="Options to pass to the transcriber. Should be in the format of key=value,key2=value2,...",
    )
    args = parser.parse_args()

    if args.model_path is not None:
        model_path = args.model_path
        model_arch = ModelArch(args.model_arch)
    elif args.language is None:
        model_path = str(get_model_path("tiny-en"))
        model_arch = ModelArch.TINY
    else:
        model_path, model_arch = get_model_for_language(
            wanted_language=args.language, wanted_model_arch=args.model_arch
        )

    if args.wav_path is None:
        wav_path = os.path.join(get_assets_path(), "two_cities.wav")
    else:
        wav_path = args.wav_path

    options = {}
    if args.options is not None:
        for option in args.options.split(","):
            key, value = option.split("=")
            options[key] = value

    transcriber = Transcriber(
        model_path=model_path, model_arch=model_arch, options=options
    )

    transcriber.start()

    class TestListener(TranscriptEventListener):
        def on_line_started(self, event):
            print(f"Line started: {event.line.text}")

        def on_line_text_changed(self, event):
            print(f"Line text changed: {event.line.text}")

        def on_line_completed(self, event):
            print(f"Line completed: {event.line.text}")

    listener = TestListener()
    transcriber.add_listener(listener)

    audio_data, sample_rate = load_wav_file(wav_path)

    # Loop through the audio data in chunks to simulate live streaming
    # from a microphone or other source.
    chunk_duration = 0.1
    chunk_size = int(chunk_duration * sample_rate)
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        transcriber.add_audio(chunk, sample_rate)

    transcriber.stop()
    transcriber.close()
