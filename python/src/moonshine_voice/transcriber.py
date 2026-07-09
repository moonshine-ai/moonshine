"""Transcriber module for Moonshine Voice."""

import ctypes
from abc import ABC
from dataclasses import dataclass
import os
import sys
from typing import Callable, List, Optional
from pathlib import Path

from moonshine_voice.moonshine_api import (
    _MoonshineLib,
    ModelArch,
    SpeakerSpan,
    Transcript,
    TranscriptLine,
    TranscriptC,
    TranscriberOptionC,
    WordTiming,
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
class LineSpeakersChanged(TranscriptEvent):
    """Event emitted when the speaker spans of a transcription line change.

    Only fired when the ``identify_speakers`` option is enabled. Note that
    this can fire for lines that are already complete, since diarization
    refines speaker assignments retroactively as more audio arrives.
    """

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


# Module-level mirror of the Transcriber class attributes so callers can
# write ``from moonshine_voice.transcriber import MOONSHINE_FLAG_SPELLING_MODE``
# without poking at the class. The class attributes below stay for
# backwards-compat with existing callers that read them as
# ``Transcriber.MOONSHINE_FLAG_SPELLING_MODE``.
MOONSHINE_HEADER_VERSION = 20000
MOONSHINE_FLAG_FORCE_UPDATE = 1 << 0
# See ``MOONSHINE_FLAG_SPELLING_MODE`` in core/moonshine-c-api.h.
MOONSHINE_FLAG_SPELLING_MODE = 1 << 1


class Transcriber:
    """Main transcriber class for Moonshine Voice."""

    MOONSHINE_HEADER_VERSION = MOONSHINE_HEADER_VERSION
    MOONSHINE_FLAG_FORCE_UPDATE = MOONSHINE_FLAG_FORCE_UPDATE
    MOONSHINE_FLAG_SPELLING_MODE = MOONSHINE_FLAG_SPELLING_MODE

    def __init__(
        self,
        model_path: str | Path,
        model_arch: ModelArch = ModelArch.BASE,
        update_interval: float = 0.5,
        options: dict = None,
        spelling_model_path: Optional[str] = None,
    ):
        """
        Initialize a transcriber.

        Args:
            model_path: Path to the directory containing model files.
            model_arch: Model architecture to use.
            update_interval: Default update interval (seconds) used by the
              streaming convenience APIs.
            options: Optional dict of advanced C API options (see
              ``parse_transcriber_options`` in core/moonshine-c-api.cpp).
            spelling_model_path: Optional path to a ``spelling_cnn.ort``
              file. When provided, the transcriber will run alphanumeric
              fusion on completed lines whenever
              ``MOONSHINE_FLAG_SPELLING_MODE`` is passed to a transcribe
              call. Equivalent to setting the C option
              ``"spelling_model_path"`` directly.
        """
        self._lib_wrapper = _MoonshineLib()
        self._lib = self._lib_wrapper.lib
        self._handle = None
        self._model_path = model_path
        self._model_arch = model_arch
        self._update_interval = update_interval if update_interval is not None else 0.5
        self._default_stream = None

        merged_options = dict(options) if options else {}
        # Surface the spelling-model kwarg as a C API option string so
        # callers can also pass it through ``options=`` if they prefer.
        if spelling_model_path is not None:
            merged_options.setdefault("spelling_model_path", spelling_model_path)

        # Load the transcriber
        model_path_bytes = str(model_path).encode("utf-8")
        if merged_options:
            options_count = len(merged_options)
            # Create a ctypes array from the options list
            options_array = (TranscriberOptionC * options_count)(
                *[
                    TranscriberOptionC(
                        name=name.encode("utf-8"), value=str(value).encode("utf-8")
                    )
                    for name, value in merged_options.items()
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

            # Extract word timestamps if available
            words = None
            if line_c.words and line_c.word_count > 0:
                words = []
                for j in range(line_c.word_count):
                    word_c = line_c.words[j]
                    word_text = ""
                    if word_c.text:
                        word_text = ctypes.string_at(word_c.text).decode(
                            "utf-8", errors="ignore"
                        )
                    words.append(
                        WordTiming(
                            word=word_text,
                            start=word_c.start,
                            end=word_c.end,
                            confidence=word_c.confidence,
                        )
                    )

            # Extract speaker spans if available
            speaker_spans = None
            if line_c.speaker_spans and line_c.speaker_span_count > 0:
                speaker_spans = []
                for j in range(line_c.speaker_span_count):
                    span_c = line_c.speaker_spans[j]
                    speaker_spans.append(
                        SpeakerSpan(
                            start_time=span_c.start_time,
                            duration=span_c.duration,
                            speaker_id=span_c.speaker_id,
                            speaker_index=span_c.speaker_index,
                            start_char=span_c.start_char,
                            end_char=span_c.end_char,
                        )
                    )

            line = TranscriptLine(
                text=text,
                start_time=line_c.start_time,
                duration=line_c.duration,
                line_id=line_c.id,
                is_complete=bool(line_c.is_complete),
                is_updated=bool(line_c.is_updated),
                is_new=bool(line_c.is_new),
                has_text_changed=bool(line_c.has_text_changed),
                have_speakers_changed=bool(line_c.have_speakers_changed),
                speaker_spans=speaker_spans,
                audio_data=audio_data,
                last_transcription_latency_ms=line_c.last_transcription_latency_ms,
                words=words,
            )
            lines.append(line)

        return Transcript(lines=lines)

    def get_version(self) -> int:
        """Get the version of the loaded Moonshine library."""
        return self._lib.moonshine_get_version()

    def create_stream(
        self,
        update_interval: float = None,
        flags: int = 0,
        transcribe_flags: int = 0,
    ) -> "Stream":
        """
        Create a new stream for real-time transcription.

        Args:
            flags: Flags for stream creation (default: 0).
            update_interval: Interval in seconds between updates (default: 0.5).
            transcribe_flags: Flags applied to every implicit
                ``update_transcription`` call the stream issues from
                ``add_audio`` / ``stop``.  Pass
                ``MOONSHINE_FLAG_SPELLING_MODE`` to drive the C++
                transcriber's spelling-CNN fusion path on live mic
                audio (default: 0).

        Returns:
            Stream object for real-time transcription
        """
        if update_interval is None:
            update_interval = self._update_interval
        return Stream(self, update_interval, flags, transcribe_flags=transcribe_flags)

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

    def push_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        """Push a temporary listener on the default stream."""
        self.get_default_stream().push_listener(listener)

    def pop_listener(self) -> None:
        """Restore previous listeners on the default stream."""
        self.get_default_stream().pop_listener()

    def pop_all_listeners(self) -> None:
        """Unwind the entire listener stack on the default stream."""
        self.get_default_stream().pop_all_listeners()


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

    def on_line_speakers_changed(self, event: LineSpeakersChanged) -> None:
        """Called when the speaker spans of a transcription line change.

        Can be called for lines that are already complete.
        """
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
        self,
        transcriber: Transcriber,
        update_interval: float = 0.5,
        flags: int = 0,
        *,
        transcribe_flags: int = 0,
    ):
        """Initialize a stream.

        ``transcribe_flags`` are forwarded to every
        ``update_transcription`` call the stream triggers implicitly
        (from ``add_audio`` and ``stop``); pass
        ``MOONSHINE_FLAG_SPELLING_MODE`` to enable the C++
        spelling-CNN fusion path on streamed audio.
        """
        self._transcriber = transcriber
        self._lib = transcriber._lib
        self._handle = None
        self._listeners: List[Callable[[TranscriptEvent], None]] = []
        self._update_interval = update_interval
        self._stream_time = 0.0
        self._last_update_time = 0.0
        self._transcribe_flags = int(transcribe_flags)
        handle = self._lib.moonshine_create_stream(transcriber._handle, flags)
        check_error(handle)
        self._handle = handle

    @property
    def transcribe_flags(self) -> int:
        """Flags currently applied to implicit ``update_transcription`` calls."""
        return self._transcribe_flags

    def set_transcribe_flags(self, flags: int) -> None:
        """Update the flags forwarded to implicit ``update_transcription`` calls.

        Use this to dynamically toggle e.g.
        ``MOONSHINE_FLAG_SPELLING_MODE`` mid-stream (DialogFlow flips it on
        only for ``SPELLED`` / ``DIGITS`` prompts so the spelling-CNN
        fusion doesn't perturb free-form recognition).
        """
        self._transcribe_flags = int(flags)

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
            result = self.update_transcription(self._transcribe_flags)
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
            self.update_transcription(self._transcribe_flags)
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

    def push_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        """Push a temporary listener, saving the current listeners on a stack.

        While the pushed listener is active, only it receives events.
        Call :meth:`pop_listener` to restore the previous set.
        """
        if not hasattr(self, "_listener_stack"):
            self._listener_stack: List[List[Callable[[TranscriptEvent], None]]] = []
        self._listener_stack.append(list(self._listeners))
        self._listeners.clear()
        self._listeners.append(listener)

    def pop_listener(self) -> None:
        """Restore the listeners that were active before the last :meth:`push_listener`."""
        if not hasattr(self, "_listener_stack") or not self._listener_stack:
            return
        self._listeners.clear()
        self._listeners.extend(self._listener_stack.pop())

    def pop_all_listeners(self) -> None:
        """Unwind the entire listener stack, restoring the original listeners."""
        if not hasattr(self, "_listener_stack") or not self._listener_stack:
            return
        self._listeners.clear()
        self._listeners.extend(self._listener_stack[0])
        self._listener_stack.clear()

    def _notify_from_transcript(self, transcript: Transcript) -> None:
        """Emit events based on transcript line properties."""
        for line in transcript.lines:
            if line.is_new:
                self._emit(LineStarted(line=line, stream_handle=self._handle))
            if line.is_updated and not line.is_new and not line.is_complete:
                self._emit(LineUpdated(line=line, stream_handle=self._handle))
            if line.has_text_changed:
                self._emit(LineTextChanged(line=line, stream_handle=self._handle))
            if line.have_speakers_changed:
                self._emit(LineSpeakersChanged(line=line, stream_handle=self._handle))
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
                    elif isinstance(event, LineSpeakersChanged):
                        listener.on_line_speakers_changed(event)
                    elif isinstance(event, LineCompleted):
                        listener.on_line_completed(event)
                    elif isinstance(event, Error):
                        listener.on_error(event)
                else:
                    # Otherwise, treat it as a callable that takes the event
                    listener(event)
            except Exception as e:
                print(f"Exception in TranscriberEventListener: {e}", file=sys.stderr)
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
        "--language", type=str, default="en", help="Language to use for transcription"
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
    parser.add_argument(
        "--quiet", action="store_true", help="Quiet output"
    )
    parser.add_argument(
        "--speaker-ids",
        action="store_true",
        help="Run speaker diarization and print a speaker-labeled conversation summary at the end",
    )
    parser.add_argument(
        "--word-timestamps", action="store_true", help="Show word timestamps"
    )
    args = parser.parse_args()

    def span_text_from_words(line: TranscriptLine, span: SpeakerSpan) -> str:
        """Map words onto this span by maximum temporal overlap."""
        span_end = span.start_time + span.duration
        words = []
        for word in line.words or []:
            overlap = min(word.end, span_end) - max(word.start, span.start_time)
            if overlap > 0.0:
                words.append((overlap, word.start, word.word))
        words.sort(key=lambda item: (item[1], -item[0]))
        return " ".join(word for _, _, word in words).strip()

    def assign_words_to_spans(
        line: TranscriptLine, spans: List[SpeakerSpan]
    ) -> List[str]:
        """Give each word to the span it overlaps most, so text is not duplicated."""
        buckets: List[List[str]] = [[] for _ in spans]
        for word in line.words or []:
            best_index = None
            best_overlap = 0.0
            for index, span in enumerate(spans):
                span_end = span.start_time + span.duration
                overlap = min(word.end, span_end) - max(word.start, span.start_time)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_index = index
            if best_index is not None and best_overlap > 0.0:
                buckets[best_index].append(word.word)
        return [" ".join(words).strip() for words in buckets]

    def span_text(
        line: TranscriptLine,
        span: SpeakerSpan,
        *,
        span_index: int,
        span_texts: Optional[List[str]] = None,
    ) -> str:
        if span_texts is not None and span_index < len(span_texts):
            snippet = span_texts[span_index].strip()
            if snippet:
                return snippet
        if span.end_char > span.start_char:
            raw = line.text.encode("utf-8")
            snippet = raw[span.start_char : span.end_char].decode("utf-8").strip()
            if snippet:
                return snippet
        return span_text_from_words(line, span)

    def format_span_line(span: SpeakerSpan, text: str) -> str:
        if text:
            return f"Speaker {span.speaker_index}: {text}"
        return (
            f"Speaker {span.speaker_index} "
            f"[{span.start_time:.2f}s +{span.duration:.2f}s]"
        )

    def format_line_by_speaker_spans(line: TranscriptLine) -> List[str]:
        """Split one VAD line into one output row per speaker span."""
        text = (line.text or "").strip()
        spans = line.speaker_spans or []
        if not text or not spans:
            return [text] if text else []

        span_texts = assign_words_to_spans(line, spans)

        if len(spans) == 1:
            snippet = span_text(line, spans[0], span_index=0, span_texts=span_texts)
            return [format_span_line(spans[0], snippet or text)]

        rows = []
        for index, span in enumerate(spans):
            rows.append(
                format_span_line(
                    span,
                    span_text(line, span, span_index=index, span_texts=span_texts),
                )
            )
        return rows

    def format_conversation_with_speakers(transcript: Transcript) -> str:
        """Build a transcript with one row per speaker span, in time order."""
        entries: List[tuple[float, str]] = []
        for line in transcript.lines:
            text = (line.text or "").strip()
            spans = line.speaker_spans or []
            if not spans:
                if text:
                    entries.append((line.start_time, text))
                continue
            for index, span in enumerate(spans):
                span_texts = assign_words_to_spans(line, spans)
                entries.append(
                    (
                        span.start_time,
                        format_span_line(
                            span,
                            span_text(
                                line, span, span_index=index, span_texts=span_texts
                            ),
                        ),
                    )
                )

        entries.sort(key=lambda item: item[0])
        return "\n".join(row for _, row in entries)

    if args.model_path is not None:
        model_path = args.model_path
        if args.model_arch is None:
            raise ValueError("--model-arch is required when --model-path is specified")
        model_arch = ModelArch(args.model_arch)
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

    if args.word_timestamps:
        options["word_timestamps"] = "true"

    if args.speaker_ids:
        options["identify_speakers"] = "true"

    transcriber = Transcriber(
        model_path=model_path, model_arch=model_arch, options=options
    )

    transcriber.start()

    class TestListener(TranscriptEventListener):
        def on_line_started(self, event: LineStarted):
            if not args.quiet:
                print(f"Line started: {event.line.text}", file=sys.stderr)

        def on_line_text_changed(self, event: LineTextChanged):
            if not args.quiet:
                print(f"Line text changed: {event.line.text}", file=sys.stderr)

        def on_line_speakers_changed(self, event: LineSpeakersChanged):
            if not args.quiet and args.speaker_ids:
                spans = ", ".join(str(span) for span in event.line.speaker_spans or [])
                print(
                    f"Line speakers changed: [{spans}] {event.line.text}",
                    file=sys.stderr,
                )

        def on_line_completed(self, event: LineCompleted):
            if args.quiet:
                return
            if args.speaker_ids and event.line.speaker_spans:
                for row in format_line_by_speaker_spans(event.line):
                    print(row, file=sys.stderr)
                if args.word_timestamps:
                    for word in event.line.words or []:
                        print(
                            f"  {word.word} [{word.start:.3f}s - {word.end:.3f}s] "
                            f"(conf: {word.confidence:.2f})",
                            file=sys.stderr,
                        )
            elif args.word_timestamps:
                print(event.line.text, file=sys.stderr)
                for word in event.line.words or []:
                    print(
                        f"  {word.word} [{word.start:.3f}s - {word.end:.3f}s] "
                        f"(conf: {word.confidence:.2f})",
                        file=sys.stderr,
                    )
            else:
                print(event.line.text, file=sys.stderr)

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

    transcript = transcriber.stop()
    if args.speaker_ids and transcript is not None:
        conversation = format_conversation_with_speakers(transcript)
        if conversation:
            print("--- Conversation (speaker IDs) ---", file=sys.stderr)
            print(conversation, file=sys.stderr)
    transcriber.close()
