"""Transcriber module for Moonshine Voice."""

import ctypes
from typing import List

from moonshine_voice.moonshine_api import (
    _MoonshineLib,
    ModelArch,
    Transcript,
    TranscriptLine,
    TranscriptC,
)
from moonshine_voice.errors import MoonshineError, check_error


class Transcriber:
    """Main transcriber class for Moonshine Voice."""

    MOONSHINE_HEADER_VERSION = 20000
    MOONSHINE_FLAG_FORCE_UPDATE = 1 << 0

    def __init__(self, model_path: str, model_arch: ModelArch = ModelArch.BASE):
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

        # Load the transcriber
        model_path_bytes = model_path.encode("utf-8")
        handle = self._lib.moonshine_load_transcriber_from_files(
            model_path_bytes,
            model_arch.value,
            None,  # options
            0,  # options_count
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
            )
            lines.append(line)

        return Transcript(lines=lines)

    def get_version(self) -> int:
        """Get the version of the loaded Moonshine library."""
        return self._lib.moonshine_get_version()

    def create_stream(self, flags: int = 0) -> "Stream":
        """
        Create a new stream for real-time transcription.

        Args:
            flags: Flags for stream creation (default: 0)

        Returns:
            Stream object for real-time transcription
        """
        return Stream(self, flags)


# Streaming functionality
class Stream:
    """Stream for real-time transcription."""

    def __init__(self, transcriber: Transcriber, flags: int = 0):
        """Initialize a stream."""
        self._transcriber = transcriber
        self._lib = transcriber._lib
        self._handle = None

        print(f"Creating stream with handle: {transcriber._handle} and flags: {flags}")
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

    def transcribe(self, flags: int = 0) -> Transcript:
        """Get the current transcript from the stream."""
        out_transcript = ctypes.POINTER(TranscriptC)()
        error = self._lib.moonshine_transcribe_stream(
            self._transcriber._handle, self._handle, flags, ctypes.byref(out_transcript)
        )
        check_error(error)
        return self._transcriber._parse_transcript(out_transcript)

    def close(self):
        """Free the stream resources."""
        if self._handle is not None:
            self._lib.moonshine_free_stream(self._transcriber._handle, self._handle)
            self._handle = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
