"""
Moonshine Voice - Fast, accurate, on-device AI library for building interactive voice applications.

This package provides Python bindings for the Moonshine Voice C API, enabling
voice-activity detection, transcription, and other voice processing capabilities.
"""

from moonshine_voice.errors import (
    MoonshineError,
    MoonshineUnknownError,
    MoonshineInvalidHandleError,
    MoonshineInvalidArgumentError,
)

from moonshine_voice.moonshine_api import (
    ModelArch,
    Transcript,
    TranscriptLine,
)

from moonshine_voice.transcriber import (
    Transcriber,
    Stream,
    TranscriptEventListener,
    TranscriptEvent,
    LineStarted,
    LineUpdated,
    LineTextChanged,
    LineCompleted,
    Error,
)

from moonshine_voice.mic_transcriber import MicTranscriber

from moonshine_voice.utils import (
    get_assets_path,
    get_model_path,
    load_wav_file,
)

__version__ = "0.1.0"

__all__ = [
    "Transcriber",
    "MicTranscriber",
    "ModelArch",
    "TranscriptLine",
    "Transcript",
    "Stream",
    "TranscriptEventListener",
    "TranscriptEvent",
    "LineStarted",
    "LineUpdated",
    "LineTextChanged",
    "LineCompleted",
    "Error",
    "MoonshineError",
    "MoonshineUnknownError",
    "MoonshineInvalidHandleError",
    "MoonshineInvalidArgumentError",
    "get_assets_path",
    "get_model_path",
    "load_wav_file",
]

