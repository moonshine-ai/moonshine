import ctypes
import platform
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional

from moonshine_voice.errors import MoonshineError

# C structure definitions matching moonshine-c-api.h


class TranscriptLineC(ctypes.Structure):
    """C structure for transcript_line_t."""

    _fields_ = [
        ("text", ctypes.POINTER(ctypes.c_char)),
        ("audio_data", ctypes.POINTER(ctypes.c_float)),
        ("audio_data_count", ctypes.c_size_t),
        ("start_time", ctypes.c_float),
        ("duration", ctypes.c_float),
        ("id", ctypes.c_uint64),
        ("is_complete", ctypes.c_int8),
        ("is_updated", ctypes.c_int8),
        ("is_new", ctypes.c_int8),
        ("has_text_changed", ctypes.c_int8),
        ("last_transcription_latency_ms", ctypes.c_uint32),
    ]


class TranscriptC(ctypes.Structure):
    """C structure for transcript_t."""

    _fields_ = [
        ("lines", ctypes.POINTER(TranscriptLineC)),
        ("line_count", ctypes.c_uint64),
    ]


class TranscriberOptionC(ctypes.Structure):
    """C structure for transcriber_option_t."""

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("value", ctypes.c_char_p),
    ]


class ModelArch(IntEnum):
    """Model architecture types."""

    TINY = 0
    BASE = 1
    TINY_STREAMING = 2
    BASE_STREAMING = 3
    MEDIUM_STREAMING = 4
    SMALL_STREAMING = 5

def model_arch_to_string(model_arch: ModelArch) -> str:
    """Convert a model architecture to a string."""
    if model_arch == ModelArch.TINY:
        return "tiny"
    elif model_arch == ModelArch.BASE:
        return "base"
    elif model_arch == ModelArch.TINY_STREAMING:
        return "tiny-streaming"
    elif model_arch == ModelArch.BASE_STREAMING:
        return "base-streaming"
    elif model_arch == ModelArch.MEDIUM_STREAMING:
        return "medium-streaming"
    elif model_arch == ModelArch.SMALL_STREAMING:
        return "small-streaming"
    else:
        raise ValueError(f"Invalid model architecture: {model_arch}")

def string_to_model_arch(model_arch_string: str) -> ModelArch:
    """Convert a string to a model architecture."""
    if model_arch_string == "tiny":
        return ModelArch.TINY
    elif model_arch_string == "base":
        return ModelArch.BASE
    elif model_arch_string == "tiny-streaming":
        return ModelArch.TINY_STREAMING
    elif model_arch_string == "base-streaming":
        return ModelArch.BASE_STREAMING
    elif model_arch_string == "small-streaming":
        return ModelArch.SMALL_STREAMING
    elif model_arch_string == "medium-streaming":
        return ModelArch.MEDIUM_STREAMING
    else:
        raise ValueError(f"Invalid model architecture string: {model_arch_string}")

@dataclass
class TranscriptLine:
    """A single line of transcription."""

    text: str
    start_time: float
    duration: float
    line_id: int
    is_complete: bool
    is_updated: bool = False
    is_new: bool = False
    has_text_changed: bool = False
    audio_data: Optional[List[float]] = None
    last_transcription_latency_ms: int = 0


@dataclass
class Transcript:
    """A complete transcript containing multiple lines."""

    lines: List[TranscriptLine]

    def __str__(self) -> str:
        """Return a string representation of the transcript."""
        return "\n".join(f"[{line.start_time:.2f}s] {line.text}" for line in self.lines)


class _MoonshineLib:
    """Internal class to load and wrap the Moonshine C library."""

    _instance = None
    _lib = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_library()
        return cls._instance

    def _load_library(self):
        """Load the Moonshine shared library."""
        if self._lib is not None:
            return

        system = platform.system()
        if system == "Darwin":
            lib_name = "libmoonshine.dylib"
        elif system == "Linux":
            lib_name = "libmoonshine.so"
        elif system == "Windows":
            lib_name = "moonshine.dll"
        else:
            raise MoonshineError(f"Unsupported platform: {system}")

        # Try to find the library in common locations
        possible_paths = [
            # In the package directory
            Path(__file__).parent / lib_name,
            Path(__file__).parent.parent.parent / lib_name,
            # In the build directory (for development)
            Path(__file__).parent.parent.parent.parent / "core" / "build" / lib_name,
            # System library paths
            Path("/usr/local/lib") / lib_name,
            Path("/usr/lib") / lib_name,
        ]

        lib_path = None
        for path in possible_paths:
            if path.exists():
                lib_path = path
                break

        if lib_path is None:
            # Try loading by name (will use system search paths)
            lib_path = lib_name

        try:
            self._lib = ctypes.CDLL(str(lib_path))
        except OSError as e:
            raise MoonshineError(
                f"Failed to load Moonshine library from {lib_path}: {e}. "
                "Make sure the library is built and available."
            ) from e

        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Setup ctypes function signatures for the C API."""
        lib = self._lib

        # Constants
        lib.moonshine_get_version.restype = ctypes.c_int32
        lib.moonshine_get_version.argtypes = []

        lib.moonshine_error_to_string.restype = ctypes.c_char_p
        lib.moonshine_error_to_string.argtypes = [ctypes.c_int32]

        # Load transcriber
        lib.moonshine_load_transcriber_from_files.restype = ctypes.c_int32
        lib.moonshine_load_transcriber_from_files.argtypes = [
            ctypes.c_char_p,  # path
            ctypes.c_uint32,  # model_arch
            ctypes.POINTER(TranscriberOptionC),  # options (can be None)
            ctypes.c_uint64,  # options_count
            ctypes.c_int32,  # moonshine_version
        ]

        lib.moonshine_free_transcriber.restype = None
        lib.moonshine_free_transcriber.argtypes = [ctypes.c_int32]

        # Transcribe without streaming
        lib.moonshine_transcribe_without_streaming.restype = ctypes.c_int32
        lib.moonshine_transcribe_without_streaming.argtypes = [
            ctypes.c_int32,  # transcriber_handle
            ctypes.POINTER(ctypes.c_float),  # audio_data
            ctypes.c_uint64,  # audio_length
            ctypes.c_int32,  # sample_rate
            ctypes.c_uint32,  # flags
            ctypes.POINTER(ctypes.POINTER(TranscriptC)),  # out_transcript
        ]

        # Streaming functions
        lib.moonshine_create_stream.restype = ctypes.c_int32
        lib.moonshine_create_stream.argtypes = [
            ctypes.c_int32,  # transcriber_handle
            ctypes.c_uint32,  # flags
        ]

        lib.moonshine_free_stream.restype = ctypes.c_int32
        lib.moonshine_free_stream.argtypes = [
            ctypes.c_int32,  # transcriber_handle
            ctypes.c_int32,  # stream_handle
        ]

        lib.moonshine_start_stream.restype = ctypes.c_int32
        lib.moonshine_start_stream.argtypes = [
            ctypes.c_int32,  # transcriber_handle
            ctypes.c_int32,  # stream_handle
        ]

        lib.moonshine_stop_stream.restype = ctypes.c_int32
        lib.moonshine_stop_stream.argtypes = [
            ctypes.c_int32,  # transcriber_handle
            ctypes.c_int32,  # stream_handle
        ]

        lib.moonshine_transcribe_add_audio_to_stream.restype = ctypes.c_int32
        lib.moonshine_transcribe_add_audio_to_stream.argtypes = [
            ctypes.c_int32,  # transcriber_handle
            ctypes.c_int32,  # stream_handle
            ctypes.POINTER(ctypes.c_float),  # new_audio_data
            ctypes.c_uint64,  # audio_length
            ctypes.c_int32,  # sample_rate
            ctypes.c_uint32,  # flags
        ]

        lib.moonshine_transcribe_stream.restype = ctypes.c_int32
        lib.moonshine_transcribe_stream.argtypes = [
            ctypes.c_int32,  # transcriber_handle
            ctypes.c_int32,  # stream_handle
            ctypes.c_uint32,  # flags
            ctypes.POINTER(ctypes.POINTER(TranscriptC)),  # out_transcript
        ]

    @property
    def lib(self):
        """Get the loaded library."""
        return self._lib
