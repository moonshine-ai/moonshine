import ctypes
import ctypes.util
import platform
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from moonshine_voice.errors import MoonshineError

# ---------------------------------------------------------------------------
# Constants (moonshine-c-api.h)
# ---------------------------------------------------------------------------

MOONSHINE_HEADER_VERSION = 20000

MOONSHINE_ERROR_NONE = 0
MOONSHINE_ERROR_UNKNOWN = -1
MOONSHINE_ERROR_INVALID_HANDLE = -2
MOONSHINE_ERROR_INVALID_ARGUMENT = -3

MOONSHINE_FLAG_FORCE_UPDATE = 1 << 0
# Mirror of ``MOONSHINE_FLAG_SPELLING_MODE`` from
# core/moonshine-c-api.h. When set, completed transcript lines are
# replaced in place with the resolved single-character output of the
# alphanumeric spelling fuser. Has no effect unless the transcriber was
# constructed with a spelling model.
MOONSHINE_FLAG_SPELLING_MODE = 1 << 1

MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M = 0


def _decode_utf8_from_c(buf: bytes) -> str:
    """Decode C malloc NUL-terminated bytes; tolerate rare invalid UTF-8 from native G2P output."""
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return buf.decode("utf-8", errors="replace")


def _load_libc():
    if sys.platform == "win32":
        return ctypes.CDLL("msvcrt")
    name = ctypes.util.find_library("c")
    if name:
        return ctypes.CDLL(name)
    if platform.system() == "Darwin":
        return ctypes.CDLL("/usr/lib/libc.dylib")
    return ctypes.CDLL("libc.so.6")


_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        _libc = _load_libc()
        _libc.free.argtypes = [ctypes.c_void_p]
        _libc.free.restype = None
    return _libc


def moonshine_free(address: Optional[int]) -> None:
    """Release memory allocated by the Moonshine C API (``malloc``). Pass the raw pointer value (integer)."""
    if address:
        _get_libc().free(ctypes.c_void_p(address))


# C structure definitions matching moonshine-c-api.h


class TranscriptWordC(ctypes.Structure):
    """C structure for transcript_word_t."""

    _fields_ = [
        ("text", ctypes.POINTER(ctypes.c_char)),
        ("start", ctypes.c_float),
        ("end", ctypes.c_float),
        ("confidence", ctypes.c_float),
    ]


class SpeakerSpanC(ctypes.Structure):
    """C structure for speaker_span_t."""

    _fields_ = [
        ("start_time", ctypes.c_float),
        ("duration", ctypes.c_float),
        ("speaker_id", ctypes.c_uint64),
        ("speaker_index", ctypes.c_uint32),
        ("start_char", ctypes.c_uint64),
        ("end_char", ctypes.c_uint64),
    ]


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
        ("have_speakers_changed", ctypes.c_int8),
        ("speaker_spans", ctypes.POINTER(SpeakerSpanC)),
        ("speaker_span_count", ctypes.c_uint64),
        ("last_transcription_latency_ms", ctypes.c_uint32),
        ("words", ctypes.POINTER(TranscriptWordC)),
        ("word_count", ctypes.c_uint64),
    ]


class TranscriptC(ctypes.Structure):
    """C structure for transcript_t."""

    _fields_ = [
        ("lines", ctypes.POINTER(TranscriptLineC)),
        ("line_count", ctypes.c_uint64),
    ]


def _require_struct_size(name, struct, expected, note=""):
    """Fail fast if a ctypes struct drifts from the compiled C ABI.

    A mismatch here means the Python binding walks native arrays with the
    wrong stride, which crashes with SIGSEGV instead of a clear error (see
    https://github.com/moonshine-ai/moonshine/issues/158). We only enforce on
    the 64-bit ABI these sizes were computed for; other ABIs (32-bit pointers,
    different size_t) legitimately produce different sizes and are skipped
    rather than misdiagnosed as binding drift.
    """
    actual = ctypes.sizeof(struct)
    if actual != expected:
        detail = f" {note}" if note else ""
        raise ImportError(
            f"moonshine_voice ABI mismatch: {name} is {actual} bytes but the "
            f"compiled C ABI expects {expected} bytes. This usually means the "
            f"installed Python binding is out of sync with libmoonshine. "
            f"(sizeof void*={ctypes.sizeof(ctypes.c_void_p)}, "
            f"size_t={ctypes.sizeof(ctypes.c_size_t)}){detail}"
        )


# Only the LP64/LLP64 layout (8-byte pointers and size_t) that these expected
# sizes were derived from is validated; skip other ABIs to avoid false alarms.
if ctypes.sizeof(ctypes.c_void_p) == 8 and ctypes.sizeof(ctypes.c_size_t) == 8:
    _require_struct_size("TranscriptWordC", TranscriptWordC, 24)
    _require_struct_size("SpeakerSpanC", SpeakerSpanC, 40)
    _require_struct_size(
        "TranscriptLineC",
        TranscriptLineC,
        88,
        "See https://github.com/moonshine-ai/moonshine/issues/158",
    )
    _require_struct_size("TranscriptC", TranscriptC, 16)


class TranscriberOptionC(ctypes.Structure):
    """C structure for moonshine_option_t."""

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("value", ctypes.c_char_p),
    ]


# Alias matching the C header name ``moonshine_option_t``.
MoonshineOptionC = TranscriberOptionC


class MoonshineIntentMatchC(ctypes.Structure):
    """C struct moonshine_intent_match_t (intent recognizer)."""

    _fields_ = [
        ("canonical_phrase", ctypes.c_char_p),
        ("similarity", ctypes.c_float),
    ]


class ModelArch(IntEnum):
    """Model architecture types."""

    TINY = 0
    BASE = 1
    TINY_STREAMING = 2
    BASE_STREAMING = 3
    SMALL_STREAMING = 4
    MEDIUM_STREAMING = 5


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
class WordTiming:
    """A single word with timing information."""

    word: str
    start: float
    end: float
    confidence: float

    def __str__(self) -> str:
        return f"[{self.start:.3f}s - {self.end:.3f}s] {self.word} (conf: {self.confidence:.2f})"


@dataclass
class SpeakerSpan:
    """One contiguous span of speech within a line attributed to one speaker.

    Only populated when the ``identify_speakers`` option is enabled. Spans
    for recent audio are mutable: streaming diarization re-clusters a
    sliding window (``diarization_cluster_window_sec``, default 120s) as more
    speech arrives, so spans can be revised on any transcription update.
    Assignments for audio older than the window are frozen. Watch
    ``TranscriptLine.have_speakers_changed`` (or the ``LineSpeakersChanged``
    event) to detect revisions.

    ``identify_speakers`` also enables word timestamps automatically. Use
    ``line.text[span.start_char:span.end_char]`` to get the UTF-8 substring
    for a span (byte offsets, like Python string slicing).
    """

    start_time: float
    duration: float
    speaker_id: int
    speaker_index: int
    start_char: int = 0
    end_char: int = 0

    def __str__(self) -> str:
        char_range = (
            f", chars {self.start_char}:{self.end_char}"
            if self.end_char > self.start_char
            else ""
        )
        return (
            f"[{self.start_time:.2f}s +{self.duration:.2f}s] "
            f"Speaker {self.speaker_index} ({self.speaker_id}){char_range}"
        )


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
    have_speakers_changed: bool = False
    speaker_spans: Optional[List[SpeakerSpan]] = None
    audio_data: Optional[List[float]] = None
    last_transcription_latency_ms: int = 0
    words: Optional[List[WordTiming]] = None

    def __str__(self) -> str:
        spans_str = (
            "[" + ", ".join(str(span) for span in self.speaker_spans) + "]"
            if self.speaker_spans
            else "[]"
        )
        return f"[{self.start_time:.2f}s]: '{self.text}', metadata: [duration={self.duration:.2f}s, line_id={self.line_id}, is_complete={self.is_complete}, is_updated={self.is_updated}, is_new={self.is_new}, has_text_changed={self.has_text_changed}, have_speakers_changed={self.have_speakers_changed}, speaker_spans={spans_str}, audio_data_len={len(self.audio_data) if self.audio_data else 0}, last_transcription_latency_ms={self.last_transcription_latency_ms}, words={len(self.words) if self.words else 0}]"


@dataclass
class Transcript:
    """A complete transcript containing multiple lines."""

    lines: List[TranscriptLine]

    def __str__(self) -> str:
        """Return a string representation of the transcript."""
        return "\n".join(f"[{line.start_time:.2f}s] {line.text}" for line in self.lines)


def moonshine_options_array(
    options: Optional[Dict[str, Union[str, int, float, bool]]],
) -> Tuple[Optional[Any], int, List[bytes]]:
    """Build a ``moonshine_option_t`` array. Keep the returned list alive until the C call completes."""
    if not options:
        return None, 0, []
    keepalive: List[bytes] = []
    structs = []
    for name, value in options.items():
        nb = name.encode("utf-8")
        vb = str(value).encode("utf-8")
        keepalive.extend((nb, vb))
        structs.append(TranscriberOptionC(name=nb, value=vb))
    arr = (TranscriberOptionC * len(structs))(*structs)
    return arr, len(structs), keepalive


def moonshine_c_string_array(
    strings: Sequence[str],
) -> Tuple[ctypes.Array, int, List[bytes]]:
    """Build ``const char *filenames[]`` for TTS/G2P create-from-files helpers."""
    encoded = [s.encode("utf-8") for s in strings]
    arr = (ctypes.c_char_p * len(encoded))(*encoded)
    return arr, len(encoded), encoded


def moonshine_memory_arrays(
    buffers: Sequence[Optional[bytes]],
) -> Tuple[ctypes.Array, ctypes.Array, List[Optional[ctypes.Array]]]:
    """Build parallel ``uint8_t*`` and ``uint64_t`` size arrays for in-memory TTS/G2P creation.

    Buffers are copied into internal ctypes buffers; keep the returned third list alive until
    the synthesizer or phonemizer is freed (per C API lifetime rules).
    """
    n = len(buffers)
    ptr_arr = (ctypes.POINTER(ctypes.c_uint8) * n)()
    size_arr = (ctypes.c_uint64 * n)()
    holders: List[Optional[ctypes.Array]] = []
    for i, buf in enumerate(buffers):
        if buf is not None and len(buf) > 0:
            raw = ctypes.create_string_buffer(buf, len(buf))
            holders.append(raw)
            ptr_arr[i] = ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8))
            size_arr[i] = len(buf)
        else:
            holders.append(None)
            ptr_arr[i] = None
            size_arr[i] = 0
    return ptr_arr, size_arr, holders


def moonshine_get_g2p_dependencies_string(
    languages: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> str:
    """Call ``moonshine_get_g2p_dependencies`` and return the comma-separated key list (UTF-8)."""
    lib = _MoonshineLib().lib
    opt_arr, opt_n, opt_keep = moonshine_options_array(options)
    lang_b = languages.encode("utf-8") if languages is not None else None
    out_p = ctypes.c_void_p()
    err = lib.moonshine_get_g2p_dependencies(lang_b, opt_arr, opt_n, ctypes.byref(out_p))
    if err != MOONSHINE_ERROR_NONE:
        raise MoonshineError(
            lib.moonshine_error_to_string(err).decode("utf-8")
            if lib.moonshine_error_to_string(err)
            else f"moonshine_get_g2p_dependencies failed ({err})"
        )
    addr = out_p.value
    if not addr:
        return ""
    try:
        return ctypes.string_at(addr).decode("utf-8")
    finally:
        moonshine_free(addr)


def moonshine_get_tts_dependencies_string(
    languages: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> str:
    """Call ``moonshine_get_tts_dependencies`` and return the JSON array string (UTF-8)."""
    lib = _MoonshineLib().lib
    opt_arr, opt_n, opt_keep = moonshine_options_array(options)
    lang_b = languages.encode("utf-8") if languages is not None else None
    out_p = ctypes.c_void_p()
    err = lib.moonshine_get_tts_dependencies(lang_b, opt_arr, opt_n, ctypes.byref(out_p))
    if err != MOONSHINE_ERROR_NONE:
        raise MoonshineError(
            lib.moonshine_error_to_string(err).decode("utf-8")
            if lib.moonshine_error_to_string(err)
            else f"moonshine_get_tts_dependencies failed ({err})"
        )
    addr = out_p.value
    if not addr:
        return ""
    try:
        return ctypes.string_at(addr).decode("utf-8")
    finally:
        moonshine_free(addr)


def moonshine_try_get_tts_voices(
    languages: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> Tuple[int, str]:
    """
    Call ``moonshine_get_tts_voices`` without raising.

    Returns ``(error_code, json_text)``. On success, ``error_code`` is ``MOONSHINE_ERROR_NONE`` and
    ``json_text`` is the JSON object. On failure, ``json_text`` is ``""``.
    """
    lib = _MoonshineLib().lib
    opt_arr, opt_n, opt_keep = moonshine_options_array(options)
    lang_b = languages.encode("utf-8") if languages is not None else None
    out_p = ctypes.c_void_p()
    err = int(lib.moonshine_get_tts_voices(lang_b, opt_arr, opt_n, ctypes.byref(out_p)))
    addr = out_p.value
    if err != MOONSHINE_ERROR_NONE:
        if addr:
            moonshine_free(addr)
        return err, ""
    if not addr:
        return err, "{}"
    try:
        text = ctypes.string_at(addr).decode("utf-8")
        return err, text if text else "{}"
    finally:
        moonshine_free(addr)


def moonshine_get_tts_voices_string(
    languages: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> str:
    """Call ``moonshine_get_tts_voices`` and return the JSON object string (UTF-8)."""
    err, text = moonshine_try_get_tts_voices(languages, options)
    if err != MOONSHINE_ERROR_NONE:
        lib = _MoonshineLib().lib
        raise MoonshineError(
            lib.moonshine_error_to_string(err).decode("utf-8")
            if lib.moonshine_error_to_string(err)
            else f"moonshine_get_tts_voices failed ({err})"
        )
    return text if text else "{}"


def moonshine_text_to_speech_samples(
    tts_synthesizer_handle: int,
    text: str,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> Tuple[List[float], int]:
    """Call ``moonshine_text_to_speech``; returns ``(samples, sample_rate_hz)``. Frees the native audio buffer."""
    lib = _MoonshineLib().lib
    opt_arr, opt_n, opt_keep = moonshine_options_array(options)
    text_b = text.encode("utf-8")
    out_audio = ctypes.POINTER(ctypes.c_float)()
    out_size = ctypes.c_uint64()
    out_sr = ctypes.c_int32()
    err = lib.moonshine_text_to_speech(
        ctypes.c_int32(tts_synthesizer_handle),
        text_b,
        opt_arr,
        opt_n,
        ctypes.byref(out_audio),
        ctypes.byref(out_size),
        ctypes.byref(out_sr),
    )
    if err != MOONSHINE_ERROR_NONE:
        raise MoonshineError(
            lib.moonshine_error_to_string(err).decode("utf-8")
            if lib.moonshine_error_to_string(err)
            else f"moonshine_text_to_speech failed ({err})"
        )
    n = int(out_size.value)
    if n <= 0 or not out_audio:
        return [], int(out_sr.value)
    try:
        chunk = ctypes.cast(out_audio, ctypes.POINTER(ctypes.c_float * n)).contents
        return list(chunk), int(out_sr.value)
    finally:
        moonshine_free(ctypes.cast(out_audio, ctypes.c_void_p).value)


def moonshine_phonemes_to_speech_samples(
    tts_synthesizer_handle: int,
    phonemes: str,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> Tuple[List[float], int]:
    """Call ``moonshine_phonemes_to_speech``; returns ``(samples, sample_rate_hz)``.

    ``phonemes`` is an IPA string as produced by
    ``moonshine_text_to_phonemes_string``. Frees the native audio buffer.
    """
    lib = _MoonshineLib().lib
    opt_arr, opt_n, opt_keep = moonshine_options_array(options)
    phonemes_b = phonemes.encode("utf-8")
    out_audio = ctypes.POINTER(ctypes.c_float)()
    out_size = ctypes.c_uint64()
    out_sr = ctypes.c_int32()
    err = lib.moonshine_phonemes_to_speech(
        ctypes.c_int32(tts_synthesizer_handle),
        phonemes_b,
        opt_arr,
        opt_n,
        ctypes.byref(out_audio),
        ctypes.byref(out_size),
        ctypes.byref(out_sr),
    )
    if err != MOONSHINE_ERROR_NONE:
        raise MoonshineError(
            lib.moonshine_error_to_string(err).decode("utf-8")
            if lib.moonshine_error_to_string(err)
            else f"moonshine_phonemes_to_speech failed ({err})"
        )
    n = int(out_size.value)
    if n <= 0 or not out_audio:
        return [], int(out_sr.value)
    try:
        chunk = ctypes.cast(out_audio, ctypes.POINTER(ctypes.c_float * n)).contents
        return list(chunk), int(out_sr.value)
    finally:
        moonshine_free(ctypes.cast(out_audio, ctypes.c_void_p).value)


def moonshine_text_to_phonemes_string(
    grapheme_to_phonemizer_handle: int,
    text: str,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> str:
    """Call ``moonshine_text_to_phonemes``; returns the IPA string (single segment)."""
    lib = _MoonshineLib().lib
    opt_arr, opt_n, opt_keep = moonshine_options_array(options)
    text_b = text.encode("utf-8")
    out_ph = ctypes.c_char_p()
    out_count = ctypes.c_uint64()
    err = lib.moonshine_text_to_phonemes(
        ctypes.c_int32(grapheme_to_phonemizer_handle),
        text_b,
        opt_arr,
        opt_n,
        ctypes.byref(out_ph),
        ctypes.byref(out_count),
    )
    if err != MOONSHINE_ERROR_NONE:
        raw = lib.moonshine_error_to_string(err)
        msg = raw.decode("utf-8") if raw else f"moonshine_text_to_phonemes failed ({err})"
        if err == MOONSHINE_ERROR_UNKNOWN and msg == "Unknown error":
            msg = (
                "G2P failed (unknown error; the native layer usually logs the cause on stderr, "
                "e.g. tokenizer / WordPiece alignment)"
            )
        raise MoonshineError(msg)
    if not out_ph.value:
        return ""
    addr = ctypes.cast(out_ph, ctypes.c_void_p).value
    try:
        return _decode_utf8_from_c(ctypes.string_at(addr))
    finally:
        moonshine_free(addr)


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

        lib.moonshine_get_version.restype = ctypes.c_int32
        lib.moonshine_get_version.argtypes = []

        lib.moonshine_error_to_string.restype = ctypes.c_char_p
        lib.moonshine_error_to_string.argtypes = [ctypes.c_int32]

        lib.moonshine_transcript_to_string.restype = ctypes.c_char_p
        lib.moonshine_transcript_to_string.argtypes = [
            ctypes.POINTER(TranscriptC),
        ]

        lib.moonshine_load_transcriber_from_files.restype = ctypes.c_int32
        lib.moonshine_load_transcriber_from_files.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.c_int32,
        ]

        lib.moonshine_load_transcriber_from_memory.restype = ctypes.c_int32
        lib.moonshine_load_transcriber_from_memory.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # encoder_model_data
            ctypes.c_size_t,                  # encoder_model_data_size
            ctypes.POINTER(ctypes.c_uint8),  # decoder_model_data
            ctypes.c_size_t,                  # decoder_model_data_size
            ctypes.POINTER(ctypes.c_uint8),  # tokenizer_data
            ctypes.c_size_t,                  # tokenizer_data_size
            # Spelling-CNN .ort buffer (NULL/0 to disable spelling mode).
            ctypes.POINTER(ctypes.c_uint8),  # spelling_model_data
            ctypes.c_size_t,                  # spelling_model_data_size
            ctypes.c_uint32,                  # model_arch
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.c_int32,
        ]

        lib.moonshine_free_transcriber.restype = None
        lib.moonshine_free_transcriber.argtypes = [ctypes.c_int32]

        lib.moonshine_transcribe_without_streaming.restype = ctypes.c_int32
        lib.moonshine_transcribe_without_streaming.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.POINTER(TranscriptC)),
        ]

        lib.moonshine_create_stream.restype = ctypes.c_int32
        lib.moonshine_create_stream.argtypes = [
            ctypes.c_int32,
            ctypes.c_uint32,
        ]

        lib.moonshine_free_stream.restype = ctypes.c_int32
        lib.moonshine_free_stream.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
        ]

        lib.moonshine_start_stream.restype = ctypes.c_int32
        lib.moonshine_start_stream.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
        ]

        lib.moonshine_stop_stream.restype = ctypes.c_int32
        lib.moonshine_stop_stream.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
        ]

        lib.moonshine_transcribe_add_audio_to_stream.restype = ctypes.c_int32
        lib.moonshine_transcribe_add_audio_to_stream.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.c_uint32,
        ]

        lib.moonshine_transcribe_stream.restype = ctypes.c_int32
        lib.moonshine_transcribe_stream.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.POINTER(TranscriptC)),
        ]

        lib.moonshine_create_intent_recognizer.restype = ctypes.c_int32
        lib.moonshine_create_intent_recognizer.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
        ]

        lib.moonshine_free_intent_recognizer.restype = None
        lib.moonshine_free_intent_recognizer.argtypes = [ctypes.c_int32]

        lib.moonshine_register_intent.restype = ctypes.c_int32
        lib.moonshine_register_intent.argtypes = [
            ctypes.c_int32,
            ctypes.c_char_p,
        ]

        lib.moonshine_unregister_intent.restype = ctypes.c_int32
        lib.moonshine_unregister_intent.argtypes = [
            ctypes.c_int32,
            ctypes.c_char_p,
        ]

        lib.moonshine_get_closest_intents.restype = ctypes.c_int32
        lib.moonshine_get_closest_intents.argtypes = [
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_float,
            ctypes.POINTER(ctypes.POINTER(MoonshineIntentMatchC)),
            ctypes.POINTER(ctypes.c_uint64),
        ]

        lib.moonshine_free_intent_matches.restype = None
        lib.moonshine_free_intent_matches.argtypes = [
            ctypes.POINTER(MoonshineIntentMatchC),
            ctypes.c_uint64,
        ]

        lib.moonshine_get_intent_count.restype = ctypes.c_int32
        lib.moonshine_get_intent_count.argtypes = [ctypes.c_int32]

        lib.moonshine_clear_intents.restype = ctypes.c_int32
        lib.moonshine_clear_intents.argtypes = [ctypes.c_int32]

        lib.moonshine_create_tts_synthesizer_from_files.restype = ctypes.c_int32
        lib.moonshine_create_tts_synthesizer_from_files.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_uint64,
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.c_int32,
        ]

        lib.moonshine_create_tts_synthesizer_from_memory.restype = ctypes.c_int32
        lib.moonshine_create_tts_synthesizer_from_memory.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.c_int32,
        ]

        lib.moonshine_free_tts_synthesizer.restype = None
        lib.moonshine_free_tts_synthesizer.argtypes = [ctypes.c_int32]

        lib.moonshine_get_g2p_dependencies.restype = ctypes.c_int32
        lib.moonshine_get_g2p_dependencies.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
        ]

        lib.moonshine_get_tts_dependencies.restype = ctypes.c_int32
        lib.moonshine_get_tts_dependencies.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
        ]

        lib.moonshine_get_tts_voices.restype = ctypes.c_int32
        lib.moonshine_get_tts_voices.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
        ]

        lib.moonshine_text_to_speech.restype = ctypes.c_int32
        lib.moonshine_text_to_speech.argtypes = [
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
        ]

        lib.moonshine_phonemes_to_speech.restype = ctypes.c_int32
        lib.moonshine_phonemes_to_speech.argtypes = [
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
        ]

        lib.moonshine_create_grapheme_to_phonemizer_from_files.restype = ctypes.c_int32
        lib.moonshine_create_grapheme_to_phonemizer_from_files.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_uint64,
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.c_int32,
        ]

        lib.moonshine_create_grapheme_to_phonemizer_from_memory.restype = ctypes.c_int32
        lib.moonshine_create_grapheme_to_phonemizer_from_memory.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.c_int32,
        ]

        lib.moonshine_free_grapheme_to_phonemizer.restype = None
        lib.moonshine_free_grapheme_to_phonemizer.argtypes = [ctypes.c_int32]

        lib.moonshine_text_to_phonemes.restype = ctypes.c_int32
        lib.moonshine_text_to_phonemes.argtypes = [
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.POINTER(TranscriberOptionC),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_uint64),
        ]

    @property
    def lib(self):
        """Get the loaded library."""
        return self._lib
