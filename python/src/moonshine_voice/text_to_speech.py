"""Text-to-speech module for Moonshine Voice.

This module provides text-to-speech synthesis using the Moonshine TTS C API.
Grapheme-to-phoneme conversion is handled internally by the C library using
the Phonemis G2P engine.
"""

import ctypes
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from moonshine_voice.moonshine_api import _MoonshineLib
from moonshine_voice.errors import MoonshineError, check_error
from moonshine_voice.download import TTSModelArch


class TtsResultC(ctypes.Structure):
    """C structure for tts_result_t."""

    _fields_ = [
        ("audio_data", ctypes.POINTER(ctypes.c_int16)),
        ("audio_length", ctypes.c_uint64),
        ("sample_rate", ctypes.c_int32),
    ]


@dataclass
class TTSResult:
    """Result of a text-to-speech generation call.

    Attributes:
        audio_data: NumPy array of signed 16-bit PCM samples.
        sample_rate: Sample rate in Hz (24000).
    """

    audio_data: NDArray[np.int16]
    sample_rate: int


class TextToSpeech:
    """Text-to-speech synthesizer using the Moonshine TTS engine.

    This class wraps the Moonshine C API for text-to-speech. It accepts
    plain text strings and produces 24 kHz signed 16-bit PCM audio.
    Grapheme-to-phoneme conversion is handled internally by the C library.

    Example usage::

        >>> tts = TextToSpeech("path/to/tts-model")
        >>> result = tts.generate("Hello, world!")
        >>> # result.audio_data is a numpy int16 array at 24 kHz
        >>> tts.close()

    Context-manager usage::

        >>> with TextToSpeech("path/to/tts-model") as tts:
        ...     result = tts.generate("Hello, world!")
    """

    def __init__(
        self,
        model_path: str,
        model_arch: TTSModelArch = TTSModelArch.TSUKI,
    ):
        """
        Initialize a text-to-speech model.

        Args:
            model_path: Path to the directory containing the TTS model files
                       (model.onnx and vocab.json).
            model_arch: The TTS model architecture to use.
                       Currently only TSUKI is supported.
        """
        self._lib_wrapper = _MoonshineLib()
        self._lib = self._lib_wrapper.lib
        self._handle: Optional[int] = None
        self._setup_function_signatures()

        model_path_bytes = model_path.encode("utf-8")

        handle = self._lib.moonshine_create_text_to_speech(
            model_path_bytes,
            model_arch.value,
        )

        if handle < 0:
            check_error(handle)

        self._handle = handle

    def _setup_function_signatures(self):
        """Setup ctypes function signatures for the TTS C API."""
        lib = self._lib

        # Create text-to-speech
        lib.moonshine_create_text_to_speech.restype = ctypes.c_int32
        lib.moonshine_create_text_to_speech.argtypes = [
            ctypes.c_char_p,  # model_path
            ctypes.c_uint32,  # model_arch
        ]

        # Free text-to-speech
        lib.moonshine_free_text_to_speech.restype = None
        lib.moonshine_free_text_to_speech.argtypes = [ctypes.c_int32]

        # Generate
        lib.moonshine_text_to_speech_generate.restype = ctypes.c_int32
        lib.moonshine_text_to_speech_generate.argtypes = [
            ctypes.c_int32,  # tts_handle
            ctypes.c_char_p,  # text
            ctypes.POINTER(TtsResultC),  # out_result
        ]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Free the text-to-speech model resources."""
        if self._handle is not None:
            self._lib.moonshine_free_text_to_speech(self._handle)
            self._handle = None

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_handle"):
            self.close()

    def generate(self, text: str) -> TTSResult:
        """
        Generate speech audio from a text string.

        Grapheme-to-phoneme conversion is performed internally by the C
        library using the Phonemis G2P engine.

        Args:
            text: A UTF-8-encoded plain text string.

        Returns:
            A TTSResult containing the generated PCM audio data (int16 numpy
            array) and sample rate.

        Raises:
            MoonshineError: If the model is not initialized.
            MoonshineInvalidArgumentError: If text is empty or None.
        """
        if self._handle is None:
            raise MoonshineError("Text-to-speech model is not initialized")

        text_bytes = text.encode("utf-8")
        result_c = TtsResultC()

        error = self._lib.moonshine_text_to_speech_generate(
            self._handle,
            text_bytes,
            ctypes.byref(result_c),
        )
        check_error(error)

        # Copy the audio data into a numpy array.  The C library owns the
        # underlying buffer and may invalidate it on the next call.
        audio_length = result_c.audio_length
        audio_array = np.ctypeslib.as_array(
            result_c.audio_data, shape=(audio_length,)
        ).copy()

        return TTSResult(
            audio_data=audio_array,
            sample_rate=result_c.sample_rate,
        )
