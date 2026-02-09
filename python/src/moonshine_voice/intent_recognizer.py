"""Intent recognition module for Moonshine Voice.

This module provides intent recognition capabilities using semantic embeddings.
It can be used standalone or as a TranscriptEventListener to automatically
recognize intents from transcribed speech.
"""

import ctypes
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from moonshine_voice.moonshine_api import _MoonshineLib
from moonshine_voice.errors import MoonshineError, check_error
from moonshine_voice.transcriber import (
    TranscriptEventListener,
    LineCompleted,
    Error,
)
from moonshine_voice.download import EmbeddingModelArch


# Callback type for intent handlers
# Signature: (trigger_phrase: str, utterance: str, similarity: float) -> None
IntentHandler = Callable[[str, str, float], None]

# C callback function type
_INTENT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,  # user_data
    ctypes.c_char_p,  # trigger_phrase
    ctypes.c_char_p,  # utterance
    ctypes.c_float,  # similarity
)


@dataclass
class IntentMatch:
    """Represents a matched intent."""

    trigger_phrase: str
    utterance: str
    similarity: float


class IntentRecognizer(TranscriptEventListener):
    """Intent recognizer that uses semantic embeddings to match utterances.

    This class can be used standalone by calling process_utterance(), or as
    a TranscriptEventListener to automatically process completed transcript
    lines.

    Example standalone usage:
        >>> recognizer = IntentRecognizer("path/to/embedding-model")
        >>> recognizer.register_intent("turn on the lights", lambda t, u, s: print(f"Lights on! ({s:.2f})"))
        >>> recognizer.process_utterance("switch on the lights")
        Lights on! (0.92)

    Example as TranscriptEventListener:
        >>> recognizer = IntentRecognizer("path/to/embedding-model")
        >>> recognizer.register_intent("turn on the lights", lambda t, u, s: print(f"Lights on!"))
        >>> transcriber.add_listener(recognizer)  # Now intents are recognized automatically
    """

    def __init__(
        self,
        model_path: str,
        model_arch: EmbeddingModelArch = EmbeddingModelArch.GEMMA_300M,
        model_variant: str = "fp32",
        threshold: float = 0.7,
    ):
        """
        Initialize an intent recognizer.

        Args:
            model_path: Path to the directory containing the embedding model files
                       (ONNX model and tokenizer.bin).
            model_arch: The embedding model architecture to use.
                       Currently only GEMMA_300M is supported.
            model_variant: Model variant to load: "fp32", "fp16", "q8", "q4",
                          or "q4f16". Default is "q4" for efficiency.
            threshold: The minimum similarity threshold to trigger an intent
                      (default 0.7, range 0.0-1.0).
        """
        self._lib_wrapper = _MoonshineLib()
        self._lib = self._lib_wrapper.lib
        self._handle = None
        self._handlers: Dict[str, IntentHandler] = {}
        self._c_callbacks: Dict[str, _INTENT_CALLBACK] = {}
        self._on_intent_callback: Optional[Callable[[IntentMatch], None]] = None
        self._setup_function_signatures()

        # Create the intent recognizer
        model_path_bytes = model_path.encode("utf-8")
        model_variant_bytes = model_variant.encode("utf-8") if model_variant else None

        handle = self._lib.moonshine_create_intent_recognizer(
            model_path_bytes,
            model_arch.value,
            model_variant_bytes,
            threshold,
        )

        if handle < 0:
            raise MoonshineError(
                f"Failed to create intent recognizer from {model_path}"
            )

        self._handle = handle

    def _setup_function_signatures(self):
        """Setup ctypes function signatures for the intent recognizer C API."""
        lib = self._lib

        # Create intent recognizer
        lib.moonshine_create_intent_recognizer.restype = ctypes.c_int32
        lib.moonshine_create_intent_recognizer.argtypes = [
            ctypes.c_char_p,  # model_path
            ctypes.c_uint32,  # model_arch
            ctypes.c_char_p,  # model_variant
            ctypes.c_float,  # threshold
        ]

        # Free intent recognizer
        lib.moonshine_free_intent_recognizer.restype = None
        lib.moonshine_free_intent_recognizer.argtypes = [ctypes.c_int32]

        # Register intent
        lib.moonshine_register_intent.restype = ctypes.c_int32
        lib.moonshine_register_intent.argtypes = [
            ctypes.c_int32,  # intent_recognizer_handle
            ctypes.c_char_p,  # trigger_phrase
            _INTENT_CALLBACK,  # callback
            ctypes.c_void_p,  # user_data
        ]

        # Unregister intent
        lib.moonshine_unregister_intent.restype = ctypes.c_int32
        lib.moonshine_unregister_intent.argtypes = [
            ctypes.c_int32,  # intent_recognizer_handle
            ctypes.c_char_p,  # trigger_phrase
        ]

        # Process utterance
        lib.moonshine_process_utterance.restype = ctypes.c_int32
        lib.moonshine_process_utterance.argtypes = [
            ctypes.c_int32,  # intent_recognizer_handle
            ctypes.c_char_p,  # utterance
        ]

        # Set threshold
        lib.moonshine_set_intent_threshold.restype = ctypes.c_int32
        lib.moonshine_set_intent_threshold.argtypes = [
            ctypes.c_int32,  # intent_recognizer_handle
            ctypes.c_float,  # threshold
        ]

        # Get threshold
        lib.moonshine_get_intent_threshold.restype = ctypes.c_float
        lib.moonshine_get_intent_threshold.argtypes = [ctypes.c_int32]

        # Get intent count
        lib.moonshine_get_intent_count.restype = ctypes.c_int32
        lib.moonshine_get_intent_count.argtypes = [ctypes.c_int32]

        # Clear intents
        lib.moonshine_clear_intents.restype = ctypes.c_int32
        lib.moonshine_clear_intents.argtypes = [ctypes.c_int32]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Free the intent recognizer resources."""
        if self._handle is not None:
            self._lib.moonshine_free_intent_recognizer(self._handle)
            self._handle = None
            self._handlers.clear()
            self._c_callbacks.clear()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_handle"):
            self.close()

    def register_intent(self, trigger_phrase: str, handler: IntentHandler) -> None:
        """
        Register an intent with a trigger phrase and handler.

        When an utterance is processed that is similar enough to the trigger
        phrase (above the threshold), the handler will be invoked.

        Args:
            trigger_phrase: The phrase that triggers this intent.
            handler: A callable that takes (trigger_phrase, utterance, similarity)
                    and handles the recognized intent.
        """
        if self._handle is None:
            raise MoonshineError("Intent recognizer is not initialized")

        # Store the Python handler
        self._handlers[trigger_phrase] = handler

        # Create a C callback that invokes the Python handler
        def c_callback(user_data, c_trigger, c_utterance, similarity):
            trigger = c_trigger.decode("utf-8") if c_trigger else ""
            utterance = c_utterance.decode("utf-8") if c_utterance else ""
            if trigger in self._handlers:
                self._handlers[trigger](trigger, utterance, similarity)
            # Also call the general on_intent callback if set
            if self._on_intent_callback:
                match = IntentMatch(
                    trigger_phrase=trigger,
                    utterance=utterance,
                    similarity=similarity,
                )
                self._on_intent_callback(match)

        # Keep a reference to prevent garbage collection
        c_callback_func = _INTENT_CALLBACK(c_callback)
        self._c_callbacks[trigger_phrase] = c_callback_func

        trigger_bytes = trigger_phrase.encode("utf-8")
        error = self._lib.moonshine_register_intent(
            self._handle, trigger_bytes, c_callback_func, None
        )
        check_error(error)

    def unregister_intent(self, trigger_phrase: str) -> bool:
        """
        Remove a registered intent.

        Args:
            trigger_phrase: The trigger phrase of the intent to remove.

        Returns:
            True if the intent was found and removed, False otherwise.
        """
        if self._handle is None:
            raise MoonshineError("Intent recognizer is not initialized")

        trigger_bytes = trigger_phrase.encode("utf-8")
        result = self._lib.moonshine_unregister_intent(self._handle, trigger_bytes)

        if result == 0:
            self._handlers.pop(trigger_phrase, None)
            self._c_callbacks.pop(trigger_phrase, None)
            return True
        return False

    def process_utterance(self, utterance: str) -> bool:
        """
        Process an utterance and invoke the handler of the most similar intent.

        Args:
            utterance: The utterance to process.

        Returns:
            True if an intent was recognized and handler invoked, False otherwise.
        """
        if self._handle is None:
            raise MoonshineError("Intent recognizer is not initialized")

        utterance_bytes = utterance.encode("utf-8")
        result = self._lib.moonshine_process_utterance(self._handle, utterance_bytes)

        if result < 0:
            check_error(result)

        return result == 1

    @property
    def threshold(self) -> float:
        """Get the current similarity threshold."""
        if self._handle is None:
            raise MoonshineError("Intent recognizer is not initialized")
        return self._lib.moonshine_get_intent_threshold(self._handle)

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the similarity threshold."""
        if self._handle is None:
            raise MoonshineError("Intent recognizer is not initialized")
        error = self._lib.moonshine_set_intent_threshold(self._handle, value)
        check_error(error)

    @property
    def intent_count(self) -> int:
        """Get the number of registered intents."""
        if self._handle is None:
            raise MoonshineError("Intent recognizer is not initialized")
        count = self._lib.moonshine_get_intent_count(self._handle)
        if count < 0:
            check_error(count)
        return count

    def clear_intents(self) -> None:
        """Clear all registered intents."""
        if self._handle is None:
            raise MoonshineError("Intent recognizer is not initialized")
        error = self._lib.moonshine_clear_intents(self._handle)
        check_error(error)
        self._handlers.clear()
        self._c_callbacks.clear()

    def set_on_intent(self, callback: Optional[Callable[[IntentMatch], None]]) -> None:
        """
        Set a callback that is invoked for any recognized intent.

        This is useful when using the IntentRecognizer as a TranscriptEventListener,
        as it allows you to handle all intents in one place rather than
        registering individual handlers.

        Args:
            callback: A callable that takes an IntentMatch, or None to clear.
        """
        self._on_intent_callback = callback

    # TranscriptEventListener implementation

    def on_line_completed(self, event: LineCompleted) -> None:
        """
        Called when a transcription line is completed.

        This implements the TranscriptEventListener interface, allowing the
        IntentRecognizer to automatically process completed transcript lines.

        Args:
            event: The LineCompleted event containing the transcript line.
        """
        if event.line and event.line.text:
            # Strip whitespace and process non-empty utterances
            utterance = event.line.text.strip()
            if utterance:
                self.process_utterance(utterance)

    def on_error(self, event: Error) -> None:
        """
        Called when an error occurs.

        Args:
            event: The Error event.
        """
        # Log or handle errors as needed
        pass


if __name__ == "__main__":
    import argparse
    import sys
    from moonshine_voice import get_model_for_language, get_embedding_model
    from moonshine_voice import (
        MicTranscriber,
        Transcriber,
        load_wav_file,
        TranscriptEventListener,
    )
    import time

    parser = argparse.ArgumentParser(
        description="Intent recognition with Moonshine Voice"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language to use for transcription (default: en)",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default=None,
        help="Model architecture to use for transcription",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="embeddinggemma-300m",
        help="Embedding model name (default: embeddinggemma-300m)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for intent matching (default: 0.6)",
    )
    parser.add_argument(
        "--intents",
        type=str,
        default="turn on the lights,turn off the lights,what is the weather,set a timer,play some music,stop the music",
        help="Actions to handle, separated by commas",
    )
    parser.add_argument(
        "--wav-file",
        type=str,
        default=None,
        help="WAV file to process instead of using the microphone",
    )
    args = parser.parse_args()

    def on_intent_triggered_on(trigger: str, utterance: str, similarity: float):
        """Handler for when an intent is triggered."""
        print(
            f"\n'{trigger.upper()}' triggered by '{utterance}' with {similarity:.0%} confidence"
        )

    class TranscriptPrinter(TranscriptEventListener):
        """Listener that prints transcript updates to the terminal."""

        def __init__(self):
            self.last_line_text_length = 0

        def update_last_terminal_line(self, new_text: str):
            print(f"\r{new_text}", end="", flush=True)
            if len(new_text) < self.last_line_text_length:
                diff = self.last_line_text_length - len(new_text)
                print(f"{' ' * diff}", end="", flush=True)
            self.last_line_text_length = len(new_text)

        def on_line_started(self, event):
            self.last_line_text_length = 0

        def on_line_text_changed(self, event):
            self.update_last_terminal_line(f"📝 {event.line.text}")

        def on_line_completed(self, event):
            self.update_last_terminal_line(f"📝 {event.line.text}")
            print()  # New line after completion

    # Load the transcription model
    print("Loading transcription model...", file=sys.stderr)
    model_path, model_arch = get_model_for_language(args.language, args.model_arch)

    # Download and load the embedding model for intent recognition
    print(
        f"Loading embedding model ({args.embedding_model}, variant={args.quantization})...",
        file=sys.stderr,
    )
    embedding_model_path, embedding_model_arch = get_embedding_model(
        args.embedding_model, args.quantization
    )

    # Create the intent recognizer (implements TranscriptEventListener)
    print(
        f"Creating intent recognizer (threshold={args.threshold})...", file=sys.stderr
    )
    intent_recognizer = IntentRecognizer(
        model_path=embedding_model_path,
        model_arch=embedding_model_arch,
        model_variant=args.quantization,
        threshold=args.threshold,
    )

    # Register intents with their trigger phrases and handlers
    intents = args.intents.split(",")
    intents = [intent.strip() for intent in intents]
    if len(intents) == 0:
        raise ValueError("No intents provided")
    for intent in intents:
        intent_recognizer.register_intent(intent, on_intent_triggered_on)

    print(f"Registered {intent_recognizer.intent_count} intents", file=sys.stderr)

    # Create the microphone transcriber or use the WAV file if provided
    if args.wav_file is None:
        transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch)
    else:
        transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
        audio_data, sample_rate = load_wav_file(args.wav_file)

    # Add both the transcript printer and intent recognizer as listeners
    # The intent recognizer will process completed lines and trigger handlers
    transcript_printer = TranscriptPrinter()
    transcriber.add_listener(transcript_printer)
    transcriber.add_listener(intent_recognizer)

    print("\n" + "=" * 60, file=sys.stderr)
    print("🎤 Listening for voice commands...", file=sys.stderr)
    print("Try saying phrases with the same meaning as these actions:", file=sys.stderr)
    for intent in intents:
        print(f"  - '{intent}'", file=sys.stderr)
    print(
        "We're doing fuzzy matching of natural language, so phrases like 'Switch on the lights' or 'Lights on' or 'Let there be light' will trigger the 'turn on the lights' action, for example."
    )
    print("=" * 60, file=sys.stderr)
    print("Press Ctrl+C to stop.\n", file=sys.stderr)

    transcriber.start()
    try:
        # Use the microphone if no WAV file is provided
        if args.wav_file is None:
            while True:
                time.sleep(0.1)
        else:
            # Process the WAV file in chunks
            chunk_duration = 0.2
            chunk_size = int(chunk_duration * sample_rate)
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                transcriber.add_audio(chunk, sample_rate)
    except KeyboardInterrupt:
        print("\n\nStopping...", file=sys.stderr)
    finally:
        intent_recognizer.close()
        transcriber.stop()
        transcriber.close()
