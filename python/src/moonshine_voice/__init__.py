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
    model_arch_to_string,
    string_to_model_arch,
)

from moonshine_voice.download import (
    get_model_for_language,
    log_model_info,
    supported_languages,
    supported_languages_friendly,
    # Embedding model functions
    EmbeddingModelArch,
    get_embedding_model,
    supported_embedding_models,
    supported_embedding_models_friendly,
    get_embedding_model_variants,
)

from moonshine_voice.utils import (
    get_assets_path,
    get_model_path,
    load_wav_file,
)

__version__ = "0.1.0"

# Lazy imports to avoid RuntimeWarning when running modules as scripts
# These will be imported on first access via __getattr__
_transcriber_imported = False
_mic_transcriber_imported = False
_intent_recognizer_imported = False


def __getattr__(name):
    """Lazy import for transcriber, mic_transcriber, and intent_recognizer modules."""
    global _transcriber_imported, _mic_transcriber_imported, _intent_recognizer_imported

    # Lazy import transcriber module
    if name in (
        "Transcriber",
        "Stream",
        "TranscriptEventListener",
        "TranscriptEvent",
        "LineStarted",
        "LineUpdated",
        "LineTextChanged",
        "LineCompleted",
        "Error",
    ):
        if not _transcriber_imported:
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

            # Store in globals for this module
            globals()["Transcriber"] = Transcriber
            globals()["Stream"] = Stream
            globals()["TranscriptEventListener"] = TranscriptEventListener
            globals()["TranscriptEvent"] = TranscriptEvent
            globals()["LineStarted"] = LineStarted
            globals()["LineUpdated"] = LineUpdated
            globals()["LineTextChanged"] = LineTextChanged
            globals()["LineCompleted"] = LineCompleted
            globals()["Error"] = Error
            _transcriber_imported = True
        return globals()[name]

    # Lazy import mic_transcriber module
    if name == "MicTranscriber":
        if not _mic_transcriber_imported:
            from moonshine_voice.mic_transcriber import MicTranscriber

            globals()["MicTranscriber"] = MicTranscriber
            _mic_transcriber_imported = True
        return globals()[name]

    # Lazy import intent_recognizer module
    # Note: EmbeddingModelArch is now imported directly from download module above
    if name in ("IntentRecognizer", "IntentMatch"):
        if not _intent_recognizer_imported:
            from moonshine_voice.intent_recognizer import (
                IntentRecognizer,
                IntentMatch,
            )

            globals()["IntentRecognizer"] = IntentRecognizer
            globals()["IntentMatch"] = IntentMatch
            _intent_recognizer_imported = True
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "IntentRecognizer",
    "EmbeddingModelArch",
    "IntentMatch",
    "MoonshineError",
    "MoonshineUnknownError",
    "MoonshineInvalidHandleError",
    "MoonshineInvalidArgumentError",
    "get_assets_path",
    "get_model_path",
    "load_wav_file",
    "get_model_for_language",
    "log_model_info",
    "supported_languages",
    "supported_languages_friendly",
    "model_arch_to_string",
    "string_to_model_arch",
    # Embedding model functions
    "get_embedding_model",
    "supported_embedding_models",
    "supported_embedding_models_friendly",
    "get_embedding_model_variants",
]
