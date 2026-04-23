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
    MoonshineAudioOutputError,
    MoonshineTtsLanguageError,
    MoonshineTtsVoiceError,
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
    # TTS / G2P asset helpers
    TTS_CDN_BASE_URL,
    tts_asset_cache_path,
    cdn_url_for_tts_asset_key,
    download_g2p_assets,
    download_tts_assets,
    is_downloadable_tts_asset_key,
    list_g2p_dependency_keys,
    list_tts_dependency_keys,
    normalize_moonshine_language_tag,
    TtsVoiceEntry,
    TtsVoicesByAvailability,
    get_tts_voice_catalog,
    list_tts_languages,
    list_tts_voices,
    validate_tts_language,
    validate_tts_voice_downloaded,
    validate_tts_voice_known,
    ensure_tts_voice_downloaded,
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
_alphanumeric_listener_imported = False
_tts_imported = False
_g2p_imported = False
_dialog_flow_imported = False


def __getattr__(name):
    """Lazy import for transcriber, mic_transcriber, and intent_recognizer modules."""
    global _transcriber_imported, _mic_transcriber_imported, _intent_recognizer_imported
    global _alphanumeric_listener_imported, _tts_imported, _g2p_imported
    global _dialog_flow_imported

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

    # Lazy import TTS / G2P
    if name == "TextToSpeech":
        if not _tts_imported:
            from moonshine_voice.tts import TextToSpeech

            globals()["TextToSpeech"] = TextToSpeech
            _tts_imported = True
        return globals()[name]

    if name == "GraphemeToPhonemizer":
        if not _g2p_imported:
            from moonshine_voice.g2p import GraphemeToPhonemizer

            globals()["GraphemeToPhonemizer"] = GraphemeToPhonemizer
            _g2p_imported = True
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

    # Lazy import alphanumeric_listener module
    if name in (
        "AlphanumericListener",
        "AlphanumericEvent",
        "AlphanumericEventType",
        "AlphanumericMatch",
        "AlphanumericMatcher",
        "letters_only_matcher",
        "digits_only_matcher",
    ):
        if not _alphanumeric_listener_imported:
            from moonshine_voice.alphanumeric_listener import (
                AlphanumericListener,
                AlphanumericEvent,
                AlphanumericEventType,
                AlphanumericMatch,
                AlphanumericMatcher,
                letters_only_matcher,
                digits_only_matcher,
            )

            globals()["AlphanumericListener"] = AlphanumericListener
            globals()["AlphanumericEvent"] = AlphanumericEvent
            globals()["AlphanumericEventType"] = AlphanumericEventType
            globals()["AlphanumericMatch"] = AlphanumericMatch
            globals()["AlphanumericMatcher"] = AlphanumericMatcher
            globals()["letters_only_matcher"] = letters_only_matcher
            globals()["digits_only_matcher"] = digits_only_matcher
            _alphanumeric_listener_imported = True
        return globals()[name]

    # Lazy import cached_embeddings module
    if name in ("CachedEmbeddings", "default_cached_embeddings_path"):
        from moonshine_voice.cached_embeddings import (
            CachedEmbeddings,
            default_cached_embeddings_path,
        )

        globals()["CachedEmbeddings"] = CachedEmbeddings
        globals()["default_cached_embeddings_path"] = default_cached_embeddings_path
        return globals()[name]

    # Lazy import dialog_flow module
    if name in (
        "DialogFlow",
        "Dialog",
        "Prompt",
        "Say",
        "Ask",
        "Confirm",
        "Choose",
        "DialogError",
        "DialogCancelled",
        "DialogRestart",
        "NoInputError",
        "NoMatchError",
        "FREE",
        "SPELLED",
        "DIGITS",
        "PHRASE",
        "spell_out",
        "PhraseMatcher",
        "EmbeddingBackend",
    ):
        if not _dialog_flow_imported:
            from moonshine_voice.dialog_flow import (
                DialogFlow,
                Dialog,
                Prompt,
                Say,
                Ask,
                Confirm,
                Choose,
                DialogError,
                DialogCancelled,
                DialogRestart,
                NoInputError,
                NoMatchError,
                FREE,
                SPELLED,
                DIGITS,
                PHRASE,
                spell_out,
                PhraseMatcher,
                EmbeddingBackend,
            )

            globals()["DialogFlow"] = DialogFlow
            globals()["Dialog"] = Dialog
            globals()["Prompt"] = Prompt
            globals()["Say"] = Say
            globals()["Ask"] = Ask
            globals()["Confirm"] = Confirm
            globals()["Choose"] = Choose
            globals()["DialogError"] = DialogError
            globals()["DialogCancelled"] = DialogCancelled
            globals()["DialogRestart"] = DialogRestart
            globals()["NoInputError"] = NoInputError
            globals()["NoMatchError"] = NoMatchError
            globals()["FREE"] = FREE
            globals()["SPELLED"] = SPELLED
            globals()["DIGITS"] = DIGITS
            globals()["PHRASE"] = PHRASE
            globals()["spell_out"] = spell_out
            globals()["PhraseMatcher"] = PhraseMatcher
            globals()["EmbeddingBackend"] = EmbeddingBackend
            _dialog_flow_imported = True
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
    "MoonshineAudioOutputError",
    "MoonshineTtsLanguageError",
    "MoonshineTtsVoiceError",
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
    # TTS / G2P
    "TextToSpeech",
    "GraphemeToPhonemizer",
    "TTS_CDN_BASE_URL",
    "tts_asset_cache_path",
    "cdn_url_for_tts_asset_key",
    "download_tts_assets",
    "download_g2p_assets",
    "is_downloadable_tts_asset_key",
    "list_tts_dependency_keys",
    "list_g2p_dependency_keys",
    "normalize_moonshine_language_tag",
    "TtsVoiceEntry",
    "TtsVoicesByAvailability",
    "get_tts_voice_catalog",
    "list_tts_languages",
    "list_tts_voices",
    "validate_tts_language",
    "validate_tts_voice_downloaded",
    "validate_tts_voice_known",
    "ensure_tts_voice_downloaded",
    # Alphanumeric listener / matcher
    "AlphanumericListener",
    "AlphanumericEvent",
    "AlphanumericEventType",
    "AlphanumericMatch",
    "AlphanumericMatcher",
    "letters_only_matcher",
    "digits_only_matcher",
    # Dialog flow
    "DialogFlow",
    "Dialog",
    "Prompt",
    "Say",
    "Ask",
    "Confirm",
    "Choose",
    "DialogError",
    "DialogCancelled",
    "DialogRestart",
    "NoInputError",
    "NoMatchError",
    "FREE",
    "SPELLED",
    "DIGITS",
    "PHRASE",
    "spell_out",
    "PhraseMatcher",
    "EmbeddingBackend",
    # Cached embeddings
    "CachedEmbeddings",
    "default_cached_embeddings_path",
]
