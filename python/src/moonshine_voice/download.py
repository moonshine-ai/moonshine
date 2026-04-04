import json
import os
import sys
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import quote

from moonshine_voice.download_file import download_file, download_model, get_cache_dir
from moonshine_voice.moonshine_api import (
    ModelArch,
    moonshine_get_g2p_dependencies_string,
    moonshine_get_tts_dependencies_string,
)


# Define EmbeddingModelArch here to avoid circular import with intent_recognizer
class EmbeddingModelArch(IntEnum):
    """Supported embedding model architectures."""
    GEMMA_300M = 0  # embeddinggemma-300m (768-dim embeddings)


MODEL_INFO = {
    "ar": {
        "english_name": "Arabic",
        "models": [
            {
                "model_name": "base-ar",
                "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-ar/quantized/base-ar",
            }
        ],
    },
    "es": {
        "english_name": "Spanish",
        "models": [
            {
                "model_name": "base-es",
                "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-es/quantized/base-es",
            }
        ],
    },
    "en": {
        "english_name": "English",
        "models": [
            {
                "model_name": "medium-streaming-en",
                "model_arch": ModelArch.MEDIUM_STREAMING,
                "download_url": "https://download.moonshine.ai/model/medium-streaming-en/quantized",
            },
            {
                "model_name": "small-streaming-en",
                "model_arch": ModelArch.SMALL_STREAMING,
                "download_url": "https://download.moonshine.ai/model/small-streaming-en/quantized",
            },
            {
                "model_name": "base-en",
                "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-en/quantized/base-en",
            },
            {
                "model_name": "tiny-streaming-en",
                "model_arch": ModelArch.TINY_STREAMING,
                "download_url": "https://download.moonshine.ai/model/tiny-streaming-en/quantized",
            },
            {
                "model_name": "tiny-en",
                "model_arch": ModelArch.TINY,
                "download_url": "https://download.moonshine.ai/model/tiny-en/quantized/tiny-en",
            },
        ],
    },
    "ja": {
        "english_name": "Japanese",
        "models": [
            {
                "model_name": "base-ja",
                "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-ja/quantized/base-ja",
            },
            {
                "model_name": "tiny-ja",
                "model_arch": ModelArch.TINY,
                "download_url": "https://download.moonshine.ai/model/tiny-ja/quantized/tiny-ja",
            },
        ],
    },
    "ko": {
        "english_name": "Korean",
        "models": [
            {
                "model_name": "base-ko",
                "model_arch": ModelArch.TINY,
                "download_url": "https://download.moonshine.ai/model/tiny-ko/quantized/tiny-ko",
            }
        ],
    },
    "vi": {
        "english_name": "Vietnamese",
        "models": [
            {
                "model_name": "base-vi",
                "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-vi/quantized/base-vi",
            }
        ],
    },
    "uk": {
        "english_name": "Ukrainian",
        "models": [
            {
                "model_name": "base-uk",
                "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-uk/quantized/base-uk",
            }
        ],
    },
    "zh": {
        "english_name": "Chinese",
        "models": [
            {
                "model_name": "base-zh",
                "model_arch": ModelArch.BASE,
                "download_url": "https://download.moonshine.ai/model/base-zh/quantized/base-zh",
            }
        ],
    },
}

# Embedding models are stored separately since they use a different arch enum
# and have a different component structure (variants like q4, fp32, etc.)
EMBEDDING_MODEL_INFO = {
    "embeddinggemma-300m": {
        "english_name": "Embedding Gemma 300M",
        "model_arch": EmbeddingModelArch.GEMMA_300M,
        "download_url": "https://download.moonshine.ai/model/embeddinggemma-300m",
        "variants": ["q4", "q8", "fp16", "fp32", "q4f16"],
        "default_variant": "q4",
    },
}


def find_model_info(language: str = "en", model_arch: ModelArch = None) -> dict:
    if language in MODEL_INFO.keys():
        language_key = language
    else:
        language_key = None
        for key, info in MODEL_INFO.items():
            if language.lower() == info["english_name"].lower():
                language_key = key
                break
        if language_key is None:
            raise ValueError(
                f"Language not found: {language}. Supported languages: {supported_languages_friendly()}"
            )

    model_info = MODEL_INFO[language_key]
    available_models = model_info["models"]
    if model_arch is None:
        result = available_models[0]
        result["language"] = language_key
        return result
    for model in available_models:
        if model["model_arch"] == model_arch:
            model["language"] = language_key
            return model
    raise ValueError(
        f"Model not found for language: {language} and model arch: {model_arch}. Available models: {available_models}"
    )


def supported_languages_friendly() -> str:
    return ", ".join(
        [f"{key} ({info['english_name']})" for key, info in MODEL_INFO.items()]
    )


def supported_languages() -> list[str]:
    return list(MODEL_INFO.keys())


def get_components_for_model_info(model_info: dict) -> list[str]:
    model_arch = model_info["model_arch"]
    if model_arch in [
        ModelArch.TINY_STREAMING,
        ModelArch.BASE_STREAMING,
        ModelArch.SMALL_STREAMING,
        ModelArch.MEDIUM_STREAMING,
    ]:
        result = [
            "adapter.ort",
            "cross_kv.ort",
            "decoder_kv.ort",
            "encoder.ort",
            "frontend.ort",
            "streaming_config.json",
            "tokenizer.bin",
        ]
        if model_info["language"] == "en":
            result.append("decoder_kv_with_attention.ort")
    else:
        result = ["encoder_model.ort", "decoder_model_merged.ort", "tokenizer.bin"]
        if model_info["language"] == "en":
            result.append("decoder_with_attention.ort")

    return result


def download_model_from_info(model_info: dict) -> tuple[str, ModelArch]:
    cache_dir = get_cache_dir()
    model_download_url = model_info["download_url"]
    model_folder_name = model_download_url.replace("https://", "")
    root_model_path = os.path.join(cache_dir, model_folder_name)
    components = get_components_for_model_info(model_info)
    for component in components:
        component_download_url = f"{model_download_url}/{component}"
        component_path = os.path.join(root_model_path, component)
        download_model(component_download_url, component_path)
    return str(root_model_path), model_info["model_arch"]


# ============================================================================
# Embedding Model Functions
# ============================================================================


def supported_embedding_models() -> list[str]:
    """Return list of supported embedding model names."""
    return list(EMBEDDING_MODEL_INFO.keys())


def supported_embedding_models_friendly() -> str:
    """Return a friendly string listing supported embedding models."""
    return ", ".join(
        [f"{key} ({info['english_name']})" for key, info in EMBEDDING_MODEL_INFO.items()]
    )


def get_embedding_model_variants(model_name: str = "embeddinggemma-300m") -> list[str]:
    """Return list of available variants for an embedding model."""
    if model_name not in EMBEDDING_MODEL_INFO:
        raise ValueError(
            f"Embedding model not found: {model_name}. "
            f"Supported models: {supported_embedding_models_friendly()}"
        )
    return EMBEDDING_MODEL_INFO[model_name]["variants"]


def get_embedding_model(
    model_name: str = "embeddinggemma-300m",
    variant: str = "fp32",
) -> tuple[str, EmbeddingModelArch]:
    """
    Download an embedding model and return (path, arch).

    Args:
        model_name: Name of the embedding model (e.g., "embeddinggemma-300m")
        variant: Model variant - one of "q4", "q8", "fp16", "fp32", "q4f16".
                 If None, uses the default variant (q4).

    Returns:
        Tuple of (model_path, model_arch) for use with IntentRecognizer.

    Example:
        >>> model_path, model_arch = get_embedding_model("embeddinggemma-300m", "q4")
        >>> recognizer = IntentRecognizer(model_path=model_path, model_arch=model_arch)
    """
    if model_name not in EMBEDDING_MODEL_INFO:
        raise ValueError(
            f"Embedding model not found: {model_name}. "
            f"Supported models: {supported_embedding_models_friendly()}"
        )

    model_info = EMBEDDING_MODEL_INFO[model_name]

    if variant is None:
        variant = model_info["default_variant"]

    if variant not in model_info["variants"]:
        raise ValueError(
            f"Variant '{variant}' not available for {model_name}. "
            f"Available variants: {model_info['variants']}"
        )

    # Determine components based on variant
    if variant == "fp32":
        components = ["model.onnx", "tokenizer.bin", "model.onnx_data"]
    else:
        components = [f"model_{variant}.onnx", "tokenizer.bin", f"model_{variant}.onnx_data"]

    # Download the model
    cache_dir = get_cache_dir()
    download_url = model_info["download_url"]
    model_folder_name = download_url.replace("https://", "")
    root_model_path = os.path.join(cache_dir, model_folder_name)

    for component in components:
        component_download_url = f"{download_url}/{component}"
        component_path = os.path.join(root_model_path, component)
        download_model(component_download_url, component_path)

    return str(root_model_path), model_info["model_arch"]


# ============================================================================
# Transcription Model Functions
# ============================================================================


def get_model_for_language(
    wanted_language: str = "en", wanted_model_arch: ModelArch = None
) -> tuple[str, ModelArch]:
    model_info = find_model_info(wanted_language, wanted_model_arch)
    if wanted_language != "en":
        print(
            "Using a model released under the non-commercial Moonshine Community License. See https://www.moonshine.ai/license for details.",
            file=sys.stderr,
        )
    return download_model_from_info(model_info)


def log_model_info(
    wanted_language: str = "en", wanted_model_arch: ModelArch = None
) -> None:
    model_info = find_model_info(wanted_language, wanted_model_arch)
    model_root_path, model_arch = download_model_from_info(model_info)
    print(f"Model download url: {model_info['download_url']}")
    print(f"Model components: {get_components_for_model_info(model_info)}")
    print(f"Model arch: {model_arch}")
    print(f"Downloaded model path: {model_root_path}")


# ============================================================================
# TTS / G2P assets (moonshine_get_tts_dependencies / moonshine_get_g2p_dependencies)
# ============================================================================

TTS_CDN_BASE_URL = "https://download.moonshine.ai/tts/"


def normalize_moonshine_language_tag(language: str) -> str:
    """Normalize a user language tag to the form expected by the Moonshine C API (e.g. en_us)."""
    s = language.strip().lower().replace("-", "_").replace(" ", "_")
    return s


def _tts_asset_cache_root(override: Optional[Path] = None) -> Path:
    if override is not None:
        return Path(override)
    return get_cache_dir() / "download.moonshine.ai" / "tts"


def _merge_tts_query_options(
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    *,
    voice: Optional[str] = None,
    engine: Optional[str] = None,
    vocoder_engine: Optional[str] = None,
) -> Dict[str, Union[str, int, float, bool]]:
    """Options passed to ``moonshine_get_tts_dependencies`` (voice, vocoder_engine, path overrides, …)."""
    merged: Dict[str, Union[str, int, float, bool]] = dict(options) if options else {}
    if voice is not None:
        merged["voice"] = voice
    ve = vocoder_engine if vocoder_engine is not None else engine
    if ve is not None:
        merged["vocoder_engine"] = ve
    return merged


def is_downloadable_tts_asset_key(key: str) -> bool:
    """Return False for G2P override labels (no path) returned when custom options are set."""
    k = key.strip()
    if not k or "/" not in k:
        return False
    return True


def cdn_url_for_tts_asset_key(key: str) -> str:
    """HTTPS URL for a canonical asset key under ``TTS_CDN_BASE_URL``."""
    parts = key.strip().split("/")
    encoded = "/".join(quote(segment, safe="") for segment in parts)
    return f"{TTS_CDN_BASE_URL}{encoded}"


def list_tts_dependency_keys(
    languages: str,
    *,
    voice: Optional[str] = None,
    engine: Optional[str] = None,
    vocoder_engine: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> List[str]:
    """
    Resolve required TTS asset paths via the native ``moonshine_get_tts_dependencies`` API.

    ``languages`` may be a comma-separated list (e.g. ``\"en_us,de\"``), matching the C API.
    """
    lang_arg = languages.strip() if languages else ""
    opts = _merge_tts_query_options(
        options, voice=voice, engine=engine, vocoder_engine=vocoder_engine
    )
    raw = moonshine_get_tts_dependencies_string(
        lang_arg if lang_arg else None,
        opts if opts else None,
    )
    if not raw.strip():
        return []
    keys = json.loads(raw)
    if not isinstance(keys, list):
        raise ValueError("moonshine_get_tts_dependencies did not return a JSON array")
    return [str(k) for k in keys]


def list_g2p_dependency_keys(
    languages: str,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> List[str]:
    """Resolve G2P-only asset paths via ``moonshine_get_g2p_dependencies`` (comma-separated from C)."""
    lang_arg = languages.strip() if languages else ""
    csv = moonshine_get_g2p_dependencies_string(
        lang_arg if lang_arg else None,
        options if options else None,
    )
    if not csv.strip():
        return []
    return [k.strip() for k in csv.split(",") if k.strip()]


def download_tts_assets(
    language: str,
    *,
    voice: Optional[str] = None,
    engine: Optional[str] = None,
    vocoder_engine: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    cache_root: Optional[Path] = None,
    show_progress: bool = True,
) -> Path:
    """
    Download every file required for TTS for the given language (and voice/engine) into the cache.

    Files are stored under the same relative paths as canonical asset keys (e.g. ``en_us/dict.tsv``).
    Pass the returned directory as ``g2p_root`` when creating a synthesizer.
    """
    lang_tag = normalize_moonshine_language_tag(language)
    keys = list_tts_dependency_keys(
        lang_tag,
        voice=voice,
        engine=engine,
        vocoder_engine=vocoder_engine,
        options=options,
    )
    root = _tts_asset_cache_root(cache_root)
    for key in keys:
        if not is_downloadable_tts_asset_key(key):
            continue
        url = cdn_url_for_tts_asset_key(key)
        dest = root / key
        download_file(url, dest, show_progress=show_progress)
    return root.resolve()


def download_g2p_assets(
    language: str,
    *,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    cache_root: Optional[Path] = None,
    show_progress: bool = True,
) -> Path:
    """Download G2P lexicon/model files into the TTS asset cache layout (same CDN tree as TTS)."""
    lang_tag = normalize_moonshine_language_tag(language)
    keys = list_g2p_dependency_keys(lang_tag, options=options)
    root = _tts_asset_cache_root(cache_root)
    for key in keys:
        if not is_downloadable_tts_asset_key(key):
            continue
        url = cdn_url_for_tts_asset_key(key)
        dest = root / key
        download_file(url, dest, show_progress=show_progress)
    return root.resolve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Moonshine STT, TTS, or G2P assets"
    )
    parser.add_argument(
        "--language", type=str, default="en", help="Language (STT, TTS, or G2P tag, e.g. en_us)"
    )
    parser.add_argument(
        "--model-arch",
        type=int,
        default=None,
        help="Model architecture to use for transcription",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Download TTS assets from download.moonshine.ai/tts (uses native dependency API)",
    )
    parser.add_argument(
        "--g2p",
        action="store_true",
        help="Download G2P-only assets (lexicons, ONNX bundles, …)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="TTS voice id (e.g. Kokoro voice or Piper basename)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="TTS vocoder engine: typically kokoro or piper (maps to vocoder_engine)",
    )
    args = parser.parse_args()

    if args.tts:
        root = download_tts_assets(
            args.language, voice=args.voice, engine=args.engine
        )
        print(f"TTS assets root (use as g2p_root): {root}", file=sys.stderr)
        print(root)
    elif args.g2p:
        root = download_g2p_assets(args.language)
        print(f"G2P assets root (use as g2p_root): {root}", file=sys.stderr)
        print(root)
    else:
        get_model_for_language(args.language, args.model_arch)
        log_model_info(args.language, args.model_arch)
