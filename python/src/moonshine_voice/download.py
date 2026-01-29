import os
import sys
from enum import IntEnum

from moonshine_voice.moonshine_api import ModelArch
from moonshine_voice.transcriber import Transcriber
from moonshine_voice.download_file import download_model, get_cache_dir
from moonshine_voice.utils import get_assets_path, load_wav_file


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
        return available_models[0]
    for model in available_models:
        if model["model_arch"] == model_arch:
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
        return [
            "adapter.ort",
            "cross_kv.ort",
            "decoder_kv.ort",
            "encoder.ort",
            "frontend.ort",
            "streaming_config.json",
            "tokenizer.bin",
        ]
    else:
        return ["encoder_model.ort", "decoder_model_merged.ort", "tokenizer.bin"]


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
        model_name: Name of the embedding model (e.g., "gemma-300m")
        variant: Model variant - one of "q4", "q8", "fp16", "fp32", "q4f16".
                 If None, uses the default variant (q4).

    Returns:
        Tuple of (model_path, model_arch) for use with IntentRecognizer.

    Example:
        >>> model_path, model_arch = get_embedding_model("gemma-300m", "q4")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model info example")
    parser.add_argument(
        "--language", type=str, default="en", help="Language to use for transcription"
    )
    parser.add_argument(
        "--model-arch",
        type=int,
        default=None,
        help="Model architecture to use for transcription",
    )
    args = parser.parse_args()

    model_path, model_arch = get_model_for_language(args.language, args.model_arch)

    log_model_info(args.language, args.model_arch)
