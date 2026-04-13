import json
import os
import re
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Set, TypedDict, Union
from urllib.parse import quote

from moonshine_voice.download_file import download_file, download_model, get_cache_dir
from moonshine_voice.errors import (
    MoonshineError,
    MoonshineTtsLanguageError,
    MoonshineTtsVoiceError,
)
from moonshine_voice.moonshine_api import (
    MOONSHINE_ERROR_INVALID_ARGUMENT,
    MOONSHINE_ERROR_NONE,
    ModelArch,
    moonshine_get_g2p_dependencies_string,
    moonshine_get_tts_voices_string,
    moonshine_get_tts_dependencies_string,
    moonshine_try_get_tts_voices,
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


def download_model_from_info(
    model_info: dict, *, cache_root: Optional[Path] = None
) -> tuple[str, ModelArch]:
    cache_dir = Path(cache_root).resolve() if cache_root is not None else get_cache_dir()
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
    variant: str = "q4",
    *,
    cache_root: Optional[Path] = None,
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
    cache_dir = Path(cache_root).resolve() if cache_root is not None else get_cache_dir()
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
    wanted_language: str = "en",
    wanted_model_arch: ModelArch = None,
    *,
    cache_root: Optional[Path] = None,
) -> tuple[str, ModelArch]:
    model_info = find_model_info(wanted_language, wanted_model_arch)
    if wanted_language != "en":
        print(
            "Using a model released under the non-commercial Moonshine Community License. See https://www.moonshine.ai/license for details.",
            file=sys.stderr,
        )
    return download_model_from_info(model_info, cache_root=cache_root)


def log_model_info(
    wanted_language: str = "en",
    wanted_model_arch: ModelArch = None,
    *,
    cache_root: Optional[Path] = None,
) -> None:
    model_info = find_model_info(wanted_language, wanted_model_arch)
    model_root_path, model_arch = download_model_from_info(
        model_info, cache_root=cache_root
    )
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


def _normalize_tts_language_tag_display(tag: str) -> str:
    """Lowercase language tag with hyphens only (underscores and spaces → ``-``, collapse repeats)."""
    s = tag.strip().lower().replace("_", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def dedupe_tts_language_tags_for_display(tags: List[str]) -> List[str]:
    """Drop aliases that differ only by ``_`` vs ``-`` / spaces; return sorted hyphenated tags."""
    seen = {_normalize_tts_language_tag_display(t) for t in tags if t and str(t).strip()}
    seen.discard("")
    # Piper CLI alias; canonical tag is ``ko-kr`` (same as ``ko_kr`` / ``ko`` in the C catalog).
    if "korean" in seen:
        seen.discard("korean")
        seen.add("ko-kr")
    return sorted(seen)


def _tts_asset_cache_root(override: Optional[Path] = None) -> Path:
    if override is not None:
        return Path(override)
    return get_cache_dir() / "download.moonshine.ai" / "tts"


def tts_asset_cache_path(cache_root: Optional[Path] = None) -> Path:
    """Resolved directory for the TTS/G2P on-disk layout (same root `download_tts_assets` uses)."""
    return _tts_asset_cache_root(cache_root).resolve()


def _merge_tts_query_options(
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    *,
    voice: Optional[str] = None,
) -> Dict[str, Union[str, int, float, bool]]:
    """Options passed to ``moonshine_get_tts_dependencies`` (``voice`` with optional ``kokoro_`` / ``piper_`` prefix, path overrides, …)."""
    merged: Dict[str, Union[str, int, float, bool]] = dict(options) if options else {}
    if voice is not None:
        merged["voice"] = voice
    return merged


_TTS_G2P_ROOT_OPTION_KEYS = frozenset({"g2p_root", "path_root", "tts_root", "model_root"})


def _options_specify_asset_root(opts: Dict[str, Union[str, int, float, bool]]) -> bool:
    """True when the caller already set a C API root key (see ``MoonshineTTSOptions::parse_options``)."""
    for k in _TTS_G2P_ROOT_OPTION_KEYS:
        if k not in opts:
            continue
        v = opts[k]
        if isinstance(v, str) and v.strip():
            return True
    return False


def _voice_query_options(
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    *,
    voice: Optional[str] = None,
    root_path: Optional[Path] = None,
    g2p_root: Optional[Path] = None,
) -> Dict[str, Union[str, int, float, bool]]:
    """
    Options for ``moonshine_get_tts_voices`` (and related voice-list queries).

    When no asset root is set in ``options``, sets ``g2p_root`` so the native layer scans the
    same tree Python downloads use: ``<cache>/download.moonshine.ai/tts`` by default, or
    ``root_path`` / ``g2p_root`` when provided (``root_path`` wins if both are set).
    """
    merged = _merge_tts_query_options(options, voice=voice)
    if _options_specify_asset_root(merged):
        return merged
    if root_path is not None:
        merged["g2p_root"] = str(Path(root_path).resolve())
    elif g2p_root is not None:
        merged["g2p_root"] = str(Path(g2p_root).resolve())
    else:
        merged["g2p_root"] = str(tts_asset_cache_path())
    return merged


@dataclass(frozen=True)
class TtsVoiceEntry:
    """One TTS voice id and whether it is available under the asset root passed as ``g2p_root`` (native ``state``)."""

    id: str
    state: str  # "found" | "missing"


class TtsVoicesByAvailability(TypedDict):
    """Return shape of `list_tts_voices`: on-disk ids vs catalog ids not yet present."""

    present: List[str]
    downloadable: List[str]


def _entries_to_present_and_downloadable(entries: List[TtsVoiceEntry]) -> TtsVoicesByAvailability:
    present = sorted({e.id for e in entries if e.state == "found"})
    downloadable = sorted({e.id for e in entries if e.state == "missing"})
    return {"present": present, "downloadable": downloadable}


def _tts_voices_json_to_catalog(raw: str) -> Dict[str, List[TtsVoiceEntry]]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("moonshine_get_tts_voices did not return a JSON object")
    out: Dict[str, List[TtsVoiceEntry]] = {}
    for lang, arr in data.items():
        rows: List[TtsVoiceEntry] = []
        if isinstance(arr, list):
            for item in arr:
                if not isinstance(item, dict):
                    continue
                vid = item.get("id")
                st = item.get("state")
                if isinstance(vid, str) and isinstance(st, str):
                    rows.append(TtsVoiceEntry(id=vid, state=st))
        out[str(lang)] = rows
    return out


def list_tts_languages(
    *,
    voice: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    root_path: Optional[Path] = None,
    g2p_root: Optional[Path] = None,
) -> List[str]:
    """
    TTS language tags supported by the native catalog for the given path/voice options
    (same rules as ``moonshine_get_tts_voices`` with an empty language list).

    ``root_path`` defaults to the package TTS cache (``get_cache_dir()/download.moonshine.ai/tts``)
    when no ``g2p_root`` / ``path_root`` / ``tts_root`` / ``model_root`` is set in ``options``.
    ``g2p_root`` is a legacy alias for ``root_path``; if both are set, ``root_path`` is used.

    Tags are deduplicated and shown with hyphens (e.g. ``en-us``) for display; any alias still
    works when passed to the C API via `normalize_moonshine_language_tag`.
    """
    opts = _voice_query_options(options, voice=voice, root_path=root_path, g2p_root=g2p_root)
    raw = moonshine_get_tts_voices_string(None, opts)
    return dedupe_tts_language_tags_for_display(list(_tts_voices_json_to_catalog(raw).keys()))


def get_tts_voice_catalog(
    *,
    languages: Optional[str] = None,
    voice: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    root_path: Optional[Path] = None,
    g2p_root: Optional[Path] = None,
) -> Dict[str, List[TtsVoiceEntry]]:
    """
    Full map of language tag → voice entries (``id`` + ``state``: ``found`` or ``missing``).

    By default, ``g2p_root`` is sent to the C API as the package TTS cache so ``found`` matches
    `download_tts_assets` / `ensure_tts_voice_downloaded`. Override with ``root_path`` or any of
    ``g2p_root`` / ``path_root`` / ``tts_root`` / ``model_root`` in ``options``.
    ``g2p_root`` (keyword) is a legacy alias for ``root_path`` when neither appears in ``options``.
    ``languages`` may be comma-separated or ``None`` / ``\"\"`` for all catalog languages.
    """
    opts = _voice_query_options(options, voice=voice, root_path=root_path, g2p_root=g2p_root)
    lang_arg = languages.strip() if isinstance(languages, str) and languages.strip() else None
    raw = moonshine_get_tts_voices_string(lang_arg, opts)
    return _tts_voices_json_to_catalog(raw)


def list_tts_voices(
    language: str,
    *,
    voice: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    root_path: Optional[Path] = None,
    g2p_root: Optional[Path] = None,
) -> TtsVoicesByAvailability:
    """
    Voice ids for one language, split by on-disk availability (raises `MoonshineTtsLanguageError`
    if ``language`` is unknown).

    Returns ``{"present": [...], "downloadable": [...]}``: voice names sorted alphabetically.
    ``present`` maps to the native ``found`` state; ``downloadable`` to ``missing`` (in catalog
    but not under the resolved asset root).

    ``root_path`` defaults to the package TTS cache when no asset root is set in ``options``;
    the value is passed through as the C option ``g2p_root``. ``g2p_root`` (keyword) is a legacy
    alias for ``root_path``.
    """
    lang_tag = normalize_moonshine_language_tag(language)
    opts = _voice_query_options(options, voice=voice, root_path=root_path, g2p_root=g2p_root)
    err, raw = moonshine_try_get_tts_voices(lang_tag, opts)
    if err == MOONSHINE_ERROR_INVALID_ARGUMENT:
        alts = list_tts_languages(
            voice=voice, options=options, root_path=root_path, g2p_root=g2p_root
        )
        raise MoonshineTtsLanguageError(lang_tag, alts)
    if err != MOONSHINE_ERROR_NONE:
        raise MoonshineError(f"moonshine_get_tts_voices failed ({err})")
    cat = _tts_voices_json_to_catalog(raw)
    if lang_tag not in cat:
        alts = list_tts_languages(
            voice=voice, options=options, root_path=root_path, g2p_root=g2p_root
        )
        raise MoonshineTtsLanguageError(lang_tag, alts)
    return _entries_to_present_and_downloadable(cat[lang_tag])


def validate_tts_language(
    language: str,
    *,
    voice: Optional[str] = None,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    root_path: Optional[Path] = None,
    g2p_root: Optional[Path] = None,
) -> str:
    """
    Normalize and validate a TTS language tag against the native catalog.

    Uses the same default asset root as `list_tts_voices` (TTS cache unless overridden).

    Raises `MoonshineTtsLanguageError` with ``alternatives`` when the tag is unknown or has no TTS layout.
    """
    lang_tag = normalize_moonshine_language_tag(language)
    opts = _voice_query_options(options, voice=voice, root_path=root_path, g2p_root=g2p_root)
    err, _ = moonshine_try_get_tts_voices(lang_tag, opts)
    if err == MOONSHINE_ERROR_INVALID_ARGUMENT:
        alts = list_tts_languages(
            voice=voice, options=options, root_path=root_path, g2p_root=g2p_root
        )
        raise MoonshineTtsLanguageError(lang_tag, alts)
    if err != MOONSHINE_ERROR_NONE:
        raise MoonshineError(f"moonshine_get_tts_voices failed ({err})")
    return lang_tag


def _normalize_tts_voice_stem(stem: str) -> str:
    t = stem.strip()
    low = t.lower()
    if low.endswith(".onnx"):
        return t[: -len(".onnx")].strip()
    if low.endswith(".kokorovoice"):
        return t[: -len(".kokorovoice")].strip()
    return t


def _tts_voice_want_aliases(voice: str) -> Set[str]:
    """Normalized ids the user may mean (prefixed catalog id vs bare stem)."""
    want = _normalize_tts_voice_stem(voice)
    aliases = {want}
    low = want.lower()
    if low and not low.startswith("kokoro_") and not low.startswith("piper_"):
        aliases.add(_normalize_tts_voice_stem(f"kokoro_{want}"))
        aliases.add(_normalize_tts_voice_stem(f"piper_{want}"))
    return aliases


def validate_tts_voice_downloaded(
    language: str,
    voice: str,
    asset_root: Path,
    *,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> None:
    """
    Ensure ``voice`` is present (native ``state`` ``found``) for ``language`` under ``asset_root``.

    Raises `MoonshineTtsVoiceError` with downloaded voice ids as ``alternatives`` and catalog
    ``missing`` voice ids as ``alternatives_available_for_download`` when the voice is not on disk.
    """
    lang_tag = normalize_moonshine_language_tag(language)
    want_aliases = _tts_voice_want_aliases(voice)
    qopts: Dict[str, Union[str, int, float, bool]] = dict(options) if options else {}
    qopts.pop("voice", None)
    qopts["g2p_root"] = str(Path(asset_root).resolve())
    by_avail = list_tts_voices(lang_tag, options=qopts)
    found_ids = by_avail["present"]
    missing_ids = by_avail["downloadable"]
    found_norm = {_normalize_tts_voice_stem(x) for x in found_ids}
    if want_aliases & found_norm:
        return
    raise MoonshineTtsVoiceError(
        voice,
        lang_tag,
        sorted(found_ids),
        alternatives_available_for_download=missing_ids,
    )


def ensure_tts_voice_downloaded(
    language: str,
    voice: str,
    asset_root: Path,
    *,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    download_missing: bool = False,
    show_progress: bool = True,
) -> None:
    """
    Ensure ``voice`` is on disk under ``asset_root``, like `validate_tts_voice_downloaded`.

    When ``download_missing`` is True and validation fails only because the voice is absent while the
    native catalog still lists it under ``missing`` (fetchable), downloads the TTS dependency keys for
    that language and voice from the CDN into ``asset_root``, then validates again.
    """
    try:
        validate_tts_voice_downloaded(language, voice, asset_root, options=options)
        return
    except MoonshineTtsVoiceError as e:
        if not download_missing:
            raise
        want_aliases = _tts_voice_want_aliases(voice)
        downloadable_norm = {
            _normalize_tts_voice_stem(x) for x in (e.alternatives_available_for_download or [])
        }
        if not (want_aliases & downloadable_norm):
            raise

    lang_tag = normalize_moonshine_language_tag(language)
    root = Path(asset_root).resolve()
    opts: Dict[str, Union[str, int, float, bool]] = dict(options) if options else {}
    opts["g2p_root"] = str(root)
    keys = list_tts_dependency_keys(lang_tag, voice=voice, options=opts)
    for key in keys:
        if not is_downloadable_tts_asset_key(key):
            continue
        url = cdn_url_for_tts_asset_key(key)
        dest = root / key
        download_file(url, dest, show_progress=show_progress)
    validate_tts_voice_downloaded(language, voice, asset_root, options=options)


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
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> List[str]:
    """
    Resolve required TTS asset paths via the native ``moonshine_get_tts_dependencies`` API.

    ``languages`` may be a comma-separated list (e.g. ``\"en_us,de\"``), matching the C API.
    """
    lang_arg = languages.strip() if languages else ""
    opts = _merge_tts_query_options(options, voice=voice)
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
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    cache_root: Optional[Path] = None,
    show_progress: bool = True,
) -> Path:
    """
    Download every file required for TTS for the given language (and optional prefixed ``voice``) into the cache.

    Files are stored under the same relative paths as canonical asset keys (e.g. ``en_us/dict.tsv``).
    Pass the returned directory as ``g2p_root`` when creating a synthesizer.
    """
    if voice is not None:
        vs = str(voice).strip()
        voice = vs if vs else None
    root = tts_asset_cache_path(cache_root)
    lang_tag = validate_tts_language(
        language,
        voice=voice,
        options=options,
        root_path=root,
    )
    # Only include the voice in dependency resolution when the catalog recognises it,
    # so we never attempt to download a non-existent voice file (HTTP 404).
    download_voice = voice
    if voice is not None:
        try:
            by_avail = list_tts_voices(lang_tag, voice=voice, options=options, root_path=root)
            want = _tts_voice_want_aliases(voice)
            known = {_normalize_tts_voice_stem(v) for v in by_avail["present"]} | {
                _normalize_tts_voice_stem(v) for v in by_avail["downloadable"]
            }
            if not (want & known):
                download_voice = None
        except MoonshineTtsLanguageError:
            download_voice = None
    dep_opts = _merge_tts_query_options(options, voice=download_voice)
    if not _options_specify_asset_root(dep_opts):
        dep_opts = dict(dep_opts)
        dep_opts["g2p_root"] = str(root)
    keys = list_tts_dependency_keys(
        lang_tag,
        voice=download_voice,
        options=dep_opts,
    )
    for key in keys:
        if not is_downloadable_tts_asset_key(key):
            continue
        url = cdn_url_for_tts_asset_key(key)
        dest = root / key
        download_file(url, dest, show_progress=show_progress)
    return root


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
    root = tts_asset_cache_path(cache_root)
    for key in keys:
        if not is_downloadable_tts_asset_key(key):
            continue
        url = cdn_url_for_tts_asset_key(key)
        dest = root / key
        download_file(url, dest, show_progress=show_progress)
    return root


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
        "--intent",
        action="store_true",
        help="Download intent recognition assets (embedding model)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="TTS voice: kokoro_* or piper_* prefix plus id/stem (e.g. kokoro_af_heart)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Download under DIR with the same relative layout as the default cache "
            "(STT/embedding: DIR/download.moonshine.ai/model/...; TTS/G2P: DIR is the asset root "
            "with language subdirs, e.g. en_us/...)"
        ),
    )
    args = parser.parse_args()

    dl_root: Optional[Path] = args.root

    if args.tts:
        root = download_tts_assets(
            args.language, voice=args.voice, cache_root=dl_root
        )
        print(f"TTS assets root (use as g2p_root): {root}", file=sys.stderr)
        print(root)
    elif args.g2p:
        root = download_g2p_assets(args.language, cache_root=dl_root)
        print(f"G2P assets root (use as g2p_root): {root}", file=sys.stderr)
        print(root)
    elif args.intent:
        model_path, model_arch = get_embedding_model(cache_root=dl_root)
        print(f"Embedding model path: {model_path}", file=sys.stderr)
        print(model_path)
    else:
        get_model_for_language(args.language, args.model_arch, cache_root=dl_root)
        log_model_info(args.language, args.model_arch, cache_root=dl_root)
