import functools
import json
import os
import re
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, TypedDict, Union
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
    moonshine_get_diarization_dependencies_string,
    moonshine_get_embedding_catalog_string,
    moonshine_get_g2p_dependencies_string,
    moonshine_get_embedding_dependencies_string,
    moonshine_get_stt_catalog_string,
    moonshine_get_stt_dependencies_string,
    moonshine_get_tts_voices_string,
    moonshine_get_tts_dependencies_string,
    moonshine_try_get_tts_voices,
)


# Define EmbeddingModelArch here to avoid circular import with embedding_model
class EmbeddingModelArch(IntEnum):
    """Supported embedding model architectures."""
    GEMMA_300M = 0  # embeddinggemma-300m (768-dim embeddings)


# The speech-to-text and embedding model tables used to be duplicated here. They
# now come from the single source of truth in the C catalog, exposed via
# moonshine_get_stt_catalog / moonshine_get_embedding_catalog, and the per-file
# download list (URLs, sizes, CRC32C checksums) comes from
# moonshine_get_stt_dependencies / moonshine_get_embedding_dependencies. Nothing in
# this file hardcodes filenames, URLs, or sizes anymore.


@functools.lru_cache(maxsize=1)
def _stt_catalog() -> Dict[str, dict]:
    """Language code -> {english_name, models:[{model_arch, download_url, is_default}]}.

    Built from the native STT catalog so this module never keeps its own copy of
    the language/model tables.
    """
    raw = json.loads(moonshine_get_stt_catalog_string())
    catalog: Dict[str, dict] = {}
    for language in raw.get("languages", []):
        models = [
            {
                "model_arch": ModelArch(model["model_arch"]),
                "download_url": model["download_url"],
                "is_default": bool(model.get("is_default")),
            }
            for model in language.get("models", [])
        ]
        catalog[language["code"]] = {
            "english_name": language["english_name"],
            "models": models,
        }
    return catalog


@functools.lru_cache(maxsize=1)
def _embedding_catalog() -> Dict[str, dict]:
    """Embedding model name -> {english_name, model_arch, download_url, variants,
    default_variant}, built from the native embedding catalog."""
    raw = json.loads(moonshine_get_embedding_catalog_string())
    catalog: Dict[str, dict] = {}
    for model in raw.get("models", []):
        catalog[model["name"]] = {
            "english_name": model["english_name"],
            # Only one embedding architecture exists today.
            "model_arch": EmbeddingModelArch.GEMMA_300M,
            "download_url": model["download_url"],
            "variants": list(model.get("variants", [])),
            "default_variant": model.get("default_variant", ""),
        }
    return catalog


def _cache_root_dir(cache_root: Optional[Path]) -> Path:
    return Path(cache_root).resolve() if cache_root is not None else get_cache_dir()


# Reports how far a download has got, as a fraction from 0 to 1 plus the file
# currently being fetched. Matches Swift's `(Double, String)` handler and Java's
# ProgressCallback, so the same handler reads the same way in all four bindings.
ProgressCallback = Callable[[float, str], None]

# Smallest change in the fraction worth another call into user code. Chunks are
# 8KB, so a large model would otherwise fire tens of thousands of callbacks.
_PROGRESS_EPSILON = 0.002


class _ProgressTracker:
    """Turns per-file byte deltas into one monotonic ``0..1`` fraction.

    When the caller knows the total size up front (the STT manifests declare a
    size per file) the fraction is byte-weighted. When it does not (the TTS and
    G2P dependency lists are bare asset keys with no sizes) it falls back to
    counting files, which is coarser but still monotonic and still ends at 1.
    """

    def __init__(
        self,
        callback: ProgressCallback,
        *,
        total_bytes: int = 0,
        total_files: int = 0,
    ) -> None:
        self._callback = callback
        self._total_bytes = max(total_bytes, 0)
        self._total_files = max(total_files, 1)
        self._done_bytes = 0
        self._done_files = 0
        self._current_bytes = 0
        self._current_size = 0
        self._name = ""
        self._last_reported = -1.0

    @classmethod
    def for_groups(
        cls, callback: ProgressCallback, groups: List[dict]
    ) -> "_ProgressTracker":
        """Builds a tracker sized for every file across ``groups``."""
        files = [f for group in groups for f in group.get("files", [])]
        total = 0
        for file_info in files:
            size = file_info.get("size")
            if isinstance(size, int) and size > 0:
                total += size
            else:
                # One unsized file makes the byte total a lie, so fall back to
                # counting files rather than reporting a fraction that jumps.
                total = 0
                break
        return cls(callback, total_bytes=total, total_files=len(files))

    def start(self, name: str, size: Optional[int] = None) -> None:
        self._name = name
        self._current_bytes = 0
        self._current_size = size if isinstance(size, int) and size > 0 else 0
        self._emit(force=True)

    def add_bytes(self, count: int) -> None:
        self._current_bytes += count
        if self._current_size:
            self._current_bytes = min(self._current_bytes, self._current_size)
        self._emit()

    def finish(self) -> None:
        self._done_files += 1
        self._done_bytes += self._current_size or self._current_bytes
        self._current_bytes = 0
        self._current_size = 0
        self._emit(force=True)

    def _emit(self, force: bool = False) -> None:
        if self._total_bytes > 0:
            fraction = (self._done_bytes + self._current_bytes) / self._total_bytes
        else:
            fraction = self._done_files / self._total_files
        fraction = min(max(fraction, 0.0), 1.0)
        if not force and abs(fraction - self._last_reported) < _PROGRESS_EPSILON:
            return
        self._last_reported = fraction
        self._callback(fraction, self._name)


def _download_manifest_group(
    group: dict, cache_dir: Path, tracker: Optional[_ProgressTracker] = None
) -> str:
    """Downloads every file in one manifest group (each file object carries a
    fully-qualified ``url`` plus optional ``size`` / ``checksum``), verifying
    size and CRC32C, and returns the group's local root directory
    (``cache_dir`` / ``base_url`` without its scheme)."""
    base_url = group["base_url"]
    root = os.path.join(cache_dir, base_url.replace("https://", ""))
    for file_info in group.get("files", []):
        name = file_info["name"]
        url = file_info.get("url") or f"{base_url}/{name}"
        size = file_info.get("size")
        checksum = file_info.get("checksum") or None
        checksum_type = file_info.get("checksum_type") or ""
        dest = os.path.join(root, name)
        if tracker is not None:
            tracker.start(name, size)
        download_model(
            url,
            dest,
            expected_size=size if isinstance(size, int) and size >= 0 else None,
            expected_crc32c=checksum if checksum_type == "crc32c" else None,
            # An application drawing its own progress bar does not also want
            # tqdm scribbling on stderr underneath it.
            show_progress=tracker is None,
            on_bytes=tracker.add_bytes if tracker is not None else None,
        )
        if tracker is not None:
            tracker.finish()
    return str(root)


def _stt_dependency_manifest(
    language: str, model_arch: Optional[ModelArch], *, include_spelling: bool = False
) -> dict:
    """Fetches and parses the native STT download manifest for a language+arch."""
    options: Dict[str, Union[str, int, float, bool]] = {}
    if model_arch is not None:
        options["model_arch"] = int(model_arch)
    if include_spelling:
        options["include_spelling"] = True
    return json.loads(moonshine_get_stt_dependencies_string(language, options))


def find_model_info(language: str = "en", model_arch: ModelArch = None) -> dict:
    """Resolves a language (code or English name) and optional arch to a model
    entry ``{model_arch, download_url, language}`` from the native catalog."""
    catalog = _stt_catalog()
    if language in catalog:
        language_key = language
    else:
        language_key = None
        for key, info in catalog.items():
            if language.lower() == info["english_name"].lower():
                language_key = key
                break
        if language_key is None:
            raise ValueError(
                f"Language not found: {language}. Supported languages: {supported_languages_friendly()}"
            )

    available_models = catalog[language_key]["models"]
    if model_arch is None:
        result = dict(available_models[0])
        result["language"] = language_key
        return result
    for model in available_models:
        if model["model_arch"] == model_arch:
            result = dict(model)
            result["language"] = language_key
            return result
    raise ValueError(
        f"Model not found for language: {language} and model arch: {model_arch}. Available models: {available_models}"
    )


def supported_languages_friendly() -> str:
    return ", ".join(
        [f"{key} ({info['english_name']})" for key, info in _stt_catalog().items()]
    )


def supported_languages() -> list[str]:
    return list(_stt_catalog().keys())


def get_components_for_model_info(model_info: dict) -> list[str]:
    """Returns the canonical filenames for a model, sourced from the native
    dependency manifest (so streaming vs non-streaming layouts are never
    hardcoded here)."""
    manifest = _stt_dependency_manifest(
        model_info["language"], model_info["model_arch"]
    )
    result: list[str] = []
    for group in manifest.get("groups", []):
        for file_info in group.get("files", []):
            result.append(file_info["name"])
    return result


def _primary_stt_group(model_info: dict) -> dict:
    """The manifest group holding the model itself, which is always the first.

    Its files carry fully-qualified URLs plus expected size / CRC32C, which
    ``_download_manifest_group`` verifies.
    """
    manifest = _stt_dependency_manifest(
        model_info["language"], model_info["model_arch"]
    )
    groups = manifest.get("groups", [])
    if not groups:
        raise MoonshineError(
            f"Empty download manifest for {model_info['download_url']}"
        )
    return groups[0]


def download_model_from_info(
    model_info: dict,
    *,
    cache_root: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
    tracker: Optional[_ProgressTracker] = None,
) -> tuple[str, ModelArch]:
    """Downloads the model named by ``model_info`` and returns its root path.

    ``tracker`` is for callers spanning several downloads under one percentage
    (see :func:`get_model_for_language`); pass ``on_progress`` instead to get a
    fraction covering just this model.
    """
    group = _primary_stt_group(model_info)
    if tracker is None and on_progress is not None:
        tracker = _ProgressTracker.for_groups(on_progress, [group])
    root_model_path = _download_manifest_group(
        group, _cache_root_dir(cache_root), tracker
    )
    return root_model_path, model_info["model_arch"]


# ============================================================================
# Embedding Model Functions
# ============================================================================


def supported_embedding_models() -> list[str]:
    """Return list of supported embedding model names."""
    return list(_embedding_catalog().keys())


def supported_embedding_models_friendly() -> str:
    """Return a friendly string listing supported embedding models."""
    return ", ".join(
        [f"{key} ({info['english_name']})" for key, info in _embedding_catalog().items()]
    )


def get_embedding_model_variants(model_name: str = "embeddinggemma-300m") -> list[str]:
    """Return list of available variants for an embedding model."""
    catalog = _embedding_catalog()
    if model_name not in catalog:
        raise ValueError(
            f"Embedding model not found: {model_name}. "
            f"Supported models: {supported_embedding_models_friendly()}"
        )
    return list(catalog[model_name]["variants"])


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
        Tuple of (model_path, model_arch).

    DialogFlow downloads this model on demand, so calling this directly is only
    needed to warm the cache ahead of time (e.g. before going offline).

    Example:
        >>> model_path, model_arch = get_embedding_model("embeddinggemma-300m", "q4")
    """
    catalog = _embedding_catalog()
    if model_name not in catalog:
        raise ValueError(
            f"Embedding model not found: {model_name}. "
            f"Supported models: {supported_embedding_models_friendly()}"
        )

    model_info = catalog[model_name]

    if variant is None:
        variant = model_info["default_variant"]

    if variant not in model_info["variants"]:
        raise ValueError(
            f"Variant '{variant}' not available for {model_name}. "
            f"Available variants: {model_info['variants']}"
        )

    # The file list (single all-in-one ``.ort`` plus ``tokenizer.bin``), their
    # URLs, sizes, and CRC32C checksums all come from the native manifest.
    manifest = json.loads(
        moonshine_get_embedding_dependencies_string(model_name, {"variant": variant})
    )
    groups = manifest.get("groups", [])
    if not groups:
        raise MoonshineError(
            f"Empty download manifest for embedding model {model_name} ({variant})"
        )
    cache_dir = _cache_root_dir(cache_root)
    root_model_path = _download_manifest_group(groups[0], cache_dir)
    return root_model_path, model_info["model_arch"]


def get_diarization_model(*, cache_root: Optional[Path] = None) -> str:
    """
    Download the speaker diarization models and return the directory holding them.

    Pass that directory to a transcriber as the ``diarization_model_dir`` option
    whenever you set ``identify_speakers``. MicTranscriber does this for you, so
    calling it directly is only needed to warm the cache ahead of time (e.g.
    before going offline).

    These two files were compiled into the library before version 26.8, when
    they became a download; they are 8.2 MB together and only some callers
    diarize. See ``docs/diarization-models.md``.

    Returns:
        Path to the directory containing ``segmentation.ort`` and
        ``embedding.ort``.

    Example:
        >>> model_dir = get_diarization_model()
    """
    manifest = json.loads(moonshine_get_diarization_dependencies_string())
    groups = manifest.get("groups", [])
    if not groups:
        raise MoonshineError("Empty download manifest for the diarization models")
    cache_dir = _cache_root_dir(cache_root)
    return _download_manifest_group(groups[0], cache_dir)


# ============================================================================
# Transcription Model Functions
# ============================================================================


def _spelling_group_from_manifest(manifest: dict) -> Optional[dict]:
    """Returns the spelling-model group from an ``include_spelling`` STT
    manifest (the group whose base URL/files identify the alphanumeric spelling
    model), or ``None`` when the manifest has no spelling group."""
    for group in manifest.get("groups", []):
        base_url = group.get("base_url", "")
        if "/spelling" in base_url or any(
            file_info.get("name", "").startswith("spelling")
            for file_info in group.get("files", [])
        ):
            return group
    return None


def _spelling_group_for_language(language: str) -> Optional[dict]:
    """The spelling-model manifest group for ``language``, or ``None`` when the
    language has no spelling model (or is not a language we know)."""
    try:
        manifest = _stt_dependency_manifest(language, None, include_spelling=True)
    except Exception:
        # Unknown language, etc. No spelling model to fetch.
        return None
    return _spelling_group_from_manifest(manifest)


def download_spelling_model_for_language(
    language: str = "en",
    *,
    cache_root: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
    tracker: Optional[_ProgressTracker] = None,
) -> Optional[str]:
    """
    Download the alphanumeric spelling model for ``language`` if one is
    available, and return the path to the ``spelling_cnn.ort`` file.

    Returns ``None`` (without an error) when no spelling model is
    published for the requested language. Already-downloaded files are
    left alone, matching :func:`download_model_from_info`. The file list and
    URLs come from the native STT manifest (``include_spelling``), so nothing is
    hardcoded here.
    """
    group = _spelling_group_for_language(language)
    if group is None:
        return None
    if tracker is None and on_progress is not None:
        tracker = _ProgressTracker.for_groups(on_progress, [group])
    root = _download_manifest_group(group, _cache_root_dir(cache_root), tracker)
    for file_info in group.get("files", []):
        name = file_info["name"]
        if name.endswith(".ort"):
            return os.path.join(root, name)
    return None


def get_spelling_model_path(
    language: str = "en",
    *,
    cache_root: Optional[Path] = None,
) -> Optional[str]:
    """
    Return the on-disk path to the spelling model for ``language``.

    Wraps :func:`download_spelling_model_for_language` so callers that
    just want a path don't have to know the URL layout. Returns
    ``None`` when no spelling model is published for the language.
    """
    return download_spelling_model_for_language(language, cache_root=cache_root)


def get_model_for_language(
    wanted_language: str = "en",
    wanted_model_arch: ModelArch = None,
    *,
    cache_root: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> tuple[str, ModelArch]:
    """Downloads the transcription model for a language and returns
    ``(model_root_path, model_arch)``.

    ``on_progress`` is called with a ``0..1`` fraction and the file being
    fetched. The fraction spans the spelling model as well as the transcriber,
    so it rises once to 1 rather than filling twice. Supplying it also silences
    the default tqdm bars, on the assumption that you are drawing your own.
    """
    model_info = find_model_info(wanted_language, wanted_model_arch)
    if wanted_language != "en":
        print(
            "Using a model released under the non-commercial Moonshine Community License. See https://www.moonshine.ai/license for details.",
            file=sys.stderr,
        )

    tracker = None
    if on_progress is not None:
        groups = [_primary_stt_group(model_info)]
        spelling = _spelling_group_for_language(model_info["language"])
        if spelling is not None:
            groups.append(spelling)
        tracker = _ProgressTracker.for_groups(on_progress, groups)

    result = download_model_from_info(
        model_info, cache_root=cache_root, tracker=tracker
    )
    # Best-effort: pre-fetch the alphanumeric spelling model alongside
    # the transcriber when one is published for this language. Failures
    # are non-fatal because the spelling path is optional.
    try:
        download_spelling_model_for_language(
            model_info["language"], cache_root=cache_root, tracker=tracker
        )
    except Exception as exc:  # pragma: no cover - best-effort prefetch
        print(
            f"Warning: failed to download alphanumeric spelling model: {exc}",
            file=sys.stderr,
        )
    return result


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
    """Strips a file suffix off a voice name, mirroring ``piper_voice_stem``.

    A voice may be named the upstream Piper way (``<stem>.onnx``) or after one
    of the files we actually ship. Longest suffix first, so ``.ort`` does not
    swallow the ``.weights`` in ``<stem>.weights.ort``.
    """
    t = stem.strip()
    low = t.lower()
    for suffix in (".weights.ort", ".model.ort", ".kokorovoice", ".onnx", ".ort"):
        if low.endswith(suffix):
            return t[: -len(suffix)].strip()
    return t


def _is_bare_tts_engine_selector(voice: str) -> bool:
    """True for a bare engine keyword like ``zipvoice`` (no built-in voice id).

    The native option parser (``MoonshineTTSOptions::apply_voice_engine_prefix``)
    treats a bare ``zipvoice`` as an *engine* selector for zero-shot cloning
    rather than a voice id. Its dependency keys are shared model files
    (``zipvoice/tokens.txt``, encoders, vocoder, …) that always exist on the
    CDN, so it must not be dropped by the catalog membership check used for
    real per-voice files (which guards against 404s for unknown voice ids).
    """
    return _normalize_tts_voice_stem(voice).strip().lower() == "zipvoice"


def _tts_voice_want_aliases(voice: str) -> Set[str]:
    """Normalized ids the user may mean (prefixed catalog id vs bare stem)."""
    want = _normalize_tts_voice_stem(voice)
    aliases = {want}
    low = want.lower()
    if (
        low
        and not low.startswith("kokoro_")
        and not low.startswith("piper_")
        and not low.startswith("zipvoice_")
    ):
        aliases.add(_normalize_tts_voice_stem(f"kokoro_{want}"))
        aliases.add(_normalize_tts_voice_stem(f"piper_{want}"))
        aliases.add(_normalize_tts_voice_stem(f"zipvoice_{want}"))
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


def validate_tts_voice_known(
    language: str,
    voice: str,
    *,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    root_path: Optional[Path] = None,
    g2p_root: Optional[Path] = None,
) -> None:
    """
    Ensure ``voice`` appears in the native TTS catalog for ``language``.

    Unlike `validate_tts_voice_downloaded`, the voice does not have to be on disk: either
    ``present`` (downloaded) or ``downloadable`` (listed by the catalog but not yet fetched)
    is accepted. This is intended for fail-fast validation before `download_tts_assets`,
    which would otherwise ask the CDN for a voice file that does not exist (HTTP 404).

    Raises `MoonshineTtsVoiceError` with present voices as ``alternatives`` and catalog
    ``missing`` voices as ``alternatives_available_for_download`` when ``voice`` is unknown.
    """
    lang_tag = normalize_moonshine_language_tag(language)
    want_aliases = _tts_voice_want_aliases(voice)
    qopts: Dict[str, Union[str, int, float, bool]] = dict(options) if options else {}
    qopts.pop("voice", None)
    by_avail = list_tts_voices(
        lang_tag, options=qopts, root_path=root_path, g2p_root=g2p_root
    )
    known_norm = {_normalize_tts_voice_stem(v) for v in by_avail["present"]} | {
        _normalize_tts_voice_stem(v) for v in by_avail["downloadable"]
    }
    if want_aliases & known_norm:
        return
    raise MoonshineTtsVoiceError(
        voice,
        lang_tag,
        sorted(by_avail["present"]),
        alternatives_available_for_download=sorted(by_avail["downloadable"]),
    )


def ensure_tts_voice_downloaded(
    language: str,
    voice: str,
    asset_root: Path,
    *,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    download_missing: bool = False,
    show_progress: bool = True,
    on_progress: Optional[ProgressCallback] = None,
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
    _download_asset_keys(
        keys, root, show_progress=show_progress, on_progress=on_progress
    )
    validate_tts_voice_downloaded(language, voice, asset_root, options=options)


def _download_asset_keys(
    keys: List[str],
    root: Path,
    *,
    show_progress: bool = True,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Fetches every downloadable key in ``keys`` into ``root``.

    The TTS and G2P dependency lists are bare asset keys with no sizes, so
    progress here is per file rather than per byte.
    """
    wanted = [key for key in keys if is_downloadable_tts_asset_key(key)]
    tracker = (
        _ProgressTracker(on_progress, total_files=len(wanted))
        if on_progress is not None and wanted
        else None
    )
    for key in wanted:
        if tracker is not None:
            tracker.start(key)
        download_file(
            cdn_url_for_tts_asset_key(key),
            root / key,
            show_progress=show_progress and tracker is None,
            on_bytes=tracker.add_bytes if tracker is not None else None,
        )
        if tracker is not None:
            tracker.finish()


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
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """
    Download every file required for TTS for the given language (and optional prefixed ``voice``) into the cache.

    Files are stored under the same relative paths as canonical asset keys (e.g. ``en_us/dict.tsv``).
    Pass the returned directory as ``g2p_root`` when creating a synthesizer.

    ``on_progress`` is called with a ``0..1`` fraction and the asset being
    fetched, and silences the default tqdm bars.
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
    if voice is not None and not _is_bare_tts_engine_selector(voice):
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
    _download_asset_keys(
        keys, root, show_progress=show_progress, on_progress=on_progress
    )
    return root


def download_g2p_assets(
    language: str,
    *,
    options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    cache_root: Optional[Path] = None,
    show_progress: bool = True,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Download G2P lexicon/model files into the TTS asset cache layout (same CDN tree as TTS)."""
    lang_tag = normalize_moonshine_language_tag(language)
    keys = list_g2p_dependency_keys(lang_tag, options=options)
    root = tts_asset_cache_path(cache_root)
    _download_asset_keys(
        keys, root, show_progress=show_progress, on_progress=on_progress
    )
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
        "--embedding",
        action="store_true",
        help="Download text embedding assets (embedding model)",
    )
    parser.add_argument(
        "--stt",
        action="store_true",
        help="Download STT assets (transcription model)",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="Download the speaker diarization models (needed for identify_speakers)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="TTS voice: kokoro_*, piper_*, or zipvoice_* prefix plus id/stem (e.g. kokoro_af_heart, zipvoice_american_female)",
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

    if (
        not args.tts
        and not args.g2p
        and not args.embedding
        and not args.stt
        and not args.diarization
    ):
        print(
            "Please specify at least one of the following: --tts, --g2p, --embedding, "
            "--stt, --diarization",
            file=sys.stderr,
        )
        parser.print_help()
        sys.exit(1)

    if args.tts:
        root = download_tts_assets(
            args.language, voice=args.voice, cache_root=dl_root
        )
        print(f"TTS assets root (use as g2p_root): {root}", file=sys.stderr)
        print(root)
    if args.g2p:
        root = download_g2p_assets(args.language, cache_root=dl_root)
        print(f"G2P assets root (use as g2p_root): {root}", file=sys.stderr)
        print(root)
    if args.embedding:
        model_path, model_arch = get_embedding_model(cache_root=dl_root)
        print(f"Embedding model path: {model_path}", file=sys.stderr)
        print(model_path)
    if args.diarization:
        model_dir = get_diarization_model(cache_root=dl_root)
        print(f"Diarization model path: {model_dir}", file=sys.stderr)
        print(model_dir)
    if args.stt:
        get_model_for_language(args.language, args.model_arch, cache_root=dl_root)
        log_model_info(args.language, args.model_arch, cache_root=dl_root)
