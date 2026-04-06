#!/usr/bin/env python3
"""
Evaluate TTS intelligibility: Moonshine synthesis (Kokoro / Piper vocoders) vs Whisper large-v3 transcripts.

Ground-truth lines come from ``scripts/data/wiki-text/<language_tag>.txt`` (default), a sectioned
``wiki-text.txt`` file (``[en_us]`` / ``[de_de]`` sections), or any path you pass to ``--wiki-text``.

By default the script downloads **WikiText-2** (``wikitext-2-raw-v1``) for English locales
(``en_us``, ``en_gb``). Other Moonshine TTS languages use the same line-cleaning heuristics on
**Wikimedia Wikipedia** snapshots (``wikimedia/wikipedia``, config ``20231101.<lang>``), because
WikiText-2 is English-only on Hugging Face.

With no arguments, the script evaluates **all** TTS languages from the native catalog, refreshes
per-language ``wiki-text/*.txt`` from Hugging Face (unless ``--no-wiki-download``), scores **10 lines**
per language by default (override with ``--max-lines-per-lang``), and writes JSON to
``scripts/data/tts_g2p_intelligibility_report.json``. Synthesized audio for each track is cached as
16-bit mono ``.wav`` files under ``scripts/data/tts_g2p_intelligibility_wav_cache/`` (see
``--tts-wav-cache-dir`` / ``--no-tts-wav-cache``) so reruns skip Kokoro/Piper/Moonshine resynthesis;
Whisper transcription still runs every time. Pass ``--no-cache`` to always resynthesize while still
writing cache files (read is skipped). Cache files live under
``<cache-dir>/<lang>/<first-two-chars-of-stem>/`` with basenames
``<first-12-chars-of-line>_<engine>_<sha256>.wav`` (engines: ``moonshine``, ``kokoro``, ``piper``).

By default, ``--tts-root`` points at ``core/moonshine-tts/data``; after voice assets are downloaded,
files from ``<tts-root>/<language_tag>/`` (e.g. ``ru/dict.tsv``) are copied over the cache so G2P
uses the repo lexicon unless you pass ``--no-tts-root-overlay``.

Progress and per-language detail go to stderr; one tab-separated summary line per language is printed
to stdout (Moonshine internal G2P vs ``pip install kokoro`` vs ``pip install piper-tts`` CER columns).

**Three synthesis tracks:** (1) Moonshine native ``TextToSpeech`` (internal G2P + Kokoro or Piper ONNX
as bundled). (2) **hexgrad/kokoro** ``KPipeline`` from PyPI (misaki + espeak-ng G2P inside the
package). (3) **piper-tts** ``PiperVoice`` (phonemization via espeak-ng data shipped with the wheel).
For (3), the script patches Piper’s espeak IPA step to use Unicode **NFC** instead of upstream **NFD**,
so combining marks (e.g. cedilla) stay on their base letters and match typical ONNX ``phoneme_id_map``
entries—reducing ``Missing phoneme from id map`` noise (often seen for German) on stderr next to tqdm.

Dependencies (not part of the core moonshine-voice package)::

    pip install -r scripts/tts-g2p-intelligibility-requirements.txt

On **Python 3.14+**, ``datasets`` must be **4.4.0 or newer** (older releases hit a pickle error when
loading any split; see https://github.com/huggingface/datasets/issues/7839).

Example::

    python scripts/tts_g2p_intelligibility.py
    python scripts/tts_g2p_intelligibility.py --languages en_us,de_de --json-out report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Repo layout: moonshine_voice lives under python/src
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_WIKI_TEXT_DIR = _REPO_ROOT / "scripts" / "data" / "wiki-text"
_DEFAULT_JSON_REPORT = _REPO_ROOT / "scripts" / "data" / "tts_g2p_intelligibility_report.json"
_DEFAULT_TTS_WAV_CACHE_DIR = _REPO_ROOT / "scripts" / "data" / "tts_g2p_intelligibility_wav_cache"
_DEFAULT_TTS_DATA_ROOT = _REPO_ROOT / "core" / "moonshine-tts" / "data"
_TTS_WAV_CACHE_FORMAT = 2
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

import numpy as np
from tqdm import tqdm

from moonshine_voice.download import (
    ensure_tts_voice_downloaded,
    list_tts_languages,
    list_tts_voices,
    normalize_moonshine_language_tag,
    tts_asset_cache_path,
    download_tts_assets,
)
from moonshine_voice.errors import MoonshineError, MoonshineTtsLanguageError
from moonshine_voice.g2p import GraphemeToPhonemizer
from moonshine_voice.tts import TextToSpeech

WHISPER_TARGET_SR = 16000

# Moonshine TTS tag → Whisper ``language`` argument (ISO-639-1 where applicable)
_MOONSHINE_TO_WHISPER: Dict[str, str] = {
    "en_us": "en",
    "en_gb": "en",
    "cmn_hans_cn": "zh",
    "zh_cn": "zh",
    "ko_kr": "ko",
    "ja_jp": "ja",
    "es_es": "es",
    "es_mx": "es",
    "es_ar": "es",
    "pt_br": "pt",
    "uk_ua": "uk",
    "ar_msa": "ar",
    "vi_vn": "vi",
}

# Moonshine TTS tag → hexgrad/kokoro ``KPipeline`` lang_code (misaki + espeak-ng G2P in the pip package).
_MOONSHINE_TO_UPSTREAM_KOKORO_LANG: Dict[str, str] = {
    "en_us": "a",
    "en_gb": "b",
    "es_es": "e",
    "es_mx": "e",
    "es_ar": "e",
    "fr_fr": "f",
    "hi_in": "h",
    "it_it": "i",
    "ja_jp": "j",
    "pt_br": "p",
    "pt_pt": "p",
    "zh_hans": "z",
    "zh_cn": "z",
    "cmn_hans_cn": "z",
}

# Moonshine TTS tag → Wikimedia Wikipedia dump language code (20231101.<code>).
# English uses WikiText-2 instead (see ``uses_wikitext2``).
_MOONSHINE_TO_WIKIPEDIA_LANG: Dict[str, str] = {
    "ar": "ar",
    "ar_msa": "ar",
    "de": "de",
    "de_de": "de",
    "es": "es",
    "es_ar": "es",
    "es_es": "es",
    "es_mx": "es",
    "fr": "fr",
    "fr_fr": "fr",
    "hi": "hi",
    "hi_in": "hi",
    "it": "it",
    "it_it": "it",
    "ja": "ja",
    "ja_jp": "ja",
    "ko": "ko",
    "ko_kr": "ko",
    "nl": "nl",
    "nl_nl": "nl",
    "pt": "pt",
    "pt_br": "pt",
    "pt_pt": "pt",
    "ru": "ru",
    "ru_ru": "ru",
    "tr": "tr",
    "tr_tr": "tr",
    "uk": "uk",
    "uk_ua": "uk",
    "vi": "vi",
    "vi_vn": "vi",
    "zh": "zh",
    "zh_cn": "zh",
    "zh_hans": "zh",
    "cmn_hans_cn": "zh",
}

_FALLBACK_MOONSHINE_TTS_LANGS: List[str] = [
    "ar_msa",
    "de_de",
    "en_gb",
    "en_us",
    "es_ar",
    "es_es",
    "es_mx",
    "fr_fr",
    "hi_in",
    "it_it",
    "ja_jp",
    "ko_kr",
    "nl_nl",
    "pt_br",
    "pt_pt",
    "ru_ru",
    "tr_tr",
    "uk_ua",
    "vi_vn",
    "zh_hans",
]


def uses_wikitext2_corpus(tag: str) -> bool:
    """WikiText-2 on Hugging Face is English-only; use it for US/UK (and generic ``en``) tags."""
    t = normalize_moonshine_language_tag(tag)
    if t == "en" or t.startswith("en_"):
        return True
    return False


def moonshine_tag_to_wikipedia_lang(tag: str) -> Optional[str]:
    t = normalize_moonshine_language_tag(tag)
    if uses_wikitext2_corpus(t):
        return None
    if t in _MOONSHINE_TO_WIKIPEDIA_LANG:
        return _MOONSHINE_TO_WIKIPEDIA_LANG[t]
    parts = t.split("_")
    if len(parts) >= 1 and len(parts[0]) == 2 and parts[0].isalpha():
        return parts[0]
    return None


def moonshine_tag_to_whisper_language(tag: str) -> Optional[str]:
    t = normalize_moonshine_language_tag(tag)
    if t in _MOONSHINE_TO_WHISPER:
        return _MOONSHINE_TO_WHISPER[t]
    if t.startswith("cmn") or t.startswith("zh_") or t == "zh":
        return "zh"
    base = t.split("_")[0]
    if len(base) == 2 and base.isalpha():
        return base
    return None


def moonshine_tag_to_upstream_kokoro_lang_code(tag: str) -> Optional[str]:
    """Locale supported by ``pip install kokoro`` / ``KPipeline`` (not all Moonshine TTS languages)."""
    t = normalize_moonshine_language_tag(tag)
    if t in _MOONSHINE_TO_UPSTREAM_KOKORO_LANG:
        return _MOONSHINE_TO_UPSTREAM_KOKORO_LANG[t]
    base = t.split("_")[0]
    return {
        "en": "a",
        "es": "e",
        "fr": "f",
        "hi": "h",
        "it": "i",
        "ja": "j",
        "pt": "p",
        "zh": "z",
    }.get(base)


def load_wiki_text_directory(dir_path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for p in sorted(dir_path.glob("*.txt")):
        lang = normalize_moonshine_language_tag(p.stem)
        lines = _read_nonempty_lines(p.read_text(encoding="utf-8"))
        if lines:
            out[lang] = normalize_eval_lines(lines, lang=lang)
    return out


def _read_nonempty_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def load_wiki_text_file(path: Path) -> Dict[str, List[str]]:
    text = path.read_text(encoding="utf-8")
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    buf: List[str] = []
    header_re = re.compile(r"^\s*\[\s*([^\]]+?)\s*\]\s*$")

    def flush() -> None:
        nonlocal current, buf
        if current is not None and buf:
            lang = normalize_moonshine_language_tag(current)
            sections[lang] = normalize_eval_lines(list(buf), lang=lang)
        buf = []

    for line in text.splitlines():
        m = header_re.match(line)
        if m:
            flush()
            current = m.group(1).strip()
            buf = []
            continue
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if current is None:
            continue
        buf.append(s)
    flush()
    return sections


def load_wiki_texts(path: Path) -> Dict[str, List[str]]:
    if path.is_dir():
        return load_wiki_text_directory(path)
    return load_wiki_text_file(path)


def _wiki_max_graphemes_for_moonshine_tag(tag: Optional[str], base_max: int = 450) -> int:
    """
    Cap line length for locales whose native G2P tokenizer expands text (e.g. Chinese WordPiece
    inserts spaces between CJK characters, so ~450 graphemes can exceed ~1000 internal codepoints
    and trigger ``basic token alignment failed`` in C++).
    """
    if not tag:
        return base_max
    t = normalize_moonshine_language_tag(tag)
    if t.startswith("zh_") or t.startswith("cmn_"):
        return min(base_max, 200)
    return base_max


def clean_wiki_text_line(
    raw: str,
    *,
    min_chars: int = 20,
    max_chars: int = 450,
) -> Optional[str]:
    s = " ".join(raw.split())
    if not s or len(s) < min_chars or len(s) > max_chars:
        return None
    if re.match(r"^=+.+=+$", s):
        return None
    if s.startswith("[[") or s.startswith("{|"):
        return None
    if s.startswith("|") and re.match(r"^\|[\w\-]+\s*=", s):
        return None
    alnum = sum(1 for ch in s if ch.isalnum())
    if alnum < max(12, min_chars // 2):
        return None
    return s


def _split_oversized_wiki_piece(
    s: str,
    *,
    min_chars: int = 20,
    max_chars: int = 450,
) -> List[str]:
    """
    Wikipedia dumps often use a few very long lines per article. ``clean_wiki_text_line`` caps length,
    so without chunking we collect no candidates for many non-English wikis.
    """
    s = " ".join(s.split())
    if len(s) < min_chars:
        return []
    if len(s) <= max_chars:
        c = clean_wiki_text_line(s, min_chars=min_chars, max_chars=max_chars)
        return [c] if c else []
    out: List[str] = []
    # Word boundaries when the line has enough spaces (Latin scripts, etc.)
    if s.count(" ") >= max(2, len(s) // 100):
        words = s.split()
        buf: List[str] = []
        length = 0
        for w in words:
            add = len(w) + (1 if buf else 0)
            if length + add > max_chars and length >= min_chars:
                piece = " ".join(buf)
                c = clean_wiki_text_line(piece, min_chars=min_chars, max_chars=max_chars)
                if c:
                    out.append(c)
                buf = [w]
                length = len(w)
            else:
                buf.append(w)
                length += add
        if buf:
            piece = " ".join(buf)
            c = clean_wiki_text_line(piece, min_chars=min_chars, max_chars=max_chars)
            if c:
                out.append(c)
        return out
    # CJK and other compact text: fixed character windows
    i = 0
    while i < len(s):
        j = min(i + max_chars, len(s))
        piece = s[i:j].strip()
        if len(piece) >= min_chars:
            c = clean_wiki_text_line(piece, min_chars=min_chars, max_chars=max_chars)
            if c:
                out.append(c)
        if j == len(s):
            break
        i = j
    return out


def normalize_eval_lines(
    raw_lines: List[str],
    *,
    min_chars: int = 20,
    max_chars: int = 450,
    lang: Optional[str] = None,
) -> List[str]:
    """
    Chunk or filter lines so each fits Moonshine G2P / TTS (same bounds as wiki download).

    Files under ``wiki-text/*.txt`` are read line-by-line with no length cap; long CJK
    paragraphs can exceed what the Chinese WordPiece path accepts and trigger alignment
    failures in native code (logged as ``basic token alignment failed at offset …``).
    """
    cap = _wiki_max_graphemes_for_moonshine_tag(lang, max_chars)
    out: List[str] = []
    for raw in raw_lines:
        s = raw.strip()
        if not s:
            continue
        if len(s) <= cap:
            c = clean_wiki_text_line(s, min_chars=min_chars, max_chars=cap)
            if c:
                out.append(c)
            continue
        out.extend(
            _split_oversized_wiki_piece(s, min_chars=min_chars, max_chars=cap)
        )
    return out


def wikipedia_paragraph_to_candidates(
    para: str,
    *,
    min_chars: int = 20,
    max_chars: int = 450,
) -> List[str]:
    para = para.strip()
    if not para:
        return []
    if para.startswith(("{|", "|}", "[[File:", "[[Image:")):
        return []
    c = clean_wiki_text_line(para, min_chars=min_chars, max_chars=max_chars)
    if c:
        return [c]
    # Split on sentence-like boundaries, then chunk anything still too long
    parts = re.split(r"(?<=[.!?。！？．؟।])\s+", para)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        c = clean_wiki_text_line(p, min_chars=min_chars, max_chars=max_chars)
        if c:
            out.append(c)
            continue
        out.extend(
            _split_oversized_wiki_piece(p, min_chars=min_chars, max_chars=max_chars)
        )
    return out


def try_import_datasets():
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_datasets_version_tuple(version: str) -> Tuple[int, int, int]:
    """Best-effort parse for ``datasets.__version__`` (handles ``4.4.0rc1``)."""
    main = (version or "").split("+", 1)[0].strip()
    nums: List[int] = []
    for part in main.split("."):
        buf = ""
        for ch in part:
            if ch.isdigit():
                buf += ch
            else:
                break
        if buf:
            nums.append(int(buf))
        if len(nums) >= 3:
            break
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def require_datasets_compatible_with_python() -> None:
    """
    ``datasets`` before 4.4.0 breaks on Python 3.14 inside dill/pickle (Hasher.hash / load_dataset).
    """
    if sys.version_info < (3, 14):
        return
    try:
        import datasets
    except ImportError:
        return
    ver = getattr(datasets, "__version__", "") or "0"
    if _parse_datasets_version_tuple(ver) < (4, 4, 0):
        raise MoonshineError(
            f"Python {sys.version_info.major}.{sys.version_info.minor} requires huggingface "
            f"``datasets`` >= 4.4.0 (found {ver!r}). The older stack raises:\n"
            "  TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given\n"
            "Upgrade with:\n"
            "  pip install -U 'datasets>=4.4.0'\n"
            "Context: https://github.com/huggingface/datasets/issues/7839"
        )


def fetch_lines_wikitext2(*, target_lines: int, seed: int) -> Tuple[List[str], str]:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    candidates: List[str] = []
    for row in ds:
        chunk = row.get("text") or ""
        for piece in chunk.split("\n"):
            c = clean_wiki_text_line(piece)
            if c:
                candidates.append(c)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    lines = candidates[:target_lines]
    meta = "wikitext/wikitext-2-raw-v1 (WikiText-2 raw train split)"
    return lines, meta


def fetch_lines_wikipedia(
    wiki_lang: str,
    *,
    target_lines: int,
    seed: int,
    moonshine_tag: Optional[str] = None,
) -> Tuple[List[str], str]:
    from datasets import load_dataset

    config = f"20231101.{wiki_lang}"
    ds = load_dataset(
        "wikimedia/wikipedia",
        config,
        split="train",
        streaming=True,
    )
    rng = random.Random(seed)
    candidates: List[str] = []
    chunk_max = _wiki_max_graphemes_for_moonshine_tag(moonshine_tag)
    it = iter(ds)
    skip = rng.randint(0, 80)
    for _ in range(skip):
        try:
            next(it)
        except StopIteration:
            break
    articles = 0
    max_articles = 4000
    while len(candidates) < target_lines * 6 and articles < max_articles:
        try:
            article = next(it)
        except StopIteration:
            break
        articles += 1
        text = (article.get("text") or "").replace("\r\n", "\n")
        for para in text.split("\n"):
            for cand in wikipedia_paragraph_to_candidates(
                para, max_chars=chunk_max
            ):
                candidates.append(cand)
    rng.shuffle(candidates)
    lines = candidates[:target_lines]
    meta = f"wikimedia/wikipedia ({config})"
    return lines, meta


def fetch_lines_for_moonshine_language(
    lang_tag: str,
    *,
    target_lines: int,
) -> Tuple[List[str], str]:
    t = normalize_moonshine_language_tag(lang_tag)
    seed = (hash(t) % (2**31)) ^ 0x9E3779B9
    if uses_wikitext2_corpus(t):
        return fetch_lines_wikitext2(target_lines=target_lines, seed=seed)
    wl = moonshine_tag_to_wikipedia_lang(t)
    if wl is None:
        raise ValueError(f"No Wikipedia language code for Moonshine tag {t!r}")
    return fetch_lines_wikipedia(
        wl, target_lines=target_lines, seed=seed, moonshine_tag=t
    )


def all_moonshine_tts_language_tags() -> List[str]:
    try:
        tags = [
            normalize_moonshine_language_tag(x) for x in list_tts_languages()
        ]
        return sorted(set(tags))
    except Exception:
        return sorted(set(_FALLBACK_MOONSHINE_TTS_LANGS))


def ensure_wiki_text_files(
    output_dir: Path,
    languages: Set[str],
    *,
    lines_per_lang: int,
    allow_download: bool,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Ensure ``output_dir/<lang>.txt`` exists for each language.

    Returns ``(sections_subset, corpus_description_by_lang)``.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_meta: Dict[str, str] = {}
    for lang in sorted(languages):
        path = output_dir / f"{lang}.txt"
        need_fetch = False
        if not path.is_file():
            need_fetch = True
        else:
            existing = _read_nonempty_lines(path.read_text(encoding="utf-8"))
            if len(existing) < max(5, min(lines_per_lang // 4, 25)):
                need_fetch = True
        if not need_fetch:
            corpus_meta[lang] = "existing local file (not refreshed)"
            continue
        if not allow_download:
            corpus_meta[lang] = "missing or short file; skipped (--no-wiki-download)"
            continue
        if not try_import_datasets():
            raise MoonshineError(
                "``datasets`` is required to download wiki text. "
                "Install: pip install -r scripts/tts-g2p-intelligibility-requirements.txt"
            )
        try:
            lines, meta = fetch_lines_for_moonshine_language(
                lang, target_lines=lines_per_lang
            )
        except Exception as e:
            corpus_meta[lang] = f"download failed: {e}"
            continue
        if not lines:
            corpus_meta[lang] = "download produced no lines"
            continue
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        corpus_meta[lang] = meta

    loaded = load_wiki_text_directory(output_dir)
    sections = {k: v for k, v in loaded.items() if k in languages}
    return sections, corpus_meta


def resample_linear(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples.astype(np.float32, copy=False)
    n = len(samples)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    duration = n / float(orig_sr)
    n_out = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, duration, num=n, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False, dtype=np.float64)
    y = np.interp(x_new, x_old, np.asarray(samples, dtype=np.float64))
    return y.astype(np.float32)


def _tts_cache_key(payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _tts_track_to_engine_slug(track: str) -> str:
    if track == "moonshine":
        return "moonshine"
    if track == "upstream_kokoro":
        return "kokoro"
    if track == "upstream_piper":
        return "piper"
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in track)
    s = safe.strip("_")
    return s if s else "engine"


def _tts_sentence_prefix_for_cache(text: str, max_chars: int = 12) -> str:
    """First ``max_chars`` Unicode scalars of the line, scrubbed for use in filenames."""
    s = (text or "").strip()
    if not s:
        return "utt"
    s = s[:max_chars]
    forbidden = '\\/:*?"<>|\n\r\t\x00'
    out = "".join("_" if ch in forbidden else ch for ch in s)
    out = out.strip().rstrip(". ")
    return out if out else "utt"


def _tts_cache_stem(text: str, track: str, payload: Dict[str, Any]) -> str:
    prefix = _tts_sentence_prefix_for_cache(text)
    engine = _tts_track_to_engine_slug(track)
    digest = _tts_cache_key(payload)
    return f"{prefix}_{engine}_{digest}"


def _tts_cache_wav_and_meta_paths(root: Path, lang: str, stem: str) -> Tuple[Path, Path]:
    shard = stem[:2] if len(stem) >= 2 else (stem + "_")[:2]
    d = root / lang / shard
    base = d / stem
    return base.with_suffix(".wav"), base.with_suffix(".meta.json")


def _write_wav_int16_mono(path: Path, samples: List[float], sample_rate: int) -> None:
    arr = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.round(arr * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _read_wav_int16_mono(path: Path) -> Optional[Tuple[List[float], int]]:
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                return None
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
    except (OSError, EOFError, wave.Error):
        return None
    if not raw:
        return [], sr
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.tolist(), sr


def _load_tts_wav_meta(meta_path: Path) -> Optional[str]:
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("format") != _TTS_WAV_CACHE_FORMAT:
        return None
    ph = data.get("phonemes")
    return str(ph) if ph is not None else None


def _save_tts_wav_meta(meta_path: Path, phonemes: Optional[str]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {"format": _TTS_WAV_CACHE_FORMAT, "phonemes": phonemes},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _try_read_tts_wav_cache(
    cache_root: Optional[Path],
    lang: str,
    payload: Dict[str, Any],
) -> Optional[Tuple[List[float], int, Optional[str]]]:
    if cache_root is None:
        return None
    track = payload.get("track")
    if not isinstance(track, str) or not track:
        return None
    text = payload.get("text", "")
    if not isinstance(text, str):
        text = ""
    stem = _tts_cache_stem(text, track, payload)
    wav_path, meta_path = _tts_cache_wav_and_meta_paths(cache_root, lang, stem)
    if not wav_path.is_file():
        return None
    got = _read_wav_int16_mono(wav_path)
    if got is None:
        return None
    samples, sr = got
    if not samples:
        return None
    ph = _load_tts_wav_meta(meta_path)
    return samples, sr, ph


def _write_tts_wav_cache(
    cache_root: Optional[Path],
    lang: str,
    payload: Dict[str, Any],
    samples: Optional[List[float]],
    sample_rate: int,
    phonemes: Optional[str],
) -> None:
    if cache_root is None or not samples:
        return
    track = payload.get("track")
    if not isinstance(track, str) or not track:
        return
    text = payload.get("text", "")
    if not isinstance(text, str):
        text = ""
    stem = _tts_cache_stem(text, track, payload)
    wav_path, meta_path = _tts_cache_wav_and_meta_paths(cache_root, lang, stem)
    _write_wav_int16_mono(wav_path, samples, sample_rate)
    _save_tts_wav_meta(meta_path, phonemes)


def try_import_cer():
    try:
        from jiwer import cer  # type: ignore

        return cer
    except ImportError:
        return None


def try_import_whisper_model():
    try:
        from faster_whisper import WhisperModel  # type: ignore

        return WhisperModel
    except ImportError:
        return None


@dataclass
class VoicePair:
    kokoro: Optional[str] = None
    piper: Optional[str] = None


def discover_voices(language: str, asset_root: Path) -> VoicePair:
    lang = normalize_moonshine_language_tag(language)
    opts = {"g2p_root": str(asset_root.resolve())}
    try:
        v = list_tts_voices(lang, options=opts)
    except MoonshineTtsLanguageError:
        return VoicePair()
    ordered = sorted(set(v["present"]) | set(v["downloadable"]))
    kok = next((x for x in ordered if x.lower().startswith("kokoro_")), None)
    pip = next((x for x in ordered if x.lower().startswith("piper_")), None)
    return VoicePair(kokoro=kok, piper=pip)


def prepare_asset_root(
    language: str,
    voices: List[str],
    cache_root: Optional[Path],
) -> Path:
    if not voices:
        raise MoonshineError(f"No Kokoro/Piper voices to download for {language}")
    root = download_tts_assets(
        language,
        voice=voices[0],
        cache_root=cache_root,
        show_progress=True,
    )
    for v in voices[1:]:
        ensure_tts_voice_downloaded(
            language,
            v,
            root,
            download_missing=True,
            show_progress=True,
        )
    return root


def overlay_repo_tts_data_into_asset_root(
    asset_root: Path,
    tts_data_root: Path,
    lang: str,
) -> int:
    """
    Copy files from ``tts_data_root/<lang>/`` onto the downloaded asset tree so repo
    ``dict.tsv`` and other linguistic data override CDN copies (same relative paths).
    """
    src_dir = (tts_data_root / lang).resolve()
    if not src_dir.is_dir():
        return 0
    dst_dir = (asset_root / lang).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.iterdir()):
        # Lexicons / tables only (skip README, stray configs at language root).
        if p.is_file() and p.suffix.lower() == ".tsv":
            shutil.copy2(p, dst_dir / p.name)
            n += 1
    return n


_UPSTREAM_KOKORO_SAMPLE_RATE_HZ = 24000
_UPSTREAM_KOKORO_REPO_ID = "hexgrad/Kokoro-82M"


def moonshine_kokoro_catalog_voice_to_package_voice(catalog_id: str) -> str:
    s = catalog_id.strip()
    if s.lower().startswith("kokoro_"):
        return s[7:]
    return s


def try_import_k_pipeline():
    try:
        from kokoro import KPipeline

        return KPipeline
    except ImportError:
        return None


def try_import_piper_voice():
    try:
        from piper import PiperVoice

        return PiperVoice
    except ImportError:
        return None


_PIPER_ESPEAK_NFC_PATCH_APPLIED = False


def ensure_piper_espeak_phonemizer_uses_nfc_for_eval() -> None:
    """
    ``piper-tts`` decomposes espeak IPA with Unicode NFD, so combining marks (e.g. U+0327 cedilla)
    become separate "phonemes". Many Piper ONNX ``phoneme_id_map`` tables omit those codepoints,
    which triggers ``Missing phoneme from id map`` and garbles tqdm on stderr (German is a common case).

    Normalizing each IPA clause to **NFC** keeps typical Latin/IPA letters + diacritics composed so
    lookups match the bundled maps. This only affects the in-process ``EspeakPhonemizer`` used by
    ``pip install piper-tts`` during this script.
    """
    global _PIPER_ESPEAK_NFC_PATCH_APPLIED
    if _PIPER_ESPEAK_NFC_PATCH_APPLIED:
        return
    try:
        from piper.phonemize_espeak import EspeakPhonemizer
    except ImportError:
        return

    def phonemize_nfc(self: Any, voice: str, text: str) -> list[list[str]]:
        import unicodedata as ucd

        from piper import espeakbridge

        espeakbridge.set_voice(voice)

        all_phonemes: list[list[str]] = []
        sentence_phonemes: list[str] = []

        clause_phonemes = espeakbridge.get_phonemes(text)
        for phonemes_str, terminator_str, end_of_sentence in clause_phonemes:
            phonemes_str = re.sub(r"\([^)]+\)", "", phonemes_str)
            phonemes_str += terminator_str
            if terminator_str in (",", ":", ";"):
                phonemes_str += " "
            # Upstream uses NFD here; NFC matches more ONNX phoneme_id_map keys (e.g. de_DE-thorsten).
            sentence_phonemes.extend(list(ucd.normalize("NFC", phonemes_str)))

            if end_of_sentence:
                all_phonemes.append(sentence_phonemes)
                sentence_phonemes = []

        if sentence_phonemes:
            all_phonemes.append(sentence_phonemes)

        return all_phonemes

    EspeakPhonemizer.phonemize = phonemize_nfc  # type: ignore[method-assign]
    _PIPER_ESPEAK_NFC_PATCH_APPLIED = True
    logging.getLogger("piper.phoneme_ids").setLevel(logging.ERROR)


def resolve_piper_onnx_path(asset_root: Path, piper_voice_catalog_id: str) -> Optional[Path]:
    stem = piper_voice_catalog_id.strip()
    if stem.lower().startswith("piper_"):
        stem = stem[6:]
    onnx_name = stem if stem.lower().endswith(".onnx") else f"{stem}.onnx"
    matches = sorted(asset_root.resolve().rglob(onnx_name))
    if not matches:
        return None
    preferred = [p for p in matches if "piper-voices" in p.parts]
    return preferred[0] if preferred else matches[0]


def build_upstream_kokoro_pipeline(
    lang_code: str,
    *,
    device: Optional[str],
    repo_id: str = _UPSTREAM_KOKORO_REPO_ID,
) -> Tuple[Any, Optional[str]]:
    KPipeline = try_import_k_pipeline()
    if KPipeline is None:
        return None, "kokoro package not installed (pip install kokoro)"
    kwargs: Dict[str, Any] = {"lang_code": lang_code, "repo_id": repo_id}
    if device:
        kwargs["device"] = device
    try:
        return KPipeline(**kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def synthesize_upstream_kokoro_package(
    pipeline: Any,
    text: str,
    voice: str,
) -> Tuple[Optional[List[float]], int, Optional[str]]:
    """hexgrad/kokoro ``KPipeline`` (misaki + espeak-ng G2P inside the package)."""
    aud_parts: List[np.ndarray] = []
    phone_parts: List[str] = []
    for result in pipeline(text, voice=voice, split_pattern=None):
        if getattr(result, "phonemes", None):
            ps = str(result.phonemes).strip()
            if ps:
                phone_parts.append(ps)
        aud = getattr(result, "audio", None)
        if aud is None:
            continue
        if hasattr(aud, "detach"):
            aud = aud.detach().cpu().numpy()
        arr = np.asarray(aud, dtype=np.float64).reshape(-1)
        if arr.size:
            aud_parts.append(arr)
    phone_str = " ".join(phone_parts).strip() or None
    if not aud_parts:
        return None, _UPSTREAM_KOKORO_SAMPLE_RATE_HZ, phone_str
    full = np.concatenate(aud_parts).astype(np.float32)
    return full.tolist(), _UPSTREAM_KOKORO_SAMPLE_RATE_HZ, phone_str


def synthesize_upstream_piper_package(
    piper_voice: Any,
    text: str,
) -> Tuple[Optional[List[float]], int, Optional[str]]:
    """``piper-tts`` ``PiperVoice`` (phonemize + espeak-ng data bundled with the package)."""
    chunks = list(piper_voice.synthesize(text))
    if not chunks:
        return None, 22050, None
    arrs = [np.asarray(c.audio_float_array, dtype=np.float32) for c in chunks]
    full = np.concatenate(arrs)
    sr = int(chunks[0].sample_rate)
    phoneme_bits: List[str] = []
    for c in chunks:
        ph = c.phonemes
        if isinstance(ph, list):
            # Piper stores one UTF-8 codepoint (or occasional " ") per list element (see
            # ``phonemize_espeak``: ``list(unicodedata.normalize("NFD", phonemes_str))``).
            # Joining with spaces would insert a gap between every character.
            phoneme_bits.append("".join(str(t) for t in ph))
        elif ph:
            phoneme_bits.append(str(ph))
    pstr = " ".join(x.strip() for x in phoneme_bits if x.strip()).strip() or None
    return full.tolist(), sr, pstr


@dataclass
class LineResult:
    ground_truth: str
    moonshine_g2p_ipa: str
    upstream_kokoro_package_phonemes: Optional[str] = None
    upstream_piper_package_phonemes: Optional[str] = None
    whisper_recognition_moonshine: Optional[str] = None
    whisper_recognition_upstream_kokoro: Optional[str] = None
    whisper_recognition_upstream_piper: Optional[str] = None
    cer_moonshine: Optional[float] = None
    cer_upstream_kokoro: Optional[float] = None
    cer_upstream_piper: Optional[float] = None


def transcribe_whisper(
    model: Any,
    samples: List[float],
    sample_rate: int,
    whisper_lang: Optional[str],
) -> str:
    audio = resample_linear(np.asarray(samples, dtype=np.float32), sample_rate, WHISPER_TARGET_SR)
    segments, _info = model.transcribe(
        audio,
        language=whisper_lang,
        task="transcribe",
        vad_filter=False,
    )
    parts = [s.text for s in segments]
    return " ".join(p.strip() for p in parts if p).strip()


def run_language(
    language: str,
    lines: List[str],
    *,
    kokoro_voice: Optional[str],
    piper_voice: Optional[str],
    cache_root: Optional[Path],
    tts_data_root: Optional[Path],
    whisper_model: Any,
    whisper_lang: Optional[str],
    cer_fn: Any,
    upstream_kokoro_device: Optional[str],
    tts_wav_cache_dir: Optional[Path],
    tts_skip_wav_cache_read: bool,
) -> Tuple[Dict[str, Any], List[LineResult]]:
    lang = normalize_moonshine_language_tag(language)
    tmp_root = tts_asset_cache_path(cache_root)

    vp = VoicePair(kokoro=kokoro_voice, piper=piper_voice)
    if vp.kokoro is None and vp.piper is None:
        vp = discover_voices(lang, tmp_root)
    if vp.kokoro is None and vp.piper is None:
        raise MoonshineError(
            f"No kokoro_* or piper_* voice listed for {lang}. "
            "Pass --kokoro-voice / --piper-voice or populate the TTS cache."
        )

    voices_to_fetch = [v for v in [vp.kokoro, vp.piper] if v is not None]
    asset_root = prepare_asset_root(lang, voices_to_fetch, cache_root)

    overlay_n = 0
    if tts_data_root is not None:
        tr = tts_data_root.resolve()
        if not tr.is_dir():
            print(
                f"Warning: --tts-root {tr} is not a directory; skipping repo overlay for {lang}.",
                file=sys.stderr,
            )
        else:
            overlay_n = overlay_repo_tts_data_into_asset_root(asset_root, tr, lang)
            if overlay_n:
                print(
                    f"Overlayed {overlay_n} file(s) from {tr / lang} → {asset_root / lang}",
                    file=sys.stderr,
                )

    # ``prepare_asset_root`` already populated ``asset_root``; avoid ``download=True`` here or
    # ``download_g2p_assets`` would re-fetch and overwrite ``--tts-root`` lexicon overlays.
    g2p = GraphemeToPhonemizer(lang, asset_root=asset_root, download=False)

    moonshine_voice = vp.kokoro if vp.kokoro is not None else vp.piper
    assert moonshine_voice is not None
    tts_moonshine = TextToSpeech(
        lang,
        voice=moonshine_voice,
        asset_root=asset_root,
        download=False,
    )

    kokoro_lang = moonshine_tag_to_upstream_kokoro_lang_code(lang)
    upstream_k_pipe: Any = None
    upstream_k_pkg_voice: Optional[str] = None
    upstream_k_skip: Optional[str] = None
    if vp.kokoro and kokoro_lang:
        upstream_k_pipe, upstream_k_skip = build_upstream_kokoro_pipeline(
            kokoro_lang,
            device=upstream_kokoro_device,
        )
        if upstream_k_pipe is not None:
            upstream_k_pkg_voice = moonshine_kokoro_catalog_voice_to_package_voice(vp.kokoro)
    elif not vp.kokoro:
        upstream_k_skip = "no Moonshine kokoro_* voice for this locale"
    else:
        upstream_k_skip = (
            f"upstream kokoro KPipeline has no lang_code mapping for {lang!r} "
            "(see _MOONSHINE_TO_UPSTREAM_KOKORO_LANG)"
        )

    PiperVoiceCls = try_import_piper_voice()
    upstream_piper_voice: Any = None
    upstream_p_skip: Optional[str] = None
    piper_onnx: Optional[Path] = None
    if vp.piper and PiperVoiceCls is not None:
        piper_onnx = resolve_piper_onnx_path(asset_root, vp.piper)
        if piper_onnx is None:
            upstream_p_skip = f"could not find ONNX for {vp.piper!r} under {asset_root}"
        else:
            try:
                upstream_piper_voice = PiperVoiceCls.load(piper_onnx)
            except Exception as e:  # noqa: BLE001
                upstream_p_skip = str(e)
                upstream_piper_voice = None
    elif not vp.piper:
        upstream_p_skip = "no Moonshine piper_* voice for this locale"
    else:
        upstream_p_skip = "piper-tts package not installed (pip install piper-tts)"

    line_results: List[LineResult] = []
    sum_m: List[float] = []
    sum_uk: List[float] = []
    sum_up: List[float] = []

    line_iter = tqdm(
        lines,
        desc=f"Evaluating {lang}",
        unit="line",
        leave=False,
        file=sys.stderr,
    )
    for text in line_iter:
        moon_ipa = g2p.to_ipa(text)

        hyp_m: Optional[str] = None
        payload_m = {
            "v": _TTS_WAV_CACHE_FORMAT,
            "track": "moonshine",
            "lang": lang,
            "voice": moonshine_voice,
            "text": text,
        }
        cached_m = (
            None
            if tts_skip_wav_cache_read
            else _try_read_tts_wav_cache(tts_wav_cache_dir, lang, payload_m)
        )
        moon_meta_ph = moon_ipa.strip() if moon_ipa and moon_ipa.strip() else None
        if cached_m is not None:
            samples_m, sr_m, _ph_m = cached_m
        else:
            samples_m, sr_m = tts_moonshine.synthesize(text)
            _write_tts_wav_cache(
                tts_wav_cache_dir, lang, payload_m, samples_m, sr_m, moon_meta_ph
            )
        hyp_m = transcribe_whisper(whisper_model, samples_m, sr_m, whisper_lang)

        hyp_uk: Optional[str] = None
        uk_ph: Optional[str] = None
        if upstream_k_pipe is not None and upstream_k_pkg_voice:
            payload_uk = {
                "v": _TTS_WAV_CACHE_FORMAT,
                "track": "upstream_kokoro",
                "lang": lang,
                "kokoro_lang": kokoro_lang or "",
                "pkg_voice": upstream_k_pkg_voice,
                "device": upstream_kokoro_device or "",
                "repo": _UPSTREAM_KOKORO_REPO_ID,
                "text": text,
            }
            cached_uk = (
                None
                if tts_skip_wav_cache_read
                else _try_read_tts_wav_cache(tts_wav_cache_dir, lang, payload_uk)
            )
            if cached_uk is not None:
                samps, sr_k, uk_ph = cached_uk
            else:
                samps, sr_k, uk_ph = synthesize_upstream_kokoro_package(
                    upstream_k_pipe, text, upstream_k_pkg_voice
                )
                if samps is not None:
                    _write_tts_wav_cache(
                        tts_wav_cache_dir, lang, payload_uk, samps, sr_k, uk_ph
                    )
            if samps is not None:
                hyp_uk = transcribe_whisper(whisper_model, samps, sr_k, whisper_lang)

        hyp_up: Optional[str] = None
        up_ph: Optional[str] = None
        if upstream_piper_voice is not None:
            payload_up: Dict[str, Any] = {
                "v": _TTS_WAV_CACHE_FORMAT,
                "track": "upstream_piper",
                "lang": lang,
                "text": text,
            }
            if piper_onnx is not None:
                payload_up["onnx"] = str(piper_onnx.resolve())
            cached_up = (
                None
                if tts_skip_wav_cache_read
                else _try_read_tts_wav_cache(tts_wav_cache_dir, lang, payload_up)
            )
            if cached_up is not None:
                samps, sr_p, up_ph = cached_up
            else:
                samps, sr_p, up_ph = synthesize_upstream_piper_package(upstream_piper_voice, text)
                if samps is not None:
                    _write_tts_wav_cache(
                        tts_wav_cache_dir, lang, payload_up, samps, sr_p, up_ph
                    )
            if samps is not None:
                hyp_up = transcribe_whisper(whisper_model, samps, sr_p, whisper_lang)

        cm = cer_fn(text, hyp_m) if hyp_m is not None else None
        cuk = cer_fn(text, hyp_uk) if hyp_uk is not None else None
        cup = cer_fn(text, hyp_up) if hyp_up is not None else None
        if cm is not None:
            sum_m.append(float(cm))
        if cuk is not None:
            sum_uk.append(float(cuk))
        if cup is not None:
            sum_up.append(float(cup))

        line_results.append(
            LineResult(
                ground_truth=text,
                moonshine_g2p_ipa=moon_ipa,
                upstream_kokoro_package_phonemes=uk_ph,
                upstream_piper_package_phonemes=up_ph,
                whisper_recognition_moonshine=hyp_m,
                whisper_recognition_upstream_kokoro=hyp_uk,
                whisper_recognition_upstream_piper=hyp_up,
                cer_moonshine=cm,
                cer_upstream_kokoro=cuk,
                cer_upstream_piper=cup,
            )
        )

    g2p.close()
    tts_moonshine.close()

    summary = {
        "language": lang,
        "moonshine_tts_voice": moonshine_voice,
        "kokoro_voice_catalog": vp.kokoro,
        "piper_voice_catalog": vp.piper,
        "upstream_kokoro_lang_code": kokoro_lang,
        "upstream_kokoro_package_voice": upstream_k_pkg_voice,
        "upstream_kokoro_skip_reason": upstream_k_skip,
        "upstream_piper_onnx": str(piper_onnx) if piper_onnx else None,
        "upstream_piper_skip_reason": upstream_p_skip,
        "whisper_language": whisper_lang,
        "num_lines": len(lines),
        "average_cer_moonshine_internal_g2p": float(np.mean(sum_m)) if sum_m else None,
        "average_cer_upstream_kokoro_python": float(np.mean(sum_uk)) if sum_uk else None,
        "average_cer_upstream_piper_python": float(np.mean(sum_up)) if sum_up else None,
        "tts_data_root": str(tts_data_root.resolve()) if tts_data_root is not None else None,
        "tts_root_overlay_files": overlay_n,
        "tts_wav_cache_skip_read": bool(tts_skip_wav_cache_read),
    }
    return summary, line_results


def line_result_to_dict(r: LineResult) -> Dict[str, Any]:
    return {
        "ground_truth": r.ground_truth,
        "moonshine_g2p_ipa": r.moonshine_g2p_ipa,
        "upstream_kokoro_package_phonemes": r.upstream_kokoro_package_phonemes,
        "upstream_piper_package_phonemes": r.upstream_piper_package_phonemes,
        "whisper_recognition_moonshine_internal_g2p": r.whisper_recognition_moonshine,
        "whisper_recognition_upstream_kokoro_python": r.whisper_recognition_upstream_kokoro,
        "whisper_recognition_upstream_piper_python": r.whisper_recognition_upstream_piper,
        "character_error_rate_moonshine_internal_g2p": r.cer_moonshine,
        "character_error_rate_upstream_kokoro_python": r.cer_upstream_kokoro,
        "character_error_rate_upstream_piper_python": r.cer_upstream_piper,
    }


def _fmt_metric(x: Optional[float]) -> str:
    return f"{x:.6f}" if x is not None else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Moonshine TTS (internal G2P) vs pip kokoro / pip piper-tts, scored with Whisper."
        )
    )
    parser.add_argument(
        "--wiki-text",
        type=Path,
        default=_DEFAULT_WIKI_TEXT_DIR,
        help="Directory of <language_tag>.txt files (default) or a sectioned wiki-text.txt file",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="",
        help=(
            "Comma-separated Moonshine TTS tags (default: all catalog languages; "
            "downloads one corpus file per language when using the default wiki-text directory)"
        ),
    )
    parser.add_argument(
        "--wiki-lines-per-lang",
        type=int,
        default=200,
        metavar="N",
        help="Lines to extract per language when downloading into wiki-text/ (default: %(default)s)",
    )
    parser.add_argument(
        "--max-lines-per-lang",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Max lines per language to synthesize and score with Whisper (default: %(default)s; "
            "0 = use every line loaded from the wiki-text file)"
        ),
    )
    parser.add_argument(
        "--no-wiki-download",
        action="store_true",
        help="Do not download from Hugging Face; use only existing wiki-text files",
    )
    parser.add_argument(
        "--kokoro-voice",
        type=str,
        default=None,
        help="Force Kokoro voice id (e.g. kokoro_af_heart). Default: first catalog match.",
    )
    parser.add_argument(
        "--piper-voice",
        type=str,
        default=None,
        help="Force Piper voice id. Default: first catalog match.",
    )
    parser.add_argument(
        "--upstream-kokoro-device",
        type=str,
        default="",
        metavar="DEVICE",
        help=(
            "Torch device for pip ``kokoro`` KPipeline (cpu, cuda, mps, …). "
            "Empty = Kokoro package default (often CUDA if available)."
        ),
    )
    parser.add_argument(
        "--asset-cache",
        type=Path,
        default=None,
        help="Download/cache root for voice ONNX and CDN linguistic files (same as moonshine download --root)",
    )
    parser.add_argument(
        "--tts-root",
        type=Path,
        default=_DEFAULT_TTS_DATA_ROOT,
        help=(
            "Repo linguistic bundle: after voices download, copy ``<lang>/*`` from here over the cache "
            f"(default: {_DEFAULT_TTS_DATA_ROOT}) so e.g. ``dict.tsv`` matches this tree"
        ),
    )
    parser.add_argument(
        "--no-tts-root-overlay",
        action="store_true",
        help="Skip copying ``--tts-root`` into the asset cache (use CDN lexicons only)",
    )
    parser.add_argument(
        "--tts-wav-cache-dir",
        type=Path,
        default=_DEFAULT_TTS_WAV_CACHE_DIR,
        help=(
            "Directory for per-line synthesized audio (.wav) and phoneme sidecars (.meta.json); "
            "reuse speeds up reruns (Whisper still runs each time). "
            "Files are named <first-12-chars-of-line>_<engine>_<sha256>.wav under <lang>/<2-char-shard>/. "
            f"Default: {_DEFAULT_TTS_WAV_CACHE_DIR}"
        ),
    )
    parser.add_argument(
        "--no-tts-wav-cache",
        action="store_true",
        help="Disable reading/writing cached TTS wave files",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Do not load cached .wav files (always resynthesize), but still write cache after each line. "
            "Ignored when --no-tts-wav-cache is set."
        ),
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="faster-whisper model name or path (default: large-v3)",
    )
    parser.add_argument(
        "--whisper-device",
        type=str,
        default="cpu",
        help="faster-whisper device (e.g. cpu, cuda, auto)",
    )
    parser.add_argument(
        "--whisper-compute-type",
        type=str,
        default="default",
        help="faster-whisper compute_type (e.g. int8_float16 for GPU, default for CPU)",
    )
    parser.add_argument(
        "--whisper-language",
        type=str,
        default="",
        help="Override Whisper language code (empty → derive from Moonshine tag)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_DEFAULT_JSON_REPORT,
        help=f"Write full per-line JSON report (default: {_DEFAULT_JSON_REPORT})",
    )
    args = parser.parse_args()

    WhisperModel = try_import_whisper_model()
    cer_fn = try_import_cer()
    if WhisperModel is None:
        print(
            "Missing faster-whisper. Install: pip install -r scripts/tts-g2p-intelligibility-requirements.txt",
            file=sys.stderr,
        )
        return 2
    if cer_fn is None:
        print(
            "Missing jiwer. Install: pip install -r scripts/tts-g2p-intelligibility-requirements.txt",
            file=sys.stderr,
        )
        return 2

    path = args.wiki_text.resolve()

    if args.languages.strip():
        want_langs: Set[str] = {
            normalize_moonshine_language_tag(x.strip())
            for x in args.languages.split(",")
            if x.strip()
        }
    else:
        want_langs = set(all_moonshine_tts_language_tags())

    wiki_corpora: Dict[str, str] = {}

    if path.is_dir():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if not args.no_wiki_download:
            try:
                require_datasets_compatible_with_python()
            except MoonshineError as e:
                print(e, file=sys.stderr)
                return 2
        try:
            all_sections, wiki_corpora = ensure_wiki_text_files(
                path,
                want_langs,
                lines_per_lang=args.wiki_lines_per_lang,
                allow_download=not args.no_wiki_download,
            )
        except MoonshineError as e:
            print(e, file=sys.stderr)
            return 2
        sections = {k: v for k, v in all_sections.items() if k in want_langs}
        missing = want_langs - set(sections.keys())
        if missing:
            print(
                f"Warning: no wiki-text lines for: {sorted(missing)}",
                file=sys.stderr,
            )
            for m in sorted(missing):
                reason = wiki_corpora.get(m, "unknown (not attempted or no metadata)")
                print(f"  — {m}: {reason}", file=sys.stderr)
    else:
        if not path.is_file():
            print(f"wiki-text file not found: {path}", file=sys.stderr)
            return 2
        all_sections = load_wiki_texts(path)
        for k in all_sections:
            wiki_corpora[k] = f"sectioned file {path}"
        if args.languages.strip():
            sections = {k: v for k, v in all_sections.items() if k in want_langs}
            missing = want_langs - set(sections.keys())
            if missing:
                print(
                    f"Warning: no wiki-text lines for: {sorted(missing)}",
                    file=sys.stderr,
                )
                for m in sorted(missing):
                    reason = wiki_corpora.get(m, "no section for this language in the wiki-text file")
                    print(f"  — {m}: {reason}", file=sys.stderr)
        else:
            sections = dict(all_sections)

    if not sections:
        print("No lines to evaluate.", file=sys.stderr)
        return 2

    compute_type = args.whisper_compute_type
    if compute_type == "default":
        compute_type = "int8" if args.whisper_device == "cpu" else "float16"

    print(
        f"Loading Whisper model {args.whisper_model!r} (device={args.whisper_device!r}, "
        f"compute_type={compute_type!r})…",
        file=sys.stderr,
    )
    model = WhisperModel(
        args.whisper_model,
        device=args.whisper_device,
        compute_type=compute_type,
    )

    ensure_piper_espeak_phonemizer_uses_nfc_for_eval()

    tts_wav_cache_dir: Optional[Path] = None
    if not args.no_tts_wav_cache:
        tts_wav_cache_dir = args.tts_wav_cache_dir.resolve()
        tts_wav_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"TTS wav cache directory: {tts_wav_cache_dir}", file=sys.stderr)
        if args.no_cache:
            print(
                "TTS wav cache: write-only (--no-cache); skipping reads, still saving .wav and .meta.json.",
                file=sys.stderr,
            )

    tts_data_root: Optional[Path] = None
    if not args.no_tts_root_overlay:
        tts_data_root = args.tts_root.resolve()

    report: Dict[str, Any] = {
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "whisper_compute_type": compute_type,
        "wiki_text_path": str(path),
        "wiki_lines_per_lang": args.wiki_lines_per_lang,
        "max_lines_per_lang": args.max_lines_per_lang,
        "tts_wav_cache_dir": str(tts_wav_cache_dir) if tts_wav_cache_dir is not None else None,
        "tts_wav_cache_skip_read": bool(tts_wav_cache_dir is not None and args.no_cache),
        "tts_root": str(tts_data_root) if tts_data_root is not None else None,
        "tts_root_overlay_disabled": bool(args.no_tts_root_overlay),
        "wiki_text_corpus_by_language": {
            lang: wiki_corpora.get(lang, "unknown") for lang in sorted(sections.keys())
        },
        "languages": {},
    }

    for lang in sorted(sections.keys()):
        loaded = sections[lang]
        if args.max_lines_per_lang > 0:
            lines = loaded[: args.max_lines_per_lang]
        else:
            lines = loaded
        whisper_lang = (
            args.whisper_language.strip() or moonshine_tag_to_whisper_language(lang) or None
        )
        if len(lines) != len(loaded):
            line_note = f"{len(lines)} of {len(loaded)} loaded lines"
        else:
            line_note = f"{len(lines)} lines"
        print(f"\n=== {lang} ({line_note}) whisper_lang={whisper_lang!r} ===", file=sys.stderr)
        try:
            summary, line_rows = run_language(
                lang,
                lines,
                kokoro_voice=args.kokoro_voice,
                piper_voice=args.piper_voice,
                cache_root=args.asset_cache,
                tts_data_root=tts_data_root,
                whisper_model=model,
                whisper_lang=whisper_lang,
                cer_fn=cer_fn,
                upstream_kokoro_device=(
                    args.upstream_kokoro_device.strip()
                    if args.upstream_kokoro_device.strip()
                    else None
                ),
                tts_wav_cache_dir=tts_wav_cache_dir,
                tts_skip_wav_cache_read=bool(tts_wav_cache_dir is not None and args.no_cache),
            )
        except MoonshineError as e:
            print(f"Skipping {lang}: {e}", file=sys.stderr)
            report["languages"][lang] = {"error": str(e)}
            continue

        report["languages"][lang] = {
            **summary,
            "lines": [line_result_to_dict(r) for r in line_rows],
        }

        print(f"  Moonshine TTS voice: {summary['moonshine_tts_voice']}", file=sys.stderr)
        print(f"  Catalog kokoro_*: {summary['kokoro_voice_catalog']}", file=sys.stderr)
        print(f"  Catalog piper_*: {summary['piper_voice_catalog']}", file=sys.stderr)
        if summary.get("upstream_kokoro_skip_reason"):
            print(
                f"  Upstream kokoro (pip): skipped — {summary['upstream_kokoro_skip_reason']}",
                file=sys.stderr,
            )
        if summary.get("upstream_piper_skip_reason"):
            print(
                f"  Upstream piper (pip): skipped — {summary['upstream_piper_skip_reason']}",
                file=sys.stderr,
            )

        am = summary["average_cer_moonshine_internal_g2p"]
        ak = summary["average_cer_upstream_kokoro_python"]
        ap = summary["average_cer_upstream_piper_python"]
        if am is not None:
            print(
                f"  Average CER (Moonshine internal G2P + native TTS): {am:.4f}",
                file=sys.stderr,
            )
        else:
            print("  Average CER (Moonshine internal G2P): n/a", file=sys.stderr)
        if ak is not None:
            print(
                f"  Average CER (pip kokoro / KPipeline G2P): {ak:.4f}",
                file=sys.stderr,
            )
        else:
            print("  Average CER (pip kokoro): n/a", file=sys.stderr)
        if ap is not None:
            print(
                f"  Average CER (pip piper-tts): {ap:.4f}",
                file=sys.stderr,
            )
        else:
            print("  Average CER (pip piper-tts): n/a", file=sys.stderr)

        print(
            f"{lang}\tlines={summary['num_lines']}\t"
            f"cer_moonshine_internal_g2p={_fmt_metric(am)}\t"
            f"cer_pip_kokoro={_fmt_metric(ak)}\t"
            f"cer_pip_piper={_fmt_metric(ap)}",
            flush=True,
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote JSON report to {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
