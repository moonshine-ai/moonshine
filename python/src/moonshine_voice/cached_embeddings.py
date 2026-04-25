"""Load pre-computed sentence embeddings from a packaged TSV file.

Embedding library-level string constants (the default yes/no phrases used
by ``Confirm`` prompts, for example) is relatively slow – the first
``PhraseMatcher`` for them can take a couple of seconds to build at
runtime while the embedding model forward-passes each phrase.  That cost
is entirely deterministic: the phrases and the model don't change between
runs, so there's no reason to pay for it repeatedly.

This module ships those embeddings as *data* instead of *computation*:

1. ``scripts/build-cached-embeddings.py`` embeds the library phrases once
   with a chosen embedding model/variant and writes the result to
   ``assets/cached_embeddings.tsv`` inside the package.
2. At import time, :class:`CachedEmbeddings` reads that TSV into a
   ``{phrase: vector}`` dict.
3. Anywhere the library needs an :class:`EmbeddingBackend`, it can wrap
   the real recognizer with a :class:`CachedEmbeddings` instance:
   cache hits return instantly, cache misses (typically incoming user
   utterances) fall through to the recognizer.

TSV format
----------
``# key: value`` header lines followed by one phrase per data line::

    # model_name: embeddinggemma-300m
    # model_variant: q4
    # embedding_dim: 768
    # phrase_count: 18
    yes\\t0.0123\\t-0.0456\\t...
    yeah\\t0.0987\\t0.0321\\t...

Fields are tab-separated; the first column is the phrase (must not
contain tabs or newlines), the remaining columns are the embedding
components as decimal floats.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


class _EmbeddingBackend(Protocol):
    """Narrow form of :class:`moonshine_voice.EmbeddingBackend`.

    Redefined here so this module doesn't import ``dialog_flow`` (which
    would create a cycle).
    """

    def calculate_embedding(self, sentence: str) -> Sequence[float]: ...

    def distance(
        self, embedding_a: Sequence[float], embedding_b: Sequence[float]
    ) -> float: ...


_DEFAULT_TSV_FILENAME = "cached_embeddings.tsv"


def default_cached_embeddings_path() -> str:
    """Return the absolute path of the packaged cached embeddings TSV."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", _DEFAULT_TSV_FILENAME)


class CachedEmbeddings:
    """A lookup table of pre-computed sentence embeddings.

    Instances duck-type as an embedding backend: call
    :meth:`calculate_embedding` with a phrase and you get back a vector.
    Hits are served from memory in O(1); misses fall through to the
    optional ``fallback`` backend (typically the real
    :class:`IntentRecognizer`), or raise :class:`KeyError` if none is
    configured.

    Parameters
    ----------
    path:
        Optional path to the TSV file.  Defaults to the TSV packaged with
        :mod:`moonshine_voice`.  If the file is missing the cache is
        simply empty – behaviour is then governed entirely by
        ``fallback``.
    fallback:
        Optional embedding backend consulted on cache misses.  If
        ``None``, :meth:`calculate_embedding` raises on misses.
    expected_model_name / expected_model_variant:
        Optional strings.  When the TSV metadata does not match these,
        the cache is *disabled* (wiped in-memory and a warning is logged
        to stderr) so you never accidentally mix embeddings from
        different models.  Leave ``None`` to skip validation.
    """

    def __init__(
        self,
        *,
        path: Optional[str] = None,
        fallback: Optional[_EmbeddingBackend] = None,
        expected_model_name: Optional[str] = None,
        expected_model_variant: Optional[str] = None,
    ):
        self._fallback = fallback
        self._cache: Dict[str, List[float]] = {}
        self._metadata: Dict[str, str] = {}
        self._active = False
        self._path = path or default_cached_embeddings_path()

        if not os.path.exists(self._path):
            return

        try:
            self._load(self._path)
        except Exception as e:
            print(
                f"CachedEmbeddings: failed to load {self._path}: {e}",
                file=sys.stderr,
            )
            self._cache.clear()
            self._metadata.clear()
            return

        if expected_model_name is not None:
            got = self._metadata.get("model_name")
            if got != expected_model_name:
                print(
                    f"CachedEmbeddings: disabling cache – expected model "
                    f"{expected_model_name!r} but TSV has {got!r}",
                    file=sys.stderr,
                )
                self._cache.clear()
                return

        if expected_model_variant is not None:
            got = self._metadata.get("model_variant")
            if got != expected_model_variant:
                print(
                    f"CachedEmbeddings: disabling cache – expected variant "
                    f"{expected_model_variant!r} but TSV has {got!r}",
                    file=sys.stderr,
                )
                self._cache.clear()
                return

        self._active = len(self._cache) > 0

    # -- properties --------------------------------------------------------

    @property
    def active(self) -> bool:
        """``True`` when the cache has at least one usable entry."""
        return self._active

    @property
    def path(self) -> str:
        return self._path

    @property
    def metadata(self) -> Dict[str, str]:
        return dict(self._metadata)

    @property
    def phrases(self) -> List[str]:
        """Return the list of cached phrases (normalized keys)."""
        return list(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, sentence: str) -> bool:
        return self._normalize(sentence) in self._cache

    # -- lookup ------------------------------------------------------------

    def get(self, sentence: str) -> Optional[List[float]]:
        """Return the cached embedding or ``None`` if not present."""
        return self._cache.get(self._normalize(sentence))

    def calculate_embedding(self, sentence: str) -> List[float]:
        """Return the embedding for ``sentence``.

        Consults the in-memory cache first, then falls through to the
        configured ``fallback`` backend.  Raises ``KeyError`` if neither
        yields a result.
        """
        cached = self._cache.get(self._normalize(sentence))
        if cached is not None:
            return list(cached)
        if self._fallback is not None:
            return list(self._fallback.calculate_embedding(sentence))
        raise KeyError(
            f"CachedEmbeddings: no entry for {sentence!r} and no fallback backend"
        )

    def distance(
        self,
        embedding_a: Sequence[float],
        embedding_b: Sequence[float],
    ) -> float:
        """Return the cosine similarity between two embedding vectors.

        The cache itself does not implement similarity scoring – it
        delegates to the configured ``fallback`` backend (typically an
        :class:`~moonshine_voice.IntentRecognizer`) so the underlying
        native ``moonshine_calculate_embedding_distance`` C function is
        used.  Raises :class:`RuntimeError` when no fallback is
        configured.
        """
        if self._fallback is None:
            raise RuntimeError(
                "CachedEmbeddings.distance requires a fallback backend "
                "that implements distance()"
            )
        return float(self._fallback.distance(embedding_a, embedding_b))

    # -- I/O ---------------------------------------------------------------

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith("#"):
                    self._parse_meta(line[1:].strip())
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                phrase = parts[0]
                try:
                    vec = [float(x) for x in parts[1:]]
                except ValueError:
                    # Skip malformed rows rather than aborting load.
                    continue
                self._cache[self._normalize(phrase)] = vec

    def _parse_meta(self, s: str) -> None:
        if ":" not in s:
            return
        k, v = s.split(":", 1)
        self._metadata[k.strip()] = v.strip()

    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize a phrase for lookup.

        We store and look up phrases in a case-folded, whitespace-stripped
        form so the cache is forgiving about minor casing differences in
        the caller's registration.
        """
        return s.strip().lower()


# ---------------------------------------------------------------------------
# Writer – used by scripts/build-cached-embeddings.py
# ---------------------------------------------------------------------------


def write_cached_embeddings_tsv(
    path: str,
    entries: Iterable[Tuple[str, Sequence[float]]],
    *,
    metadata: Optional[Dict[str, str]] = None,
    float_format: str = "{:.7g}",
) -> int:
    """Write a TSV compatible with :class:`CachedEmbeddings`.

    Returns the number of phrase rows written.  Raises ``ValueError`` if a
    phrase contains a tab or newline character.
    """
    meta = dict(metadata or {})
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            if "\n" in str(v) or "\r" in str(v):
                raise ValueError(f"metadata value must not contain newlines: {k}")
            f.write(f"# {k}: {v}\n")
        for phrase, embedding in entries:
            if "\t" in phrase or "\n" in phrase or "\r" in phrase:
                raise ValueError(
                    f"phrase cannot contain tab/newline characters: {phrase!r}"
                )
            vals = "\t".join(float_format.format(float(x)) for x in embedding)
            f.write(f"{phrase}\t{vals}\n")
            count += 1
    return count
