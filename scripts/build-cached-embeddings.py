#!/usr/bin/env python3
"""Pre-compute embeddings for Moonshine Voice's library-level phrases.

Writes ``python/src/moonshine_voice/assets/cached_embeddings.tsv``, which
is loaded at runtime by :class:`moonshine_voice.CachedEmbeddings` so that
``DialogFlow``'s default yes/no ``Confirm`` matcher (and any other
library-level embedding consumer) does not need to embed its string
constants at process startup.

Usage
-----
    python scripts/build-cached-embeddings.py
    python scripts/build-cached-embeddings.py --model embeddinggemma-300m --quantization q4
    python scripts/build-cached-embeddings.py --output /tmp/cached.tsv

Run this whenever the library-level phrase sets change (e.g.
``_DEFAULT_YES_PHRASES`` / ``_DEFAULT_NO_PHRASES`` in ``dialog_flow.py``)
or when you want to ship embeddings for a different quantization.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
PY_SRC = os.path.join(REPO_ROOT, "python", "src")
if os.path.isdir(PY_SRC) and PY_SRC not in sys.path:
    sys.path.insert(0, PY_SRC)

from moonshine_voice.alphanumeric_listener import (  # noqa: E402
    _CLEAR_WORDS,
    _DIGIT_WORDS,
    _LETTER_WORDS,
    _SPECIAL_CHAR_WORDS,
    _SPELL_OUT_DIGITS,
    _SPELL_OUT_LETTERS,
    _SPELL_OUT_SYMBOLS,
    _STOP_WORDS,
    _UNDO_WORDS,
    _UPPER_MODIFIERS,
)
from moonshine_voice.cached_embeddings import (  # noqa: E402
    default_cached_embeddings_path,
    write_cached_embeddings_tsv,
)
from moonshine_voice.dialog_flow import (  # noqa: E402
    _DEFAULT_NO_PHRASES,
    _DEFAULT_YES_PHRASES,
)
from moonshine_voice.download import get_embedding_model  # noqa: E402
from moonshine_voice.intent_recognizer import IntentRecognizer  # noqa: E402


def _collect_library_phrases() -> List[Tuple[str, str]]:
    """Return ``[(group, phrase)]`` pairs for every library-level constant.

    Covers the yes/no defaults used by ``Confirm`` as well as every
    spoken form recognised by :class:`AlphanumericMatcher` (letter,
    digit, special-character, upper-case modifier, undo, clear and stop
    words).  Pre-embedding these makes them instantly available to any
    consumer that wires a :class:`CachedEmbeddings` backend – notably a
    future fuzzy-matching path for alphanumeric commands – without the
    user paying an embedding cost on their first utterance.

    Deduplicates by (normalized) phrase while keeping the first group
    each phrase was seen in, so the TSV doesn't contain duplicate rows.
    """
    seen: Dict[str, Tuple[str, str]] = {}

    def _add(group: str, phrases) -> None:
        for p in phrases:
            key = p.strip().lower()
            if not key or key in seen:
                continue
            seen[key] = (group, p)

    _add("confirm_yes", _DEFAULT_YES_PHRASES)
    _add("confirm_no", _DEFAULT_NO_PHRASES)
    _add("alpha_letter", _LETTER_WORDS.keys())
    _add("alpha_digit", _DIGIT_WORDS.keys())
    _add("alpha_special", _SPECIAL_CHAR_WORDS.keys())
    _add("alpha_upper_modifier", _UPPER_MODIFIERS)
    _add("alpha_undo", _UNDO_WORDS)
    _add("alpha_clear", _CLEAR_WORDS)
    _add("alpha_stop", _STOP_WORDS)

    # Spoken-form phrases used by :class:`AlphanumericListener`'s TTS
    # repeat-back and :func:`moonshine_voice.dialog_flow.spell_out`.
    # ``_SPELL_OUT_LETTERS`` values are the bare phonetic form ("ay",
    # "bee"); the ``capital <form>`` variants are what gets spoken for
    # upper-case characters, so both are embedded.
    _add("spell_out_letter", _SPELL_OUT_LETTERS.values())
    _add(
        "spell_out_letter_capital",
        (f"capital {v}" for v in _SPELL_OUT_LETTERS.values()),
    )
    _add("spell_out_digit", _SPELL_OUT_DIGITS.values())
    _add("spell_out_symbol", _SPELL_OUT_SYMBOLS.values())

    return list(seen.values())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-compute embeddings for library-level phrases and write "
            "them to cached_embeddings.tsv."
        ),
    )
    parser.add_argument(
        "--model",
        default="embeddinggemma-300m",
        help="Embedding model name (default: embeddinggemma-300m).",
    )
    parser.add_argument(
        "--quantization",
        default="q4",
        help="Embedding model variant (default: q4).",
    )
    parser.add_argument(
        "--output",
        default=default_cached_embeddings_path(),
        help=(
            "Output TSV path. Defaults to the packaged location so the "
            "file is shipped with moonshine_voice."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be embedded without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    entries = _collect_library_phrases()
    if not entries:
        print("No phrases to embed; aborting.", file=sys.stderr)
        sys.exit(1)

    group_counts: Dict[str, int] = {}
    for group, _phrase in entries:
        group_counts[group] = group_counts.get(group, 0) + 1
    print(f"Collected {len(entries)} library phrase(s):")
    for group, count in sorted(group_counts.items()):
        print(f"  [{group}] {count}")

    if args.dry_run:
        for group, phrase in entries:
            print(f"  [{group}] {phrase!r}")
        return

    print(
        f"\nLoading embedding model {args.model!r} "
        f"(variant={args.quantization!r}) – first run may download it..."
    )
    model_path, model_arch = get_embedding_model(args.model, args.quantization)
    recognizer = IntentRecognizer(
        model_path=model_path,
        model_arch=model_arch,
        model_variant=args.quantization,
    )

    try:
        rows: List[Tuple[str, List[float]]] = []
        total = len(entries)
        for idx, (group, phrase) in enumerate(entries, start=1):
            emb = recognizer.calculate_embedding(phrase)
            rows.append((phrase, emb))
            if idx == 1 or idx == total or idx % 25 == 0:
                print(
                    f"  [{idx}/{total}] embedded {phrase!r} "
                    f"-> dim={len(emb)} (group={group})"
                )
    finally:
        recognizer.close()

    dim = len(rows[0][1]) if rows else 0
    metadata = {
        "model_name": args.model,
        "model_variant": args.quantization,
        "model_arch": getattr(model_arch, "name", str(model_arch)),
        "embedding_dim": str(dim),
        "phrase_count": str(len(rows)),
    }

    written = write_cached_embeddings_tsv(
        args.output, rows, metadata=metadata
    )
    print(f"\nWrote {written} row(s) to {args.output}")
    print(
        f"  model={args.model} variant={args.quantization} dim={dim} "
        f"arch={metadata['model_arch']}"
    )


if __name__ == "__main__":
    main()
