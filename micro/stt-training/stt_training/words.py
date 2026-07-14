"""Vocabulary handling: load ``words.txt`` and turn it into model classes.

The word list is the single source of truth for the class set. Everything else
(data folders, model outputs, firmware labels) is derived from it.
"""

from __future__ import annotations

from pathlib import Path

# Reserved reject class, appended automatically to every model. Clips for it
# come from generic People's Speech utterances that contain none of the command
# words (see tools/mine_peoples_speech.py --unknown). On device, a prediction of
# this label means "no command was spoken" and should be ignored.
UNKNOWN_LABEL = "_unknown_"


def folder_for_word(word: str) -> str:
    """Filesystem-safe label folder for a word.

    Lower-cases and strips characters that are illegal in folder names so the
    data-gathering scripts and the dataset loader agree on the on-disk layout
    ``<corpus>/<label>/*.wav``.
    """
    safe = word.strip().lower()
    for ch in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        safe = safe.replace(ch, "_")
    return safe or "unknown"


def load_words(path: str | Path) -> list[str]:
    """Load command words from ``words.txt`` (one per line, ``#`` comments)."""
    words: list[str] = []
    seen: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label = folder_for_word(line)
        if label == UNKNOWN_LABEL:
            # The reject class is reserved and added automatically.
            continue
        if label in seen:
            print(f"  skip duplicate word: {line!r}")
            continue
        seen.add(label)
        words.append(label)
    if not words:
        raise ValueError(f"No command words loaded from {path}")
    return words


def resolve_classes(path: str | Path, include_unknown: bool = True) -> list[str]:
    """Return the ordered class list: command words plus the reject class.

    The class order defines the model's output-logit order and is preserved all
    the way to the firmware's ``classes.h``.
    """
    classes = load_words(path)
    if include_unknown:
        classes.append(UNKNOWN_LABEL)
    return classes
