"""Standalone training recipe for the moonshine-micro on-device word classifier.

This package trains a compact MobileNetV2-style log-mel classifier (``WordCNN``)
over a user-defined command vocabulary and exports it to the int8 LiteRT format
consumed by the RP2350 firmware in ``moonshine-micro/stt``.

The class list is driven entirely by ``words.txt``; a reserved ``_unknown_``
reject class is appended automatically so the model can ignore out-of-vocabulary
speech.
"""

from .words import UNKNOWN_LABEL, folder_for_word, load_words, resolve_classes

__all__ = [
    "UNKNOWN_LABEL",
    "folder_for_word",
    "load_words",
    "resolve_classes",
]
