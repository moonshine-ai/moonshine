"""Training-set manifests. Stdlib only: safe to import without the ``[lora]`` extra."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass
class Utterance:
    """One training or eval clip.

    ``audio`` is a filesystem path. Hub-backed corpora (ATCOSIM, replay) leave
    it empty and fill ``shard`` / ``row`` instead so the audio can be decoded
    from parquet without copying every file out first.
    """

    text: str
    audio: Optional[str] = None
    seconds: Optional[float] = None
    utterance_id: Optional[str] = None
    speaker: Optional[str] = None
    shard: Optional[str] = None
    row: Optional[int] = None
    extra: dict = field(default_factory=dict)


def choose_text_mode(texts: Sequence[str], requested: str = "auto") -> str:
    """Match a corpus's transcript style to what Moonshine emits.

    A corpus that is more than 90% uppercase (AMI is the usual case) is
    lowercased, because training on ALL-CAPS spends the adapter on a
    typography the WER normalizer then throws away. Everything else is left
    alone. ``requested`` of ``none`` or ``lower`` skips the heuristic.
    """
    if requested != "auto":
        return requested
    if not texts:
        return "none"
    upper = sum(1 for text in texts if text.isupper())
    if upper / len(texts) > 0.9:
        return "lower"
    return "none"


def apply_text_mode(text: str, mode: str) -> str:
    if mode == "lower":
        return text.lower()
    return text


def _resolve_audio(raw: str, data_root: Path) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((data_root / path).resolve())


def _utterance_from_mapping(row: dict, data_root: Path) -> Utterance:
    text = row.get("text") or row.get("transcript") or row.get("transcription")
    if not text or not str(text).strip():
        raise ValueError("each utterance needs a non-empty 'text' field")
    audio = row.get("audio") or row.get("path") or row.get("wav")
    seconds = row.get("seconds") or row.get("duration")
    return Utterance(
        text=str(text).strip(),
        audio=_resolve_audio(str(audio), data_root) if audio else None,
        seconds=float(seconds) if seconds is not None else None,
        utterance_id=row.get("id") or row.get("utterance_id"),
        speaker=row.get("speaker") or row.get("group"),
        shard=row.get("shard"),
        row=row.get("row"),
        extra={
            k: v
            for k, v in row.items()
            if k
            not in {
                "text",
                "transcript",
                "transcription",
                "audio",
                "path",
                "wav",
                "seconds",
                "duration",
                "id",
                "utterance_id",
                "speaker",
                "group",
                "shard",
                "row",
            }
        },
    )


def load_manifest(path: str, data_root: Optional[str] = None) -> List[Utterance]:
    """Load a JSONL, JSON, or TSV manifest of ``audio`` + ``text`` rows.

    Paths in the file are resolved against ``data_root``, or against the
    manifest's parent directory when that is omitted.

    Formats::

        {"audio": "clips/001.wav", "text": "the transcript"}     # JSONL, one per line
        {"utterances": [ ... ]}                                  # JSON
        clips/001.wav<TAB>the transcript                         # TSV
    """
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    root = Path(data_root) if data_root else manifest.parent
    raw = manifest.read_text(encoding="utf-8")
    suffix = manifest.suffix.lower()

    if suffix == ".tsv" or suffix == ".csv":
        rows = []
        for line_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t" if "\t" in line else ",", 1)
            if len(parts) != 2:
                raise ValueError(f"{manifest}:{line_no}: expected path<TAB>text")
            rows.append(_utterance_from_mapping(
                {"audio": parts[0].strip(), "text": parts[1].strip()}, root
            ))
        return rows

    if suffix == ".jsonl" or suffix == ".ndjson":
        rows = []
        for line_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{manifest}:{line_no}: {error}") from error
            if not isinstance(obj, dict):
                raise ValueError(f"{manifest}:{line_no}: expected a JSON object")
            rows.append(_utterance_from_mapping(obj, root))
        return rows

    obj = json.loads(raw)
    if isinstance(obj, dict) and "utterances" in obj:
        items: Iterable = obj["utterances"]
    elif isinstance(obj, list):
        items = obj
    else:
        raise ValueError(
            f"{manifest}: expected a JSON list, a JSONL file, or "
            '{"utterances": [...]}'
        )
    return [_utterance_from_mapping(item, root) for item in items]
