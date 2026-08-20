"""Audio caches, ATCOSIM, UWB-ATCC, ATCO2, and the yodas-en-replay corpus.

Imported only after ``require_lora_deps()``.
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from moonshine_voice.lora.manifest import (
    Utterance,
    apply_text_mode,
    choose_text_mode,
)

SAMPLE_RATE = 16_000
ATCOSIM_REPO = "Jzuluaga/atcosim_corpus"
SPLITS_REPO = "moonshine-ai/atcosim-speaker-disjoint-splits"
UWB_REPO = "Jzuluaga/uwb_atcc"
UWB_SPLITS_REPO = "moonshine-ai/uwb-atcc-session-disjoint-splits"
UWB_SHARED_SESSION = "TWR-34720N"
ATCO2_REPO = "Jzuluaga/atco2_corpus_1h"
REPLAY_REPO = "moonshine-ai/yodas-en-replay"
LIBRISPEECH = ("openslr/librispeech_asr", "clean/test/0000.parquet")


def to_mono_16k(data, rate):
    """Resample to 16 kHz mono float32. ATCOSIM ships 32 kHz; skip this and the
    baseline looks worse than the model deserves."""
    if data.ndim > 1:
        data = data.mean(axis=1)
    if int(rate) != SAMPLE_RATE:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(rate), SAMPLE_RATE)
        data = resample_poly(
            data, SAMPLE_RATE // gcd, int(rate) // gcd
        ).astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def load_wave(path: str) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32")
    return to_mono_16k(data, rate)


@dataclass
class AtcosimIndex:
    """Speaker-disjoint ATCOSIM rows, no audio yet.

    ``other`` is every utterance in neither the train pool nor the scored
    test set: the glossary source in the notebook, standing in for a
    customer's own word list.
    """

    train: List[Utterance]
    scored: List[Utterance]
    other: List[Utterance]


def index_atcosim() -> AtcosimIndex:
    """Speaker-disjoint train pool and scored test utterances, no audio yet.

    ATCOSIM's widely-used Hub split is utterance-random: all four scored
    speakers also appear in train. Training on that overstates the adaptation
    win by about 6 WER points. The published split definition holds out whole
    speakers instead.
    """
    from huggingface_hub import HfFileSystem, hf_hub_download

    splits = {}
    with open(
        hf_hub_download(SPLITS_REPO, "atcosim_splits.csv", repo_type="dataset")
    ) as handle:
        for row in csv.DictReader(handle):
            splits[row["id"]] = row

    fs = HfFileSystem()
    train, scored, other = [], [], []
    for remote in sorted(fs.glob(f"datasets/{ATCOSIM_REPO}/data/*.parquet")):
        shard = remote.split("/")[-1]
        with fs.open(remote, "rb") as handle:
            table = pq.read_table(handle, columns=["id", "text", "duration"])
        ids = table.column("id").to_pylist()
        texts = table.column("text").to_pylist()
        durations = table.column("duration").to_pylist()
        for i, (utt, text, seconds) in enumerate(zip(ids, texts, durations)):
            meta = splits.get(utt)
            if meta is None:
                continue
            row = Utterance(
                text=text,
                seconds=float(seconds),
                utterance_id=utt,
                speaker=meta["speaker"],
                shard=shard,
                row=i,
            )
            in_scored = meta["scored"] == "True"
            in_train = meta["speaker_disjoint_train"] == "True"
            if in_scored:
                scored.append(row)
            if in_train:
                train.append(row)
            if not in_scored and not in_train:
                other.append(row)
    return AtcosimIndex(train=train, scored=scored, other=other)


def decode_parquet_audio(repo: str, rows: Sequence[Utterance]) -> List[np.ndarray]:
    """Decode utterances, opening each parquet file once rather than once per row."""
    from huggingface_hub import hf_hub_download

    waves: List[Optional[np.ndarray]] = [None] * len(rows)
    by_shard = {}
    for i, row in enumerate(rows):
        by_shard.setdefault(row.shard, []).append(i)
    for shard, which in by_shard.items():
        path = hf_hub_download(repo, f"data/{shard}", repo_type="dataset")
        column = pq.read_table(path, columns=["audio"]).column("audio")
        for i in which:
            blob = column[rows[i].row].as_py()
            data, rate = sf.read(io.BytesIO(blob["bytes"]), dtype="float32")
            waves[i] = to_mono_16k(data, rate)
        del column
    return waves  # type: ignore[return-value]


def decode_atcosim(rows: Sequence[Utterance]) -> List[np.ndarray]:
    """Decode ATCOSIM utterances from the Hub parquet shards."""
    return decode_parquet_audio(ATCOSIM_REPO, rows)


def decode_uwb_atcc(rows: Sequence[Utterance]) -> List[np.ndarray]:
    """Decode UWB-ATCC utterances from the Hub parquet shards."""
    return decode_parquet_audio(UWB_REPO, rows)


def decode_atco2(rows: Sequence[Utterance]) -> List[np.ndarray]:
    """Decode ATCO2-test-set-1h utterances. Eval only; never train on this."""
    return decode_parquet_audio(ATCO2_REPO, rows)


def uwb_session(utt_id: str) -> str:
    """Session from ids like ``uwb-atcc_TWR-34720N_...``."""
    parts = (utt_id or "").split("_")
    return parts[1] if len(parts) > 1 else (utt_id or "")


def session_disjoint_train(session: str, scored_sessions) -> bool:
    """True when this session is absent from the scored (test) half."""
    return session not in scored_sessions


@dataclass
class UwbAtccIndex:
    """Session-disjoint UWB-ATCC rows, no audio yet.

    The published train/test split shares one session (``TWR-34720N``).
    ``train`` drops that session so an in-domain number is domain only.
    """

    train: List[Utterance]
    scored: List[Utterance]


def _iter_parquet_meta(repo: str, columns=("id", "text", "duration")):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for remote in sorted(fs.glob(f"datasets/{repo}/data/*.parquet")):
        shard = remote.split("/")[-1]
        with fs.open(remote, "rb") as handle:
            table = pq.read_table(handle, columns=list(columns))
        yield shard, table


def _uwb_index_from_shards() -> UwbAtccIndex:
    scored, train_raw = [], []
    for shard, table in _iter_parquet_meta(UWB_REPO):
        ids = table.column("id").to_pylist()
        texts = table.column("text").to_pylist()
        durations = table.column("duration").to_pylist()
        is_test = shard.startswith("test-")
        for i, (utt, text, seconds) in enumerate(zip(ids, texts, durations)):
            row = Utterance(
                text=text,
                seconds=float(seconds),
                utterance_id=utt,
                speaker=uwb_session(utt),
                shard=shard,
                row=i,
            )
            if is_test:
                scored.append(row)
            else:
                train_raw.append(row)
    scored_sessions = {r.speaker for r in scored}
    train = [
        r for r in train_raw if session_disjoint_train(r.speaker, scored_sessions)
    ]
    return UwbAtccIndex(train=train, scored=scored)


def index_uwb_atcc() -> UwbAtccIndex:
    """Session-disjoint UWB-ATCC train pool and official test utterances.

    Prefers the published no-audio split definition when present; otherwise
    builds the same split from the Hub shards (train-* vs test-*, then drop
    any session that appears on the test side).
    """
    from huggingface_hub import hf_hub_download

    try:
        split_path = hf_hub_download(
            UWB_SPLITS_REPO, "uwb_atcc_splits.csv", repo_type="dataset"
        )
    except Exception:
        return _uwb_index_from_shards()

    splits = {}
    with open(split_path) as handle:
        for row in csv.DictReader(handle):
            splits[row["id"]] = row

    train, scored = [], []
    for shard, table in _iter_parquet_meta(UWB_REPO):
        ids = table.column("id").to_pylist()
        texts = table.column("text").to_pylist()
        durations = table.column("duration").to_pylist()
        for i, (utt, text, seconds) in enumerate(zip(ids, texts, durations)):
            meta = splits.get(utt)
            if meta is None:
                continue
            row = Utterance(
                text=text,
                seconds=float(seconds),
                utterance_id=utt,
                speaker=meta.get("session") or uwb_session(utt),
                shard=shard,
                row=i,
            )
            if meta.get("scored") == "True":
                scored.append(row)
            if meta.get("session_disjoint_train") == "True":
                train.append(row)
    if not train or not scored:
        return _uwb_index_from_shards()
    return UwbAtccIndex(train=train, scored=scored)


def uwb_split_csv_rows(index: Optional[UwbAtccIndex] = None) -> List[dict]:
    """No-audio split definition, one row per utterance. Safe to publish."""
    indexed = index or _uwb_index_from_shards()
    train_ids = {r.utterance_id for r in indexed.train}
    scored_ids = {r.utterance_id for r in indexed.scored}
    rows = []
    seen = set()
    for row in list(indexed.scored) + list(indexed.train):
        if row.utterance_id in seen:
            continue
        seen.add(row.utterance_id)
        rows.append(
            {
                "id": row.utterance_id,
                "session": row.speaker,
                "scored": str(row.utterance_id in scored_ids),
                "session_disjoint_train": str(row.utterance_id in train_ids),
            }
        )
    return rows


def index_atco2() -> List[Utterance]:
    """ATCO2-test-set-1h. Held-out transfer eval; do not train on this."""
    rows = []
    for shard, table in _iter_parquet_meta(ATCO2_REPO):
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        n = len(next(iter(columns.values())))
        ids = columns.get("id") or [f"{shard}:{i}" for i in range(n)]
        texts = columns.get("text") or columns.get("transcript") or [""] * n
        durations = columns.get("duration") or [None] * n
        for i in range(n):
            seconds = durations[i]
            rows.append(
                Utterance(
                    text=texts[i],
                    seconds=float(seconds) if seconds is not None else None,
                    utterance_id=ids[i],
                    speaker=None,
                    shard=shard,
                    row=i,
                )
            )
    return rows


def hours_of(rows: Sequence[Utterance]) -> float:
    return sum(r.seconds or 0.0 for r in rows) / 3600.0


def take_hours(rows: Sequence[Utterance], hours: Optional[float]) -> List[Utterance]:
    """Prefix of ``rows`` totalling ``hours``. ``None`` keeps all of them."""
    if hours is None:
        return list(rows)
    budget, taken, chosen = hours * 3600.0, 0.0, []
    for row in rows:
        if taken >= budget:
            break
        chosen.append(row)
        taken += row.seconds or 0.0
    return chosen


def build_cache(
    name: str,
    work_dir: Path,
    hours: float,
    source: Callable[[float], Iterable[Tuple[np.ndarray, str]]],
    encode_text: Callable[[str], List[int]],
) -> dict:
    """Decode audio into one flat int16 blob plus an index of offsets and tokens."""
    work_dir.mkdir(parents=True, exist_ok=True)
    blob_path, index_path = work_dir / f"{name}.i16", work_dir / f"{name}_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["hours"] >= hours - 0.02:
            print(f"{name}: cache hit, {index['count']} utts / {index['hours']:.2f} h")
            return index
    entries, offset, started = [], 0, time.time()
    with open(blob_path, "wb") as sink:
        for wave, text in source(hours):
            samples = np.clip(wave * 32767.0, -32768, 32767).astype(np.int16)
            sink.write(samples.tobytes())
            entries.append(
                {
                    "offset": offset,
                    "samples": int(samples.shape[0]),
                    "tokens": encode_text(text),
                }
            )
            offset += int(samples.shape[0])
            if offset / SAMPLE_RATE / 3600 >= hours:
                break
    index = {
        "count": len(entries),
        "samples": offset,
        "hours": offset / SAMPLE_RATE / 3600,
        "entries": entries,
    }
    index_path.write_text(json.dumps(index))
    print(
        f"{name}: {len(entries)} utts / {index['hours']:.2f} h "
        f"({time.time() - started:.0f}s)"
    )
    return index


def open_blob(work_dir: Path, name: str, index: dict):
    return np.memmap(
        work_dir / f"{name}.i16",
        dtype=np.int16,
        mode="r",
        shape=(index["samples"],),
    )


def file_source(
    rows: Sequence[Utterance], hours: float, text_mode: str
) -> Iterator[Tuple[np.ndarray, str]]:
    chosen = take_hours(rows, hours)
    for row in chosen:
        if not row.audio:
            raise ValueError(f"utterance {row.utterance_id!r} has no audio path")
        yield load_wave(row.audio), apply_text_mode(row.text, text_mode)


def parquet_source(
    decode, pool: Sequence[Utterance], hours: float, text_mode: str
) -> Iterator[Tuple[np.ndarray, str]]:
    chosen = take_hours(pool, hours)
    for row, wave in zip(chosen, decode(chosen)):
        yield wave, apply_text_mode(row.text, text_mode)


def atcosim_source(
    pool: Sequence[Utterance], hours: float, text_mode: str
) -> Iterator[Tuple[np.ndarray, str]]:
    yield from parquet_source(decode_atcosim, pool, hours, text_mode)


def uwb_atcc_source(
    pool: Sequence[Utterance], hours: float, text_mode: str
) -> Iterator[Tuple[np.ndarray, str]]:
    yield from parquet_source(decode_uwb_atcc, pool, hours, text_mode)


def replay_source(hours: float, repo: str = REPLAY_REPO):
    from huggingface_hub import hf_hub_download, list_repo_files

    shards = sorted(
        f
        for f in list_repo_files(repo, repo_type="dataset")
        if f.endswith(".arrow")
    )
    seconds = 0.0
    for shard in shards:
        if seconds >= hours * 3600:
            break
        table = pa.ipc.open_file(
            hf_hub_download(repo, shard, repo_type="dataset")
        ).read_all()
        audio, text, dur = (
            table.column("audio"),
            table.column("text"),
            table.column("duration"),
        )
        for i in range(table.num_rows):
            if seconds >= hours * 3600:
                break
            data, rate = sf.read(
                io.BytesIO(audio[i].as_py()["bytes"]), dtype="float32"
            )
            seconds += float(dur[i].as_py())
            yield to_mono_16k(data, rate), text[i].as_py()
        del table


def librispeech_eval(limit: Optional[int], seed: int = 0):
    from huggingface_hub import hf_hub_download

    table = pq.read_table(
        hf_hub_download(LIBRISPEECH[0], LIBRISPEECH[1], repo_type="dataset"),
        columns=["audio", "text"],
    )
    audio, text = table.column("audio"), table.column("text")
    indices = list(range(table.num_rows))
    if limit is not None and limit < len(indices):
        indices = sorted(
            np.random.default_rng(seed).choice(len(indices), limit, replace=False).tolist()
        )
    refs, waves = [], []
    for i in indices:
        data, rate = sf.read(io.BytesIO(audio[i].as_py()["bytes"]), dtype="float32")
        waves.append(to_mono_16k(data, rate))
        refs.append(text[i].as_py())
    return refs, waves
