"""Mine People's Speech for command words (and generic "unknown" speech).

Scans MLCommons/peoples_speech transcripts and, for each utterance that either
contains a command word or -- in ``--unknown`` mode -- contains none of them,
fetches the audio, resamples it to 16 kHz, and records a JSONL row for the
clip extractor to align and cut.

Data source: the Hub's auto-generated Parquet export, read anonymously (no
account or token required) over HTTP range requests. We list the shards with
``HfFileSystem`` and read them column-by-column with pyarrow: only the tiny
``text`` column is pulled while scanning, and the (large) ``audio`` column is
downloaded only for the row groups that actually contain a hit. This replaces
the old ``datasets-server.huggingface.co`` REST API, which is queue-backed and
frequently unavailable (HTTP 503).

Two modes:
  command  (default)  one row per matched command occurrence -> data/mined/commands.jsonl
  --unknown           one row per non-command utterance      -> data/mined/unknown.jsonl

Output audio cache: ``<mined-dir>/audio/<clip>.wav``. Runs resume: rows already
recorded and audio already on disk are skipped.

Requires: ``pip install datasets soundfile`` (pyarrow + huggingface_hub come
with ``datasets``).

Example (scan 200k rows of the clean train split for commands)::

    python tools/mine_peoples_speech.py --words-file words.txt \
        --split train --limit 200000 --mined-dir data/mined
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

# Read Parquet through the plain `resolve` CDN, not Xet. Anonymous Xet access
# caches a storage token that expires after ~1 h and is shared process-wide, so
# long scans die with a 401 that reopening the file can't clear. The non-Xet
# path re-signs a fresh CDN URL on every range request. Must be set before
# huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import soundfile as sf

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from stt_training.words import UNKNOWN_LABEL, load_words  # noqa: E402

_DATASET = "MLCommons/peoples_speech"
_PARQUET_REF = "refs/convert/parquet"  # Hub's auto-converted Parquet branch
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_SPEAKER_RE = re.compile(r"_SLASH_|/")
TARGET_SR = 16000


# --------------------------------------------------------------------------
# Anonymous Parquet loader (column-selective HTTP range reads, no token).
# --------------------------------------------------------------------------
def _new_fs():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def _shard_paths(config, split):
    base = f"datasets/{_DATASET}@{_PARQUET_REF}/{config}/{split}"
    return sorted(p for p in _new_fs().ls(base, detail=False) if p.endswith(".parquet"))


class _ShardReader:
    """Row-group reader for one Parquet shard, resilient to expiring creds.

    Anonymous access to the Hub's Xet/presigned storage uses short-lived
    (~1 h) credentials, so a single long-lived handle 401s partway through a
    big scan. On any read error we sleep and reopen the shard with a *fresh*
    ``HfFileSystem`` (new credentials), which also rides out transient network
    blips. Reopening is cheap: only Parquet metadata is re-fetched.
    """

    def __init__(self, shard, max_attempts=6):
        import pyarrow.parquet as pq

        self._pq = pq
        self.shard = shard
        self.max_attempts = max_attempts
        self._open()

    def _open(self):
        self._fh = _new_fs().open(self.shard)
        self.pf = self._pq.ParquetFile(self._fh)

    @property
    def num_row_groups(self):
        return self.pf.num_row_groups

    def row_group_num_rows(self, rg):
        return self.pf.metadata.row_group(rg).num_rows

    def read_columns(self, rg, columns):
        delay, last = 2.0, None
        for attempt in range(self.max_attempts):
            try:
                return self.pf.read_row_group(rg, columns=columns)
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(
                    f"  [retry] read {Path(self.shard).name} rg{rg} "
                    f"attempt {attempt + 1}/{self.max_attempts}: "
                    f"{type(exc).__name__}",
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                try:
                    self._open()
                except Exception:  # noqa: BLE001
                    pass
        raise last


def iter_peoples_speech(config, split, offset, limit):
    """Yield ``(clip_id, transcript, speaker, fetch_audio)`` rows.

    Reads the Parquet export anonymously. ``fetch_audio()`` lazily pulls and
    decodes the audio for its row (downloading that row group's ``audio``
    column only on first use), so scanning stays cheap for non-matching rows.
    """
    shards = _shard_paths(config, split)
    seen = 0          # rows yielded (counts toward ``limit``)
    skip = max(0, int(offset))
    for shard in shards:
        if limit is not None and seen >= limit:
            return
        reader = _ShardReader(shard)
        for rg in range(reader.num_row_groups):
            if limit is not None and seen >= limit:
                return
            n_rows = reader.row_group_num_rows(rg)
            if skip >= n_rows:
                skip -= n_rows
                continue
            tbl = reader.read_columns(rg, ["id", "text"])
            ids = tbl.column("id").to_pylist()
            texts = tbl.column("text").to_pylist()
            audio_cache: dict[int, list] = {}

            def _fetch(_rg=rg, _i=None):
                if _rg not in audio_cache:
                    at = reader.read_columns(_rg, ["audio"])
                    audio_cache[_rg] = at.column("audio").to_pylist()
                entry = audio_cache[_rg][_i]
                data = entry.get("bytes") if isinstance(entry, dict) else None
                if not data:
                    return None
                try:
                    arr, sr = sf.read(io.BytesIO(data), always_2d=False)
                    return arr, int(sr)
                except Exception:  # noqa: BLE001
                    return None

            start = skip
            skip = 0
            for i in range(start, len(ids)):
                if limit is not None and seen >= limit:
                    return
                clip_id = str(ids[i] or f"peoples_speech_{seen}")
                speaker = _SPEAKER_RE.split(clip_id)[0][:64] or "unknown"
                yield clip_id, str(texts[i] or ""), speaker, (
                    lambda _i=i: _fetch(_i=_i)
                )
                seen += 1


# --------------------------------------------------------------------------
# Matching + audio helpers
# --------------------------------------------------------------------------
def _tokens_with_pos(text: str):
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text.lower())]


def find_matches(text, targets):
    """Return ``[(label, char_start, char_end), ...]`` for command words in text.

    ``targets`` is ``[(label, (tok0, tok1, ...)), ...]`` supporting multi-word
    phrases; all our default commands are single tokens.
    """
    toks = _tokens_with_pos(text)
    out = []
    for label, seq in targets:
        n = len(seq)
        for i in range(len(toks) - n + 1):
            if all(toks[i + j][0] == seq[j] for j in range(n)):
                out.append((label, toks[i][1], toks[i + n - 1][2]))
    return out


def _safe_name(clip_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", clip_id)[:180]


def save_audio_16k(fetch, out_path: Path) -> bool:
    if out_path.is_file():
        return True
    payload = fetch()
    if payload is None:
        return False
    arr, sr = payload
    y = np.asarray(arr, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        import torchaudio
        import torch

        y = torchaudio.functional.resample(torch.from_numpy(y), sr, TARGET_SR).numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, TARGET_SR, subtype="PCM_16")
    return True


def _load_seen_keys(path: Path) -> set:
    keys = set()
    if not path.is_file():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add((row.get("clip_id"), row.get("char_start"), tuple(row.get("words") or [])))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--words-file", default="words.txt")
    ap.add_argument("--mined-dir", default="data/mined")
    ap.add_argument("--config", default="clean")
    ap.add_argument("--split", default="train")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=200000, help="Rows to scan.")
    ap.add_argument(
        "--unknown",
        action="store_true",
        help="Mine generic (non-command) utterances for the reject class instead.",
    )
    ap.add_argument("--report-every", type=int, default=2000)
    args = ap.parse_args()

    words = load_words(args.words_file)
    targets = [(w, tuple(w.split())) for w in words]
    command_set = set(words)

    mined_dir = Path(args.mined_dir)
    audio_dir = mined_dir / "audio"
    out_jsonl = mined_dir / ("unknown.jsonl" if args.unknown else "commands.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    seen = _load_seen_keys(out_jsonl)
    if seen:
        print(f"Resuming {out_jsonl}: {len(seen)} rows already recorded.")

    mode = "unknown" if args.unknown else "command"
    print(f"Mining People's Speech ({args.config}/{args.split}) for {mode} clips...")

    scanned = written = 0
    with out_jsonl.open("a", encoding="utf-8") as f:
        for clip_id, text, speaker, fetch in iter_peoples_speech(
            args.config, args.split, args.offset, args.limit
        ):
            scanned += 1
            if scanned % args.report_every == 0:
                print(f"  scanned {scanned}, written {written}", flush=True)

            if args.unknown:
                # Keep only utterances that mention NO command word, and are
                # long enough to sample a 1 s window from.
                toks = {t for t, _, _ in _tokens_with_pos(text)}
                if toks & command_set:
                    continue
                if len(toks) < 3:
                    continue
                key = (clip_id, 0, (UNKNOWN_LABEL,))
                if key in seen:
                    continue
                wav_path = audio_dir / f"{_safe_name(clip_id)}.wav"
                if not save_audio_16k(fetch, wav_path):
                    continue
                rows = [{
                    "clip_id": clip_id, "speaker": speaker, "transcript": text,
                    "audio_path": str(wav_path), "words": [UNKNOWN_LABEL],
                    "char_start": 0, "char_end": 0,
                }]
            else:
                matches = find_matches(text, targets)
                if not matches:
                    continue
                new = [m for m in matches if (clip_id, m[1], (m[0],)) not in seen]
                if not new:
                    continue
                wav_path = audio_dir / f"{_safe_name(clip_id)}.wav"
                if not save_audio_16k(fetch, wav_path):
                    continue
                rows = [{
                    "clip_id": clip_id, "speaker": speaker, "transcript": text,
                    "audio_path": str(wav_path), "words": [label],
                    "char_start": cs, "char_end": ce,
                } for (label, cs, ce) in new]

            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.add((row["clip_id"], row["char_start"], tuple(row["words"])))
                written += 1
            f.flush()

    print(f"Done. Scanned {scanned} rows, wrote {written} new rows to {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
