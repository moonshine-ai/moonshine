"""Cut per-word training clips from mined People's Speech utterances.

For command rows (data/mined/commands.jsonl) the full utterance is force-aligned
with torchaudio's MMS_FA model and one clip is cut around each matched command
word. For unknown rows (data/mined/unknown.jsonl) a few random ~1 s windows are
cut and labelled ``_unknown_`` (no alignment needed).

Output (Speech-Commands layout, ready for the trainer)::

    data/peoples_speech/<label>/ps-<speaker>_nohash_<n>.wav

Runs resume via an ``extracted_keys.jsonl`` sidecar in the output dir.

Requires: ``pip install torch torchaudio soundfile numpy``.

Example::

    python tools/extract_clips.py --mined-dir data/mined --output-dir data/peoples_speech
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from stt_training.words import UNKNOWN_LABEL, load_words  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class AlignedWord:
    text: str
    start_s: float
    end_s: float
    score: float


class MMSAligner:
    """Thin wrapper around ``torchaudio.pipelines.MMS_FA`` forced alignment."""

    def __init__(self, device: str, with_star: bool = True):
        import torch
        import torchaudio
        import torchaudio.functional as TF

        self.torch = torch
        self.TF = TF
        bundle = torchaudio.pipelines.MMS_FA
        self.device = torch.device(device)
        self.sample_rate = bundle.sample_rate
        print(f"[align] loading MMS_FA on {self.device}...", file=sys.stderr)
        self.model = bundle.get_model(with_star=with_star).to(self.device).eval()
        self.dictionary = bundle.get_dict(star="<star>" if with_star else None)
        self.blank_id = 0

    def _ids(self, word: str) -> list[int]:
        return [self.dictionary[c] for c in word.lower() if c in self.dictionary]

    def align(self, audio_1d, words: list[str]) -> Optional[list[Optional[AlignedWord]]]:
        """Return one ``AlignedWord`` per input word (``None`` if unpronounceable).

        Returns ``None`` if CTC alignment fails entirely.
        """
        torch, TF = self.torch, self.TF
        per_word_ids = [self._ids(w) for w in words]
        flat = [i for ids in per_word_ids for i in ids]
        if not flat:
            return [None] * len(words)
        with torch.inference_mode():
            waveform = torch.from_numpy(audio_1d).to(self.device, torch.float32).unsqueeze(0)
            emission, _ = self.model(waveform)
            emission = torch.log_softmax(emission, dim=-1)
            targets = torch.tensor([flat], dtype=torch.int32)
            try:
                aligned, scores = TF.forced_align(emission.cpu(), targets, blank=self.blank_id)
            except (RuntimeError, NotImplementedError):
                return None
            token_spans = TF.merge_tokens(aligned[0], scores[0])
        n_frames = emission.shape[1]
        spf = audio_1d.shape[0] / n_frames / self.sample_rate
        out: list[Optional[AlignedWord]] = []
        cursor = 0
        for word, ids in zip(words, per_word_ids):
            if not ids:
                out.append(None)
                continue
            spans = token_spans[cursor:cursor + len(ids)]
            cursor += len(ids)
            if not spans:
                out.append(None)
                continue
            out.append(AlignedWord(
                text=word,
                start_s=spans[0].start * spf,
                end_s=spans[-1].end * spf,
                score=sum(s.score for s in spans) / len(spans),
            ))
        return out


def _cut_natural_clip(
    audio, start_s, end_s, *, prev_end_s, next_start_s, audio_duration_s,
    pad_pre_s, pad_post_s, sample_rate, min_clip_s=0.0,
):
    """Neighbour-clamped clip covering ``[start_s, end_s]`` padded and grown."""
    safe_lo = 0.0 if prev_end_s is None else (prev_end_s + start_s) / 2.0
    safe_hi = audio_duration_s if next_start_s is None else (end_s + next_start_s) / 2.0
    safe_hi = min(safe_hi, audio_duration_s)
    safe_lo = max(safe_lo, 0.0)
    clip_start = max(safe_lo, start_s - pad_pre_s)
    clip_end = min(safe_hi, end_s + pad_post_s)
    if min_clip_s and (clip_end - clip_start) < min_clip_s:
        half = min_clip_s / 2.0
        center = (clip_start + clip_end) / 2.0
        clip_start, clip_end = center - half, center + half
        if clip_start < safe_lo:
            clip_end += safe_lo - clip_start
            clip_start = safe_lo
        if clip_end > safe_hi:
            clip_start -= clip_end - safe_hi
            clip_end = safe_hi
        clip_start, clip_end = max(clip_start, safe_lo), min(clip_end, safe_hi)
    a = max(0, int(round(clip_start * sample_rate)))
    b = min(audio.shape[0], int(round(clip_end * sample_rate)))
    out = audio[a:b]
    return out if out.size else np.zeros(1, dtype=audio.dtype)


def _resolve_device(choice: str) -> str:
    import torch

    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_rows(path: Path):
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:64]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--words-file", default="words.txt")
    ap.add_argument("--mined-dir", default="data/mined")
    ap.add_argument("--output-dir", default="data/peoples_speech")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--pad-pre-s", type=float, default=0.08)
    ap.add_argument("--pad-post-s", type=float, default=0.08)
    ap.add_argument("--min-span-s", type=float, default=0.08)
    ap.add_argument("--max-span-s", type=float, default=2.5)
    ap.add_argument("--min-clip-s", type=float, default=0.30)
    ap.add_argument("--min-align-score", type=float, default=-3.5)
    ap.add_argument("--min-peak", type=float, default=0.02)
    ap.add_argument("--unknown-per-utt", type=int, default=2,
                    help="Random 1 s windows to cut per unknown utterance.")
    ap.add_argument("--per-class-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    words = set(load_words(args.words_file))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys_path = out_dir / "extracted_keys.jsonl"

    extracted = set()
    for line in _load_rows(keys_path):
        extracted.add((line["clip_id"], line["label"], line["char_start"]))
    counts: dict[str, int] = defaultdict(int)
    for lbl_dir in out_dir.iterdir() if out_dir.is_dir() else []:
        if lbl_dir.is_dir():
            counts[lbl_dir.name] = len(list(lbl_dir.glob("*.wav")))

    command_rows = _load_rows(Path(args.mined_dir) / "commands.jsonl")
    unknown_rows = _load_rows(Path(args.mined_dir) / "unknown.jsonl")
    print(f"Loaded {len(command_rows)} command rows, {len(unknown_rows)} unknown rows.")

    keys_f = keys_path.open("a", encoding="utf-8")

    def emit(label, clip_id, char_start, speaker, clip, sample_rate):
        if args.per_class_limit and counts[label] >= args.per_class_limit:
            return False
        peak = float(np.abs(clip).max()) if clip.size else 0.0
        if peak < args.min_peak:
            return False
        n = counts[label]
        path = out_dir / label / f"ps-{_safe(speaker)}_nohash_{n}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), clip.astype(np.float32), sample_rate, subtype="PCM_16")
        counts[label] = n + 1
        keys_f.write(json.dumps({
            "clip_id": clip_id, "label": label, "char_start": char_start,
        }) + "\n")
        keys_f.flush()
        extracted.add((clip_id, label, char_start))
        return True

    written = 0

    # --- commands: force-align and cut -------------------------------------
    if command_rows:
        aligner = MMSAligner(device=_resolve_device(args.device), with_star=True)
        by_clip: dict[str, list[dict]] = defaultdict(list)
        for row in command_rows:
            by_clip[row["clip_id"]].append(row)

        for clip_id, rows in by_clip.items():
            pending = [r for r in rows if (clip_id, r["words"][0], r["char_start"]) not in extracted]
            if not pending:
                continue
            audio_path = Path(rows[0]["audio_path"])
            if not audio_path.is_file():
                continue
            try:
                audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            except Exception:  # noqa: BLE001
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            dur = audio.shape[0] / sr
            transcript = rows[0]["transcript"]
            toks = [(m.group(0), m.start()) for m in _TOKEN_RE.finditer(transcript.lower())]
            token_words = [t for t, _ in toks]
            aligned = aligner.align(audio, token_words)
            if aligned is None:
                continue

            for r in pending:
                label = r["words"][0]
                n_tok = len(label.split())
                cs = r["char_start"]
                # Find the transcript token index at this char position.
                j = next((k for k, (_, start) in enumerate(toks) if start == cs), None)
                if j is None or j + n_tok > len(aligned):
                    continue
                span = aligned[j:j + n_tok]
                if any(a is None for a in span):
                    continue
                start_s, end_s = span[0].start_s, span[-1].end_s
                score = sum(a.score for a in span) / len(span)
                if not (args.min_span_s <= (end_s - start_s) <= args.max_span_s):
                    continue
                if score < args.min_align_score:
                    continue
                prev_end = aligned[j - 1].end_s if j > 0 and aligned[j - 1] else None
                nxt = aligned[j + n_tok].start_s if j + n_tok < len(aligned) and aligned[j + n_tok] else None
                clip = _cut_natural_clip(
                    audio, start_s, end_s,
                    prev_end_s=prev_end, next_start_s=nxt, audio_duration_s=dur,
                    pad_pre_s=args.pad_pre_s, pad_post_s=args.pad_post_s,
                    sample_rate=sr, min_clip_s=args.min_clip_s,
                )
                if emit(label, clip_id, cs, r.get("speaker", "unknown"), clip, sr):
                    written += 1

    # --- unknown: random windows -------------------------------------------
    for r in unknown_rows:
        clip_id = r["clip_id"]
        if (clip_id, UNKNOWN_LABEL, 0) in extracted:
            continue
        audio_path = Path(r["audio_path"])
        if not audio_path.is_file():
            continue
        try:
            audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        except Exception:  # noqa: BLE001
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        win = sr  # 1 s
        if audio.shape[0] <= win:
            windows = [audio]
        else:
            windows = [
                audio[s:s + win]
                for s in (rng.randint(0, audio.shape[0] - win) for _ in range(args.unknown_per_utt))
            ]
        emitted_any = False
        for clip in windows:
            if emit(UNKNOWN_LABEL, clip_id, 0, r.get("speaker", "unknown"), clip, sr):
                emitted_any = True
        if emitted_any:
            written += 1

    keys_f.close()
    print(f"Extracted {written} clips into {out_dir}")
    for label in sorted(counts):
        print(f"  {label:<12} {counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
