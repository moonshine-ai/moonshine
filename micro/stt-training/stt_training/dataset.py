"""Speech-Commands-style dataset over local ``<root>/<class>/*.wav`` trees.

Loads raw fixed-length waveforms; log-mel features are computed on the GPU
inside the training step so augmentation can see the waveform. Splits are
speaker-independent (no voice appears in both train and val).
"""

from __future__ import annotations

import collections
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

_MAX_DECODE_WARNINGS = 20
_decode_warning_count = 0


def _warn_decode_failure(src: str, exc: BaseException) -> None:
    global _decode_warning_count
    _decode_warning_count += 1
    if _decode_warning_count <= _MAX_DECODE_WARNINGS:
        print(
            f"[decode] skipping undecodable clip ({src}): "
            f"{type(exc).__name__}; substituting silence "
            f"[{_decode_warning_count}/{_MAX_DECODE_WARNINGS} shown]",
            flush=True,
        )


def voice_id_from_path(path: Path) -> str:
    """Stable per-voice id for speaker-independent splits.

    Filenames follow ``<voice>_nohash_<n>.wav`` (Speech Commands convention),
    so the stem before ``_nohash_`` identifies the voice; namespacing by corpus
    keeps voices from different corpora distinct.
    """
    stem = path.stem.split("_nohash_", 1)[0]
    corpus = path.parent.parent.name  # <corpus>/<class>/<file>.wav
    return f"{corpus}:{stem}"


class SpeechCommandsDataset(Dataset):
    """Loads wavs from ``<root>/<class>/*.wav`` across one or more corpus roots."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        classes: Sequence[str],
        sample_rate: int = 16000,
        clip_seconds: float = 1.0,
    ):
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.sample_rate = sample_rate
        self.target_samples = int(round(sample_rate * clip_seconds))

        self.items: list[tuple[Path, int, str]] = []  # (path, label_idx, voice_id)
        found_roots = []
        for root in roots:
            root = Path(root)
            if not root.is_dir():
                continue
            found_roots.append(root)
            for cls in self.classes:
                cls_dir = root / cls
                if not cls_dir.is_dir():
                    continue
                for wav in sorted(cls_dir.glob("*.wav")):
                    vid = voice_id_from_path(wav)
                    self.items.append((wav, self.class_to_idx[cls], vid))

        if not found_roots:
            raise FileNotFoundError(
                f"None of the dataset roots exist: {list(roots)}. "
                "Run the synthesis / mining / extraction steps first."
            )
        if not self.items:
            raise RuntimeError(f"No .wav files found under roots: {list(roots)}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_wav(self, path: Path) -> torch.Tensor:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:  # noqa: BLE001
            _warn_decode_failure(str(path), exc)
            return torch.zeros(self.target_samples, dtype=torch.float32)
        wav = torch.from_numpy(np.ascontiguousarray(data.T))  # (channels, samples)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(dim=0)  # mono
        n = wav.shape[0]
        if n < self.target_samples:
            wav = F.pad(wav, (0, self.target_samples - n))
        elif n > self.target_samples:
            wav = wav[: self.target_samples]
        return wav

    def __getitem__(self, idx: int):
        path, label, _ = self.items[idx]
        return self._load_wav(path), label

    def voice_ids(self) -> list[str]:
        return [vid for _, _, vid in self.items]

    def load_waveform(self, idx: int) -> torch.Tensor:
        return self._load_wav(self.items[idx][0])


def speaker_independent_split(
    dataset: SpeechCommandsDataset,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Split indices so no voice appears in both train and val."""
    by_voice: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, vid) in enumerate(dataset.items):
        by_voice[vid].append(i)

    voices = sorted(by_voice.keys())
    rng = random.Random(seed)
    rng.shuffle(voices)
    n_val_voices = max(1, int(round(len(voices) * val_fraction)))
    val_voices = set(voices[:n_val_voices])

    train_idx, val_idx = [], []
    for v, idxs in by_voice.items():
        (val_idx if v in val_voices else train_idx).extend(idxs)
    return train_idx, val_idx


def build_class_balanced_sampler(
    labels: Sequence[int],
    power: float,
    *,
    seed: int = 0,
) -> torch.utils.data.WeightedRandomSampler:
    """Weighted sampler that softens class imbalance.

    Draw probability is ``count(class)^(-power)``; ``power=0`` is natural
    sampling, ``power=1`` makes every class equiprobable per draw. Useful when
    People's Speech contributes far more clips for common words than rare ones.
    """
    counts = collections.Counter(int(lab) for lab in labels)
    class_w = {c: (1.0 / n) ** power for c, n in counts.items()}
    weights = torch.tensor([class_w[int(lab)] for lab in labels], dtype=torch.double)
    g = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(labels), replacement=True, generator=g
    )


def report_class_coverage(
    dataset: SpeechCommandsDataset, classes: Sequence[str]
) -> None:
    """Print per-class clip counts and warn about empty classes.

    Not fatal: a class may legitimately lack real People's Speech clips and be
    covered by ZipVoice alone. An entirely empty class, however, cannot be
    learned, so we surface it loudly.
    """
    counts = collections.Counter(int(lab) for _, lab, _ in dataset.items)
    print("Class coverage:")
    missing = []
    for i, cls in enumerate(classes):
        n = counts.get(i, 0)
        flag = "  <-- EMPTY" if n == 0 else ""
        print(f"  {cls:<12} {n:>7}{flag}")
        if n == 0:
            missing.append(cls)
    if missing:
        print(
            "WARNING: no training clips for: "
            + ", ".join(missing)
            + ". Synthesize them with tools/synthesize.py or the model cannot "
            "learn these classes."
        )


def mixup(
    x: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard mixup. Returns (mixed_x, soft_targets)."""
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]
    y_oh = F.one_hot(y, num_classes).float()
    y_mixed = lam * y_oh + (1 - lam) * y_oh[perm]
    return x_mixed, y_mixed


def soft_cross_entropy(
    logits: torch.Tensor, soft_targets: torch.Tensor, smoothing: float = 0.0
) -> torch.Tensor:
    if smoothing > 0:
        n = soft_targets.size(-1)
        soft_targets = soft_targets * (1 - smoothing) + smoothing / n
    return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
