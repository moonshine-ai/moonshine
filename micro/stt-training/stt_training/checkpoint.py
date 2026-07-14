"""Shared checkpoint loading for export and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .model import WordCNN, normalize_stride


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a ``.pt`` file, a run directory, or the checkpoints parent."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        cand = p / "word_cnn.pt"
        if cand.is_file():
            return cand
        runs = sorted(
            (d for d in p.glob("run_*") if (d / "word_cnn.pt").is_file()),
            key=lambda d: d.name,
        )
        if runs:
            return runs[-1] / "word_cnn.pt"
    raise FileNotFoundError(f"No word_cnn.pt found at {path}")


def load_model(path: str | Path, device: torch.device | str = "cpu"):
    """Load a checkpoint. Returns ``(model, classes, cfg)``."""
    ckpt_path = resolve_checkpoint(path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = list(ckpt["classes"])
    cfg = dict(ckpt.get("args", {}))
    cfg.setdefault("sample_rate", 16000)
    cfg.setdefault("clip_seconds", 1.0)
    cfg.setdefault("n_mels", 64)
    cfg.setdefault("target_frames", 128)
    cfg.setdefault("hop_length", 125)
    cfg.setdefault("stem_stride", [2, 2])
    cfg.setdefault("pad_to_odd", True)
    cfg.setdefault("width_mult", 1.0)

    model = WordCNN(
        num_classes=len(classes),
        width_mult=float(cfg["width_mult"]),
        stem_stride=normalize_stride(cfg["stem_stride"]),
        pad_to_odd=bool(cfg["pad_to_odd"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, classes, cfg, ckpt_path


def load_representative_waveforms(
    data_roots, n, target_samples, sample_rate, seed=0
) -> list[torch.Tensor]:
    """Sample up to ``n`` fixed-length mono waveforms from ``<root>/<class>/*.wav``."""
    files = []
    for root in data_roots:
        root = Path(root)
        if root.is_dir():
            files.extend(sorted(root.rglob("*.wav")))
    rng = random.Random(seed)
    rng.shuffle(files)
    out = []
    for f in files[: n * 2]:
        try:
            data, sr = sf.read(str(f), dtype="float32", always_2d=True)
        except Exception:  # noqa: BLE001
            continue
        wav = torch.from_numpy(np.ascontiguousarray(data.T))
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.mean(dim=0)
        if wav.shape[0] < target_samples:
            wav = F.pad(wav, (0, target_samples - wav.shape[0]))
        else:
            wav = wav[:target_samples]
        out.append(wav)
        if len(out) >= n:
            break
    return out
