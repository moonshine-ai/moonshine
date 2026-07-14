"""Evaluate a trained checkpoint (and, optionally, the exported int8 .tflite).

Runs the held-out speaker-independent validation split and reports overall
accuracy, macro (mean per-class) accuracy, per-class recall, and the most
common confusions -- so you can see, e.g., whether two commands sound too alike.

Example::

    python -m stt_training.evaluate --checkpoint checkpoints/run_XXXX
    python -m stt_training.evaluate --checkpoint checkpoints/run_XXXX \
        --tflite checkpoints/run_XXXX/spelling_cnn.mel.int8.tflite
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .checkpoint import load_model
from .dataset import SpeechCommandsDataset, speaker_independent_split
from .features import LogMelSpectrogram
from .train import default_data_roots


def _confusions(y_true, y_pred, classes, top=10):
    conf = collections.Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            conf[(classes[t], classes[p])] += 1
    return conf.most_common(top)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints")
    ap.add_argument("--data-roots", nargs="*", default=None)
    ap.add_argument("--tts-dir", default="data/tts")
    ap.add_argument("--ps-dir", default="data/peoples_speech")
    ap.add_argument("--tflite", default=None, help="Also evaluate this int8 .tflite.")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, cfg, ckpt_path = load_model(args.checkpoint, device)
    print(f"Loaded {ckpt_path} ({len(classes)} classes)")

    roots = (
        [Path(r) for r in args.data_roots]
        if args.data_roots
        else default_data_roots(Path(args.tts_dir), Path(args.ps_dir))
    )
    ds = SpeechCommandsDataset(
        roots, classes=classes,
        sample_rate=int(cfg["sample_rate"]), clip_seconds=float(cfg["clip_seconds"]),
    )
    _, val_idx = speaker_independent_split(ds, args.val_fraction, args.seed)
    loader = DataLoader(
        Subset(ds, val_idx), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    print(f"Validation clips: {len(val_idx)}")

    feature_fn = LogMelSpectrogram(
        sample_rate=int(cfg["sample_rate"]), n_mels=int(cfg["n_mels"]),
        target_frames=int(cfg["target_frames"]), hop_length=int(cfg["hop_length"]),
    ).to(device)

    y_true, y_pred_pt = [], []
    tfl = None
    tfl_pred = []
    if args.tflite:
        from .export import _interpreter, _run_tflite

        tfl = _interpreter(Path(args.tflite))

    with torch.no_grad():
        for wav, label in loader:
            wav = wav.to(device)
            feats = feature_fn(wav)
            logits = model(feats)
            y_pred_pt.extend(logits.argmax(-1).cpu().tolist())
            y_true.extend(label.tolist())
            if tfl is not None:
                fnp = feats.cpu().numpy().astype(np.float32)
                for i in range(fnp.shape[0]):
                    tfl_pred.append(int(_run_tflite(tfl, fnp[i:i + 1]).argmax()))

    _report("PyTorch", y_true, y_pred_pt, classes)
    if tfl is not None:
        _report(f"int8 tflite ({Path(args.tflite).name})", y_true, tfl_pred, classes)
    return 0


def _report(name, y_true, y_pred, classes):
    n = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    per_total = collections.Counter(y_true)
    per_correct = collections.Counter(t for t, p in zip(y_true, y_pred) if t == p)
    macro = np.mean([per_correct[i] / per_total[i] for i in per_total]) if per_total else float("nan")
    print(f"\n=== {name} ===")
    print(f"accuracy:   {correct/max(n,1)*100:.2f}%  ({correct}/{n})")
    print(f"macro acc:  {macro*100:.2f}%")
    print("per-class recall:")
    for i, cls in enumerate(classes):
        tot = per_total.get(i, 0)
        rec = per_correct.get(i, 0) / tot * 100 if tot else float("nan")
        print(f"  {cls:<12} {rec:6.1f}%  (n={tot})")
    conf = _confusions(y_true, y_pred, classes)
    if conf:
        print("top confusions (true -> pred):")
        for (t, p), c in conf:
            print(f"  {t:<12} -> {p:<12} {c}")


if __name__ == "__main__":
    raise SystemExit(main())
