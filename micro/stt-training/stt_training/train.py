"""Train the WordCNN command classifier.

The vocabulary comes from ``words.txt``; a ``_unknown_`` reject class is added
automatically. Data is read from local ``<root>/<class>/*.wav`` trees produced
by the tools/ scripts: one root per ZipVoice speaker under ``data/tts/`` plus
``data/peoples_speech``. Mel geometry and stem stride default to the values the
RP2350 firmware expects, so the export is drop-in.

Example::

    python -m stt_training.train --words-file words.txt --epochs 60
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .augment import WaveformAugment
from .dataset import (
    SpeechCommandsDataset,
    build_class_balanced_sampler,
    mixup,
    report_class_coverage,
    soft_cross_entropy,
    speaker_independent_split,
)
from .features import LogMelSpectrogram, SpecAugment
from .model import build_model, normalize_stride
from .words import resolve_classes

# Mel geometry + model shape that match the shipped moonshine-micro firmware.
# Changing these means the export will no longer be drop-in for moonshine-micro/stt.
DEFAULT_N_MELS = 64
DEFAULT_TARGET_FRAMES = 128
DEFAULT_HOP_LENGTH = 125
DEFAULT_STEM_STRIDE = "2,2"
DEFAULT_PAD_TO_ODD = True


def default_data_roots(tts_dir: Path, ps_dir: Path) -> list[Path]:
    """Discover training roots: one per ZipVoice speaker, plus People's Speech."""
    roots: list[Path] = []
    if tts_dir.is_dir():
        roots.extend(sorted(p for p in tts_dir.iterdir() if p.is_dir()))
    if ps_dir.is_dir():
        roots.append(ps_dir)
    return roots


@torch.no_grad()
def evaluate(model, feature_fn, loader, device) -> dict:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    class_correct = class_total = None
    for wav, label in loader:
        wav = wav.to(device, non_blocking=True)
        logits = model(feature_fn(wav))
        # Keep labels on CPU (int64 on MPS is unreliable); compare there.
        preds = logits.argmax(dim=-1).cpu()
        loss_sum += F.cross_entropy(logits, label.to(device), reduction="sum").item()
        hit = preds == label
        correct += int(hit.sum().item())
        total += label.numel()
        n = int(logits.size(-1))
        if class_total is None:
            class_total = torch.zeros(n)
            class_correct = torch.zeros(n)
        class_total += torch.bincount(label, minlength=n).float()
        class_correct += torch.bincount(label[hit], minlength=n).float()
    seen = class_total > 0
    macro = (class_correct[seen] / class_total[seen]).mean().item() if bool(seen.any()) else float("nan")
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "macro_acc": macro,
    }


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    classes = resolve_classes(args.words_file)
    print(f"Classes ({len(classes)}): {', '.join(classes)}")

    stem_stride = normalize_stride(args.stem_stride)

    if args.smoke_test:
        model = build_model(
            len(classes), stem_stride=stem_stride, pad_to_odd=args.pad_to_odd
        )
        n_params = sum(p.numel() for p in model.parameters())
        dummy = torch.zeros(1, 1, args.n_mels, args.target_frames)
        out = model(dummy)
        print(
            f"Smoke test OK: {n_params/1e6:.2f}M params, "
            f"input {tuple(dummy.shape)} -> output {tuple(out.shape)}"
        )
        return 0

    roots = (
        [Path(r) for r in args.data_roots]
        if args.data_roots
        else default_data_roots(Path(args.tts_dir), Path(args.ps_dir))
    )
    if not roots:
        print(
            "No data roots found. Run the synthesis / mining / extraction steps "
            f"first (looked under {args.tts_dir} and {args.ps_dir})."
        )
        return 2
    print(f"Data roots: {', '.join(str(r) for r in roots)}")

    full_ds = SpeechCommandsDataset(
        roots, classes=classes,
        sample_rate=args.sample_rate, clip_seconds=args.clip_seconds,
    )
    report_class_coverage(full_ds, classes)

    train_idx, val_idx = speaker_independent_split(full_ds, args.val_fraction, args.seed)
    print(f"Dataset: {len(full_ds)} clips ({len(train_idx)} train / {len(val_idx)} val)")

    sampler = None
    if args.sampling_power > 0:
        train_labels = [int(full_ds.items[i][1]) for i in train_idx]
        sampler = build_class_balanced_sampler(train_labels, args.sampling_power, seed=args.seed)
        print(f"Class-balanced sampling: power={args.sampling_power}")

    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    feature_fn = LogMelSpectrogram(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        target_frames=args.target_frames,
        hop_length=args.hop_length,
    ).to(device)
    spec_aug = SpecAugment().to(device)

    wave_aug = None
    if args.waveform_augment:
        wave_aug = WaveformAugment(
            sample_rate=args.sample_rate,
            musan_noise_dir=Path(args.musan_noise_dir) if args.musan_noise_dir else None,
            rir_dir=Path(args.rir_dir) if args.rir_dir else None,
        ).to(device)
        extra = " (with MUSAN/RIR)" if wave_aug.has_external_data else " (synthetic noise only)"
        print(f"Waveform augmentation: {wave_aug.n_transforms} transforms{extra}")
    else:
        print("Waveform augmentation: disabled")

    model = build_model(
        len(classes), width_mult=args.width_mult,
        stem_stride=stem_stride, pad_to_odd=args.pad_to_odd,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M (stem_stride={stem_stride}, pad_to_odd={args.pad_to_odd})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        start = args.warmup_steps if args.warmup_steps > 0 else 0
        progress = (step - start) / max(1, total_steps - start)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    out = Path(args.output)
    if len(out.parts) == 1:  # bare filename -> place under a dated run dir
        run_dir = Path(args.checkpoints_dir) / time.strftime("run_%Y_%m_%d_%H_%M")
        out_path = run_dir / out.name
    else:
        out_path = out
        run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {out_path}")

    saved_args = {
        "sample_rate": args.sample_rate,
        "clip_seconds": args.clip_seconds,
        "n_mels": args.n_mels,
        "target_frames": args.target_frames,
        "hop_length": args.hop_length,
        "stem_stride": list(stem_stride),
        "pad_to_odd": bool(args.pad_to_odd),
        "width_mult": args.width_mult,
    }

    best_acc, global_step = 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_n, t0 = 0.0, 0, time.time()
        with tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", dynamic_ncols=True) as pbar:
            for wav, label in pbar:
                wav = wav.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                for g in optimizer.param_groups:
                    g["lr"] = args.lr * lr_at(global_step)
                with torch.no_grad():
                    if wave_aug is not None:
                        wav = wave_aug(wav)
                    feats = feature_fn(wav)
                feats = spec_aug(feats)
                feats, soft_y = mixup(feats, label, len(classes), args.mixup_alpha)
                logits = model(feats)
                loss = soft_cross_entropy(logits, soft_y, smoothing=args.label_smoothing)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * label.size(0)
                epoch_n += label.size(0)
                global_step += 1
                pbar.set_postfix(loss=f"{epoch_loss/max(epoch_n,1):.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        m = evaluate(model, feature_fn, val_loader, device)
        print(
            f"epoch {epoch+1:3d}/{args.epochs}  train_loss={epoch_loss/max(epoch_n,1):.4f}  "
            f"val_loss={m['loss']:.4f}  val_acc={m['acc']*100:.2f}%  "
            f"macro_acc={m['macro_acc']*100:.2f}%  ({time.time()-t0:.1f}s)"
        )

        if m["acc"] > best_acc:
            best_acc = m["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "val_acc": m["acc"],
                    "macro_acc": m["macro_acc"],
                    "epoch": epoch + 1,
                    "args": saved_args,
                },
                out_path,
            )
            print(f"  -> saved {out_path} (val_acc={m['acc']*100:.2f}%)")

    print(f"Done. Best val_acc={best_acc*100:.2f}%  ->  {out_path}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--words-file", default="words.txt", help="Vocabulary file (default: words.txt).")
    p.add_argument("--data-roots", nargs="*", default=None,
                   help="Explicit <class>/*.wav roots (default: auto-discover under --tts-dir/--ps-dir).")
    p.add_argument("--tts-dir", default="data/tts", help="ZipVoice output base (one subdir per speaker).")
    p.add_argument("--ps-dir", default="data/peoples_speech", help="Extracted People's Speech clips.")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampling-power", type=float, default=0.5,
                   help="Class-balanced sampling exponent (0=natural, 1=uniform).")
    p.add_argument("--width-mult", type=float, default=1.0)

    p.add_argument("--waveform-augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--musan-noise-dir", default="data/musan/noise", help="'' to disable.")
    p.add_argument("--rir-dir", default="data/rirs", help="'' to disable.")

    # Mel / model geometry -- defaults match the moonshine-micro firmware.
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--clip-seconds", type=float, default=1.0)
    p.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    p.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    p.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    p.add_argument("--stem-stride", default=DEFAULT_STEM_STRIDE)
    p.add_argument("--pad-to-odd", action=argparse.BooleanOptionalAction, default=DEFAULT_PAD_TO_ODD)

    p.add_argument("--checkpoints-dir", default="checkpoints")
    p.add_argument("--output", default="word_cnn.pt",
                   help="Checkpoint filename (bare name -> under a dated run dir) or full path.")
    p.add_argument("--smoke-test", action="store_true", help="Build the model, print params, exit.")
    return p


def main() -> int:
    return train(build_argparser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
