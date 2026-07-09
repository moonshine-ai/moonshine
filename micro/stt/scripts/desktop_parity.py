"""Desktop regression check for the moonshine-micro on-device embedded-clip test loop.

The Pico firmware runs an int8 mel-mode ``.tflite`` over a fixed set of
embedded clips and prints a per-clip ``exp=.. got=..`` table plus an
overall accuracy (see ``pico_monitor.log``). This script reproduces that
run on the desktop with ``ai_edge_litert`` so predictions can be checked
without a board attached.

The script mirrors ``generate_embedded_data.py`` clip selection, applies the
same int16 round-trip the firmware uses, and feeds the interpreter with
log-mel features computed the same way as the on-device front-end.

Usage::

    python stt/scripts/desktop_parity.py

    python stt/scripts/desktop_parity.py \\
        --tflite models/spelling_cnn_letters_digits_mel_int8.tflite
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "moonshine-micro" / "stt" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.log_mel_pure import (  # noqa: E402
    PureLogMelSpectrogram,
    pad_or_crop,
    read_wav_mono,
    resample_linear,
)
from generate_embedded_data import (  # noqa: E402
    _pick_clips,
    _resolve_int8_mel_tflite,
)
from export_spelling_cnn_litert import (  # noqa: E402
    _dequantize_output,
    _quantize_input,
)

_DEFAULT_CLASSES = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z", "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine",
]


def _resolve_meta(tflite: pathlib.Path) -> dict:
    """Load whichever metadata sidecar sits next to the model.

    Accepts both the current ``spelling_cnn_meta.json`` and the older
    mode-qualified ``spelling_cnn.mel_meta.json`` so this works against
    historical runs too. Missing dims are filled with the project's
    long-standing LogMelSpectrogram defaults and can be overridden on the
    CLI.
    """
    for name in ("spelling_cnn_meta.json", "spelling_cnn.mel_meta.json"):
        cand = tflite.parent / name
        if cand.is_file():
            return json.loads(cand.read_text())
    return {}


def _int16_roundtrip(samples: list[float]) -> list[float]:
    """Replicate the embedded int16 blob and the firmware's inverse.

    ``generate_embedded_data._fp32_to_int16_pcm`` stores ``round(x*32767)``
    (saturated); ``main.cc`` reads it back as ``int16 * (1/32768)``. Doing
    both here makes the fp32 waveform identical to what the board feeds its
    log-mel front-end.
    """
    out = []
    for s in samples:
        v = int(round(s * 32767.0))
        v = 32767 if v > 32767 else (-32768 if v < -32768 else v)
        out.append(v / 32768.0)
    return out


def _softmax_prob(logits: np.ndarray, idx: int) -> float:
    m = float(np.max(logits))
    exps = np.exp(logits - m)
    return float(exps[idx] / np.sum(exps))


def _parse_device_log(path: pathlib.Path) -> list[tuple[str, str]]:
    """Pull ``(exp, got)`` pairs from a pico_monitor.log per-clip table."""
    if not path.is_file():
        return []
    pairs: list[tuple[str, str]] = []
    rx = re.compile(r"^\[\s*\d+/\s*\d+\]\s+exp=(\S+)\s+got=(\S+)")
    for line in path.read_text().splitlines():
        m = rx.match(line.strip())
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tflite", default=None, help="Model path (default: auto tiny).")
    ap.add_argument(
        "--wavs-dirs",
        default="speech-data/real/captured,speech-data/real/peoples_speech",
        help="Comma-separated clip roots (relative to repo root), searched in order.",
    )
    ap.add_argument("--clips-per-class", type=int, default=2)
    ap.add_argument("--max-classes", type=int, default=None)
    ap.add_argument("--n-mels", type=int, default=None, help="Override sidecar.")
    ap.add_argument("--target-frames", type=int, default=None)
    ap.add_argument("--hop-length", type=int, default=None)
    ap.add_argument("--n-fft", type=int, default=None)
    ap.add_argument(
        "--no-int16", action="store_true",
        help="Skip the int16 round-trip (feed raw fp32 from the WAV).",
    )
    ap.add_argument(
        "--device-log", default="pico_monitor.log",
        help="pico_monitor.log to diff per-clip against (relative to repo root).",
    )
    args = ap.parse_args(argv)

    tflite = (
        pathlib.Path(args.tflite).expanduser().resolve()
        if args.tflite else _resolve_int8_mel_tflite()
    )
    meta = _resolve_meta(tflite)
    classes = list(meta.get("classes", _DEFAULT_CLASSES))
    sr = int(meta.get("sample_rate", 16000))
    clip_s = float(meta.get("clip_seconds", 1.0))
    n_mels = int(args.n_mels if args.n_mels is not None else meta.get("n_mels", 80))
    target_frames = int(
        args.target_frames if args.target_frames is not None
        else meta.get("target_frames", 200)
    )
    hop_length = int(
        args.hop_length if args.hop_length is not None
        else meta.get("hop_length", 80)
    )
    n_fft = int(args.n_fft if args.n_fft is not None else meta.get("n_fft", 512))
    n_samples = int(round(sr * clip_s))

    print(f"Model:   {tflite.relative_to(REPO_ROOT) if tflite.is_relative_to(REPO_ROOT) else tflite}")
    print(f"Mel:     n_mels={n_mels} target_frames={target_frames} "
          f"hop={hop_length} n_fft={n_fft} sr={sr}")
    print(f"int16 round-trip: {'OFF (raw fp32)' if args.no_int16 else 'ON (matches firmware)'}")

    front_end = PureLogMelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=20.0,
        f_max=sr / 2.0,
        target_frames=target_frames,
    )

    from ai_edge_litert import interpreter as litert_interp
    interp = litert_interp.Interpreter(model_path=str(tflite))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_idx, out_idx = in_d["index"], out_d["index"]
    print(f"LiteRT:  in {tuple(in_d['shape'])} {np.dtype(in_d['dtype']).name} "
          f"q={in_d['quantization']}  ->  out {np.dtype(out_d['dtype']).name} "
          f"q={out_d['quantization']}")

    wavs_roots = [
        REPO_ROOT / p.strip()
        for p in args.wavs_dirs.split(",")
        if p.strip()
    ]
    selected = _pick_clips(
        wavs_roots, classes, args.clips_per_class, args.max_classes
    )
    print(f"Clips:   {len(selected)}\n")

    device_pairs = _parse_device_log(REPO_ROOT / args.device_log)
    use_device = len(device_pairs) == len(selected)

    header = f"{'#':>3}  {'exp':<6}{'got':<6}{'p':>6}  res"
    if use_device:
        header += "   device  agree"
    print(header)
    print("-" * len(header))

    correct = 0
    device_match = 0
    fails: list[tuple[int, str, str]] = []
    for i, (label_idx, label, wav_path) in enumerate(selected):
        samples, src_rate = read_wav_mono(wav_path)
        if src_rate != sr:
            samples = resample_linear(samples, src_rate, sr)
        samples = pad_or_crop(samples, n_samples)
        if not args.no_int16:
            samples = _int16_roundtrip(samples)

        feats = front_end(samples)  # (n_mels, target_frames)
        x = np.asarray(feats, dtype=np.float32).reshape(1, 1, n_mels, target_frames)
        interp.set_tensor(in_idx, _quantize_input(x, in_d))
        interp.invoke()
        logits = _dequantize_output(interp.get_tensor(out_idx), out_d).reshape(-1)
        pred = int(np.argmax(logits))
        prob = _softmax_prob(logits, pred)
        got = classes[pred]
        ok = pred == label_idx
        correct += int(ok)

        line = f"{i+1:>3}  {label:<6}{got:<6}{prob:>6.3f}  {'OK' if ok else 'FAIL'}"
        if use_device:
            dev_got = device_pairs[i][1]
            agree = dev_got == got
            device_match += int(agree)
            line += f"   {dev_got:<6}  {'=' if agree else 'DIFF'}"
        print(line)
        if not ok:
            fails.append((i + 1, label, got))

    total = len(selected)
    print("-" * len(header))
    print(f"\nDesktop accuracy: {correct}/{total} = {100.0*correct/max(total,1):.1f}%")
    if use_device:
        print(f"Per-clip agreement with device log: {device_match}/{total} "
              f"({100.0*device_match/max(total,1):.1f}%)")
        dev_correct = sum(1 for e, g in device_pairs if e == g)
        print(f"Device log accuracy (for reference): {dev_correct}/{total} = "
              f"{100.0*dev_correct/max(total,1):.1f}%")
    if fails:
        print("\nDesktop misclassifications:")
        for n, exp, got in fails:
            print(f"  [{n:>2}] {exp} -> {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
