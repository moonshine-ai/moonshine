"""Export a trained WordCNN to the int8 LiteRT format the RP2350 firmware uses.

Produces the two artifacts ``moonshine-micro/stt`` consumes (names match what
its ``generate_embedded_data.py`` looks for, so deploying is a plain copy):

    spelling_cnn_mel_int8.tflite   int8 mel-mode classifier (weights + activations)
    spelling_cnn_meta.json         class order + audio config for the firmware

The exporter converts only the classifier head (``--mode mel``): the log-mel
front-end runs on the host / device, so quantization stays in the benign range
that survives int8. After quantization the flatbuffer's weight buffers are
inlined so TensorFlow Lite Micro can load the file.

The artifact names are kept as ``spelling_cnn.*`` for drop-in compatibility with
the existing firmware (``generate_embedded_data.py`` looks for those names).

Requires: ``pip install litert-torch ai-edge-litert ai-edge-quantizer flatbuffers``.

Example::

    python -m stt_training.export --checkpoint checkpoints/run_XXXX
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model, load_representative_waveforms
from .features import LogMelSpectrogram


def _inline_buffers(buf: bytes):
    """Re-serialize a .tflite with every external weight buffer inlined (TFLM-safe)."""
    import flatbuffers
    from ai_edge_litert import schema_py_generated as schema

    model = schema.Model.GetRootAsModel(buf, 0)
    model_t = schema.ModelT.InitFromObj(model)
    inlined = 0
    for b in model_t.buffers or []:
        if b.data is not None and len(b.data) > 0:
            continue
        if b.offset and b.size:
            off, sz = int(b.offset), int(b.size)
            chunk = buf[off:off + sz]
            if len(chunk) != sz:
                raise RuntimeError("external buffer extends past EOF")
            b.data = chunk
            b.offset = 0
            b.size = 0
            inlined += 1
    builder = flatbuffers.Builder(len(buf) + 4096)
    builder.Finish(model_t.Pack(builder), b"TFL3")
    return bytes(builder.Output()), inlined


def _interpreter(path: Path):
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def _run_tflite(interp, feats: np.ndarray) -> np.ndarray:
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = feats
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        x = np.clip(np.round(feats / scale + zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y = interp.get_tensor(out["index"])
    if out["dtype"] == np.int8:
        scale, zp = out["quantization"]
        y = (y.astype(np.float32) - zp) * scale
    return y


def export(args: argparse.Namespace) -> int:
    try:
        import litert_torch
    except ImportError as exc:
        print(f"litert-torch not installed ({exc}). Run: pip install litert-torch")
        return 2

    device = torch.device("cpu")
    model, classes, cfg, ckpt_path = load_model(args.checkpoint, device)
    n_mels = int(cfg["n_mels"])
    target_frames = int(cfg["target_frames"])
    hop_length = int(cfg["hop_length"])
    sample_rate = int(cfg["sample_rate"])
    clip_seconds = float(cfg["clip_seconds"])
    target_samples = int(round(sample_rate * clip_seconds))
    print(f"Loaded {ckpt_path} ({len(classes)} classes): {', '.join(classes)}")

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # Names chosen to match what moonshine-micro/stt/scripts/generate_embedded_data.py
    # looks for, so deploying is a plain copy of these two files into
    # moonshine-micro/models/.
    fp32_path = out_dir / "spelling_cnn_mel_fp32.tflite"
    int8_path = out_dir / "spelling_cnn_mel_int8.tflite"
    meta_path = out_dir / "spelling_cnn_meta.json"

    feature_fn = LogMelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels,
        target_frames=target_frames, hop_length=hop_length,
    ).eval()

    dummy = torch.zeros(1, 1, n_mels, target_frames)
    dyn = ({0: torch.export.Dim.AUTO},)
    print(f"Converting classifier ({tuple(dummy.shape)}) -> {fp32_path.name}")
    edge_model = litert_torch.convert(model, (dummy,), dynamic_shapes=dyn)
    edge_model.export(str(fp32_path))

    metadata = {
        "classes": list(classes),
        "sample_rate": sample_rate,
        "clip_seconds": clip_seconds,
        "n_mels": n_mels,
        "target_frames": target_frames,
        "hop_length": hop_length,
        "n_fft": 512,
        "stem_stride": list(cfg["stem_stride"]),
        "pad_to_odd": bool(cfg["pad_to_odd"]),
        "input_name": "log_mel",
        "output_name": "logits",
        "mode": "mel",
        "source_checkpoint": ckpt_path.name,
        "source_run": ckpt_path.parent.name,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {meta_path.name}")

    # --- calibration set (real log-mels) -----------------------------------
    roots = args.calibration_roots or _default_calibration_roots(args)
    waves = load_representative_waveforms(
        roots, args.num_calibration_samples, target_samples, sample_rate, seed=args.seed
    )
    if not waves:
        print(
            f"No calibration wavs found under {roots}. int8 quantization needs "
            "representative data; run the data-gathering steps first."
        )
        return 3
    with torch.no_grad():
        feats = [feature_fn(w.unsqueeze(0)).numpy().astype(np.float32) for w in waves]
    print(f"Calibrating int8 with {len(feats)} real clips.")

    # --- quantize to int8 --------------------------------------------------
    try:
        import ai_edge_quantizer as aeq
    except ImportError as exc:
        print(f"ai-edge-quantizer not installed ({exc}).")
        return 4

    sig_in = _interpreter(fp32_path).get_signature_runner("serving_default").get_input_details()
    in_name = next(iter(sig_in))
    feed = [{in_name: f} for f in feats]

    q = aeq.Quantizer(str(fp32_path), aeq.recipe.static_wi8_ai8())
    calib = q.calibrate({"serving_default": feed}) if q.need_calibration else None
    q.quantize(calibration_result=calib, serialize_to_path=str(int8_path),
               enable_progress_report=False)

    raw = int8_path.read_bytes()
    inlined_bytes, n_inlined = _inline_buffers(raw)
    if n_inlined:
        int8_path.write_bytes(inlined_bytes)
    print(
        f"Wrote {int8_path.name} ({int8_path.stat().st_size/1024:.1f} KiB, "
        f"inlined {n_inlined} buffer(s) for TFLM)"
    )

    # --- parity: PyTorch vs int8 tflite argmax agreement -------------------
    interp = _interpreter(int8_path)
    ref = torch.stack([torch.from_numpy(f)[0] for f in feats])
    with torch.no_grad():
        torch_pred = model(ref).argmax(-1).numpy()
    tfl_pred = np.array([int(_run_tflite(interp, f).argmax()) for f in feats])
    agree = int((torch_pred == tfl_pred).sum())
    print(f"int8 parity: argmax agreement {agree}/{len(feats)}")
    if agree < 0.9 * len(feats):
        print("WARNING: low int8 argmax agreement; check calibration data quality.")

    print(f"\nDeploy artifacts in {out_dir}:\n  {int8_path.name}\n  {meta_path.name}")
    return 0


def _default_calibration_roots(args: argparse.Namespace) -> list[str]:
    roots = []
    tts = Path(args.tts_dir)
    if tts.is_dir():
        roots.extend(str(p) for p in tts.iterdir() if p.is_dir())
    if Path(args.ps_dir).is_dir():
        roots.append(args.ps_dir)
    return roots


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints",
                   help="Path to word_cnn.pt, a run dir, or the checkpoints parent (newest run).")
    p.add_argument("--output-dir", default=None, help="Where to write artifacts (default: checkpoint dir).")
    p.add_argument("--calibration-roots", nargs="*", default=None)
    p.add_argument("--tts-dir", default="data/tts")
    p.add_argument("--ps-dir", default="data/peoples_speech")
    p.add_argument("--num-calibration-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> int:
    return export(build_argparser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
