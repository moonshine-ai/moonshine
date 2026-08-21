#!/usr/bin/env python3
"""Times whole vs two-stage vs chunked Kokoro decoding on the target machine.

Splitting Kokoro buys a much shorter wait for the first audio, but it is only
worth shipping if it does not slow the ordinary path down and if the extra
decoder work still fits inside real time. That second condition is where a
Raspberry Pi decides the design: padding each chunk with enough context to
sound right multiplies the decoder work, and past roughly real time the audio
underruns and streaming is worse than not streaming.

Inputs come from a .npz written by --dump on a machine with the G2P assets, so
this can run anywhere onnxruntime does.

On a Raspberry Pi it answered no: chunked decoding cut time to first audio from
57 s to 7 s but ran at 4.73x real time, so the audio would underrun anyway. The
quality measurement in scripts/kokoro-stream-prototype.py ruled the approach out
independently. Kept for the next engine that looks sliceable.

Usage:
    python scripts/kokoro-stream-timing.py --dump /tmp/kokoro-input.npz
    python scripts/kokoro-stream-timing.py --input /tmp/kokoro-input.npz \\
        --whole model.onnx --stages /tmp/kokoro-stages-shrunk
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLES_PER_FRAME = 600
SAMPLE_RATE = 24000
FRAMES_PER_SECOND = SAMPLE_RATE / SAMPLES_PER_FRAME


def session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if threads:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def best_of(runs: int, call) -> float:
    """Fastest of several runs, which is the least noisy estimate of cost."""
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        call()
        timings.append(time.perf_counter() - started)
    return min(timings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, help="Write inputs and exit")
    parser.add_argument("--text", default=(
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs, and then take a short break "
        "before the next long sentence begins in earnest."
    ))
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--whole", type=Path)
    parser.add_argument("--stages", type=Path)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--tolerance-seconds", type=float, default=0.2)
    parser.add_argument("--pad-frames", type=int, nargs="*", default=[4, 8, 16])
    args = parser.parse_args()

    if args.dump:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "proto", Path(__file__).with_name("kokoro-stream-prototype.py")
        )
        proto = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proto)
        ids = proto.phoneme_ids(args.text)
        styles = proto.load_style(args.voice)
        style = styles[min(ids.shape[1], len(styles) - 1)].astype(np.float32)
        np.savez(args.dump, input_ids=ids, style=style,
                 speed=np.array([1.0], dtype=np.float32))
        print(f"wrote {args.dump} ({ids.shape[1]} ids)")
        return 0

    if not (args.input and args.stages):
        raise SystemExit("--input and --stages are required without --dump")

    data = np.load(args.input)
    feed = {"input_ids": data["input_ids"], "style": data["style"],
            "speed": data["speed"]}

    prosody = session(args.stages / "prosody.onnx", args.threads)
    decoder = session(args.stages / "decoder.onnx", args.threads)

    asr, f0, n = prosody.run(["asr", "f0", "n"], feed)
    frames = asr.shape[-1]
    audio_seconds = frames * SAMPLES_PER_FRAME / SAMPLE_RATE
    print(f"{frames} frames, {audio_seconds:.2f} s of audio, "
          f"{args.threads or 'default'} threads")

    prosody_s = best_of(args.runs, lambda: prosody.run(["asr", "f0", "n"], feed))
    decode_feed = {"asr": asr, "f0": f0, "n": n, "style": data["style"]}
    decode_s = best_of(args.runs, lambda: decoder.run(["waveform"], decode_feed))
    two_stage = prosody_s + decode_s
    print(f"two-stage whole: {two_stage * 1000:7.0f} ms "
          f"(prosody {prosody_s * 1000:.0f} + decode {decode_s * 1000:.0f}), "
          f"RTF {two_stage / audio_seconds:.2f}")

    if args.whole:
        whole = session(args.whole, args.threads)
        whole_s = best_of(args.runs, lambda: whole.run(["waveform"], feed))
        print(f"single graph:    {whole_s * 1000:7.0f} ms, "
              f"RTF {whole_s / audio_seconds:.2f}  "
              f"(split is {two_stage / whole_s:.2f}x)")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "proto", Path(__file__).with_name("kokoro-stream-prototype.py")
    )
    proto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proto)

    target = max(1, int(round(args.chunk_seconds * FRAMES_PER_SECOND)))
    search = max(0, int(round(args.tolerance_seconds * FRAMES_PER_SECOND)))
    cost = proto.boundary_cost(f0, n, frames)
    boundaries = proto.snap(cost, target, search)

    for pad in args.pad_frames:
        chunk_times = []
        decoded = 0
        for index in range(len(boundaries) - 1):
            start, end = boundaries[index], boundaries[index + 1]
            low = max(0, start - pad)
            high = min(frames, end + pad)
            piece_feed = {
                "asr": asr[:, :, low:high],
                "f0": f0[:, 2 * low: 2 * high],
                "n": n[:, 2 * low: 2 * high],
                "style": data["style"],
            }
            began = time.perf_counter()
            decoder.run(["waveform"], piece_feed)
            chunk_times.append(time.perf_counter() - began)
            decoded += high - low
        total = prosody_s + sum(chunk_times)
        first = prosody_s + chunk_times[0]
        print(f"chunked pad={pad:2d}: {len(chunk_times):2d} chunks, "
              f"work {decoded / frames:.2f}x frames, "
              f"total {total * 1000:6.0f} ms (RTF {total / audio_seconds:.2f}), "
              f"first audio {first * 1000:5.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
