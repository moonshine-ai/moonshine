#!/usr/bin/env python3
"""Checks whether Piper's vocoder can be run a chunk at a time.

Piper is a VITS model: everything up to the latent `z` is one graph, and a
HiFi-GAN generator turns `z` into audio. One time-varying tensor crosses that
boundary, and the generator is a stack of transposed convolutions with weight
normalization rather than the instance normalization Kokoro's decoder uses, so
a slice of `z` should decode to the matching slice of audio given enough
context either side. Multi-speaker voices send a second tensor across, the
speaker embedding, but it is one column for the whole utterance rather than
one per frame, so every slice reuses it unchanged.

That "enough context" is the number this measures. It decodes a fixed target
window with growing amounts of padding and reports how close the result gets
to the same window from an unchunked render. Once the error stops falling, the
padding has covered the generator's receptive field, and the ratio of decoded
frames to emitted frames is what streaming would cost.

Piper genuinely can be streamed a chunk at a time — unlike Kokoro, see
scripts/kokoro-stream-prototype.py, which needs crossfades and an offline gain
table to hide what its decoder does to a short chunk. Measured over all four
quality tiers and a multi-speaker voice: the stages reproduce the whole graph
exactly, 16 frames of padding either side puts the seam at 1e-6 of the signal,
the two stages weigh what the whole model weighs, and they convert to ORT at
full optimization unchanged. Nothing here needs a crossfade or a level fix.

What it costs is assets, not code. The chunk policy, the stream and the C API
are already engine-agnostic, but the stages have to be built and published for
all 96 catalog voices before any of it engages. What it buys is 66-84% of time
to first audio, and the generator's share of the work *rises* as the machine
gets slower (76% to 97% for the high tier, going from all cores to one), so
the saving is largest exactly where synthesis is slow enough to notice: on a
single core here, en_US-lessac-high drops from 3.07 s to 479 ms.

Usage:
    python scripts/piper-stream-prototype.py --model en_US-amy-medium.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# Scales are [noise, length, noise_w]. Piper ships 0.667/1.0/0.8, but both
# noise terms are drawn per session, so a whole render and a staged one would
# not be comparing the same utterance. Zeroing them is what the repository's
# other parity tools do; it changes the audio, not the receptive field being
# measured.
SCALES = [0.0, 1.0, 0.0]


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def best_of(runs: int, call) -> float:
    """Fastest of several runs, which is the least noisy estimate of cost."""
    import time

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        call()
        timings.append(time.perf_counter() - started)
    return min(timings)


def depends_on_inputs(model: onnx.ModelProto) -> set[str]:
    """Tensors whose value depends on what the caller passed in.

    Everything else is folded from initializers, which matters on a quantized
    voice: its weights are dequantized by chains that sit outside the generator
    in the graph but belong to it, and telling the two apart is what keeps the
    weights inside the stage that uses them instead of crossing the seam on
    every utterance.
    """
    live = {value.name for value in model.graph.input}
    # A single pass in topological order is enough; ONNX requires nodes be
    # sorted so that producers precede consumers.
    for node in model.graph.node:
        if any(name in live for name in node.input):
            live.update(node.output)
    return live


def find_boundary(model: onnx.ModelProto) -> list[str]:
    """The tensors carrying the utterance from the VITS body into /dec."""
    producer = {output: node for node in model.graph.node for output in node.output}
    live = depends_on_inputs(model)
    crossing = []
    for node in model.graph.node:
        if not node.name.startswith("/dec"):
            continue
        for name in node.input:
            source = producer.get(name)
            if source is None or source.name.startswith("/dec"):
                continue
            if name in live and name not in crossing:
                crossing.append(name)
    if not crossing:
        raise SystemExit("Found no tensor entering the generator.")
    return crossing


def split(model_path: Path, out_dir: Path) -> tuple[Path, Path, list[str], list[str]]:
    model = onnx.load(str(model_path), load_external_data=False)
    crossing = find_boundary(model)
    # Multi-speaker voices add a `sid` input, so take the inputs off the graph
    # rather than assuming the single-speaker three.
    inputs = [value.name for value in model.graph.input]
    out_dir.mkdir(parents=True, exist_ok=True)
    upstream = out_dir / "upstream.onnx"
    generator = out_dir / "generator.onnx"
    onnx.utils.extract_model(
        str(model_path), str(upstream), inputs, crossing, check_model=False
    )
    onnx.utils.extract_model(
        str(model_path), str(generator), crossing, ["output"], check_model=False
    )
    for path in (upstream, generator):
        stage = onnx.load(str(path), load_external_data=False)
        print(f"{path.name}: {len(stage.graph.node)} nodes, "
              f"{path.stat().st_size / 1e6:.1f} MB")
    return upstream, generator, crossing, inputs


def phoneme_ids(text: str, config_path: Path) -> np.ndarray:
    """Maps Moonshine's IPA through the voice's own phoneme table."""
    from moonshine_voice.g2p import GraphemeToPhonemizer

    config = json.loads(config_path.read_text())
    table = config["phoneme_id_map"]
    ipa = GraphemeToPhonemizer("en_us").to_ipa(text)
    ids = list(table["^"][0:1]) or [1]
    for character in ipa:
        if character in table:
            ids.extend(table[character])
            ids.extend(table["_"])
    ids.extend(table.get("$", [2]))
    return np.array([ids], dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to <model>.json")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/piper-split"))
    parser.add_argument("--text", default=(
        "The quick brown fox jumps over the lazy dog, and then pauses for a "
        "moment before carrying on to the end of a rather longer sentence."
    ))
    parser.add_argument("--target-frames", type=int, default=20)
    parser.add_argument("--speaker", type=int, default=0,
                        help="Speaker id, for multi-speaker voices")
    args = parser.parse_args()

    config_path = args.config or args.model.with_suffix(".onnx.json")
    upstream_path, generator_path, crossing, inputs = split(
        args.model, args.out_dir)
    whole = session(args.model)
    upstream = session(upstream_path)
    generator = session(generator_path)

    ids = (phoneme_ids(args.text, config_path) if config_path.is_file()
           else np.array([[1] + list(range(2, 60)) + [2]], dtype=np.int64))
    feed = {
        "input": ids,
        "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
        "scales": np.array(SCALES, dtype=np.float32),
    }
    if "sid" in inputs:
        feed["sid"] = np.array([args.speaker], dtype=np.int64)
    reference = whole.run(["output"], feed)[0].reshape(-1)
    crossed = dict(zip(crossing, upstream.run(crossing, feed)))
    # The frame axis is the one that grows with the text. Anything else is a
    # per-utterance conditioning vector that every slice reuses unchanged.
    latent = max(crossed, key=lambda name: crossed[name].shape[-1])
    conditioning = {n: v for n, v in crossed.items() if n != latent}
    if conditioning:
        print(f"carried whole into every chunk: {', '.join(conditioning)}")
    latent_values = crossed[latent]
    frames = latent_values.shape[-1]
    samples_per_frame = reference.size // frames
    print(f"{ids.shape[1]} phoneme ids -> {frames} latent frames, "
          f"{reference.size} samples ({samples_per_frame} per frame)")

    joined = generator.run(["output"], dict(crossed))[0].reshape(-1)
    print(f"split reproduces the whole graph to "
          f"{float(np.max(np.abs(joined - reference))):.3e}")

    sample_rate = json.loads(config_path.read_text())["audio"]["sample_rate"]
    upstream_seconds = best_of(3, lambda: upstream.run(crossing, feed))
    generator_seconds = best_of(
        3, lambda: generator.run(["output"], dict(crossed))
    )
    audio_seconds = reference.size / sample_rate
    total = upstream_seconds + generator_seconds
    print(f"upstream {upstream_seconds * 1000:.0f} ms + generator "
          f"{generator_seconds * 1000:.0f} ms = {total * 1000:.0f} ms for "
          f"{audio_seconds:.2f} s (RTF {total / audio_seconds:.3f}); "
          f"the generator is {generator_seconds / total:.0%} of the work")

    start = frames // 3
    end = min(frames, start + args.target_frames)
    target = reference[start * samples_per_frame: end * samples_per_frame]
    print(f"\ndecoding frames [{start}, {end}) with growing context:")
    print(f"{'pad':>4} {'decoded':>8} {'work':>6} {'rel err':>9} {'level dB':>9}")
    for pad in (0, 1, 2, 4, 8, 16, 32):
        low = max(0, start - pad)
        high = min(frames, end + pad)
        sliced = dict(conditioning)
        sliced[latent] = latent_values[:, :, low:high]
        piece = generator.run(["output"], sliced)[0].reshape(-1)
        offset = (start - low) * samples_per_frame
        window = piece[offset: offset + target.size]
        length = min(window.size, target.size)
        error = float(np.sqrt(np.mean((window[:length] - target[:length]) ** 2)))
        scale = float(np.sqrt(np.mean(target[:length] ** 2))) or 1e-9
        level = 20.0 * np.log10(
            (float(np.sqrt(np.mean(window[:length] ** 2))) or 1e-9) / scale
        )
        print(f"{pad:4d} {high - low:8d} {(high - low) / (end - start):5.2f}x "
              f"{error / scale:9.4f} {level:+9.2f}")

    # What streaming would actually cost: the upstream stage runs once for the
    # whole utterance, so only the generator pays the padding multiplier.
    frames_per_second = sample_rate / samples_per_frame
    print(f"\nstreaming cost at {frames_per_second:.0f} latent frames/second:")
    print(f"{'chunk s':>8} {'pad':>4} {'gen work':>9} {'total work':>11} "
          f"{'first audio':>12}")
    for chunk_seconds in (0.25, 0.5, 1.0):
        per_chunk = max(1, int(round(chunk_seconds * frames_per_second)))
        for pad in (8, 16):
            gen_multiplier = (per_chunk + 2 * pad) / per_chunk
            streamed = upstream_seconds + generator_seconds * gen_multiplier
            first = upstream_seconds + generator_seconds * (
                (per_chunk + pad) / frames
            )
            print(f"{chunk_seconds:8.2f} {pad:4d} {gen_multiplier:8.2f}x "
                  f"{streamed / total:10.2f}x {first * 1000:11.0f} ms")
    print(f"unchunked, the first audio arrives after {total * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
