#!/usr/bin/env python3
"""Measures what chunked Kokoro decoding costs and how audible the seams are.

Given the two stages from scripts/split-kokoro-stages.py, it runs prosody once,
picks chunk boundaries at quiet unvoiced frames, decodes each chunk with padding
and overlap, crossfades them back together, and compares the result with the
unchunked render.

What it measures, on the shipped uint8 model with af_heart over a 10.8 s
paragraph:

    pad frames   spectral distance   decoder work
             0             5.11 dB          1.10x
             4             3.61 dB          1.48x
             8             3.04 dB          1.87x
            16             2.48 dB          2.65x
            32             1.88 dB          4.12x

Read that table for cost, not for quality. `spectral_distance` is a waveform
comparison, and on this model it has repeatedly disagreed with how the audio
actually sounds — the same mistake was made twice during the quantisation work,
where the build with the best spectral distance was the only one a recogniser
could not read. Chunk length was originally chosen from this table and from seam
click level, and both picked 0.5 s, which an intelligibility check then rejected.

Judged instead by word error, round-tripping the audio through Moonshine's own
base-en recogniser over 40 sentences and 4 voices (1,880 reference words):

    setting          WER     change vs whole (95% CI)   work    first audio
    whole          2.98%     --                         1.00x       408 ms
    snap 1.0 s     3.35%     +0.37  (-0.48 to +1.27)    1.47x       113 ms
    snap 0.5 s     5.37%     +2.39  (+1.16 to +3.71)    2.07x        59 ms

So a 1 s target is intelligibility-neutral and a 0.5 s target is not. What
remains at 1 s is the AdaIN drift, which a recogniser is indifferent to and
which has to be judged by ear.

By ear, at a 1 s target with snap placement and a crossfade, the level moving
between chunks is what is left. Splitting that error against the unchunked
render into a per-chunk offset and a shape change inside the chunk puts only
37% in the offset, so no gain scheme can reach the other 63%: matching each
chunk to the previous one's tail, and anchoring each chunk to the energy
contour the prosody stage predicts for the whole sentence, were both measured
and neither beat doing nothing. Feeding the decoder more context does work but
buys little until it is decoding nearly everything, since AdaIN's statistics
are genuinely global -- 0.5 s of context each side takes 4.67 dB to 4.05 dB for
twice the work, and only full past and future reaches 0.19 dB, at 10.7x.

What does work is fewer, longer chunks. Only the first chunk sets latency, so
the rest can grow (6 sentences, 1 s chunks as the baseline):

    schedule            seams   level error   step at seam    work
    uniform 1.0 s         9.2       4.82 dB        3.51 dB   1.40x
    0.6 s then double     3.0       3.54 dB        2.18 dB   1.13x
    0.7 s then remainder  1.0       1.75 dB        1.04 dB   1.04x

Growing is better on every axis at once, including cost, and word error over 4
voices and 10 sentences puts it at +1.22 against the unchunked render where the
uniform 1 s grid is +1.73. A growth factor g is safe only while g * RTF <= 1,
because chunk k has to finish decoding inside chunk k-1's playing time; the
remainder schedule is the g = infinity case and will underrun on long sentences
on slow hardware, so it needs the decoder's measured speed behind it.

This behaviour is not an artifact of the ONNX export. Feeding identical captured
decoder inputs to PyTorch and to the deployed graph gives 1.655 against 1.885 on
the same metric at a 1 s chunk, so the export costs about 14% and reproduces the
underlying effect. Note that the PyTorch decoder redraws its noise excitation on
every call (19.9 dB SNR between two identical runs) while the exported graph is
deterministic; silencing that noise moves the chunking numbers by under 1%, so
it affects the measurement floor rather than the chunking cost.

Growing is therefore what `--growth` defaults to, at 2.0. It is bounded by
decode speed rather than by quality: with a growth factor g and a decoder
costing r times realtime, g * r <= 1 keeps every chunk ahead of playback, and
the k cancels out of that inequality, so the margin neither builds nor erodes
and the first join decides it. Padding tightens it a little, since each chunk
decodes a roughly fixed extra span whatever its length, which weighs most on
the short early chunks -- at 0.5 s growing to 1.0 s that works out around 2.3x
realtime rather than 2x. `playback_margin` simulates the measured timings
instead of trusting the inequality, and `run` prints the spare time at the
tightest join.

Moonshine currently streams whole sentences and `WholeUtteranceChunkSource` is
the only source in the shipped core.

Usage:
    python scripts/kokoro-stream-prototype.py --stages /tmp/kokoro-stages-shrunk
    python scripts/kokoro-stream-prototype.py --stages ... --growth 1.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
KOKORO_DATA = REPO / "core" / "moonshine-tts" / "data" / "kokoro"

# Kokoro emits 600 audio samples per prosody frame, so 40 frames a second at
# its 24 kHz output rate.
SAMPLES_PER_FRAME = 600
SAMPLE_RATE = 24000
FRAMES_PER_SECOND = SAMPLE_RATE / SAMPLES_PER_FRAME

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs, and then take a short break "
    "before the next long sentence begins in earnest."
)


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def phoneme_ids(text: str) -> np.ndarray:
    """Phonemizes with the shipped G2P and maps through Kokoro's vocab."""
    from moonshine_voice.g2p import GraphemeToPhonemizer

    phonemes = GraphemeToPhonemizer("en_us").to_ipa(text)
    vocab = json.loads((KOKORO_DATA / "config.json").read_text())["vocab"]
    ids = [0]
    for character in phonemes:
        if character in vocab:
            ids.append(vocab[character])
    ids.append(0)
    return np.array([ids], dtype=np.int64)


def load_style(voice: str) -> np.ndarray:
    """Reads a .kokorovoice pack: "KVO1", rows, cols, then float32 style rows.

    Kokoro picks the row matching the token count, so the whole table is
    returned and the caller indexes it.
    """
    raw = (KOKORO_DATA / "voices" / f"{voice}.kokorovoice").read_bytes()
    if raw[:4] != b"KVO1":
        raise SystemExit(f"{voice}.kokorovoice is not a KVO1 pack")
    rows, cols = np.frombuffer(raw, dtype=np.uint32, count=2, offset=4)
    values = np.frombuffer(raw, dtype=np.float32, count=int(rows) * int(cols), offset=12)
    return values.reshape(int(rows), 1, int(cols))


def standardize(values: np.ndarray) -> np.ndarray:
    spread = float(values.std())
    if spread < 1e-9:
        return np.zeros_like(values)
    return (values - float(values.mean())) / spread


def boundary_cost(f0: np.ndarray, n: np.ndarray, frames: int) -> np.ndarray:
    """Quiet, unvoiced frames are the cheapest places to cut.

    F0 and N arrive at twice the frame rate, so each pair is averaged down.
    """
    f0_frames = np.asarray(f0, dtype=np.float32).reshape(-1)[: 2 * frames]
    n_frames = np.asarray(n, dtype=np.float32).reshape(-1)[: 2 * frames]
    f0_frames = f0_frames.reshape(frames, 2).mean(axis=1)
    n_frames = n_frames.reshape(frames, 2).mean(axis=1)
    return standardize(np.abs(n_frames)) + 0.5 * standardize(
        np.minimum(f0_frames, 200.0)
    )


def snap(cost: np.ndarray, target: int, search: int,
         growth: float = 1.0) -> list[int]:
    """Chunk boundaries at the quietest frame near each nominal cut.

    ``growth`` multiplies the target length after every chunk, so ``target`` is
    the first chunk only and 1.0 gives the uniform grid. Growing is close to
    free: only the first chunk is decoded before anything can play, so the ones
    after it are produced while their predecessor is still sounding, and longer
    chunks carry AdaIN statistics closer to the sentence's own. Where the
    growth has to stop is set by decode speed, not by quality — see
    ``playback_margin``.
    """
    frames = len(cost)
    boundaries, current, step = [0], 0, float(target)
    while current + max(1, int(round(step))) < frames:
        nominal = current + max(1, int(round(step)))
        low = max(current + 2, nominal - search)
        high = min(frames - 2, nominal + search + 1)
        if high <= low:
            nxt = min(nominal, frames - 1)
        else:
            nxt = int(low + np.argmin(cost[low:high]))
        if nxt <= current:
            break
        boundaries.append(nxt)
        current = nxt
        step *= growth
    boundaries.append(frames)
    return sorted(set(boundaries))


def playback_margin(
    first_wait: float, decode_seconds: list[float], durations: list[float]
) -> tuple[float, int]:
    """Spare time at each join, and how many joins run out of it.

    A chunk has to finish decoding before its predecessor finishes playing.
    Growing the chunks grows the decode time too, but it grows the buffer that
    hides it by the same factor, so for uniform decode speed the margin is
    constant in relative terms and the very first join is the one that decides
    whether a growth factor is usable. Padding upsets that slightly, costing a
    roughly fixed amount per chunk that weighs most on the short early ones,
    which is why this simulates the real timings rather than trusting g * RTF.
    """
    if not decode_seconds:
        return float("nan"), 0
    decoded_at = first_wait + decode_seconds[0]
    playing_until = decoded_at + durations[0]
    worst, underruns = float("inf"), 0
    for index in range(1, len(decode_seconds)):
        decoded_at += decode_seconds[index]
        slack = playing_until - decoded_at
        worst = min(worst, slack)
        if slack < 0.0:
            underruns += 1
        # A late chunk stalls playback rather than being skipped.
        playing_until = max(playing_until, decoded_at) + durations[index]
    return (float("nan") if worst == float("inf") else worst), underruns


def overlap_gain(
    accumulated: np.ndarray, piece: np.ndarray, overlap: int, limit_db: float
) -> float:
    """Gain that makes the new chunk as loud as the tail it is joining.

    The decoder normalizes over whatever time span it is given, so a chunk can
    come back at a slightly different level than the same audio would have had
    in a full render. The two sides of a crossfade cover the same moment, so
    their loudness ratio measures that offset directly. Clamped, because a
    large ratio means the two really are different audio, not a gain error.
    """
    if overlap <= 0 or accumulated.size < overlap or piece.size < overlap:
        return 1.0
    tail = float(np.sqrt(np.mean(accumulated[-overlap:] ** 2)))
    head = float(np.sqrt(np.mean(piece[:overlap] ** 2)))
    floor = 1e-4
    if tail < floor or head < floor:
        return 1.0
    limit = 10.0 ** (limit_db / 20.0)
    return float(np.clip(tail / head, 1.0 / limit, limit))


def crossfade_append(
    accumulated: np.ndarray, piece: np.ndarray, overlap: int
) -> np.ndarray:
    """Sine-law overlap-add, which holds power constant through the seam."""
    if accumulated.size == 0 or overlap <= 0:
        return np.concatenate([accumulated, piece])
    overlap = min(overlap, accumulated.size, piece.size)
    ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
    fade_out = np.cos(ramp * np.pi / 2.0)
    fade_in = np.sin(ramp * np.pi / 2.0)
    head = accumulated[:-overlap]
    seam = accumulated[-overlap:] * fade_out + piece[:overlap] * fade_in
    return np.concatenate([head, seam, piece[overlap:]])


def seam_click_level(streamed: np.ndarray, seams: list[int]) -> float:
    """How much the joins stand out from the waveform's own roughness.

    A chunk cannot reproduce the unchunked waveform sample for sample: the
    vocoder's noise excitation is drawn from the input length, so every chunk
    gets a different (equally valid) realization. Comparing against the whole
    render therefore measures nothing audible. What is audible is a step at the
    join, so compare the largest sample-to-sample jump near each seam against
    how large the jumps get everywhere else. Zero dB means the seams look like
    ordinary speech; well below zero means they are invisible.
    """
    if not seams:
        return float("-inf")
    step = np.abs(np.diff(streamed))
    if step.size == 0:
        return float("-inf")
    ordinary = float(np.percentile(step, 99.9)) or 1e-12
    window = int(0.005 * SAMPLE_RATE)
    worst = 0.0
    for seam in seams:
        low = max(0, seam - window)
        high = min(step.size, seam + window)
        if high > low:
            worst = max(worst, float(np.max(step[low:high])))
    return 20.0 * np.log10(max(worst, 1e-12) / ordinary)


def mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Log mel magnitudes, as a stand-in for what the ear resolves.

    Deliberately crude — 32 triangular bands over a 512-point STFT — because
    the question is only whether two renders sound alike, not what they sound
    like. Phase is dropped, which is the point: the vocoder redraws its noise
    excitation for every input length, so two renders of the same speech differ
    completely sample by sample while being the same sound.
    """
    window_size, hop, bands = 512, 128, 32
    if audio.size < window_size:
        return np.zeros((0, bands), dtype=np.float32)
    window = np.hanning(window_size).astype(np.float32)
    frames = 1 + (audio.size - window_size) // hop
    strided = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frames, window_size),
        strides=(audio.strides[0] * hop, audio.strides[0]),
    )
    power = np.abs(np.fft.rfft(strided * window, axis=1)) ** 2

    def to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    edges = np.linspace(to_mel(np.array(50.0)), to_mel(np.array(SAMPLE_RATE / 2)),
                        bands + 2)
    hz_edges = 700.0 * (10.0 ** (edges / 2595.0) - 1.0)
    bin_hz = np.fft.rfftfreq(window_size, 1.0 / SAMPLE_RATE)
    filters = np.zeros((power.shape[1], bands), dtype=np.float32)
    for band in range(bands):
        low, mid, high = hz_edges[band: band + 3]
        rising = (bin_hz >= low) & (bin_hz <= mid)
        falling = (bin_hz > mid) & (bin_hz <= high)
        filters[rising, band] = (bin_hz[rising] - low) / max(mid - low, 1e-9)
        filters[falling, band] = (high - bin_hz[falling]) / max(high - mid, 1e-9)
    return np.log10(power @ filters + 1e-10).astype(np.float32)


def spectral_distance(streamed: np.ndarray, reference: np.ndarray) -> float:
    """Mean absolute log-mel difference between two renders, in dB.

    Counts only the loud parts. Near silence the log of a tiny number swings
    wildly for no audible reason, and Kokoro redraws its noise excitation for
    every input length, so an unweighted average mostly measures which noise
    happened to be drawn. Weighting by the reference's own level asks the
    narrower question that matters: where there is speech to hear, is it the
    same speech?
    """
    shared = min(streamed.size, reference.size)
    if shared == 0:
        return float("nan")
    a = mel_spectrogram(np.ascontiguousarray(streamed[:shared]))
    b = mel_spectrogram(np.ascontiguousarray(reference[:shared]))
    rows = min(a.shape[0], b.shape[0])
    if rows == 0:
        return float("nan")
    a, b = a[:rows], b[:rows]
    # Reference power per bin, with everything more than 40 dB below the peak
    # discarded as inaudible.
    power = 10.0**b
    weight = np.where(power >= power.max() * 1e-4, power, 0.0)
    if weight.sum() <= 0.0:
        return float("nan")
    return float(np.sum(np.abs(a - b) * weight) / np.sum(weight)) * 10.0


def level_step(audio: np.ndarray, position: int, window: int) -> float | None:
    """Short-term loudness change across ``position``, in dB."""
    before = audio[max(0, position - window): position]
    after = audio[position: position + window]
    if before.size < window // 2 or after.size < window // 2:
        return None
    floor = 1e-4
    rms_before = max(float(np.sqrt(np.mean(before**2))), floor)
    rms_after = max(float(np.sqrt(np.mean(after**2))), floor)
    return 20.0 * np.log10(rms_after / rms_before)


def seam_level_jump(
    streamed: np.ndarray, reference: np.ndarray, seams: list[int]
) -> float:
    """Largest loudness step a join adds beyond what the speech already does.

    Instance normalization over a chunk's own statistics would show up here as
    a gain change. Boundaries are chosen at quiet frames, though, so the raw
    step across a seam is often large and entirely legitimate — the unchunked
    render has the same step. Only the excess counts.
    """
    window = int(0.02 * SAMPLE_RATE)
    worst = 0.0
    for seam in seams:
        streamed_step = level_step(streamed, seam, window)
        reference_step = level_step(reference, seam, window)
        if streamed_step is None or reference_step is None:
            continue
        worst = max(worst, abs(streamed_step - reference_step))
    return worst


def run(stages: Path, voice: str, text: str, chunk_s: float, tol_s: float,
        crossfade_s: float, pad_frames: int, match_gain_db: float,
        growth: float, out_dir: Path | None) -> None:
    prosody = session(stages / "prosody.onnx")
    decoder = session(stages / "decoder.onnx")

    ids = phoneme_ids(text)
    styles = load_style(voice)
    style = styles[min(ids.shape[1], len(styles) - 1)].astype(np.float32)
    speed = np.array([1.0], dtype=np.float32)

    started = time.perf_counter()
    asr, f0, n = prosody.run(["asr", "f0", "n"], {
        "input_ids": ids, "style": style, "speed": speed
    })
    prosody_seconds = time.perf_counter() - started
    frames = asr.shape[-1]

    started = time.perf_counter()
    reference = decoder.run(
        ["waveform"], {"asr": asr, "f0": f0, "n": n, "style": style}
    )[0].reshape(-1)
    whole_decode_seconds = time.perf_counter() - started

    target = max(1, int(round(chunk_s * FRAMES_PER_SECOND)))
    search = max(0, int(round(tol_s * FRAMES_PER_SECOND)))
    overlap_samples = int(round(crossfade_s * SAMPLE_RATE))
    # Each chunk reaches half a crossfade past its boundary on both sides, so
    # neighbours overlap by exactly one crossfade and the joined length is the
    # same as the unchunked render.
    half_overlap = overlap_samples // 2

    cost = boundary_cost(f0, n, frames)
    boundaries = snap(cost, target, search, growth)
    total_samples = frames * SAMPLES_PER_FRAME

    streamed = np.zeros(0, dtype=np.float32)
    chunk_seconds = []
    seams = []
    decoded_frames = 0
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        keep_from = max(0, start * SAMPLES_PER_FRAME - half_overlap)
        keep_to = min(total_samples, end * SAMPLES_PER_FRAME + half_overlap)
        # Decode whole frames covering the kept span, plus padding either side
        # so the convolutions see their receptive field.
        low = max(0, keep_from // SAMPLES_PER_FRAME - pad_frames)
        high = min(frames, -(-keep_to // SAMPLES_PER_FRAME) + pad_frames)

        began = time.perf_counter()
        piece = decoder.run(["waveform"], {
            "asr": asr[:, :, low:high],
            "f0": f0[:, 2 * low: 2 * high],
            "n": n[:, 2 * low: 2 * high],
            "style": style,
        })[0].reshape(-1)
        chunk_seconds.append(time.perf_counter() - began)
        decoded_frames += high - low

        offset = low * SAMPLES_PER_FRAME
        piece = piece[keep_from - offset: keep_to - offset]
        if streamed.size:
            if match_gain_db > 0.0:
                piece = piece * overlap_gain(
                    streamed, piece, overlap_samples, match_gain_db
                )
            seams.append(streamed.size - overlap_samples // 2)
        streamed = crossfade_append(streamed, piece, overlap_samples)

    audio_seconds = reference.size / SAMPLE_RATE
    durations = [
        (boundaries[i + 1] - boundaries[i]) / FRAMES_PER_SECOND
        for i in range(len(boundaries) - 1)
    ]
    print(f"text: {len(text)} chars -> {ids.shape[1]} ids -> {frames} frames "
          f"({audio_seconds:.2f} s of audio)")
    print(f"chunks: {len(boundaries) - 1} at boundaries {boundaries}")
    print(f"chunk seconds (growth {growth:g}x): "
          + ", ".join(f"{d:.2f}" for d in durations))
    print(f"spectral distance: {spectral_distance(streamed, reference):.2f} dB "
          "from the unchunked render (under ~1 dB is the same sound)")
    print(f"seam click level: {seam_click_level(streamed, seams):+.1f} dB "
          f"(0 dB = indistinguishable from ordinary speech)")
    print("worst seam level jump: "
          f"{seam_level_jump(streamed, reference, seams):.1f} dB "
          "beyond the unchunked render")
    print(f"length: streamed {streamed.size} vs whole {reference.size} samples")
    print(f"decoder work: {decoded_frames / frames:.2f}x frames")
    print(f"prosody: {prosody_seconds * 1000:.0f} ms, "
          f"whole decode: {whole_decode_seconds * 1000:.0f} ms, "
          f"chunked decode total: {sum(chunk_seconds) * 1000:.0f} ms")
    print(f"time to first audio: {(prosody_seconds + chunk_seconds[0]) * 1000:.0f} ms "
          f"(vs {(prosody_seconds + whole_decode_seconds) * 1000:.0f} ms unchunked)")

    slack, underruns = playback_margin(prosody_seconds, chunk_seconds, durations)
    rate = whole_decode_seconds / max(audio_seconds, 1e-9)
    print(f"decode speed: {1.0 / max(rate, 1e-9):.1f}x realtime "
          f"({rate * 100:.1f}% of the audio duration in compute)")
    if underruns:
        print(f"playback margin: UNDERRUNS at {underruns} join(s), "
              f"worst {slack * 1000:.0f} ms short -- growth {growth:g}x is too "
              f"fast for this decoder")
    else:
        print(f"playback margin: {slack * 1000:.0f} ms spare at the tightest "
              f"join, no underruns")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_wav(out_dir / "whole.wav", reference)
        write_wav(out_dir / "streamed.wav", streamed)
        print(f"wrote {out_dir}/whole.wav and streamed.wav")


def write_wav(path: Path, audio: np.ndarray) -> None:
    import struct
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        clipped = np.clip(audio, -1.0, 1.0)
        handle.writeframes(
            struct.pack(f"<{clipped.size}h", *(clipped * 32767).astype(np.int16))
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--text", default=PARAGRAPH)
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.5,
        help="Length of the first chunk, which is the only one that sets "
             "time-to-first-audio. Later chunks grow by --growth",
    )
    parser.add_argument("--tolerance-seconds", type=float, default=0.2)
    parser.add_argument("--crossfade-seconds", type=float, default=0.025)
    parser.add_argument(
        "--pad-frames",
        type=int,
        default=4,
        help="Extra frames decoded either side and thrown away, to give the "
             "convolutions their receptive field",
    )
    parser.add_argument(
        "--match-gain-db",
        type=float,
        default=0.0,
        help="Cap, in dB, on the gain applied to a chunk so it joins the "
             "previous one at the same loudness. 0 disables the correction",
    )
    parser.add_argument(
        "--growth",
        type=float,
        default=2.0,
        help="How much longer each chunk is than the one before it. 1.0 gives "
             "a uniform grid. Higher is better for quality and for cost, and "
             "is limited only by decode speed: a chunk must finish decoding "
             "inside its predecessor's playing time, so growth much above the "
             "decoder's realtime multiple will underrun (default: 2.0)",
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    if args.growth < 1.0:
        parser.error("--growth below 1.0 shrinks the chunks, which underruns "
                     "by construction")

    run(args.stages, args.voice, args.text, args.chunk_seconds,
        args.tolerance_seconds, args.crossfade_seconds, args.pad_frames,
        args.match_gain_db, args.growth, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
