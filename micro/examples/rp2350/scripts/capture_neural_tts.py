#!/usr/bin/env python3
"""Capture the neural_tts_test app's USB output: logs + AUDIO frames -> wavs.

Reads the CDC stream from the RP2350 (use /dev/cu.*, never /dev/tty.* on
macOS), prints every text line, and writes each AUDIO frame as a .wav.

Usage:
  python capture_neural_tts.py [--out DIR] [--utterances N] [--port PATH]
"""

from __future__ import annotations

import argparse
import glob
import struct
import sys
import time
import wave
from pathlib import Path


def find_port() -> str:
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not ports:
        raise SystemExit("no /dev/cu.usbmodem* found -- is the board attached?")
    return ports[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--out", type=Path, default=Path("neural_tts_capture"))
    ap.add_argument("--utterances", type=int, default=4,
                    help="stop after this many AUDIO frames")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    port = args.port or find_port()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[capture] reading {port}", flush=True)

    import serial  # pyserial
    ser = serial.Serial(port, 115200, timeout=1.0)

    captured = 0
    deadline = time.time() + args.timeout
    line = b""
    while captured < args.utterances and time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b != b"\n":
            line += b
            continue
        text = line.decode("utf-8", "replace").strip()
        line = b""
        if not text:
            continue
        print(f"[dev] {text}", flush=True)
        if not text.startswith("AUDIO "):
            continue
        _, rate_s, n_s = text.split()
        rate, n = int(rate_s), int(n_s)
        raw = b""
        want = n * 2
        t0 = time.time()
        while len(raw) < want and time.time() - t0 < 120:
            chunk = ser.read(want - len(raw))
            if chunk:
                raw += chunk
        if len(raw) < want:
            print(f"[capture] SHORT frame: {len(raw)}/{want} bytes", flush=True)
            continue
        path = args.out / f"utt_{captured:02d}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(raw)
        peak = max(abs(struct.unpack(f"<{n}h", raw)[i]) for i in range(0, n, 97))
        print(f"[capture] wrote {path} ({n / rate:.2f} s, peak ~{peak})",
              flush=True)
        captured += 1
    ser.close()
    return 0 if captured == args.utterances else 1


if __name__ == "__main__":
    sys.exit(main())
