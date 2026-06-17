#!/usr/bin/env python3
"""Speak text on the RP2350 firmware and save the streamed PCM as a WAV.

The firmware (built with SPELLING_TINY_TTS=ON, the default) runs its spelling
sweep on boot and then drops into a USB text-to-speech command loop. This script
drives that loop over the USB CDC serial port:

    host  -> "SPEAK the quick brown fox\\n"
    board -> "AUDIO <sample_rate> <num_samples>\\n"
    board -> <num_samples * int16 little-endian PCM>
    board -> "\\nEND <num_samples>\\n"

and writes the PCM to a mono 16-bit WAV.

Usage:
    python tts_speak.py "the quick brown fox" -o fox.wav
    python tts_speak.py --ipa "h\\u0259lo\\u028a" -o hi.wav
    python tts_speak.py --rate 16000 --gender 0.8 "hello there" -o she.wav

Requires pyserial:  pip install pyserial

Notes:
  * On macOS use /dev/cu.usbmodem* (auto-detected); /dev/tty.* would block on
    open() because TinyUSB CDC never asserts DCD (see monitor.sh).
  * The board only reads commands once it has printed "[tts] ready"; this script
    waits for that, and also resends the command if a reset reprints it.
"""

import argparse
import glob
import sys
import time
import wave


def find_port() -> str:
    ports = sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        sys.exit("error: no /dev/cu.usbmodem* (or /dev/ttyACM*) port found.\n"
                 "       Is the board plugged in and flashed?")
    return ports[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", help="Text (or IPA with --ipa) to synthesize.")
    ap.add_argument("-o", "--output", default="board_tts.wav", help="Output WAV path.")
    ap.add_argument("--port", default="", help="Serial port (default: auto-detect).")
    ap.add_argument("--ipa", action="store_true", help="Send as IPA (IPA cmd), bypass G2P.")
    ap.add_argument("--rate", type=int, default=0, help="Set sample rate (Hz) before speaking.")
    ap.add_argument("--speed", type=float, default=0.0, help="Set speed multiplier before speaking.")
    ap.add_argument("--gender", type=float, default=-1.0, help="Set voice gender 0..1 before speaking.")
    ap.add_argument("--boot-timeout", type=float, default=60.0,
                    help="Seconds to wait for the board's TTS loop (boot + spelling sweep).")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="Seconds to wait for the AUDIO reply after sending the command.")
    args = ap.parse_args()

    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("error: pyserial not installed.  pip install pyserial")

    port = args.port or find_port()
    print(f"[tts_speak] port={port}", file=sys.stderr)

    with serial.Serial(port, baudrate=115200, timeout=0.5) as ser:
        cmd = ("IPA " if args.ipa else "SPEAK ") + args.text

        def send_line(s: str):
            ser.write((s + "\n").encode("utf-8", "replace"))
            ser.flush()

        # Apply optional voice settings first.
        if args.rate > 0:
            send_line(f"RATE {args.rate}")
        if args.speed > 0:
            send_line(f"SPEED {args.speed}")
        if args.gender >= 0.0:
            send_line(f"GENDER {args.gender}")

        # Wait for the board to enter its TTS loop, then send the command.
        # Re-send on every "[tts] ready" (covers reset / reconnect). Serial reads
        # tolerate USB re-enumeration during the long spelling sweep on boot.
        send_line(cmd)
        deadline = time.time() + args.boot_timeout
        rate = nsamp = None
        while time.time() < deadline:
            try:
                raw = ser.readline()
            except serial.SerialException as e:
                print(f"  board| (serial dropped: {e}; waiting for reconnect)",
                      file=sys.stderr)
                ser.close()
                time.sleep(1.0)
                p = args.port or find_port()
                ser.port = p
                ser.open()
                time.sleep(0.5)
                send_line(cmd)
                continue
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("AUDIO "):
                parts = line.split()
                rate, nsamp = int(parts[1]), int(parts[2])
                break
            if line.startswith("[tts] err"):
                sys.exit(f"error: board reported: {line}")
            if line:
                print(f"  board| {line}", file=sys.stderr)
            if "[tts] ready" in line:
                send_line(cmd)
        if rate is None:
            sys.exit("error: timed out waiting for AUDIO header from board.")

        print(f"[tts_speak] receiving {nsamp} samples @ {rate} Hz ...", file=sys.stderr)

        # Read exactly nsamp*2 raw bytes (binary follows the AUDIO line directly).
        want = nsamp * 2
        ser.timeout = args.timeout
        buf = bytearray()
        while len(buf) < want:
            chunk = ser.read(want - len(buf))
            if not chunk:
                sys.exit(f"error: timed out after {len(buf)}/{want} PCM bytes.")
            buf.extend(chunk)

        # Drain the trailing "\nEND <n>" framing line (best-effort).
        ser.timeout = 1.0
        for _ in range(3):
            tail = ser.readline().decode("utf-8", "replace").strip()
            if tail.startswith("END"):
                break

    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(buf))

    print(f"[tts_speak] wrote {args.output} ({nsamp/rate:.2f} s, {rate} Hz)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
