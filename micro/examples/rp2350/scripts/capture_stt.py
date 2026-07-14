"""Capture the exact audio the on-device STT receives, over USB CDC.

The ``moonshine_micro_echo_hardware`` firmware (I2S mic + PWM speaker) is built
with ``SPELLING_DUMP_STT_USB``: for every recognition it streams the precise,
front-aligned + level-normalized 1 s clip that the SpellingCNN log-mel front-end
consumes -- "what the model receives" -- out the USB CDC log pipe, framed as::

    STTIN <sample_rate> <num_samples>\n
    <num_samples * int16 little-endian>
    \nEND\n

This script doubles as a serial monitor (it prints every text log line) AND
saves each STTIN clip as a numbered wav so you can listen to / analyze exactly
what the classifier saw. Run it INSTEAD of monitor.sh while debugging.

On macOS the serial node MUST be ``/dev/cu.usbmodem*`` (not ``/dev/tty.*``).

Example::

    python moonshine-micro/examples/rp2350/scripts/capture_stt.py
    python moonshine-micro/examples/rp2350/scripts/capture_stt.py \
        --out-dir /tmp/stt-capture --serial /dev/cu.usbmodem1101
"""

from __future__ import annotations

import argparse
import glob
import os
import termios
import time
import tty
import wave


def _resolve_serial(explicit: str | None, timeout: float = 20.0) -> str:
    """Return the device serial node, waiting for it to (re)appear after flash."""
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        if explicit:
            if os.path.exists(explicit):
                return explicit
        else:
            ports = sorted(glob.glob("/dev/cu.usbmodem*"))
            if ports:
                return ports[0]
        if time.monotonic() > deadline:
            target = explicit or "/dev/cu.usbmodem*"
            raise SystemExit(
                f"No serial port {target} found after {timeout:.0f}s. Is the "
                "board plugged in and running echo_hardware (not BOOTSEL)?")
        if not announced:
            print("Waiting for the board's USB serial port to appear "
                  "(it re-enumerates for a second or two after a flash)...",
                  flush=True)
            announced = True
        time.sleep(0.25)


def _open_raw(dev: str) -> int:
    """Open the CDC device in raw, NON-BLOCKING mode (mirrors usb_audio_bridge)."""
    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        tty.setraw(fd)
    except termios.error:
        pass
    return fd


class SerialReader:
    """Reads the mixed text/binary stream from the device fd."""

    def __init__(self, fd: int):
        self.fd = fd
        self._buf = bytearray()

    def _fill(self) -> bool:
        try:
            chunk = os.read(self.fd, 4096)
        except BlockingIOError:
            time.sleep(0.002)
            return True
        except OSError:
            return False
        if not chunk:
            time.sleep(0.002)
            return True
        self._buf += chunk
        return True

    def readline(self) -> str:
        while b"\n" not in self._buf:
            if not self._fill():
                return ""
        line, _, rest = self._buf.partition(b"\n")
        self._buf = bytearray(rest)
        return line.decode("ascii", "replace").rstrip("\r")

    def read_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            if not self._fill():
                break
        out = bytes(self._buf[:n])
        self._buf = bytearray(self._buf[n:])
        return out


def _save_wav(path: str, raw: bytes, rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serial", default=None,
                    help="Device node (default: first /dev/cu.usbmodem*).")
    ap.add_argument("--out-dir", default="/tmp/stt-capture",
                    help="Directory for the saved STTIN clips (default %(default)s).")
    args = ap.parse_args()

    dev = _resolve_serial(args.serial)
    fd = _open_raw(dev)
    reader = SerialReader(fd)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Capture: attached to {dev}")
    print(f"         saving STT input clips to {args.out_dir}/")
    print("         (also echoing device logs; Ctrl-C to stop)\n", flush=True)

    seq = 0
    try:
        while True:
            line = reader.readline()
            if not line:
                continue
            if line.startswith("STTIN "):
                try:
                    _, rate_s, n_s = line.split()
                    rate, n = int(rate_s), int(n_s)
                except ValueError:
                    print(line)
                    continue
                raw = reader.read_exact(n * 2)
                # The device sends "\nEND\n" after the payload; consume that line.
                tail = reader.readline()
                if tail not in ("", "END"):
                    # Not the trailer we expected -- surface it but keep going.
                    print(f"  (unexpected STTIN trailer: {tail!r})")
                seq += 1
                path = os.path.join(args.out_dir, f"sttin_{seq:04d}.wav")
                _save_wav(path, raw, rate)
                got = len(raw) // 2
                print(f"  [STT input #{seq}: {got}/{n} samples @ {rate} Hz "
                      f"-> {path}]", flush=True)
            else:
                print(line, flush=True)
    except KeyboardInterrupt:
        print("\nCapture: stopped.")
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
