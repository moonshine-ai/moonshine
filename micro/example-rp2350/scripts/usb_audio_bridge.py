"""Bridge the laptop mic + speaker to the RP2350 over USB (audio peripheral sim).

The RP2350 firmware built with ``-DSPELLING_TINY_AUDIO=ON`` runs the streaming
VAD + SpellingCNN on-device and speaks the recognized letter/digit back via its
formant TTS. This script makes the laptop act as the board's microphone and
speaker:

  * captures the default mic at 16 kHz mono, frames it into 32 ms (512-sample)
    hops, each prefixed with a 0xA5 0x5A sync, and streams them to the device
    over the USB CDC serial port;
  * reads the device's event/audio stream back: ``VAD start`` / ``VAD end`` /
    ``RESULT <label> <prob>`` lines, then an ``AUDIO <rate> <n>`` header
    followed by ``n`` int16 samples of synthesized speech, which it plays on
    the laptop speaker.

It is turn-based: while the device is classifying + speaking (it stops reading
USB during the ~0.5 s STT + the TTS), the host pauses sending and flushes the
mic so stale audio isn't replayed, then resumes after the reply finishes.

Requires ``sounddevice`` + ``numpy`` (``pip install sounddevice``). On macOS the
serial node MUST be ``/dev/cu.usbmodem*`` (not ``/dev/tty.*``).

Example::

    python cpp-tiny/example-rp2350/scripts/usb_audio_bridge.py
    python cpp-tiny/example-rp2350/scripts/usb_audio_bridge.py \
        --serial /dev/cu.usbmodem1101 --input-device 2
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import sys
import termios
import threading
import time
import tty

import numpy as np

SR = 16000
HOP = 512
SYNC = bytes((0xA5, 0x5A))


def _resolve_serial(explicit: str | None, timeout: float = 20.0) -> str:
    """Return the device serial node, waiting for it to (re)appear.

    After a flash the Pico reboots and re-enumerates its USB CDC port, which
    takes a second or two; if you launch the bridge straight after flash.sh the
    node isn't there yet. Poll for it instead of failing immediately.
    """
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
                "board plugged in and running the SPELLING_TINY_AUDIO firmware "
                "(not BOOTSEL)?")
        if not announced:
            print("Waiting for the board's USB serial port to appear "
                  "(it re-enumerates for a second or two after a flash)...",
                  flush=True)
            announced = True
        time.sleep(0.25)


def _open_raw(dev: str) -> int:
    """Open the CDC device in raw, NON-BLOCKING mode.

    We KEEP O_NONBLOCK for the lifetime of the fd. The device stops draining USB
    while it does STT + TTS (it isn't reading hops then), so a *blocking* write
    would park the sender thread in an uninterruptible kernel os.write -- and a
    blocking read would do the same to the reader thread when the device is
    quiet. On macOS that uninterruptible state survives Ctrl-C and wedges the
    CDC handle until you physically replug. With O_NONBLOCK, os.write/os.read
    raise BlockingIOError (EAGAIN) instead of blocking, so both threads stay
    responsive and shutdown is always clean. The sender retries EAGAIN; the
    reader treats it as "no data yet".
    """
    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        tty.setraw(fd)
    except termios.error:
        pass
    return fd


class SerialReader:
    """Reads the mixed text/binary stream from the device fd (single reader)."""

    def __init__(self, fd: int):
        self.fd = fd
        self._buf = bytearray()

    def _fill(self) -> bool:
        try:
            chunk = os.read(self.fd, 4096)
        except BlockingIOError:
            # Non-blocking fd with no data right now: not an error, just wait.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serial", default=None, help="Device node (default: first /dev/cu.usbmodem*).")
    ap.add_argument("--input-device", default=None, help="sounddevice mic id/name.")
    ap.add_argument("--output-device", default=None, help="sounddevice speaker id/name.")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--wav", default=None,
                    help="Stream this 16 kHz wav as the 'mic' instead of live audio "
                         "(for testing without speaking). Followed by ~0.6 s silence "
                         "to trigger the VAD end, then the script waits for the reply.")
    ap.add_argument("--save-reply", default=None,
                    help="Write the device's spoken reply PCM to this wav path.")
    args = ap.parse_args()

    print("Bridge: starting...", flush=True)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    def _dev(arg):
        if arg is None:
            return None
        try:
            return int(arg)
        except ValueError:
            return arg

    dev = _resolve_serial(args.serial)
    fd = _open_raw(dev)
    print(f"Bridge: serial={dev}", flush=True)

    reader = SerialReader(fd)
    paused = threading.Event()   # set => stop sending mic (device is busy)
    stop = threading.Event()
    reply_done = threading.Event()
    ready = threading.Event()    # set when the device prints its readiness banner
    mic_q: queue.Queue[bytes] = queue.Queue(maxsize=256)

    # --- mic capture: enqueue framed hops ---
    def on_audio(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            pass
        pcm = (np.clip(indata[:, 0], -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        try:
            mic_q.put_nowait(pcm)
        except queue.Full:
            pass

    # --- sender thread: write framed hops unless paused ---
    # Uses select() with a short timeout so a non-reading device never blocks
    # us in an uninterruptible os.write (which previously wedged the USB port);
    # if the device isn't draining, we drop the hop instead of stalling.
    import select

    global _tx_hops, _tx_bytes, _tx_partial
    _tx_hops = _tx_bytes = _tx_partial = 0

    def sender():
        while not stop.is_set():
            try:
                pcm = mic_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if paused.is_set():
                continue  # drop stale audio while the device is replying
            # Write the WHOLE framed hop, looping on the byte count returned by
            # os.write() (the non-blocking CDC fd does partial writes -- the USB
            # endpoint is only 64 B). We deliberately do NOT use select() for
            # write-readiness: on macOS select() on a /dev/cu.* serial fd does
            # not reliably report writability, so it falsely claimed "not
            # writable" and dropped ~75/78 hops. Instead we retry on EAGAIN with
            # a tiny sleep, bounded by a deadline so a genuinely stalled device
            # can't wedge us (the device also times out its reads and keeps
            # draining, so in practice writes never stall that long).
            frame = memoryview(SYNC + pcm)
            sent = 0
            deadline = time.monotonic() + 2.0
            try:
                while sent < len(frame) and not stop.is_set():
                    try:
                        sent += os.write(fd, frame[sent:])
                    except BlockingIOError:
                        if time.monotonic() > deadline:
                            break  # device not draining; drop the rest
                        time.sleep(0.002)
            except OSError:
                break
            global _tx_hops, _tx_bytes, _tx_partial
            _tx_hops += 1
            _tx_bytes += sent
            if sent < len(frame):
                _tx_partial += 1

    def flush_mic():
        try:
            while True:
                mic_q.get_nowait()
        except queue.Empty:
            pass

    # --- reader thread: parse events + play the spoken reply ---
    def receiver():
        while not stop.is_set():
            line = reader.readline()
            if not line:
                continue
            if line.startswith("AUDIO "):
                parts = line.split()
                rate, n = int(parts[1]), int(parts[2])
                raw = reader.read_exact(n * 2)
                samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
                print(f"  [spoken reply: {n} samples @ {rate} Hz]")
                if args.save_reply:
                    try:
                        import soundfile as sf
                        sf.write(args.save_reply, samples, rate, subtype="PCM_16")
                        print(f"  [reply saved -> {args.save_reply}]")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  (save error: {exc})")
                try:
                    import sounddevice as sd
                    sd.play(samples, samplerate=rate, device=_dev(args.output_device))
                    sd.wait()
                except Exception as exc:  # noqa: BLE001
                    print(f"  (playback skipped/error: {exc})")
                flush_mic()
                paused.clear()  # turn over: resume listening
                reply_done.set()
                print("  [resumed listening]")
            elif line.startswith("RESULT "):
                parts = line.split()
                print(f">>> recognized: {parts[1]}  (p={parts[2]})")
                # The device ignores (does not speak) recognitions below 0.5, so
                # no AUDIO reply will follow -- resume listening immediately
                # instead of staying paused waiting for one. Keep this threshold
                # in sync with kMinResultProb in audio_service.cc.
                try:
                    if float(parts[2]) < 0.5:
                        flush_mic()
                        paused.clear()
                        reply_done.set()
                        print("  [ignored <0.5; resumed listening]")
                except (IndexError, ValueError):
                    pass
            elif line.startswith("VAD end"):
                paused.set()      # device is about to do STT+TTS; stop sending
                flush_mic()
                print(f"[{line}] -> classifying...")
            elif line.startswith("VAD start"):
                print("[speech detected]")
            elif line.strip():
                # The device is ready as soon as it prints its banner / enters
                # the listening loop -- unblock startup immediately (see below).
                if "[audio] ready" in line or "[audio] listening" in line:
                    ready.set()
                print(f"[dev] {line}")

    rx = threading.Thread(target=receiver, daemon=True)
    rx.start()

    # Wait until the device reports it's listening, rather than a blind sleep.
    # The board is ready well under a second after boot, so a fixed delay just
    # added dead time before "Speak a letter or digit". The timeout is only a
    # fallback for the case where the banner was missed (e.g. attaching mid-run).
    print("Waiting for device [audio] ready...")
    t0 = time.monotonic()
    if ready.wait(timeout=10.0):
        print(f"Device ready in {time.monotonic() - t0:.2f}s.")
    else:
        print("Device readiness banner not seen in 10 s; starting anyway "
              "(is it running the SPELLING_TINY_AUDIO firmware?).")

    tx = threading.Thread(target=sender, daemon=True)
    tx.start()

    if args.wav:
        # Simulate the mic from a file: stream the clip as real-time hops, then
        # ~0.6 s of silence to make the VAD commit the segment, then wait for
        # the device's reply.
        import soundfile as sf
        wav, sr = sf.read(args.wav, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            print(f"  (note: {args.wav} is {sr} Hz; the device expects {SR} Hz)")
        n_hops = len(wav) // HOP
        print(f"Streaming {args.wav} ({n_hops} hops) as simulated mic...")
        for k in range(n_hops):
            pcm = (np.clip(wav[k * HOP:(k + 1) * HOP], -1, 1) * 32767.0).astype("<i2").tobytes()
            mic_q.put(pcm)
            time.sleep(HOP / SR)
        for _ in range(int(0.6 * SR / HOP)):  # trailing silence -> VAD end
            if paused.is_set():
                break
            mic_q.put(b"\x00\x00" * HOP)
            time.sleep(HOP / SR)
        print(f"[tx] hops={_tx_hops} bytes={_tx_bytes} partial={_tx_partial}")
        if reply_done.wait(timeout=10.0):
            print("Loop complete (got recognition + spoken reply).")
        else:
            print("Timed out waiting for the device reply.")
        stop.set()
        os.close(fd)
        return 0

    print("Speak a letter or digit. Ctrl-C to stop.")
    # Stop cleanly on Ctrl-C (SIGINT) or `kill` (SIGTERM): the handler just sets
    # the flag, the main loop notices and tears down the stream + fd. (Relying
    # on KeyboardInterrupt alone is flaky under PortAudio's stream context.)
    import signal
    def _on_signal(signum, frame):  # noqa: ARG001
        stop.set()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    import sounddevice as sd
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=HOP, device=_dev(args.input_device),
                            callback=on_audio):
            while not stop.is_set():
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        os.close(fd)
        print("\nBridge stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
