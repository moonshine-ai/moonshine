"""Regression test for issue #196.

``MicTranscriber`` feeds microphone audio to the stream from PortAudio's
capture callback, which runs on a high-priority, time-critical audio thread.
The stream performs the actual (slow) transcription every ``update_interval``.
If that transcription runs *inline* on the capture callback, the callback
blocks for the full inference time and PortAudio reports ``input overflow`` --
exactly what was observed on a Raspberry Pi in
https://github.com/moonshine-ai/moonshine/issues/196.

This test reproduces that behaviour without a real microphone, model, or the
native library:

  * ``FakeInputStream`` stands in for ``sounddevice.InputStream`` and drives
    the capture callback at the real-time block cadence, timing how long each
    callback takes to return.
  * ``FakeStream`` stands in for the real ``Stream``: ``add_audio`` buffers
    cheaply but, every ``update_interval`` worth of audio, blocks for
    ``SLOW_UPDATE_S`` to emulate a slow on-device inference.

A callback that returns slower than the block's wall-clock duration would
overflow a real capture buffer, so we assert no callback ever exceeds that
budget. With the transcription running inline (the buggy implementation) the
update-triggering callbacks block ~``SLOW_UPDATE_S`` and the test fails; once
``MicTranscriber`` drains audio to a worker thread it passes.
"""

import threading
import time

import numpy as np
import pytest

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
# One captured block spans this many seconds of wall-clock time; a callback
# that takes longer than this cannot keep up with real-time capture.
BLOCK_PERIOD_S = BLOCK_SIZE / SAMPLE_RATE  # 64 ms
UPDATE_INTERVAL_S = 0.1
# Emulated inference time per update. Chosen well above BLOCK_PERIOD_S so an
# inline update unambiguously blows the real-time budget, but small enough to
# keep the test quick.
SLOW_UPDATE_S = 0.25
NUM_BLOCKS = 10


class FakeStream:
    """Minimal stand-in for ``moonshine_voice.transcriber.Stream``.

    Mirrors the real stream's contract: ``add_audio`` is cheap except that
    every ``update_interval`` of accumulated audio it runs a "transcription"
    that blocks for ``SLOW_UPDATE_S``.
    """

    def __init__(self, update_interval):
        self._update_interval = update_interval or UPDATE_INTERVAL_S
        self.transcribe_flags = 0
        self._stream_time = 0.0
        self._last_update_time = 0.0
        self.total_samples = 0
        self.add_audio_calls = 0
        # Thread idents from which add_audio (and its blocking update) ran.
        self.add_audio_idents = set()
        self.blocking_update_idents = set()

    def start(self):
        pass

    def stop(self):
        # The real stream flushes any trailing audio on stop.
        self._run_update()

    def close(self):
        pass

    def set_transcribe_flags(self, flags):
        self.transcribe_flags = int(flags)

    def add_listener(self, listener):
        pass

    def remove_listener(self, listener):
        pass

    def remove_all_listeners(self):
        pass

    def add_audio(self, audio_data, sample_rate=SAMPLE_RATE):
        self.add_audio_idents.add(threading.get_ident())
        self.add_audio_calls += 1
        self.total_samples += len(audio_data)
        self._stream_time += len(audio_data) / sample_rate
        if self._stream_time - self._last_update_time >= self._update_interval:
            self._run_update()
            self._last_update_time = self._stream_time

    def _run_update(self):
        self.blocking_update_idents.add(threading.get_ident())
        time.sleep(SLOW_UPDATE_S)


class FakeTranscriber:
    """Stand-in for ``Transcriber`` that hands back a ``FakeStream``."""

    def __init__(self, *args, **kwargs):
        self.stream = None

    def create_stream(self, update_interval=None, flags=0, transcribe_flags=0):
        self.stream = FakeStream(update_interval)
        return self.stream

    def close(self):
        pass


class FakeInputStream:
    """Stand-in for ``sounddevice.InputStream``.

    Delivers ``NUM_BLOCKS`` blocks of audio to the capture callback at the
    real-time block cadence and records how long each callback takes. Feeding
    only begins once ``go`` is set, so the test can guarantee MicTranscriber
    has finished ``start()`` (and set its listen flag) before audio arrives.
    """

    def __init__(self, samplerate, blocksize, device, channels, dtype, callback, **_):
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._channels = channels
        self._callback = callback
        self._thread = None
        self.go = threading.Event()
        self.finished = threading.Event()
        self.callback_durations = []

    def start(self):
        self._thread = threading.Thread(target=self._feed, daemon=True)
        self._thread.start()

    def stop(self):
        pass

    def close(self):
        pass

    def _feed(self):
        self.go.wait()
        rng = np.random.default_rng(0)
        for _ in range(NUM_BLOCKS):
            block = (
                rng.standard_normal((self._blocksize, self._channels)).astype(
                    np.float32
                )
                * 0.01
            )
            start = time.perf_counter()
            self._callback(block, self._blocksize, None, None)
            self.callback_durations.append(time.perf_counter() - start)
            # Maintain a real-time-ish cadence for the next block.
            remaining = BLOCK_PERIOD_S - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)
        self.finished.set()


def test_capture_callback_is_not_blocked_by_transcription(monkeypatch):
    pytest.importorskip("sounddevice")
    from moonshine_voice import mic_transcriber

    monkeypatch.setattr(mic_transcriber, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(mic_transcriber.sd, "InputStream", FakeInputStream)

    transcriber = mic_transcriber.MicTranscriber(
        model_path="unused",
        update_interval=UPDATE_INTERVAL_S,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
    )

    transcriber.start()
    fake_input = transcriber._sd_stream
    fake_stream = transcriber.mic_stream

    # Release the feeder now that start() has completed.
    fake_input.go.set()
    assert fake_input.finished.wait(timeout=30), "capture feeder did not finish"

    transcriber.stop()
    transcriber.close()

    assert fake_input.callback_durations, "callback was never invoked"
    slowest = max(fake_input.callback_durations)

    # The core assertion: no capture callback may exceed the real-time budget
    # for a single audio block. When transcription runs inline, the callbacks
    # that trigger an update block ~SLOW_UPDATE_S and this fails.
    assert slowest < BLOCK_PERIOD_S, (
        f"slowest capture callback took {slowest * 1000:.0f} ms, exceeding the "
        f"{BLOCK_PERIOD_S * 1000:.0f} ms real-time budget for one block -- "
        "transcription is running on the capture thread (see issue #196)"
    )

    # A blocking update must have happened somewhere (otherwise the test isn't
    # actually exercising the slow path), just never on the capture thread.
    assert fake_stream.blocking_update_idents, "no transcription update ran"
    capture_idents = {fake_input._thread.ident}
    assert not (fake_stream.blocking_update_idents & capture_idents), (
        "transcription update ran on the capture callback thread"
    )

    # The decoupling must not drop audio: every captured sample reaches the
    # stream by the time we stop.
    assert fake_stream.total_samples == NUM_BLOCKS * BLOCK_SIZE

    # The worker coalesces the backlog that builds while a slow transcription
    # is running, so it should reach the stream in fewer add_audio calls than
    # there were captured blocks (otherwise it would transcribe once per
    # chunk, which is wasteful).
    assert fake_stream.add_audio_calls < NUM_BLOCKS
