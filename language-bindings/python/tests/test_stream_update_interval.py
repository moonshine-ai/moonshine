"""How often a Stream is willing to enter the engine.

The update interval is a floor rather than a cadence: a pass has to cover at
least as much audio as the last one took to make. Most of what a pass costs is
not the audio in it -- measured on the tiny model with speakers, 102ms of a pass
goes on getting started and 269ms on each second of audio it looks at -- so
asking twice a second pays that overhead twice a second, and a machine that
cannot quite afford it does not fall behind by a fixed amount, it falls behind
further every pass. Replayed as a live session, a machine three times slower
than the one those numbers came from ended a three-minute meeting 82 seconds
behind, still delivering lines 80 seconds after the audio stopped.

The clock here is the test's to say, so that none of this depends on how long
anything really takes.
"""

import ctypes

import pytest

from moonshine_voice import transcriber as transcriber_module
from moonshine_voice.transcriber import Stream

RATE = 16000
INTERVAL = 0.5
#: What a capture callback hands over, in seconds.
CHUNK = 0.02


class FakeTranscriber:
    """Just enough of a transcriber for a Stream to be built on."""

    def __init__(self, lib):
        self._lib = lib
        self._handle = 1

    def _parse_transcript(self, out_transcript):
        return transcriber_module.Transcript(lines=[])


class FakeLib:
    """Counts the passes that reach the engine, and charges for each."""

    def __init__(self, clock, cost):
        self._clock = clock
        self._cost = cost
        self.passes = 0
        self.audio_samples = 0

    def moonshine_create_stream(self, transcriber_handle, flags):
        return 1

    def moonshine_transcribe_add_audio_to_stream(
        self, transcriber_handle, stream_handle, audio, count, sample_rate, flags
    ):
        self.audio_samples += count
        return 0

    def moonshine_transcribe_stream(
        self, transcriber_handle, stream_handle, flags, out_transcript
    ):
        self.passes += 1
        self._clock.advance(self._cost(self.passes))
        return 0

    def moonshine_stop_stream(self, transcriber_handle, stream_handle):
        return 0


class Clock:
    """A monotonic clock that only moves when the test says so."""

    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def paced(monkeypatch):
    """Builds streams whose passes cost exactly what the test asks."""

    clock = Clock()
    monkeypatch.setattr(transcriber_module.time, "monotonic", lambda: clock.now)

    def build(cost):
        charge = cost if callable(cost) else (lambda _pass: cost)
        lib = FakeLib(clock, charge)
        stream = Stream(FakeTranscriber(lib), update_interval=INTERVAL)
        return stream, lib, clock

    return build


def feed(stream, seconds, clock):
    """Hands over `seconds` of audio, chunk by chunk, as it would arrive."""

    chunk = [0.0] * int(CHUNK * RATE)
    for _ in range(round(seconds / CHUNK)):
        stream.add_audio(chunk, RATE)
        clock.advance(CHUNK)


def test_a_pass_must_cover_at_least_as_much_audio_as_the_last_one_cost(paced):
    # Every pass takes two seconds, which is four intervals' worth of audio.
    stream, lib, clock = paced(2.0)

    feed(stream, 0.5, clock)
    assert lib.passes == 1, "the first pass has only the floor to clear"

    feed(stream, 1.4, clock)
    assert lib.passes == 1, "a second pass should not be made on half a second"

    feed(stream, 0.8, clock)
    assert lib.passes == 2, "and should be made once it has two seconds to cover"

    # Which is four times less often than the floor alone would have asked.
    feed(stream, 20, clock)
    assert 9 <= lib.passes <= 13, (
        f"22.7s of audio at two seconds a pass should be about 11 passes, "
        f"got {lib.passes}"
    )
    assert lib.audio_samples == round(22.7 / CHUNK) * int(CHUNK * RATE), (
        "every sample should still have reached the engine"
    )


def test_a_stream_with_time_to_spare_keeps_to_the_interval(paced):
    # A pass costing a tenth of the interval is a machine with headroom, and
    # nothing about it should change.
    stream, lib, clock = paced(INTERVAL / 10)
    feed(stream, 10, clock)
    wanted = 10 / INTERVAL
    assert wanted - 2 <= lib.passes <= wanted + 1, (
        f"10s of audio should still be about {wanted} passes, got {lib.passes}"
    )


def test_one_freak_pass_does_not_leave_the_transcript_silent_behind_it(paced):
    # A pass that somehow took a minute -- a collection, a laptop lid -- must not
    # hold the next one back for a minute of audio.
    stream, lib, clock = paced(lambda n: 60.0 if n == 1 else 0.05)
    feed(stream, 0.5, clock)
    assert lib.passes == 1, "the freak pass itself"

    feed(stream, INTERVAL * 10 + 0.1, clock)
    assert lib.passes == 2, "the wait should be capped, not a minute long"

    # And with the freak behind it, the floor governs again.
    feed(stream, 2, clock)
    assert lib.passes >= 5, f"should be back to the interval, got {lib.passes}"


def test_stopping_still_flushes_however_long_the_last_pass_took(paced):
    stream, lib, clock = paced(5.0)
    feed(stream, 0.5, clock)
    assert lib.passes == 1

    feed(stream, 0.1, clock)
    assert lib.passes == 1, "a chunk is not five seconds of audio"

    stream.stop()
    assert lib.passes == 2, "stopping should transcribe what was left regardless"
