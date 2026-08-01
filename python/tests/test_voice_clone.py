"""Tests for VoiceClone, the reference-clip capture behind voice cloning.

The window search runs the voice-activity detector compiled into the library,
so these exercise the real thing: no model files, no downloads, no microphone.
"""

import pytest

import moonshine_voice as mv
from moonshine_voice.moonshine_api import moonshine_extract_speech_clip
from moonshine_voice.voice_clone import VoiceClone


@pytest.fixture(scope="module")
def speech():
    """A recording with several seconds of speech in it."""
    return mv.load_wav_file(str(mv.get_assets_path() / "two_cities.wav"))


def feed(clone, audio, sample_rate, seconds=0.25):
    """Push audio in as a microphone would, stopping once there is a clip."""
    chunk = max(int(sample_rate * seconds), 1)
    for start in range(0, len(audio), chunk):
        clone.add_audio(audio[start:start + chunk], sample_rate)
        if clone.is_ready:
            return
    return


# ---------------------------------------------------------------------------
# The underlying clip search
# ---------------------------------------------------------------------------


def test_a_clip_comes_back_at_16_khz_however_it_was_recorded(speech):
    audio, sample_rate = speech

    clip = moonshine_extract_speech_clip(audio, sample_rate)

    assert clip.is_complete
    assert len(clip.audio) == 4 * VoiceClone.CLIP_SAMPLE_RATE
    assert clip.speech_duration >= 2.0


def test_the_window_length_is_configurable(speech):
    audio, sample_rate = speech

    clip = moonshine_extract_speech_clip(
        audio, sample_rate, clip_duration_seconds=2, minimum_speech_seconds=1
    )

    assert len(clip.audio) == 2 * VoiceClone.CLIP_SAMPLE_RATE


def test_silence_yields_no_clip():
    clip = moonshine_extract_speech_clip([0.0] * (16000 * 6), 16000)

    assert not clip.is_complete
    assert clip.audio is None


# ---------------------------------------------------------------------------
# Incremental capture
# ---------------------------------------------------------------------------


def test_it_becomes_ready_once_it_has_heard_enough(speech):
    audio, sample_rate = speech
    clone = VoiceClone()

    feed(clone, audio, sample_rate)

    assert clone.is_ready
    assert len(clone.audio) == 4 * VoiceClone.CLIP_SAMPLE_RATE
    assert clone.sample_rate == VoiceClone.CLIP_SAMPLE_RATE


def test_it_stays_unready_through_silence():
    clone = VoiceClone()

    feed(clone, [0.0] * (16000 * 6), 16000)

    assert not clone.is_ready
    assert clone.audio is None
    assert clone.recorded_seconds == pytest.approx(6.0)


def test_on_ready_fires_once(speech):
    audio, sample_rate = speech
    fired = []
    clone = VoiceClone().on_ready(lambda: fired.append(True))

    feed(clone, audio, sample_rate)
    # Anything arriving after the clip is found is ignored.
    clone.add_audio(audio[:sample_rate], sample_rate)

    assert fired == [True]


def test_on_ready_fires_immediately_when_it_is_already_ready(speech):
    audio, sample_rate = speech
    clone = VoiceClone()
    feed(clone, audio, sample_rate)
    fired = []

    clone.on_ready(lambda: fired.append(True))

    assert fired == [True]


def test_on_progress_reports_what_it_has_heard(speech):
    audio, sample_rate = speech
    seen = []
    clone = VoiceClone().on_progress(lambda recorded, spoken: seen.append((recorded, spoken)))

    feed(clone, audio, sample_rate)

    assert len(seen) > 1
    recorded, spoken = seen[-1]
    assert recorded == pytest.approx(clone.recorded_seconds, abs=0.3)
    assert spoken == pytest.approx(clone.speech_seconds)
    # Recording runs ahead of the speech found inside it.
    assert spoken <= recorded


def test_the_search_only_runs_a_few_times_a_second(speech):
    """Running the detector on every buffer would be wasteful."""
    audio, sample_rate = speech
    searches = []
    clone = VoiceClone().on_progress(lambda recorded, spoken: searches.append(recorded))

    chunk = sample_rate // 100  # 10 ms buffers, as a mic would deliver
    for start in range(0, sample_rate * 2, chunk):
        clone.add_audio(audio[start:start + chunk], sample_rate)

    assert 4 <= len(searches) <= 12


def test_a_change_of_sample_rate_starts_the_recording_over(speech):
    audio, sample_rate = speech
    clone = VoiceClone()

    clone.add_audio(audio[:sample_rate], sample_rate)
    clone.add_audio([0.0] * 8000, 8000)

    assert clone.recorded_seconds == pytest.approx(1.0)


def test_reset_throws_away_what_it_captured(speech):
    audio, sample_rate = speech
    clone = VoiceClone()
    feed(clone, audio, sample_rate)

    clone.reset()

    assert not clone.is_ready
    assert clone.audio is None
    assert clone.recorded_seconds == 0.0
    assert clone.speech_seconds == 0.0


def test_empty_audio_is_ignored():
    clone = VoiceClone()

    clone.add_audio([], 16000)
    clone.add_audio([0.1, 0.2], 0)

    assert clone.recorded_seconds == 0.0


def test_a_shorter_window_needs_less_speech(speech):
    """The thresholds start_cloning() passes through actually take effect."""
    audio, sample_rate = speech
    clone = VoiceClone(clip_duration_seconds=2, minimum_speech_seconds=1)

    feed(clone, audio, sample_rate)

    assert len(clone.audio) == 2 * VoiceClone.CLIP_SAMPLE_RATE
