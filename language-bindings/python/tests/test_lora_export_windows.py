"""Sliding-window conventions in the exporter. Skipped without the ``[lora]`` extra."""

import pytest

pytest.importorskip("torch")

from moonshine_voice.lora.export import inclusive_windows  # noqa: E402

# How the English checkpoints store the windows: inclusive frame counts.
ENGLISH_TINY = [(16, 4), (16, 4), (16, 0), (16, 0), (16, 4), (16, 4)]
# The same layers in the half-open form the later checkpoints use.
HALF_OPEN_TINY = [(17, 5), (17, 5), (17, 1), (17, 1), (17, 5), (17, 5)]
HALF_OPEN_SMALL = [(17, 5), (17, 5)] + [(17, 1)] * 6 + [(17, 5), (17, 5)]


def test_inclusive_form_is_left_alone():
    assert inclusive_windows(ENGLISH_TINY) == ENGLISH_TINY


def test_half_open_form_is_converted():
    assert inclusive_windows(HALF_OPEN_TINY) == ENGLISH_TINY
    assert inclusive_windows(HALF_OPEN_SMALL) == (
        [(16, 4), (16, 4)] + [(16, 0)] * 6 + [(16, 4), (16, 4)]
    )


def test_both_forms_agree_on_lookahead_and_left_context():
    # The runtime feeds total_lookahead frames past the emitted ones and the
    # cached encoder keeps `past` frames per layer; both must come out the same
    # whichever way the checkpoint wrote its config.
    for windows in (HALF_OPEN_TINY, ENGLISH_TINY):
        normalized = inclusive_windows(windows)
        assert sum(future for _, future in normalized) == 16
        assert {past for past, _ in normalized} == {16}
