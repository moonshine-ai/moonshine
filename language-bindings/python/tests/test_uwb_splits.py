"""UWB-ATCC session parsing and the hold-out of the shared test session.

No Hub download: the split rule is a function of utterance ids.
"""

import pytest

pyarrow = pytest.importorskip("pyarrow")

from moonshine_voice.lora.data import (  # noqa: E402
    UWB_SHARED_SESSION,
    session_disjoint_train,
    uwb_session,
)


def test_uwb_session_from_id():
    assert uwb_session("uwb-atcc_TWR-34720N_0001") == "TWR-34720N"
    assert uwb_session("uwb-atcc_APP-1_x") == "APP-1"
    assert uwb_session("no-underscore") == "no-underscore"


def test_shared_session_is_held_out_of_train():
    scored = {UWB_SHARED_SESSION, "TWR-other"}
    assert not session_disjoint_train(UWB_SHARED_SESSION, scored)
    assert session_disjoint_train("APP-99", scored)
    assert session_disjoint_train("TWR-1", scored)


def test_help_lists_uwb_and_adapt_flags():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "moonshine_voice.lora", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for flag in ("uwb_atcc", "--sites", "--adapt", "--eval-dataset"):
        assert flag in result.stdout
