"""Runs the ``__main__`` sections of the most significant modules.

Each module in moonshine_voice doubles as a command-line demo through its
``if __name__ == "__main__"`` block, so invoking ``python -m`` on it
exercises the module end to end: model loading, the native library, and the
public API. These tests run the headless-safe ones for real and check their
output, and smoke-test the microphone-only ones by starting them and failing
only on an argument parsing error (the same trick as test_docs.py).

The suite is meant to stay under roughly three minutes with a warm model
cache, so it uses the short bundled audio clips and the quantized embedding
model.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FULL_RUN_TIMEOUT_SECONDS = 300
SMOKE_TIMEOUT_SECONDS = 10

ARGPARSE_USAGE_ERROR_EXIT_CODE = 2


def run_module(module, *args, cwd=None, timeout=FULL_RUN_TIMEOUT_SECONDS):
    return subprocess.run(
        [sys.executable, "-m", f"moonshine_voice.{module}", *args],
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


def describe(result):
    return (
        f"exit code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def assets_path():
    moonshine_voice = pytest.importorskip("moonshine_voice")
    return Path(moonshine_voice.get_assets_path())


def test_transcriber_transcribes_bundled_audio():
    wav_path = assets_path() / "beckett.wav"
    result = run_module("transcriber", "--wav-path", str(wav_path), "--quiet")
    assert result.returncode == 0, describe(result)
    # The clip says "Ever tried, ever failed. No matter. Try again.
    # Fail again. Fail better."
    combined = (result.stdout + result.stderr).lower()
    assert "fail" in combined, describe(result)


def test_tts_synthesizes_wav(tmp_path):
    out_path = tmp_path / "out.wav"
    result = run_module(
        "tts",
        "--language", "en_us",
        "--text", "Hello world",
        "--out", str(out_path),
        cwd=tmp_path,
    )
    assert result.returncode == 0, describe(result)
    assert out_path.exists(), describe(result)
    # A WAV header alone is 44 bytes; real speech is far larger.
    assert out_path.stat().st_size > 10000, describe(result)


def test_g2p_prints_ipa():
    result = run_module("g2p", "--language", "en_us", "--text", "Hello world")
    assert result.returncode == 0, describe(result)
    assert result.stdout.strip(), describe(result)


def test_intent_recognizer_triggers_intents():
    wav_path = REPO_ROOT / "test-assets" / "intent.wav"
    result = run_module(
        "intent_recognizer",
        "--wav-file", str(wav_path),
        "--quantization", "q4",
        "--intents", "move forward,move backward,turn left,turn right",
    )
    assert result.returncode == 0, describe(result)
    combined = result.stdout + result.stderr
    assert "triggered" in combined.lower(), describe(result)


def test_download_g2p_assets():
    result = run_module("download", "--g2p", "--language", "en_us")
    assert result.returncode == 0, describe(result)
    root = result.stdout.strip().splitlines()[-1]
    assert Path(root).exists(), describe(result)


def test_dialog_flow_lists_output_devices():
    result = run_module("dialog_flow", "--list-output-devices")
    assert result.returncode == 0, describe(result)


@pytest.mark.parametrize(
    "module,args",
    [
        ("mic_transcriber", ["--language", "en"]),
        ("alphanumeric_listener", ["--language", "en"]),
    ],
)
def test_mic_module_arguments_parse(module, args):
    """Microphone modules can't run headlessly, so just start them and make
    sure the documented arguments are accepted (argparse failures exit 2)."""
    try:
        result = run_module(module, *args, timeout=SMOKE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # Still running after the timeout means the arguments parsed.
        return
    assert result.returncode != ARGPARSE_USAGE_ERROR_EXIT_CODE, describe(result)
