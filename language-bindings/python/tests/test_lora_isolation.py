"""LoRA training must not load PyTorch on the inference import path."""

import subprocess
import sys

import pytest

from moonshine_voice.lora._deps import lora_deps_error


ISOLATION_SNIPPET = """
import sys
import moonshine_voice
assert "torch" not in sys.modules, "import moonshine_voice loaded torch"
assert "transformers" not in sys.modules, "import moonshine_voice loaded transformers"
import moonshine_voice.lora
assert "torch" not in sys.modules, "import moonshine_voice.lora loaded torch"
assert "moonshine_voice.lora.train" not in sys.modules
assert "moonshine_voice.lora.adapter" not in sys.modules
from moonshine_voice.lora.manifest import load_manifest, choose_text_mode
assert "torch" not in sys.modules
print("ok")
"""


def test_inference_import_does_not_load_training_stack():
    """A subprocess so a parent pytest that already imported torch cannot leak."""
    result = subprocess.run(
        [sys.executable, "-c", ISOLATION_SNIPPET],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_lora_deps_error_points_at_the_extra():
    error = lora_deps_error(["torch", "transformers>=5.15"])
    message = str(error)
    assert "moonshine-voice[lora]" in message
    assert "torch" in message
    assert "inference wheel" in message


def test_help_does_not_need_the_extra():
    result = subprocess.run(
        [sys.executable, "-m", "moonshine_voice.lora", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dataset" in result.stdout
    assert "--train-manifest" in result.stdout
