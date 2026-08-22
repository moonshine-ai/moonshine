"""Importing the inference package must not pull in download-only dependencies."""

import subprocess
import sys


ISOLATION_SNIPPET = """
import sys
import moonshine_voice
assert "requests" not in sys.modules, "import moonshine_voice loaded requests"
assert "urllib3" not in sys.modules, "import moonshine_voice loaded urllib3"
assert "tqdm" not in sys.modules, "import moonshine_voice loaded tqdm"
assert "filelock" not in sys.modules, "import moonshine_voice loaded filelock"
print("ok")
"""


def test_inference_import_does_not_load_requests():
    """A subprocess so a parent pytest that already imported requests cannot leak."""
    result = subprocess.run(
        [sys.executable, "-c", ISOLATION_SNIPPET],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout
