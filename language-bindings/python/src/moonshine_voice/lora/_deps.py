"""Optional-extra checks for the LoRA training path.

This module is stdlib-only so ``import moonshine_voice.lora`` stays cheap for
inference installs. The heavy packages (PyTorch, Transformers, …) are listed
here and imported only after ``require_lora_deps()`` succeeds.
"""

from __future__ import annotations

# (import name, pip specifier shown in the error). Keep this in sync with
# ``[project.optional-dependencies] lora`` in pyproject.toml.
LORA_PACKAGES = (
    ("torch", "torch"),
    ("transformers", "transformers>=5.15"),
    ("safetensors", "safetensors"),
    ("soundfile", "soundfile"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("huggingface_hub", "huggingface_hub"),
)

INSTALL_HINT = "pip install 'moonshine-voice[finetune]'"
INSTALL_HINT_ALIASES = "pip install 'moonshine-voice[finetune]'  (or 'moonshine-voice[lora]')"


def missing_lora_packages():
    """Return pip specifiers for extra packages that are not importable."""
    missing = []
    for module, spec in LORA_PACKAGES:
        try:
            __import__(module)
        except ImportError:
            missing.append(spec)
    return missing


def lora_deps_error(missing):
    names = ", ".join(missing)
    return ImportError(
        "Fine-tuning needs extra packages that are not installed with the "
        "default moonshine-voice inference wheel.\n"
        f"  {INSTALL_HINT_ALIASES}\n"
        f"Missing: {names}"
    )


def require_lora_deps():
    """Raise ``ImportError`` with an install hint if the extra is missing."""
    missing = missing_lora_packages()
    if missing:
        raise lora_deps_error(missing)
