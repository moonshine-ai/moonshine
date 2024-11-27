from pathlib import Path
from .version import __version__

ASSETS_DIR = Path(__file__).parents[0] / "assets"

from .model import load_model, Moonshine
from .transcribe import (
    transcribe,
    benchmark,
    load_tokenizer,
    load_audio,
)
