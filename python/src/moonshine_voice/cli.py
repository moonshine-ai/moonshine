"""Console-script entry point for the ``moonshine-voice`` package.

Installed as the ``moonshine-voice`` and ``moonshine`` commands via the
``[project.scripts]`` table in ``pyproject.toml``. This is a thin dispatcher:
each subcommand runs the corresponding module's existing ``__main__`` demo, so
``moonshine-voice mic --language en`` is equivalent to the long-standing
``python -m moonshine_voice.mic_transcriber --language en``. Argument parsing,
help text, and behaviour therefore live entirely in the individual modules and
stay identical between the two invocation styles.
"""

from __future__ import annotations

import runpy
import sys
import warnings
from typing import Dict, List, Optional, Tuple

PROG = "moonshine-voice"

# Ordered map of user-facing subcommand -> (module to run, one-line summary).
COMMANDS: Dict[str, Tuple[str, str]] = {
    "mic": (
        "moonshine_voice.mic_transcriber",
        "Transcribe live microphone input to the terminal.",
    ),
    "transcribe": (
        "moonshine_voice.transcriber",
        "Transcribe a WAV file (with optional speaker IDs / word timestamps).",
    ),
    "tts": (
        "moonshine_voice.tts",
        "Synthesize speech from text to a WAV file or audio device.",
    ),
    "intent": (
        "moonshine_voice.intent_recognizer",
        "Recognize spoken intents from the mic or a WAV file.",
    ),
    "download": (
        "moonshine_voice.download",
        "Download STT, TTS, G2P, or intent model assets.",
    ),
    "g2p": (
        "moonshine_voice.g2p",
        "Convert text to phonemes (IPA) with the G2P engine.",
    ),
}


def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("moonshine-voice")
        except PackageNotFoundError:
            pass
    except ImportError:  # pragma: no cover - Python < 3.8 has no importlib.metadata
        pass
    try:
        from moonshine_voice import __version__

        return __version__
    except Exception:  # pragma: no cover - defensive fallback
        return "unknown"


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        f"usage: {PROG} <command> [options]",
        "",
        "Fast, accurate, on-device AI tools for building voice applications.",
        "",
        "commands:",
    ]
    lines += [
        f"  {name.ljust(width)}  {summary}" for name, (_, summary) in COMMANDS.items()
    ]
    lines += [
        "",
        "options:",
        "  -h, --help     Show this help message and exit.",
        "  -V, --version  Show the moonshine-voice version and exit.",
        "",
        f"Run '{PROG} <command> --help' for command-specific options.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Dispatch to a subcommand. Returns a process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version", "version"):
        print(f"{PROG} {_package_version()}")
        return 0

    command, rest = argv[0], argv[1:]
    entry = COMMANDS.get(command)
    if entry is None:
        print(f"{PROG}: unknown command '{command}'\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    module_name = entry[0]
    # Run the target module exactly as ``python -m <module>`` would. Setting
    # argv[0] to "moonshine-voice <command>" makes each module's argparse
    # ``usage:`` line show the full command prefix (run_module leaves argv[0]
    # untouched, unlike run_path). Modules that the package ``__init__`` already
    # imported (e.g. ``download``) trigger a benign runpy RuntimeWarning about
    # re-running an already-imported module; it does not affect these
    # self-contained demo blocks, so we silence just that message.
    saved_argv = sys.argv
    sys.argv = [f"{PROG} {command}", *rest]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*found in sys\.modules after import of package.*",
                category=RuntimeWarning,
            )
            runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = saved_argv
    return 0


if __name__ == "__main__":
    sys.exit(main())
