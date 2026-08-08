"""Tests the ``moonshine-voice`` console-script entry point.

The pip package installs a ``moonshine-voice`` command (and a ``moonshine``
alias) via ``[project.scripts]`` in ``pyproject.toml``. These tests exercise the
installed script directly so that a broken entry point, a renamed subcommand, or
a stale help table is caught before a release is published. They only touch the
dispatcher (help / version / argument parsing) and never load models, so they
run in well under a second.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from moonshine_voice.cli import COMMANDS

ARGPARSE_USAGE_ERROR_EXIT_CODE = 2
TIMEOUT_SECONDS = 60


def console_script(name="moonshine-voice"):
    """Locate the installed console script, falling back to ``python -m``.

    Console scripts are installed next to the interpreter running the tests, so
    we look there first (this is the wheel-under-test in CI). If it is missing
    we fall back to running the module, which still exercises the dispatcher.
    """
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return [str(candidate)]
    found = shutil.which(name)
    if found:
        return [found]
    return [sys.executable, "-m", "moonshine_voice.cli"]


def run(*args, name="moonshine-voice"):
    return subprocess.run(
        console_script(name) + list(args),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )


def describe(result):
    return (
        f"exit code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


@pytest.mark.parametrize("name", ["moonshine-voice", "moonshine"])
def test_console_scripts_are_installed(name):
    """Both the primary command and the short alias must be on PATH."""
    assert (
        Path(sys.executable).parent / name
    ).exists() or shutil.which(name), f"{name} console script is not installed"


def test_help_lists_every_command():
    result = run("--help")
    assert result.returncode == 0, describe(result)
    for command in COMMANDS:
        assert command in result.stdout, describe(result)


def test_version_reports_package_name():
    result = run("--version")
    assert result.returncode == 0, describe(result)
    assert "moonshine-voice" in result.stdout, describe(result)


def test_unknown_command_is_a_usage_error():
    result = run("not-a-real-command")
    assert result.returncode == ARGPARSE_USAGE_ERROR_EXIT_CODE, describe(result)


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_subcommand_help_parses(command):
    """``<command> --help`` must succeed and show the friendly command prefix.

    This routes through the module's own argparse, so it also proves the
    dispatcher hands arguments off correctly and that the module's flags parse.
    """
    result = run(command, "--help")
    assert result.returncode == 0, describe(result)
    assert f"moonshine-voice {command}" in result.stdout, describe(result)
