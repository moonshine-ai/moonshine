"""Tests that the code blocks in the documentation actually work.

Extracts fenced code blocks from the markdown files listed in DOC_FILES and
checks each one, so that stale commands (for example a CLI flag that has been
renamed) are caught before a release is published.

How a block is treated is controlled by an HTML comment placed on the line
immediately before the opening fence (invisible in rendered markdown):

    <!-- doc-test: skip -->         Not tested at all.
    <!-- doc-test: parse-only -->   The command is started with a short
                                    timeout. The test fails only if it exits
                                    with code 2 (an argparse usage error), so
                                    stale flags are caught even for commands
                                    that need a microphone or run forever.
    <!-- doc-test: expect-error --> The command must exit with a nonzero code.
    <!-- doc-test: run -->          Full execution (opt-in for python blocks).

Unannotated ``bash`` blocks are executed in full and must succeed.
Unannotated ``python`` blocks are only checked for valid syntax, since many
are illustrative fragments that reference variables defined in prose.
"""

import dataclasses
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DOC_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "python" / "README.md",
]

ANNOTATION_PREFIX = "<!-- doc-test:"

# Modes a block can be tested in.
SKIP = "skip"
PARSE_ONLY = "parse-only"
EXPECT_ERROR = "expect-error"
RUN = "run"
SYNTAX = "syntax"

VALID_ANNOTATIONS = {SKIP, PARSE_ONLY, EXPECT_ERROR, RUN}

# Generous, since some commands download models on a cold cache.
FULL_RUN_TIMEOUT_SECONDS = 900
# Just long enough to get past argument parsing.
PARSE_ONLY_TIMEOUT_SECONDS = 15

ARGPARSE_USAGE_ERROR_EXIT_CODE = 2


@dataclasses.dataclass
class DocBlock:
    path: Path
    start_line: int
    language: str
    code: str
    mode: str

    @property
    def test_id(self):
        rel_path = self.path.relative_to(REPO_ROOT)
        return f"{rel_path}:{self.start_line}-{self.language}-{self.mode}"


def parse_annotation(line):
    """Returns the doc-test mode from an annotation comment, or None."""
    stripped = line.strip()
    if not stripped.startswith(ANNOTATION_PREFIX):
        return None
    value = stripped[len(ANNOTATION_PREFIX):].removesuffix("-->").strip()
    if value not in VALID_ANNOTATIONS:
        raise ValueError(f"Unknown doc-test annotation: {line!r}")
    return value


def extract_blocks(path):
    """Yields a DocBlock for every fenced code block in a markdown file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    pending_annotation = None
    in_block = False
    language = ""
    start_line = 0
    block_lines = []
    block_annotation = None
    for line_number, line in enumerate(lines, start=1):
        if not in_block:
            annotation = parse_annotation(line)
            if annotation is not None:
                pending_annotation = annotation
                continue
            if line.startswith("```"):
                in_block = True
                language = line[3:].strip()
                start_line = line_number
                block_lines = []
                block_annotation = pending_annotation
            pending_annotation = None
        else:
            if line.startswith("```"):
                in_block = False
                yield DocBlock(
                    path=path,
                    start_line=start_line,
                    language=language,
                    code="\n".join(block_lines) + "\n",
                    mode=resolve_mode(language, block_annotation),
                )
            else:
                block_lines.append(line)


def resolve_mode(language, annotation):
    if annotation is not None:
        return annotation
    if language == "bash":
        return RUN
    if language == "python":
        return SYNTAX
    return SKIP


def collect_all_blocks():
    blocks = []
    for path in DOC_FILES:
        blocks.extend(extract_blocks(path))
    return [block for block in blocks if block.mode != SKIP]


def run_bash_block(block, cwd, timeout):
    """Runs a bash block as a script, returning the CompletedProcess.

    Raises subprocess.TimeoutExpired if the timeout is hit; callers decide
    whether that counts as a failure.
    """
    # Make sure `python`/`python3` in the block resolve to the interpreter
    # running the tests, even when its venv hasn't been activated.
    env = os.environ.copy()
    interpreter_dir = str(Path(sys.executable).parent)
    env["PATH"] = interpreter_dir + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        ["bash", "-e", "-c", block.code],
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        env=env,
    )


def check_python_syntax(block):
    code = textwrap.dedent(block.code)
    try:
        compile(code, str(block.path), "exec")
        return
    except SyntaxError:
        pass
    # Many fragments use yield/return at the top level because they are
    # excerpts from a function body; retry with the code wrapped in one.
    wrapped = "def _doc_test_wrapper():\n" + textwrap.indent(code, "    ")
    compile(wrapped, str(block.path), "exec")


def describe(result):
    return (
        f"exit code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


ALL_BLOCKS = collect_all_blocks()


@pytest.mark.parametrize(
    "block", ALL_BLOCKS, ids=[block.test_id for block in ALL_BLOCKS]
)
def test_doc_block(block, tmp_path):
    if block.mode == SYNTAX:
        check_python_syntax(block)
    elif block.mode == RUN and block.language == "python":
        script = tmp_path / "doc_block.py"
        script.write_text(textwrap.dedent(block.code), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            timeout=FULL_RUN_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, describe(result)
    elif block.mode == RUN:
        result = run_bash_block(block, tmp_path, FULL_RUN_TIMEOUT_SECONDS)
        assert result.returncode == 0, describe(result)
    elif block.mode == EXPECT_ERROR:
        result = run_bash_block(block, tmp_path, FULL_RUN_TIMEOUT_SECONDS)
        assert result.returncode != 0, (
            "Expected a nonzero exit code but the command succeeded.\n"
            + describe(result)
        )
    elif block.mode == PARSE_ONLY:
        try:
            result = run_bash_block(block, tmp_path, PARSE_ONLY_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # Still running after the timeout means the arguments parsed.
            return
        assert result.returncode != ARGPARSE_USAGE_ERROR_EXIT_CODE, (
            "Command failed with an argument parsing (usage) error, so the "
            "documented flags are probably stale.\n" + describe(result)
        )
    else:
        raise AssertionError(f"Unhandled doc-test mode: {block.mode}")
