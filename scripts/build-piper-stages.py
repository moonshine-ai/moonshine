#!/usr/bin/env python3
"""Rebuilds every shipped Piper voice as the two stages streaming needs.

Each voice currently ships as one model, ``<stem>.model.ort`` plus its
``<stem>.weights.ort``. This replaces that with four files: the same graph cut
in two, each half in the same split-weights form. The four come to the same
total size, because they hold the same weights, and the pair renders a whole
utterance about as fast as the single model did, so nothing is given up by
carrying only the stages. What is gained is that the generator can be asked for
a range of frames, which is what lets a reply start playing before it has been
synthesized. See ``scripts/split-piper-stages.py`` for the cut itself.

The source is the quantized ``.onnx`` on the CDN, which is what the shipped ORT
files were converted from. Splitting that, rather than requantizing the float32
original, is what makes the stages numerically the voice that ships today; the
verification step checks it rather than assuming it.

    scripts/build-piper-stages.py --cache-dir /tmp/piper-onnx
    scripts/build-piper-stages.py --voices en_US-amy-medium --keep-monolith

Nothing is written into the data tree until a voice has passed verification, so
an interrupted run leaves a tree that still works.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

_splitter = import_module("split-piper-stages")
split_voice = _splitter.split_voice
phoneme_ids = _splitter.phoneme_ids
VERIFY_SCALES = _splitter.VERIFY_SCALES

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "core" / "moonshine-tts" / "data"
CDN = "https://download.moonshine.ai/tts"
STAGES = ("upstream", "generator")


def voices_in_tree() -> list[tuple[str, Path]]:
    """Every installed voice, as (stem, its piper-voices directory)."""
    found = []
    for config in sorted(DATA_ROOT.glob("*/piper-voices/*.onnx.json")):
        found.append((config.name[: -len(".onnx.json")], config.parent))
    return found


def download(stem: str, directory: Path, cache: Path) -> Path:
    """The quantized ``.onnx`` this voice's shipped ORT was converted from."""
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{stem}.onnx"
    if local.is_file() and local.stat().st_size > 0:
        return local
    language = directory.parent.name
    # A couple of voices have a non-ASCII character in their name.
    url = (f"{CDN}/{language}/piper-voices/"
           f"{urllib.parse.quote(stem)}.onnx")
    partial = local.with_suffix(".onnx.part")
    # The CDN turns away the default urllib agent.
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    partial.rename(local)
    return local


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options,
                                providers=["CPUExecutionProvider"])


def weights_of(stem_path: Path) -> dict:
    """The tensors a split-weights model expects to be handed each call."""
    # Not with_suffix: a stage's name already has a dot in it, and that would
    # take ".upstream" for an extension and replace it.
    if Path(f"{stem_path}.ort").is_file():
        return {}
    model = session(Path(f"{stem_path}.weights.ort"))
    names = [output.name for output in model.get_outputs()]
    return dict(zip(names, model.run(names, {})))


def graph_of(stem_path: Path) -> ort.InferenceSession:
    single = Path(f"{stem_path}.ort")
    if single.is_file():
        return session(single)
    return session(Path(f"{stem_path}.model.ort"))


def render(graph: ort.InferenceSession, weights: dict,
           feed: dict) -> np.ndarray:
    return graph.run(["output"], {**feed, **weights})[0].reshape(-1)


def text_inputs(onnx_path: Path) -> dict:
    ids = phoneme_ids(onnx_path)
    feed = {
        "input": ids,
        "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
        "scales": np.array(VERIFY_SCALES, dtype=np.float32),
    }
    config = json.loads(onnx_path.with_suffix(".onnx.json").read_text())
    if int(config.get("num_speakers", 1)) > 1:
        feed["sid"] = np.array([0], dtype=np.int64)
    return feed


def check_against_shipped(onnx_path: Path, staged: Path, shipped: Path,
                          feed: dict) -> tuple[float, int]:
    """Compares the new stages with the voice as it ships today.

    Returns the largest sample difference and the length difference. Both being
    zero is what says a listener will not hear this change.
    """
    reference = render(graph_of(shipped), weights_of(shipped), feed)

    body = graph_of(Path(f"{staged}.upstream"))
    generator = graph_of(Path(f"{staged}.generator"))
    generator_weights = weights_of(Path(f"{staged}.generator"))
    seam = [value.name for value in generator.get_inputs()
            if value.name not in generator_weights]
    crossed = dict(zip(seam, body.run(
        seam, {**feed, **weights_of(Path(f"{staged}.upstream"))})))
    joined = render(generator, generator_weights, crossed)

    count = min(joined.size, reference.size)
    if count == 0:
        return float("inf"), joined.size - reference.size
    difference = float(np.max(np.abs(joined[:count] - reference[:count])))
    return difference, int(joined.size - reference.size)


def convert_to_ort(directory: Path) -> None:
    # The converter's growth budget is meant to stop a conversion bloating a
    # download, and it judges one file at a time. A stage is a fraction of the
    # voice but carries a whole graph's worth of structure, so it grows by more
    # in percentage terms while the voice as a whole does not grow at all. The
    # budget that matters is the four files against the two they replace, which
    # is checked after the conversion instead.
    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "convert-models-to-ort.py"),
         str(directory), "--max-growth", "25"],
        capture_output=True, text=True)
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()
        raise RuntimeError("convert-models-to-ort: " +
                           " / ".join(tail[-3:] or ["no output"]))


def installed_names(stem: str, work: Path) -> list[Path]:
    """The ORT files the conversion produced, in either of its two forms."""
    produced = []
    for stage in STAGES:
        single = work / f"{stem}.{stage}.ort"
        if single.is_file():
            produced.append(single)
            continue
        for part in ("model", "weights"):
            produced.append(work / f"{stem}.{stage}.{part}.ort")
    return produced


MONOLITH_SUFFIXES = (".ort", ".model.ort", ".weights.ort")

# What the four stage files may add over the two they replace. They hold the
# same weights, so the difference is graph structure counted twice; anything
# beyond this means the cut landed somewhere that duplicates real work.
SIZE_TOLERANCE = 0.05


def monolith_bytes(directory: Path, stem: str) -> int:
    return sum((directory / f"{stem}{suffix}").stat().st_size
               for suffix in MONOLITH_SUFFIXES
               if (directory / f"{stem}{suffix}").is_file())


def already_built(directory: Path, stem: str) -> bool:
    """Whether this voice has its stages and nothing left to replace."""
    if monolith_bytes(directory, stem) > 0:
        return False
    return any((directory / f"{stem}.{stage}.model.ort").is_file() or
               (directory / f"{stem}.{stage}.ort").is_file()
               for stage in STAGES)


def build(stem: str, directory: Path, cache: Path,
          keep_monolith: bool) -> tuple[bool, str]:
    if already_built(directory, stem):
        return True, "already built"
    before = monolith_bytes(directory, stem)
    onnx_path = download(stem, directory, cache)
    # The config sits beside the model for phoneme_ids and speaker count.
    config = cache / f"{stem}.onnx.json"
    if not config.is_file():
        config.write_bytes((directory / f"{stem}.onnx.json").read_bytes())

    with tempfile.TemporaryDirectory(prefix=f"piper-{stem}-") as scratch:
        work = Path(scratch)
        try:
            split_voice(onnx_path, work)
        except SystemExit as failure:
            return False, f"split failed: {failure}"
        convert_to_ort(work)
        produced = installed_names(stem, work)
        missing = [path.name for path in produced if not path.is_file()]
        if missing:
            return False, f"conversion produced no {', '.join(missing)}"

        feed = text_inputs(onnx_path)
        difference, length = check_against_shipped(
            onnx_path, work / stem, directory / stem, feed)
        if length != 0:
            return False, f"length differs by {length} samples"
        if difference != 0.0:
            return False, f"samples differ by up to {difference:.3e}"

        after = sum(path.stat().st_size for path in produced)
        if before > 0 and after > before * (1.0 + SIZE_TOLERANCE):
            return False, (f"stages are {after / 1e6:.1f} MB against "
                           f"{before / 1e6:.1f} MB shipped")

        for path in produced:
            (directory / path.name).write_bytes(path.read_bytes())

    if not keep_monolith:
        for suffix in MONOLITH_SUFFIXES:
            stale = directory / f"{stem}{suffix}"
            if stale.is_file():
                stale.unlink()
    growth = f", {(after / before - 1) * 100:+.1f}%" if before > 0 else ""
    return True, f"{after / 1e6:.1f} MB in {len(produced)} files{growth}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/piper-onnx"),
                        help="where downloaded .onnx voices are kept")
    parser.add_argument("--voices", nargs="*",
                        help="stems to build, default every installed voice")
    parser.add_argument("--keep-monolith", action="store_true",
                        help="leave the single-model files in place")
    args = parser.parse_args()

    wanted = set(args.voices or [])
    voices = [(stem, directory) for stem, directory in voices_in_tree()
              if not wanted or stem in wanted]
    if not voices:
        print("No voices to build.", file=sys.stderr)
        return 1

    failures = []
    for index, (stem, directory) in enumerate(voices, start=1):
        print(f"[{index}/{len(voices)}] {stem}", end=" ", flush=True)
        try:
            ok, detail = build(stem, directory, args.cache_dir,
                               args.keep_monolith)
        except (urllib.error.URLError, subprocess.CalledProcessError,
                Exception) as failure:  # noqa: BLE001 - report and carry on
            ok, detail = False, f"{type(failure).__name__}: {failure}"
        print(detail if ok else f"FAILED: {detail}", flush=True)
        if not ok:
            failures.append(stem)

    print(f"\n{len(voices) - len(failures)}/{len(voices)} voices built")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
