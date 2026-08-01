#!/usr/bin/env python3
"""Convert ``.onnx`` models to ORT format (``.ort``) beside the originals.

ORT format is the only format Moonshine loads, on any platform; see
``docs/ort-only-models.md``. This is how a model gets into it.

An ORT-format model has its graph optimizations baked in at conversion time,
and ORT will not re-apply them at load time. That makes the optimization level
used here a permanent property of the file, and it splits our models into two
groups:

* Models with float weights (Kokoro, the English OOV model) convert at full
  optimization for about the same size as the ``.onnx``, and load faster
  because there is no protobuf parse. Straight win.
* Models that store int8 weights and cast them to float at inference (every
  Piper voice, the Chinese and Arabic G2P models) get those weights folded to
  float32 during optimization, so the ``.ort`` lands roughly 4x larger.
  Converting them without optimization keeps the size but costs 3-5x on
  inference, since the folding is exactly what makes them fast.

A model that fits ``--max-growth`` at full optimization is simply converted.
For the rest we try the split form instead (see
``scripts/split-model-weights.py``), which keeps the fusions and the original
size by moving the weights into a second ORT model that runs once at startup.
That trade only works when nothing pre-packs the weights: ORT rearranges a
constant ``MatMul`` operand into a blocked layout at load time and cannot do so
for a graph input, which costs about 2.2x on inference for the transformer G2P
models. ``Conv`` has no such step, so the Piper voices split for free. We pick
between them by looking at which ops consume the dequantized weights, and any
model that would lose is left as ``.onnx``.

Full optimization is also what rules out CoreML and NNAPI, since it fuses whole
regions into ``com.microsoft`` operators that no compiling execution provider
recognises. That was measured and accepted rather than overlooked; see
``docs/execution-providers.md``.

Usage:
    python scripts/convert-models-to-ort.py core/moonshine-tts/data
    python scripts/convert-models-to-ort.py --dry-run <path>...
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

CONVERTED = "converted"
SPLIT = "split"
TOO_LARGE = "too-large"
FAILED = "failed"

SPLITTER = Path(__file__).with_name("split-model-weights.py")

#: Ops that rearrange a constant operand into a kernel-specific layout at load
#: time, which they cannot do once the operand is a graph input.
PREPACKING_OPS = frozenset({"MatMul", "FusedMatMul", "Gemm", "Attention"})

#: Share of dequantized weight bytes reaching a pre-packing op above which
#: splitting costs more on inference than it saves at startup.
PREPACK_SHARE_LIMIT = 0.10


def _levels():
    import onnxruntime as ort

    return {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }


def _splitter():
    spec = importlib.util.spec_from_file_location("split_model_weights", SPLITTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepacking_share(onnx: Path) -> float:
    """Fraction of dequantized weight bytes that feed a pre-packing op.

    Reads the shrink-ray dequantize chains straight off the ``.onnx``; the op mix
    is a property of the architecture and survives graph optimization, so there
    is no need to build the optimized graph just to classify a model.
    """
    import onnx as onnx_mod
    from onnx import numpy_helper

    splitter = _splitter()
    model = onnx_mod.load(str(onnx), load_external_data=False)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    chain_bytes: dict[str, int] = {}
    for chain in splitter.find_dequantize_chains(graph):
        base = chain["output"][: -len(splitter.DEQUANT_OUTPUT_SUFFIX)]
        init = inits.get(base + splitter.QUANT_SUFFIX)
        if init is not None:
            chain_bytes[chain["output"]] = numpy_helper.to_array(init).size * 4
    total = sum(chain_bytes.values())
    if not total:
        return 0.0
    prepacked = sum(
        chain_bytes[inp]
        for node in graph.node
        if node.op_type in PREPACKING_OPS
        for inp in node.input
        if inp in chain_bytes
    )
    return prepacked / total


@dataclass
class Result:
    onnx: Path
    status: str
    level: str = ""
    onnx_size: int = 0
    ort_size: int = 0
    detail: str = ""

    @property
    def growth(self) -> float:
        if not self.onnx_size:
            return 0.0
        return 100.0 * (self.ort_size - self.onnx_size) / self.onnx_size


def _convert_at_level(src: Path, dst: Path, level: str) -> None:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = _levels()[level]
    opts.optimized_model_filepath = str(dst)
    opts.add_session_config_entry("session.save_model_format", "ORT")
    opts.log_severity_level = 3
    ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"])


def _split_one(onnx: Path, onnx_size: int, budget: float) -> Result:
    """Emit the split pair for a model whose optimized ``.ort`` busts the budget."""
    share = prepacking_share(onnx)
    if share > PREPACK_SHARE_LIMIT:
        return Result(
            onnx,
            TOO_LARGE,
            level="all",
            onnx_size=onnx_size,
            detail=f"{share:.0%} of weight bytes feed a pre-packing op",
        )
    _, model_size, weights_size = _splitter().split_model(onnx, force=True)
    size = model_size + weights_size
    if size > budget:
        for stale in (onnx.with_suffix(".model.ort"), onnx.with_suffix(".weights.ort")):
            stale.unlink(missing_ok=True)
        return Result(
            onnx,
            TOO_LARGE,
            level="split",
            onnx_size=onnx_size,
            ort_size=size,
            detail="split model exceeds the size budget",
        )
    return Result(onnx, SPLIT, level="split", onnx_size=onnx_size, ort_size=size)


def convert_one(args: tuple[str, float, bool]) -> Result:
    path_str, max_growth, allow_split = args
    onnx = Path(path_str)
    target = onnx.with_suffix(".ort")
    onnx_size = onnx.stat().st_size
    budget = onnx_size * (1.0 + max_growth / 100.0)

    with tempfile.TemporaryDirectory(dir=str(onnx.parent)) as tmpdir:
        candidate = Path(tmpdir) / f"{onnx.stem}.ort"
        try:
            _convert_at_level(onnx, candidate, "all")
        except Exception as exc:  # noqa: BLE001 - report rather than abort the batch
            return Result(onnx, FAILED, onnx_size=onnx_size, detail=str(exc)[:200])
        size = candidate.stat().st_size

        if size > budget:
            if not allow_split:
                return Result(
                    onnx,
                    TOO_LARGE,
                    level="all",
                    onnx_size=onnx_size,
                    ort_size=size,
                    detail="optimized model exceeds the size budget",
                )
            try:
                return _split_one(onnx, onnx_size, budget)
            except Exception as exc:  # noqa: BLE001
                return Result(onnx, FAILED, onnx_size=onnx_size, detail=str(exc)[:200])

        os.replace(candidate, target)
        return Result(onnx, CONVERTED, level="all", onnx_size=onnx_size, ort_size=size)


def _already_done(onnx: Path) -> bool:
    if onnx.with_suffix(".ort").exists():
        return True
    return (
        onnx.with_suffix(".model.ort").exists()
        and onnx.with_suffix(".weights.ort").exists()
    )


def find_models(roots: list[str], force: bool) -> tuple[list[Path], list[Path]]:
    todo: list[Path] = []
    have_ort: list[Path] = []
    for root in roots:
        root_path = Path(root)
        candidates = (
            [root_path]
            if root_path.is_file()
            else sorted(p for p in root_path.rglob("*.onnx"))
        )
        for onnx in candidates:
            if _already_done(onnx) and not force:
                have_ort.append(onnx)
            else:
                todo.append(onnx)
    return todo, have_ort


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Files or directories to scan")
    parser.add_argument(
        "--max-growth",
        type=float,
        default=8.0,
        help="Largest acceptable %% size increase over the .onnx (default: 8, "
        "which is what the smallest Piper voices need: their graphs carry a "
        "fixed amount of float weight that shrink-ray leaves alone, so the "
        "split overhead is proportionally largest there)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip models whose optimized .ort busts the budget rather than "
        "emitting the split .model.ort/.weights.ort pair for them",
    )
    parser.add_argument(
        "--force", action="store_true", help="Reconvert even if a .ort exists"
    )
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would be converted"
    )
    args = parser.parse_args()

    todo, have_ort = find_models(args.roots, args.force)
    print(
        f"Found {len(todo) + len(have_ort)} .onnx models: {len(todo)} to convert, "
        f"{len(have_ort)} already have a .ort"
    )
    if args.dry_run:
        for onnx in todo:
            print(f"  would convert {onnx}")
        return 0
    if not todo:
        return 0

    started = time.time()
    payload = [(str(p), args.max_growth, not args.no_split) for p in todo]
    results: list[Result] = []
    with multiprocessing.Pool(args.jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(convert_one, payload), 1):
            prefix = f"[{i}/{len(todo)}]"
            if result.status == FAILED:
                print(f"{prefix} FAILED {result.onnx}: {result.detail}")
            elif result.status == TOO_LARGE:
                print(f"{prefix} left as .onnx {result.onnx.name} ({result.detail})")
            elif result.status == SPLIT:
                print(f"{prefix} {result.onnx.name} split {result.growth:+.1f}%")
            else:
                print(f"{prefix} {result.onnx.name} "
                      f"level={result.level} {result.growth:+.1f}%")
            results.append(result)

    converted = [r for r in results if r.status in (CONVERTED, SPLIT)]
    too_large = [r for r in results if r.status == TOO_LARGE]
    failed = [r for r in results if r.status == FAILED]
    onnx_total = sum(r.onnx_size for r in converted)
    ort_total = sum(r.ort_size for r in converted)

    print(f"\nConverted {len(converted)}/{len(todo)} in {time.time() - started:.0f}s")
    if converted:
        by_level: dict[str, int] = {}
        for r in converted:
            by_level[r.level] = by_level.get(r.level, 0) + 1
        print(f"  .onnx total: {human(onnx_total)}")
        print(f"  .ort  total: {human(ort_total)} "
              f"({100.0 * (ort_total - onnx_total) / max(onnx_total, 1):+.1f}%)")
        print(f"  forms used: {by_level}")
    if too_large:
        print(f"  {len(too_large)} left as .onnx:")
        for r in too_large:
            print(f"    {r.onnx}: {r.detail}")
    for r in failed:
        print(f"  FAILED {r.onnx}: {r.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
