#!/usr/bin/env python3
"""Report how a compiling execution provider partitions our ``.ort`` models.

CoreML and NNAPI do not run a model op by op. They take contiguous subgraphs of
ops they recognise, compile each into one of their own graphs, and hand the rest
back to the CPU provider. Every boundary between a compiled subgraph and the CPU
costs a synchronisation and, on the Neural Engine, a copy. A model split into a
hundred partitions runs slower than the same model on the CPU alone, so "how
many nodes does the provider support" is the wrong question: what matters is how
few pieces they land in.

Our ``.ort`` files are converted at full optimization (see
``convert-models-to-ort.py``), which fuses whole regions into ``com.microsoft``
ops such as ``FusedConv``, ``MultiHeadAttention`` and ``MatMulNBits``. Those are
ORT's own CPU kernels and no compiling provider recognises them, so they land
between the pieces the provider does recognise and shatter the graph. This
script measures that; see docs/execution-providers.md for the numbers it
produced and what we concluded.

CoreML stands in for NNAPI here. Both take ai.onnx ops only, so a graph that is
fragmented for one is fragmented for the other, and CoreML runs on the machine
you are reading this on.

Usage:
    python scripts/check-ep-partitioning.py <model.ort>...
    python scripts/check-ep-partitioning.py --provider CoreML core/**/*.ort
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# The one line the provider logs from GetCapability, e.g. "number of partitions
# supported by CoreML: 141 number of nodes in the graph: 2549 number of nodes
# supported by CoreML: 891".
SUMMARY = re.compile(
    r"number of partitions supported by \w+: (\d+) "
    r"number of nodes in the graph: (\d+) "
    r"number of nodes supported by \w+: (\d+)"
)


def load_in_child(model: Path, provider: str) -> None:
    """Create one session, letting the provider log its partitioning to stderr."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    # GetCapability logs its summary at WARNING when it takes something and at
    # INFO when it takes nothing, so INFO is the level that catches both.
    opts.log_severity_level = 1
    ort.InferenceSession(str(model), opts, providers=[provider, "CPUExecutionProvider"])


def partitions(model: Path, provider: str) -> tuple[int, int, int] | str:
    """(partitions, nodes, supported nodes), or a message if that is unavailable.

    One child process per model. ORT logs through NSLog on Apple platforms,
    which holds its own handle on the real stderr and ignores any redirection
    done inside the process, so the output has to be caught from outside.
    """
    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--load-one", provider,
         str(model)],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        last = [line for line in done.stderr.strip().splitlines() if line]
        return f"session failed: {last[-1][:120] if last else 'no output'}"

    match = SUMMARY.search(done.stderr)
    if match is None:
        return "no partition summary logged"
    return int(match[1]), int(match[2]), int(match[3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument(
        "--provider",
        default="CoreMLExecutionProvider",
        help="Execution provider to ask (default: CoreMLExecutionProvider)",
    )
    parser.add_argument(
        "--load-one",
        metavar="PROVIDER",
        help=argparse.SUPPRESS,  # Internal: the per-model child described above.
    )
    args = parser.parse_args()

    if args.load_one:
        load_in_child(args.models[0], args.load_one)
        return 0

    import onnxruntime as ort

    if args.provider not in ort.get_available_providers():
        print(
            f"{args.provider} is not in this onnxruntime build "
            f"({', '.join(ort.get_available_providers())})",
            file=sys.stderr,
        )
        return 2

    print(f"{'model':<52} {'nodes':>7} {'taken':>7} {'parts':>7} {'per part':>9}")
    worst = 0.0
    for model in args.models:
        result = partitions(model, args.provider)
        name = model.name if len(model.name) < 52 else model.name[:49] + "..."
        if isinstance(result, str):
            print(f"{name:<52} {result}")
            continue
        parts, nodes, taken = result
        per_part = taken / parts if parts else 0.0
        worst = max(worst, per_part)
        print(f"{name:<52} {nodes:>7} {taken:>7} {parts:>7} {per_part:>9.1f}")

    print(
        "\n'per part' is the average number of nodes in a compiled subgraph. "
        "Single digits\nmean the provider is being handed scraps and will lose "
        "to the CPU on its own."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
