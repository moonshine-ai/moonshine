"""CLI for ``python -m moonshine_voice.lora`` and ``moonshine-voice lora``.

Argparse lives here so ``--help`` does not import PyTorch. Training imports
happen only after the extra is confirmed present.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    if prog is None:
        argv0 = sys.argv[0] if sys.argv else ""
        prog = argv0 if argv0.startswith("moonshine-voice") else "moonshine-voice lora"
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Train a LoRA adapter or full fine-tune for Moonshine Streaming. "
            "Requires pip install 'moonshine-voice[finetune]' "
            "(or the equivalent 'moonshine-voice[lora]')."
        )
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset",
        choices=["atcosim", "uwb_atcc"],
        default=None,
        help="built-in corpus. atcosim is speaker-disjoint headset ATC "
        "(phraseology). uwb_atcc is session-disjoint real VHF "
        "(CC BY-NC-SA 4.0, research only)",
    )
    data.add_argument(
        "--train-manifest",
        default=None,
        help="JSONL / JSON / TSV of {audio, text} rows for your own data",
    )
    data.add_argument(
        "--eval-manifest",
        default=None,
        help="held-out manifest scored when --eval is set",
    )
    data.add_argument(
        "--eval-dataset",
        choices=["atco2"],
        default=None,
        help="optional transfer canary. atco2 is ATCO2-test-set-1h and is "
        "never used for training",
    )
    data.add_argument(
        "--data-root",
        default=None,
        help="resolve relative audio paths against this directory "
        "(default: the manifest's parent)",
    )
    data.add_argument(
        "--text-mode",
        default="auto",
        choices=["auto", "none", "lower"],
        help="auto lowercases a corpus that is >90%% uppercase",
    )
    data.add_argument(
        "--replay-repo",
        default="moonshine-ai/yodas-en-replay",
        help="general-domain replay corpus (HF dataset id)",
    )
    data.add_argument(
        "--no-replay",
        action="store_true",
        help="train on in-domain audio only. Not recommended: the canary "
        "usually gets worse and in-domain WER rarely improves",
    )

    hours = parser.add_argument_group("hours")
    hours.add_argument(
        "--train-hours",
        type=float,
        default=None,
        help="in-domain hours to train on (default: 2.0 for built-in "
        "datasets, all of a custom manifest except the dev slice)",
    )
    hours.add_argument("--dev-hours", type=float, default=0.25)
    hours.add_argument("--replay-hours", type=float, default=6.0)
    hours.add_argument("--replay-dev-hours", type=float, default=0.2)
    hours.add_argument("--replay-ratio", type=float, default=0.5)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        default="moonshine-ai/moonshine-streaming-medium",
        help="HF hub id or local save_pretrained directory",
    )
    model.add_argument(
        "--adapt",
        default="lora",
        choices=["lora", "full"],
        help="lora freezes the backbone (default). full unfreezes it; "
        "use for real radio, not a Colab T4",
    )
    model.add_argument(
        "--sites",
        default="decoder",
        choices=["decoder", "encoder", "both"],
        help="which self-attention stacks get LoRA (ignored when --adapt full)",
    )
    model.add_argument("--rank", type=int, default=8)
    model.add_argument("--alpha", type=float, default=None)
    model.add_argument(
        "--lr",
        type=float,
        default=None,
        help="default 1e-3 for decoder LoRA, 1e-4 when --sites "
        "includes the encoder, 1e-5 for --adapt full",
    )
    model.add_argument("--batch-size", type=int, default=8)
    model.add_argument("--max-steps", type=int, default=3000)
    model.add_argument("--eval-every", type=int, default=100)
    model.add_argument("--patience", type=int, default=4)
    model.add_argument("--warmup", type=int, default=100)
    model.add_argument("--seed", type=int, default=0)
    model.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu, or auto (cuda when available)",
    )

    io = parser.add_argument_group("output")
    io.add_argument("--output-dir", "-o", default="lora_runs")
    io.add_argument("--work-dir", default="lora_work")
    io.add_argument(
        "--prepare-only",
        action="store_true",
        help="index the data and print hours/speakers, then exit",
    )
    io.add_argument(
        "--eval",
        action="store_true",
        help="score in-domain WER after training (ATCOSIM/UWB scored "
        "split, or --eval-manifest)",
    )
    io.add_argument("--eval-limit", type=int, default=None)
    io.add_argument(
        "--canary",
        action="store_true",
        help="also score LibriSpeech test-clean, so forgetting is visible",
    )
    io.add_argument("--canary-limit", type=int, default=None)

    export = parser.add_argument_group("export")
    export.add_argument(
        "--export",
        action="store_true",
        help="export a save_pretrained directory to the runtime's ONNX graphs "
        "instead of training. Use with --model and --output-dir",
    )
    export.add_argument(
        "--graphs",
        default="all",
        help="'all' or a comma-separated subset of "
        "frontend,encoder,adapter,cross_kv,decoder_kv. decoder-only LoRA "
        "only needs decoder_kv; --sites encoder|both and --adapt full "
        "need all",
    )
    export.add_argument(
        "--tokenizer-bin",
        default=None,
        help="tokenizer.bin to copy next to the exported graphs",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from moonshine_voice.lora._deps import require_lora_deps

    try:
        require_lora_deps()
    except ImportError as error:
        print(error, file=sys.stderr)
        return 1

    if args.export:
        from moonshine_voice.lora.export import main as export_main

        export_argv = [
            "--model",
            args.model,
            "--output-dir",
            args.output_dir,
            "--graphs",
            args.graphs,
        ]
        if args.tokenizer_bin:
            export_argv += ["--tokenizer-bin", args.tokenizer_bin]
        export_main(export_argv)
        return 0

    if args.dataset is None and args.train_manifest is None:
        parser.error("one of --dataset or --train-manifest is required")

    from moonshine_voice.lora.train import train_adapter

    train_adapter(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
