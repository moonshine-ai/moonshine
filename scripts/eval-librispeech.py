#!/usr/bin/env python3
"""Evaluate Moonshine English WER on LibriSpeech test-clean.

This script reproduces the LibriSpeech (clean) numbers reported in the
Moonshine v2 paper (arXiv:2602.12241, Table 3) and lets us compare three
different code paths so we can see where any accuracy gap comes from:

  * ``moonshine_c``   - the shipped C++/ONNX library via the ``moonshine_voice``
                        Python bindings. These are the *quantized* ``.ort``
                        models that real users run. Uses
                        ``transcribe_without_streaming`` (batch, whole-utterance).
  * ``moonshine_c_streaming`` - same library, but fed in small chunks through a
                        streaming ``Stream`` (mimics issue #148's setup, which
                        reports ~11.8% WER).
  * ``hf``            - the Hugging Face Transformers reference implementation
                        with the *float* safetensors checkpoints. This is the
                        code path the paper used to measure WER (see paper
                        section 4.1.2), so it should reproduce ~4.49% for tiny.

WER is aggregated corpus-wide (total edits / total reference words), which is
the Open ASR Leaderboard convention the paper follows. We also print the
character-weighted per-utterance average that ``scripts/eval-model-accuracy.py``
uses, because that alternative aggregation is itself a source of confusion.

The VAD is disabled by default for the ``moonshine_c`` backend (the samples are
known to be single speech utterances, so VAD segmentation only adds errors):
``vad_threshold=0`` and a very large ``vad_max_segment_duration`` so the whole
clip is transcribed as one segment.

Examples
--------
Quick smoke test (25 utterances, quantized tiny streaming C library)::

    python scripts/eval-librispeech.py --backend moonshine_c \
        --model-arch tiny_streaming --limit 25

Full reproduction of the paper number with the HF float model::

    python scripts/eval-librispeech.py --backend hf \
        --hf-model UsefulSensors/moonshine-streaming-tiny

On a Mac you may need ffmpeg for audio decoding::

    brew install ffmpeg@8
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@8/lib:$DYLD_LIBRARY_PATH"
"""

import argparse
import sys
import time

import io

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from jiwer import process_words
from scipy.signal import resample_poly
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

TARGET_SAMPLE_RATE = 16000


# Map friendly arch names to the moonshine_voice ModelArch and the matching
# language string used to download the quantized C-library model.
C_ARCH_TO_LANGUAGE = {
    "tiny": "en",
    "base": "en",
    "tiny_streaming": "en",
    "small_streaming": "en",
    "medium_streaming": "en",
}

# Default HF safetensors checkpoint per arch (float reference models).
HF_DEFAULT_CHECKPOINT = {
    "tiny_streaming": "UsefulSensors/moonshine-streaming-tiny",
    "small_streaming": "UsefulSensors/moonshine-streaming-small",
    "medium_streaming": "UsefulSensors/moonshine-streaming-medium",
    "tiny": "UsefulSensors/moonshine-tiny",
    "base": "UsefulSensors/moonshine-base",
}

normalizer = EnglishTextNormalizer()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend",
        choices=["moonshine_c", "moonshine_c_streaming", "hf"],
        default="moonshine_c",
    )
    parser.add_argument(
        "--model-arch",
        default="tiny_streaming",
        choices=sorted(C_ARCH_TO_LANGUAGE.keys()),
        help="Architecture for the moonshine_c backends (default: tiny_streaming).",
    )
    parser.add_argument(
        "--hf-model",
        default=None,
        help="HF checkpoint id for the hf backend (defaults per --model-arch).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Directory of .ort models to use instead of the downloaded ones. "
        "Useful for comparing quantization recipes.",
    )
    parser.add_argument(
        "--dataset",
        default="hf-audio/esb-datasets-test-only-sorted",
        help="HF dataset id (default: the Open ASR Leaderboard test-only set).",
    )
    parser.add_argument("--dataset-config", default="librispeech")
    parser.add_argument("--split", default="test.clean")
    parser.add_argument(
        "--text-column",
        default=None,
        help="Ground-truth text column. Auto-detected if not set.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N samples."
    )
    parser.add_argument(
        "--enable-vad",
        action="store_true",
        help="Leave the VAD enabled for moonshine_c (default disables it).",
    )
    parser.add_argument(
        "--max-tokens-per-second",
        type=float,
        default=6.5,
        help="Hallucination guard for English (C library default 6.5).",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.1,
        help="Chunk size (s) for moonshine_c_streaming (default 0.1).",
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=0.5,
        help="Update interval (s) for moonshine_c_streaming (default 0.5).",
    )
    parser.add_argument(
        "--use-speculative-decoding",
        action="store_true",
        help="Enable decode_full speculative verify on streaming re-decodes. "
        "Implies --backend moonshine_c_streaming (batch has only one decode).",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Run a named panel: 'librispeech' (test-clean only), 'official' "
        "(Open ASR Leaderboard 7-set average), 'suite' (internal hourly panel), "
        "or a comma-separated list of set names. Overrides --dataset-config.",
    )
    parser.add_argument(
        "--sample-size",
        default=None,
        help="Limit per set: N for all, or 'N,librispeech=0' style overrides "
        "(0 = full split). Default: full for librispeech*, 400 for others when "
        "using --suite.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# Matches moonshine-internal-2/scripts/eval_seq2seq_librispeech.py.
ESB_REPO = "hf-audio/esb-datasets-test-only-sorted"
OFFICIAL_REPO = "hf-audio/open-asr-leaderboard"

EVAL_SETS = {
    "librispeech": {
        "dataset": ESB_REPO,
        "name": "librispeech",
        "split": "test.clean",
        "text_column": "text",
    },
    "librispeech_other": {
        "dataset": ESB_REPO,
        "name": "librispeech",
        "split": "test.other",
        "text_column": "text",
    },
    "ami": {
        "dataset": ESB_REPO,
        "name": "ami",
        "split": "test",
        "text_column": "text",
    },
    "earnings22": {
        "dataset": ESB_REPO,
        "name": "earnings22",
        "split": "test",
        "text_column": "text",
    },
    "voxpopuli": {
        "dataset": ESB_REPO,
        "name": "voxpopuli",
        "split": "test",
        "text_column": "text",
    },
    "spgispeech": {
        "dataset": OFFICIAL_REPO,
        "name": "spgispeech",
        "split": "test",
        "text_column": "text",
    },
    "common_voice": {
        "dataset": ESB_REPO,
        "name": "common_voice",
        "split": "test",
        "text_column": "text",
    },
    "gigaspeech": {
        "dataset": ESB_REPO,
        "name": "gigaspeech",
        "split": "test",
        "text_column": "text",
    },
}

SUITE = [
    "librispeech",
    "ami",
    "earnings22",
    "voxpopuli",
    "common_voice",
    "gigaspeech",
]

OFFICIAL = [
    "ami",
    "earnings22",
    "gigaspeech",
    "librispeech",
    "librispeech_other",
    "spgispeech",
    "voxpopuli",
]


def resolve_suite(suite_arg):
    if suite_arg is None:
        return None
    if suite_arg == "librispeech":
        return ["librispeech"]
    if suite_arg == "suite":
        return list(SUITE)
    if suite_arg == "official":
        return list(OFFICIAL)
    return [s.strip() for s in suite_arg.split(",") if s.strip()]


def parse_sample_size(sample_size_arg, set_names):
    """Return {set_name: limit_or_None} where None means full split."""
    defaults = {}
    for name in set_names:
        if name.startswith("librispeech"):
            defaults[name] = None  # full
        else:
            defaults[name] = 400
    if sample_size_arg is None:
        return defaults
    # Forms: "500" or "400,librispeech=0,ami=200"
    parts = [p.strip() for p in sample_size_arg.split(",") if p.strip()]
    global_n = None
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            n = int(v)
            defaults[k] = None if n == 0 else n
        else:
            global_n = int(part)
    if global_n is not None:
        for name in set_names:
            if name not in sample_size_arg:  # crude; overrides already applied
                pass
        # Apply global only where not explicitly overridden in the string.
        overridden = {
            p.split("=", 1)[0] for p in parts if "=" in p
        }
        for name in set_names:
            if name not in overridden:
                defaults[name] = None if global_n == 0 else global_n
    return defaults


def detect_text_column(sample):
    for candidate in ("text", "transcription", "sentence", "normalized_text"):
        if candidate in sample:
            return candidate
    raise ValueError(
        f"Could not find a text column in sample keys: {list(sample.keys())}"
    )


def load_eval_dataset(args):
    print(
        f"Loading {args.dataset} ({args.dataset_config}, split={args.split})...",
        file=sys.stderr,
    )
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    # Decode audio ourselves via soundfile. datasets 4.x otherwise pulls in
    # torchcodec, which needs an ffmpeg build (4-7) that isn't available here.
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def decode_audio(audio_field):
    """Return (float32 mono @16kHz, sample_rate) from a non-decoded audio field."""
    if audio_field.get("bytes") is not None:
        data, sample_rate = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
    else:
        data, sample_rate = sf.read(audio_field["path"], dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        data = resample_poly(data, TARGET_SAMPLE_RATE, sample_rate).astype(np.float32)
        sample_rate = TARGET_SAMPLE_RATE
    return data, sample_rate


# ---------------------------------------------------------------------------
# Backends: each returns a callable transcribe(audio_float32, sample_rate)->str
# ---------------------------------------------------------------------------


def make_moonshine_c_backend(args, streaming):
    from moonshine_voice import Transcriber, get_model_for_language, ModelArch

    arch = getattr(ModelArch, args.model_arch.upper())
    if args.model_path:
        path = args.model_path
    else:
        language = C_ARCH_TO_LANGUAGE[args.model_arch]
        path, arch = get_model_for_language(language, arch)

    options = {"max_tokens_per_second": args.max_tokens_per_second}
    if getattr(args, "use_speculative_decoding", False):
        options["use_speculative_decoding"] = True
    if not args.enable_vad:
        # Disable VAD: threshold 0 turns off speech gating, and a huge max
        # segment duration stops the transcriber chopping the clip into
        # fixed-length pieces (default 15s), so the whole utterance is one
        # segment.
        options["vad_threshold"] = 0.0
        options["vad_max_segment_duration"] = 100000.0

    transcriber = Transcriber(path, arch, options=options)
    print(f"Loaded C library model from {path} (arch={arch})", file=sys.stderr)
    if options.get("use_speculative_decoding"):
        print("Speculative decoding: ENABLED", file=sys.stderr)

    def transcribe_batch(audio, sample_rate):
        transcript = transcriber.transcribe_without_streaming(
            audio.tolist(), sample_rate
        )
        return " ".join(line.text for line in transcript.lines).strip()

    def transcribe_streaming(audio, sample_rate):
        stream = transcriber.create_stream(update_interval=args.update_interval)
        stream.start()
        chunk_size = max(1, int(args.chunk_duration * sample_rate))
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            stream.add_audio(chunk.tolist(), sample_rate)
        transcript = stream.stop()
        stream.close()
        if transcript is None:
            return ""
        return " ".join(line.text for line in transcript.lines).strip()

    return transcribe_streaming if streaming else transcribe_batch


def make_hf_backend(args):
    import torch
    from transformers import AutoProcessor

    checkpoint = args.hf_model or HF_DEFAULT_CHECKPOINT[args.model_arch]

    # Streaming (v2) checkpoints need MoonshineStreamingForConditionalGeneration;
    # older v1 checkpoints use MoonshineForConditionalGeneration.
    model_cls = None
    try:
        from transformers import MoonshineStreamingForConditionalGeneration

        model_cls = MoonshineStreamingForConditionalGeneration
    except ImportError:
        pass

    processor = AutoProcessor.from_pretrained(checkpoint)
    try:
        if model_cls is not None:
            model = model_cls.from_pretrained(checkpoint)
        else:
            raise ImportError
    except Exception:
        from transformers import MoonshineForConditionalGeneration

        model = MoonshineForConditionalGeneration.from_pretrained(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    sr = processor.feature_extractor.sampling_rate
    print(
        f"Loaded HF model {checkpoint} ({model.__class__.__name__}) on {device}",
        file=sys.stderr,
    )

    def transcribe(audio, sample_rate):
        assert sample_rate == sr, f"expected {sr} Hz, got {sample_rate}"
        inputs = processor(
            audio, return_tensors="pt", sampling_rate=sample_rate
        ).to(device)
        with torch.no_grad():
            # token_limit_factor mirrors the library's tokens-per-second guard;
            # generous cap here since these are clean utterances.
            generated = model.generate(**inputs, max_new_tokens=256)
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    return transcribe


def evaluate_one(args, transcribe, dataset_config, split, text_column_override,
                 limit, label):
    print(
        f"Loading {args.dataset} ({dataset_config}, split={split})"
        + (f" limit={limit}" if limit else "")
        + "...",
        file=sys.stderr,
    )
    dataset = load_dataset(args.dataset, dataset_config, split=split)
    if limit is not None:
        # ESB splits are sorted longest-first; a plain head() would only score
        # the longest clips. Seeded shuffle matches the internal eval harness.
        dataset = dataset.shuffle(seed=0).select(range(min(limit, len(dataset))))
    dataset = dataset.cast_column("audio", Audio(decode=False))

    text_column = text_column_override or detect_text_column(dataset[0])

    references = []
    hypotheses = []
    total_audio_seconds = 0.0

    start_time = time.time()
    for sample in tqdm(dataset, desc=label):
        audio, sample_rate = decode_audio(sample["audio"])
        total_audio_seconds += len(audio) / sample_rate

        reference = normalizer(sample[text_column])
        hypothesis = normalizer(transcribe(audio, sample_rate))

        if not reference:
            continue

        references.append(reference)
        hypotheses.append(hypothesis)

        if args.verbose:
            print(f"\nREF: {reference}", file=sys.stderr)
            print(f"HYP: {hypothesis}", file=sys.stderr)

    elapsed = time.time() - start_time
    if not references:
        return {
            "label": label,
            "n": 0,
            "wer": None,
            "words": 0,
            "audio_s": total_audio_seconds,
            "elapsed_s": elapsed,
        }

    corpus = process_words(references, hypotheses)
    corpus_errors = corpus.substitutions + corpus.deletions + corpus.insertions
    corpus_words = corpus.hits + corpus.substitutions + corpus.deletions
    corpus_wer = corpus_errors / max(1, corpus_words)
    return {
        "label": label,
        "n": len(references),
        "wer": corpus_wer,
        "words": corpus_words,
        "subs": corpus.substitutions,
        "dels": corpus.deletions,
        "ins": corpus.insertions,
        "audio_s": total_audio_seconds,
        "elapsed_s": elapsed,
    }


def main():
    args = parse_args()

    if args.use_speculative_decoding and args.backend != "moonshine_c_streaming":
        print(
            "Note: --use-speculative-decoding forces moonshine_c_streaming "
            "(batch path only decodes once).",
            file=sys.stderr,
        )
        args.backend = "moonshine_c_streaming"

    suite = resolve_suite(args.suite)

    if args.backend == "hf":
        transcribe = make_hf_backend(args)
    else:
        transcribe = make_moonshine_c_backend(
            args, streaming=(args.backend == "moonshine_c_streaming")
        )

    if suite is None:
        # Legacy single-set mode.
        dataset = load_eval_dataset(args)
        text_column = args.text_column or detect_text_column(dataset[0])
        # Reuse evaluate_one via a thin adapter: reload is fine / consistent.
        result = evaluate_one(
            args,
            transcribe,
            args.dataset_config,
            args.split,
            text_column,
            args.limit,
            args.dataset_config,
        )
        results = [result]
    else:
        limits = parse_sample_size(args.sample_size, suite)
        if args.limit is not None:
            for k in list(limits.keys()):
                limits[k] = args.limit
        results = []
        for name in suite:
            if name not in EVAL_SETS:
                raise SystemExit(f"Unknown eval set '{name}'. Known: {sorted(EVAL_SETS)}")
            cfg = EVAL_SETS[name]
            # Temporarily point args.dataset at the set's repo.
            saved_dataset = args.dataset
            args.dataset = cfg["dataset"]
            result = evaluate_one(
                args,
                transcribe,
                cfg["name"],
                cfg["split"],
                cfg.get("text_column"),
                limits.get(name),
                name,
            )
            args.dataset = saved_dataset
            results.append(result)
            wer_s = f"{result['wer']:.2%}" if result["wer"] is not None else "n/a"
            print(
                f"{name} WER = {wer_s} (n={result['n']})",
                file=sys.stderr,
            )

    print("\n" + "=" * 60)
    print(f"Backend:            {args.backend}")
    print(f"Model arch:         {args.model_arch}")
    print(f"Speculative:        {args.use_speculative_decoding}")
    if args.backend == "hf":
        print(f"HF checkpoint:      {args.hf_model or HF_DEFAULT_CHECKPOINT[args.model_arch]}")
    else:
        print(f"VAD:                {'enabled' if args.enable_vad else 'DISABLED'}")
        print(f"max_tokens_per_sec: {args.max_tokens_per_second}")
        if args.backend == "moonshine_c_streaming":
            print(f"update_interval:    {args.update_interval}s")
            print(f"chunk_duration:     {args.chunk_duration}s")
    print("-" * 60)
    wers = []
    for r in results:
        if r["wer"] is None:
            print(f"{r['label']:20s}  n/a (n=0)")
            continue
        print(
            f"{r['label']:20s}  WER={r['wer']:.2%}  n={r['n']}  "
            f"words={r['words']}  "
            f"S/D/I={r.get('subs',0)}/{r.get('dels',0)}/{r.get('ins',0)}  "
            f"RTF={r['elapsed_s'] / max(1e-9, r['audio_s']):.3f}"
        )
        wers.append(r["wer"])
    if len(wers) > 1:
        macro = float(np.mean(wers))
        print("-" * 60)
        print(f"{'macro average':20s}  WER={macro:.2%}  ({len(wers)} sets)")
    print("=" * 60)


if __name__ == "__main__":
    main()
