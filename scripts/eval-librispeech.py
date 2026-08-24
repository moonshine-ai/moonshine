#!/usr/bin/env python3
"""Evaluate Moonshine speech-to-text accuracy.

English defaults to LibriSpeech ``test-clean`` and reproduces the numbers in
the Moonshine v2 paper (arXiv:2602.12241, Table 3). Other languages default to
FLEURS, which is the fast public set that every published streaming model can
run. Neither path is part of ``scripts/test-*.sh`` or the release build.

Three backends:

  * ``moonshine_c``   - the shipped C++/ONNX library via the ``moonshine_voice``
                        Python bindings. These are the *quantized* ``.ort``
                        models that real users run. Uses
                        ``transcribe_without_streaming`` (batch, whole-utterance).
  * ``moonshine_c_streaming`` - same library, but fed in small chunks through a
                        streaming ``Stream`` (mimics issue #148's setup).
  * ``hf``            - Hugging Face Transformers with the *float* safetensors
                        checkpoints. English streaming models should reproduce
                        the paper (~4.49% tiny on LibriSpeech clean).

Error rate is corpus-wide (total edits / total reference units), the Open ASR
Leaderboard convention. Japanese and Mandarin report no-space CER; everything
else reports WER. ``scripts/eval-model-accuracy.py`` still exists for the older
character-weighted FLEURS average used on the deprecated non-streaming models.

The VAD is disabled by default (``vad_threshold=0`` and a very large
``vad_max_segment_duration``) because these eval clips are already single
utterances.

Examples
--------
Quick smoke test (25 utterances, quantized tiny streaming C library)::

    python scripts/eval-librispeech.py --backend moonshine_c \
        --model-arch tiny_streaming --limit 25

Full reproduction of the paper number with the HF float model::

    python scripts/eval-librispeech.py --backend hf \
        --hf-model moonshine-ai/moonshine-streaming-tiny

Quick FLEURS check for a non-English streaming model (400 seeded clips)::

    python scripts/eval-librispeech.py --language ar --model-arch tiny_streaming --quick

Every published streaming model, FLEURS (or LibriSpeech for English), 400 clips::

    python scripts/eval-librispeech.py --all-streaming --quick

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
from jiwer import process_characters, process_words
from scipy.signal import resample_poly
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

TARGET_SAMPLE_RATE = 16000
QUICK_SAMPLE_SIZE = 400
ESB_REPO = "hf-audio/esb-datasets-test-only-sorted"
OFFICIAL_REPO = "hf-audio/open-asr-leaderboard"
FLEURS_REPO = "google/fleurs"

# Per-language defaults. ``fleurs`` is the google/fleurs config id. Japanese
# and Mandarin are scored with no-space CER because the writing systems do not
# mark word boundaries. ``max_tokens_per_second`` matches the transcriber
# guidance for Latin vs non-Latin tokenizers.
LANGUAGE_CONFIGS = {
    "en": {
        "name": "English",
        "fleurs": "en_us",
        "metric": "wer",
        "normalizer": "english",
        "strip_spaces": False,
        "max_tokens_per_second": 6.5,
        "latin_script": True,
    },
    "ar": {
        "name": "Arabic",
        "fleurs": "ar_eg",
        "metric": "wer",
        "normalizer": "basic",
        "strip_spaces": False,
        "max_tokens_per_second": 13.0,
        "latin_script": False,
    },
    "de": {
        "name": "German",
        "fleurs": "de_de",
        "metric": "wer",
        "normalizer": "basic",
        "strip_spaces": False,
        "max_tokens_per_second": 6.5,
        "latin_script": True,
    },
    "es": {
        "name": "Spanish",
        "fleurs": "es_419",
        "metric": "wer",
        "normalizer": "basic",
        "strip_spaces": False,
        "max_tokens_per_second": 6.5,
        "latin_script": True,
    },
    "ja": {
        "name": "Japanese",
        "fleurs": "ja_jp",
        "metric": "cer",
        "normalizer": "basic",
        "strip_spaces": True,
        "max_tokens_per_second": 13.0,
        "latin_script": False,
    },
    "ko": {
        "name": "Korean",
        "fleurs": "ko_kr",
        "metric": "wer",
        "normalizer": "basic",
        "strip_spaces": False,
        "max_tokens_per_second": 13.0,
        "latin_script": False,
    },
    "tl": {
        "name": "Tagalog",
        "fleurs": "fil_ph",
        "metric": "wer",
        "normalizer": "basic",
        "strip_spaces": False,
        "max_tokens_per_second": 6.5,
        "latin_script": True,
    },
    "uk": {
        "name": "Ukrainian",
        "fleurs": "uk_ua",
        "metric": "wer",
        "normalizer": "basic",
        "strip_spaces": False,
        "max_tokens_per_second": 13.0,
        "latin_script": False,
    },
    "vi": {
        "name": "Vietnamese",
        "fleurs": "vi_vn",
        "metric": "wer",
        "normalizer": "basic",
        "strip_spaces": False,
        "max_tokens_per_second": 6.5,
        "latin_script": True,
    },
    "zh": {
        "name": "Mandarin",
        "fleurs": "cmn_hans_cn",
        "metric": "cer",
        "normalizer": "basic",
        "strip_spaces": True,
        "max_tokens_per_second": 13.0,
        "latin_script": False,
    },
}

# Every streaming (language, arch) pair the catalog publishes. Used by
# --all-streaming. English stays on LibriSpeech; the rest use FLEURS.
STREAMING_JOBS = [
    ("ar", "tiny_streaming"),
    ("de", "small_streaming"),
    ("de", "tiny_streaming"),
    ("en", "medium_streaming"),
    ("en", "small_streaming"),
    ("en", "tiny_streaming"),
    ("es", "small_streaming"),
    ("es", "tiny_streaming"),
    ("ja", "small_streaming"),
    ("ja", "tiny_streaming"),
    ("tl", "tiny_streaming"),
    ("vi", "tiny_streaming"),
    ("zh", "tiny_streaming"),
]

ARCH_CHOICES = sorted(
    {
        "tiny",
        "base",
        "tiny_streaming",
        "small_streaming",
        "medium_streaming",
        *(arch for _, arch in STREAMING_JOBS),
    }
)

# Default HF safetensors checkpoint per arch (English float reference models).
HF_DEFAULT_CHECKPOINT = {
    "tiny_streaming": "moonshine-ai/moonshine-streaming-tiny",
    "small_streaming": "moonshine-ai/moonshine-streaming-small",
    "medium_streaming": "moonshine-ai/moonshine-streaming-medium",
    "tiny": "moonshine-ai/moonshine-tiny",
    "base": "moonshine-ai/moonshine-base",
}

english_normalizer = EnglishTextNormalizer()
basic_normalizer = BasicTextNormalizer()


def _parse_pad_ms(value):
    """Parse a comma-separated list of trailing-pad lengths in milliseconds."""
    if isinstance(value, list):
        return value
    pads = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        pad = int(part)
        if pad < 0:
            raise argparse.ArgumentTypeError("pad-ms values must be >= 0")
        pads.append(pad)
    if not pads:
        raise argparse.ArgumentTypeError("need at least one pad-ms value")
    return pads


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
        "--language",
        default="en",
        choices=sorted(LANGUAGE_CONFIGS.keys()),
        help="Language code (default: en). Non-English defaults to FLEURS.",
    )
    parser.add_argument(
        "--model-arch",
        default="tiny_streaming",
        choices=ARCH_CHOICES,
        help="Architecture for the moonshine_c backends (default: tiny_streaming).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Evaluate {QUICK_SAMPLE_SIZE} seeded clips (a fast check, not the "
        "paper or shipping-panel number).",
    )
    parser.add_argument(
        "--all-streaming",
        action="store_true",
        help="Run a quick eval of every published streaming language/arch pair. "
        "Implies --quick unless --limit is set. Not used by CI or release.",
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
        "--pad-ms",
        type=_parse_pad_ms,
        default=[0],
        help="Trailing zero-padding in milliseconds after each clip. Comma-separated "
        "to sweep, e.g. 0,100,250,500,1000,2000.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Keep only clips at least this many seconds long.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Keep only clips at most this many seconds long. Use this to score "
        "short utterances, where trailing silence has the largest effect.",
    )
    parser.add_argument(
        "--enable-vad",
        action="store_true",
        help="Leave the VAD enabled for moonshine_c (default disables it).",
    )
    parser.add_argument(
        "--max-tokens-per-second",
        type=float,
        default=None,
        help="Hallucination guard. Default is 6.5 for Latin-script languages "
        "and 13.0 otherwise.",
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
        "--keyterms",
        default=None,
        help="Comma-separated key terms to bias the decoder towards. Use this "
        "to measure what contextual biasing costs general accuracy.",
    )
    parser.add_argument(
        "--keyterms-file",
        default=None,
        help="Read key terms from a file of comma or newline separated terms. "
        "A list of any size fits here, unlike an argument.",
    )
    parser.add_argument(
        "--keyterm-boost",
        type=float,
        default=None,
        help="Boost for --keyterms (default: the library's own default).",
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


def language_config(language):
    try:
        return LANGUAGE_CONFIGS[language]
    except KeyError as exc:
        raise SystemExit(
            f"Unknown language '{language}'. Known: {sorted(LANGUAGE_CONFIGS)}"
        ) from exc


def make_text_normalizer(lang_cfg):
    if lang_cfg["normalizer"] == "english":
        def normalize(text):
            return english_normalizer(text)

        return normalize

    def normalize(text):
        out = basic_normalizer(text)
        if lang_cfg.get("strip_spaces"):
            out = "".join(out.split())
        return out

    return normalize


def apply_language_defaults(args):
    """Fill dataset / token-guard defaults for ``args.language``.

    English keeps LibriSpeech test-clean unless the caller overrode the
    dataset flags. Every other language switches the still-default LibriSpeech
    pointing at FLEURS.
    """
    lang_cfg = language_config(args.language)
    args.lang_config = lang_cfg
    args.normalize = make_text_normalizer(lang_cfg)
    args.metric = lang_cfg["metric"]
    if args.max_tokens_per_second is None:
        args.max_tokens_per_second = lang_cfg["max_tokens_per_second"]
    using_english_default = (
        args.dataset == ESB_REPO
        and args.dataset_config == "librispeech"
        and args.split == "test.clean"
    )
    if args.language != "en" and using_english_default:
        args.dataset = FLEURS_REPO
        args.dataset_config = lang_cfg["fleurs"]
        args.split = "test"
    if args.quick and args.limit is None:
        args.limit = QUICK_SAMPLE_SIZE
    return lang_cfg


# Matches moonshine-internal-2/scripts/eval_seq2seq_librispeech.py.
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


def audio_duration_seconds(audio_field):
    """Duration of an undecoded datasets audio field, without resampling."""
    if audio_field.get("bytes") is not None:
        info = sf.info(io.BytesIO(audio_field["bytes"]))
    else:
        info = sf.info(audio_field["path"])
    return float(info.duration)


def apply_trailing_pad(audio, sample_rate, pad_ms):
    """Append `pad_ms` milliseconds of digital zeros after `audio`."""
    if pad_ms <= 0:
        return audio
    n = int(round(pad_ms * sample_rate / 1000.0))
    if n <= 0:
        return audio
    return np.concatenate([audio, np.zeros(n, dtype=audio.dtype)])


def summarize_durations(durations):
    arr = np.asarray(durations, dtype=np.float64)
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size} mean={arr.mean():.2f}s median={np.median(arr):.2f}s "
        f"p10={np.percentile(arr, 10):.2f}s p90={np.percentile(arr, 90):.2f}s "
        f"<2s={(arr < 2).mean():.1%} <3s={(arr < 3).mean():.1%} <5s={(arr < 5).mean():.1%}"
    )


def corpus_error(references, hypotheses, metric="wer"):
    if metric == "cer":
        corpus = process_characters(references, hypotheses)
    else:
        corpus = process_words(references, hypotheses)
    errors = corpus.substitutions + corpus.deletions + corpus.insertions
    units = corpus.hits + corpus.substitutions + corpus.deletions
    return {
        "error": errors / max(1, units),
        "wer": errors / max(1, units),
        "words": units,
        "subs": corpus.substitutions,
        "dels": corpus.deletions,
        "ins": corpus.insertions,
        "metric": metric.upper(),
    }


# ---------------------------------------------------------------------------
# Backends: each returns a callable transcribe(audio_float32, sample_rate)->str
# ---------------------------------------------------------------------------


def load_keyterms(args):
    """Return the key terms named by --keyterms / --keyterms-file, in order."""
    raw = ""
    if args.keyterms_file:
        with open(args.keyterms_file) as handle:
            raw = handle.read()
    elif args.keyterms:
        raw = args.keyterms
    terms = [term.strip() for term in raw.replace("\n", ",").split(",")]
    return [term for term in terms if term]


def make_moonshine_c_backend(args, streaming):
    from moonshine_voice import Transcriber, get_model_for_language, ModelArch

    arch = getattr(ModelArch, args.model_arch.upper())
    if args.model_path:
        path = args.model_path
    else:
        language = args.language
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

    keyterms = load_keyterms(args)
    if keyterms:
        options["keyterms"] = ",".join(keyterms)
        if args.keyterm_boost is not None:
            options["keyterm_boost"] = args.keyterm_boost

    install_start = time.time()
    transcriber = Transcriber(path, arch, options=options)
    load_seconds = time.time() - install_start
    print(f"Loaded C library model from {path} (arch={arch})", file=sys.stderr)
    if keyterms:
        # Every term is tokenized when the list is installed, so this grows with
        # the length of the list and is worth reporting alongside the WER.
        print(
            f"Key terms: {len(keyterms)} "
            f"(boost {args.keyterm_boost if args.keyterm_boost is not None else 'default'}, "
            f"load took {load_seconds:.2f}s)",
            file=sys.stderr,
        )
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

    if args.hf_model:
        checkpoint = args.hf_model
    elif args.language == "en" and args.model_arch in HF_DEFAULT_CHECKPOINT:
        checkpoint = HF_DEFAULT_CHECKPOINT[args.model_arch]
    else:
        raise SystemExit(
            "The hf backend needs --hf-model for non-English checkpoints "
            "(there is no default float id per language)."
        )

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
    pads = list(args.pad_ms)
    extra = []
    if args.min_duration is not None:
        extra.append(f"min={args.min_duration:g}s")
    if args.max_duration is not None:
        extra.append(f"max={args.max_duration:g}s")
    if limit is not None:
        extra.append(f"limit={limit}")
    extra.append("pad=" + ",".join(str(p) for p in pads) + "ms")
    print(
        f"Loading {args.dataset} ({dataset_config}, split={split})"
        + (f" [{', '.join(extra)}]" if extra else "")
        + "...",
        file=sys.stderr,
    )
    dataset = load_dataset(
        args.dataset,
        dataset_config,
        split=split,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    n_loaded = len(dataset)

    if args.min_duration is not None or args.max_duration is not None:
        keep = []
        for i, sample in enumerate(tqdm(dataset, desc=f"{label} duration filter")):
            duration = audio_duration_seconds(sample["audio"])
            if args.min_duration is not None and duration < args.min_duration:
                continue
            if args.max_duration is not None and duration > args.max_duration:
                continue
            keep.append(i)
        dataset = dataset.select(keep)
        print(
            f"{label}: kept {len(keep)}/{n_loaded} clips after duration filter",
            file=sys.stderr,
        )

    if limit is not None:
        # ESB splits are sorted longest-first; a plain head() would only score
        # the longest clips. Seeded shuffle matches the internal eval harness.
        dataset = dataset.shuffle(seed=0).select(range(min(limit, len(dataset))))

    text_column = text_column_override or detect_text_column(dataset[0])

    buckets = {
        pad: {"references": [], "hypotheses": [], "elapsed_s": 0.0} for pad in pads
    }
    durations = []
    total_audio_seconds = 0.0
    n_diff = {pad: 0 for pad in pads}
    baseline_pad = 0 if 0 in pads else pads[0]

    start_time = time.time()
    for sample in tqdm(dataset, desc=label):
        audio, sample_rate = decode_audio(sample["audio"])
        duration = len(audio) / sample_rate
        durations.append(duration)
        total_audio_seconds += duration

        reference = args.normalize(sample[text_column])
        if not reference:
            continue

        hyps_this_clip = {}
        for pad in pads:
            padded = apply_trailing_pad(audio, sample_rate, pad)
            t0 = time.time()
            hypothesis = args.normalize(transcribe(padded, sample_rate))
            buckets[pad]["elapsed_s"] += time.time() - t0
            buckets[pad]["references"].append(reference)
            buckets[pad]["hypotheses"].append(hypothesis)
            hyps_this_clip[pad] = hypothesis

        for pad in pads:
            if hyps_this_clip[pad] != hyps_this_clip[baseline_pad]:
                n_diff[pad] += 1

        if args.verbose:
            print(f"\nREF: {reference}", file=sys.stderr)
            for pad in pads:
                print(f"HYP pad={pad}ms: {hyps_this_clip[pad]}", file=sys.stderr)

    wall_s = time.time() - start_time
    print(
        f"{label} durations ({summarize_durations(durations)}); "
        f"wall={wall_s:.1f}s",
        file=sys.stderr,
    )

    results = []
    for pad in pads:
        refs = buckets[pad]["references"]
        hyps = buckets[pad]["hypotheses"]
        pad_label = label if pads == [0] else f"{label}/pad{pad}"
        if not refs:
            results.append(
                {
                    "label": pad_label,
                    "pad_ms": pad,
                    "n": 0,
                    "wer": None,
                    "words": 0,
                    "n_diff": 0,
                    "audio_s": total_audio_seconds,
                    "elapsed_s": buckets[pad]["elapsed_s"],
                }
            )
            continue
        stats = corpus_error(refs, hyps, getattr(args, "metric", "wer"))
        stats.update(
            {
                "label": pad_label,
                "pad_ms": pad,
                "n": len(refs),
                "n_diff": n_diff[pad],
                "audio_s": total_audio_seconds,
                "elapsed_s": buckets[pad]["elapsed_s"],
            }
        )
        results.append(stats)
    return results


def print_result_block(args, results):
    print("\n" + "=" * 60)
    print(f"Backend:            {args.backend}")
    print(f"Language:           {args.language}")
    print(f"Model arch:         {args.model_arch}")
    print(f"Metric:             {getattr(args, 'metric', 'wer').upper()}")
    print(f"Speculative:        {args.use_speculative_decoding}")
    print(f"pad-ms:             {','.join(str(p) for p in args.pad_ms)}")
    if args.min_duration is not None or args.max_duration is not None:
        lo = args.min_duration if args.min_duration is not None else 0
        hi = args.max_duration if args.max_duration is not None else float("inf")
        print(f"duration filter:    {lo:g}s .. {hi:g}s")
    if args.backend == "hf":
        hf_id = args.hf_model
        if hf_id is None and args.language == "en":
            hf_id = HF_DEFAULT_CHECKPOINT.get(args.model_arch)
        print(f"HF checkpoint:      {hf_id}")
    else:
        print(f"VAD:                {'enabled' if args.enable_vad else 'DISABLED'}")
        print(f"max_tokens_per_sec: {args.max_tokens_per_second}")
        keyterms = load_keyterms(args)
        boost = args.keyterm_boost if args.keyterm_boost is not None else "default"
        print(f"key terms:          {len(keyterms)} (boost {boost})")
        if args.backend == "moonshine_c_streaming":
            print(f"update_interval:    {args.update_interval}s")
            print(f"chunk_duration:     {args.chunk_duration}s")
    print("-" * 60)
    unique_sets = {r["label"].split("/pad")[0] for r in results}
    unique_pads = {r.get("pad_ms", 0) for r in results}
    errors = []
    metric_name = getattr(args, "metric", "wer").upper()
    for r in results:
        if r["wer"] is None:
            print(f"{r['label']:24s}  n/a (n=0)")
            continue
        print(
            f"{r['label']:24s}  {metric_name}={r['wer']:.2%}  n={r['n']}  "
            f"units={r['words']}  "
            f"S/D/I={r.get('subs',0)}/{r.get('dels',0)}/{r.get('ins',0)}  "
            f"diff={r.get('n_diff', 0)}  "
            f"RTF={r['elapsed_s'] / max(1e-9, r['audio_s']):.3f}"
        )
        errors.append(r["wer"])
    if len(errors) > 1 and len(unique_sets) > 1 and len(unique_pads) == 1:
        macro = float(np.mean(errors))
        print("-" * 60)
        print(f"{'macro average':24s}  {metric_name}={macro:.2%}  ({len(errors)} sets)")
    print("=" * 60)
    return errors


def run_one_eval(args):
    apply_language_defaults(args)
    if args.backend == "hf":
        transcribe = make_hf_backend(args)
    else:
        transcribe = make_moonshine_c_backend(
            args, streaming=(args.backend == "moonshine_c_streaming")
        )

    results = []
    suite = resolve_suite(args.suite)
    if suite is None:
        results.extend(
            evaluate_one(
                args,
                transcribe,
                args.dataset_config,
                args.split,
                args.text_column,
                args.limit,
                args.dataset_config,
            )
        )
    else:
        limits = parse_sample_size(args.sample_size, suite)
        if args.limit is not None:
            for k in list(limits.keys()):
                limits[k] = args.limit
        for name in suite:
            if name not in EVAL_SETS:
                raise SystemExit(
                    f"Unknown eval set '{name}'. Known: {sorted(EVAL_SETS)}"
                )
            cfg = EVAL_SETS[name]
            saved_dataset = args.dataset
            args.dataset = cfg["dataset"]
            pad_results = evaluate_one(
                args,
                transcribe,
                cfg["name"],
                cfg["split"],
                cfg.get("text_column"),
                limits.get(name),
                name,
            )
            args.dataset = saved_dataset
            results.extend(pad_results)
            for result in pad_results:
                metric_name = getattr(args, "metric", "wer").upper()
                err_s = (
                    f"{result['wer']:.2%}" if result["wer"] is not None else "n/a"
                )
                print(
                    f"{result['label']} {metric_name} = {err_s} (n={result['n']})",
                    file=sys.stderr,
                )
    print_result_block(args, results)
    return results


def main():
    args = parse_args()

    if args.use_speculative_decoding and args.backend != "moonshine_c_streaming":
        print(
            "Note: --use-speculative-decoding forces moonshine_c_streaming "
            "(batch path only decodes once).",
            file=sys.stderr,
        )
        args.backend = "moonshine_c_streaming"

    if args.all_streaming:
        if args.limit is None:
            args.quick = True
        saved_dataset = args.dataset
        saved_config = args.dataset_config
        saved_split = args.split
        saved_tokens = args.max_tokens_per_second
        summary = []
        for language, arch in STREAMING_JOBS:
            args.language = language
            args.model_arch = arch
            args.dataset = saved_dataset
            args.dataset_config = saved_config
            args.split = saved_split
            args.max_tokens_per_second = saved_tokens
            print(
                f"\n>>> {language} {arch}",
                file=sys.stderr,
            )
            results = run_one_eval(args)
            primary = next((r for r in results if r.get("wer") is not None), None)
            summary.append(
                {
                    "language": language,
                    "arch": arch,
                    "metric": getattr(args, "metric", "wer").upper(),
                    "error": None if primary is None else primary["wer"],
                    "n": 0 if primary is None else primary["n"],
                }
            )
        print("\n" + "=" * 60)
        print("Streaming models (quick)")
        print("-" * 60)
        for row in summary:
            err = "n/a" if row["error"] is None else f"{row['error']:.2%}"
            print(
                f"{row['language']:4s} {row['arch']:18s}  "
                f"{row['metric']}={err}  n={row['n']}"
            )
        print("=" * 60)
        return

    run_one_eval(args)


if __name__ == "__main__":
    main()
