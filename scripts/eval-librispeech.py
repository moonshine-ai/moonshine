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
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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
    language = C_ARCH_TO_LANGUAGE[args.model_arch]
    path, arch = get_model_for_language(language, arch)

    options = {"max_tokens_per_second": args.max_tokens_per_second}
    if not args.enable_vad:
        # Disable VAD: threshold 0 turns off speech gating, and a huge max
        # segment duration stops the transcriber chopping the clip into
        # fixed-length pieces (default 15s), so the whole utterance is one
        # segment.
        options["vad_threshold"] = 0.0
        options["vad_max_segment_duration"] = 100000.0

    transcriber = Transcriber(path, arch, options=options)
    print(f"Loaded C library model from {path} (arch={arch})", file=sys.stderr)

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


def main():
    args = parse_args()
    dataset = load_eval_dataset(args)

    text_column = args.text_column or detect_text_column(dataset[0])

    if args.backend == "hf":
        transcribe = make_hf_backend(args)
    else:
        transcribe = make_moonshine_c_backend(
            args, streaming=(args.backend == "moonshine_c_streaming")
        )

    references = []
    hypotheses = []
    per_sample_wer = []
    per_sample_weight = []
    total_audio_seconds = 0.0

    start_time = time.time()
    for sample in tqdm(dataset, desc=args.backend):
        audio, sample_rate = decode_audio(sample["audio"])
        total_audio_seconds += len(audio) / sample_rate

        reference = normalizer(sample[text_column])
        hypothesis = normalizer(transcribe(audio, sample_rate))

        if not reference:
            continue

        references.append(reference)
        hypotheses.append(hypothesis)

        measures = process_words([reference], [hypothesis])
        errors = measures.substitutions + measures.deletions + measures.insertions
        n_words = measures.hits + measures.substitutions + measures.deletions
        per_sample_wer.append(errors / max(1, n_words))
        # Weight by character count to match eval-model-accuracy.py.
        per_sample_weight.append(len(reference))

        if args.verbose:
            print(f"\nREF: {reference}", file=sys.stderr)
            print(f"HYP: {hypothesis}", file=sys.stderr)

    elapsed = time.time() - start_time

    corpus = process_words(references, hypotheses)
    corpus_errors = corpus.substitutions + corpus.deletions + corpus.insertions
    corpus_words = corpus.hits + corpus.substitutions + corpus.deletions
    corpus_wer = corpus_errors / max(1, corpus_words)

    weights = np.asarray(per_sample_weight, dtype=np.float64)
    char_weighted_wer = float(
        np.average(per_sample_wer, weights=weights)
    ) if len(per_sample_wer) else 0.0

    print("\n" + "=" * 60)
    print(f"Backend:            {args.backend}")
    print(f"Model arch:         {args.model_arch}")
    if args.backend == "hf":
        print(f"HF checkpoint:      {args.hf_model or HF_DEFAULT_CHECKPOINT[args.model_arch]}")
    else:
        print(f"VAD:                {'enabled' if args.enable_vad else 'DISABLED'}")
        print(f"max_tokens_per_sec: {args.max_tokens_per_second}")
    print(f"Utterances:         {len(references)}")
    print(f"Reference words:    {corpus_words}")
    print("-" * 60)
    print(f"Corpus WER (OpenASR method):        {corpus_wer:.2%}")
    print(f"  substitutions={corpus.substitutions} deletions={corpus.deletions} insertions={corpus.insertions}")
    print(f"Char-weighted avg WER (old script): {char_weighted_wer:.2%}")
    print("-" * 60)
    print(f"Audio duration:     {total_audio_seconds:.1f}s")
    print(f"Wall time:          {elapsed:.1f}s  (RTF={elapsed / max(1e-9, total_audio_seconds):.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
