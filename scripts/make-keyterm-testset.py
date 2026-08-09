#!/usr/bin/env python3
"""Build a key-term biasing test set out of an ordinary ASR corpus.

Tuning contextual biasing needs a corpus where the terms are the words that
actually matter, and there is no need to record one: any ASR test set already
contains them. The words a corpus rarely says are the ones a model gets wrong
and the ones a real caller would put on a list — names, places, jargon — so this
takes the rare words of each utterance as that utterance's key terms.

A list of only the words that were said would flatter the feature, because
biasing is easy when everything you are told to listen for is really there. So
each list is padded with distractors drawn from other utterances' rare words:
plausible, confusable, and definitely not spoken here. That size is the setting
that matters most, since false alarms scale with how many terms are live.

Rare is defined by frequency within the corpus itself: everything outside the
``--common-words`` most frequent words. On LibriSpeech test-clean the default
leaves about three key terms per utterance and 15% of all words spoken as key
terms, which is the right shape for tuning - common enough to measure, rare
enough to be the words you would have listed.

The output is the JSON Lines manifest and WAV files that
``scripts/eval-keyterm-biasing.py`` consumes::

    python scripts/make-keyterm-testset.py --output-dir /tmp/libri-keyterm
    python scripts/eval-keyterm-biasing.py \\
        --manifest /tmp/libri-keyterm/manifest-100.jsonl --boosts 0,1,2,3,4,6

One manifest is written per ``--distractor-count``, all sharing the same audio,
so list size can be swept without re-exporting anything.
"""

import argparse
import io
import json
import random
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000

_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(text):
    """Lower-case, strip punctuation, collapse whitespace.

    Deliberately the same rule as eval-keyterm-biasing.py applies to both sides
    of its comparison, so the terms chosen here are words that script will see.
    """
    return " ".join(text.lower().translate(_PUNCTUATION).split())


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="hf-audio/esb-datasets-test-only-sorted",
        help="HF dataset id (default: the Open ASR Leaderboard test-only set).",
    )
    parser.add_argument("--dataset-config", default="librispeech")
    parser.add_argument("--split", default="test.clean")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the WAV files and manifests into.",
    )
    parser.add_argument(
        "--common-words",
        type=int,
        default=2000,
        help="How many of the corpus's most frequent words count as common, and "
        "so are never key terms (default: 2000).",
    )
    parser.add_argument(
        "--distractor-counts",
        default="100,1000",
        help="Comma-separated list sizes to write a manifest for (default: "
        "100,1000). Each is the number of unspoken terms padding every list.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only use the first N utterances."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the distractor sample."
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Rewrite the manifests only, reusing WAV files already exported.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep the corpus order. By default utterances are written in a "
        "seeded shuffle, so evaluating a prefix of the manifest samples the "
        "whole corpus rather than whichever end it happens to be sorted by.",
    )
    return parser.parse_args(argv)


def decode_audio(audio_field):
    """Return float32 mono samples at 16 kHz from a non-decoded audio field."""
    if audio_field.get("bytes") is not None:
        samples, sample_rate = sf.read(
            io.BytesIO(audio_field["bytes"]), dtype="float32"
        )
    else:
        samples, sample_rate = sf.read(audio_field["path"], dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        samples = resample_poly(samples, TARGET_SAMPLE_RATE, sample_rate).astype(
            np.float32
        )
    return samples


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        distractor_counts = [
            int(value) for value in args.distractor_counts.split(",") if value.strip()
        ]
    except ValueError as error:
        raise SystemExit(f"--distractor-counts must be numbers: {error}") from error

    print(
        f"Loading {args.dataset} ({args.dataset_config}, split={args.split})...",
        file=sys.stderr,
    )
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    dataset = dataset.cast_column("audio", Audio(decode=False))

    raw_texts = list(dataset[args.text_column])
    texts = [normalize(text) for text in raw_texts]
    frequencies = Counter(word for text in texts for word in text.split())
    common = {word for word, _ in frequencies.most_common(args.common_words)}
    rare_pool = sorted(word for word in frequencies if word not in common)
    if not rare_pool:
        raise SystemExit("No rare words left; lower --common-words.")
    print(
        f"{len(dataset)} utterances, {sum(frequencies.values())} words, "
        f"{len(frequencies)} distinct, {len(rare_pool)} of them rare",
        file=sys.stderr,
    )

    spoken_terms = []
    for text in texts:
        # Ordered rather than a set so the manifest is stable and readable.
        seen = {}
        for word in text.split():
            if word not in common:
                seen[word] = None
        spoken_terms.append(list(seen))

    if not args.skip_audio:
        for index, sample in enumerate(tqdm(dataset, desc="exporting audio")):
            path = audio_dir / f"{index:05d}.wav"
            if path.exists():
                continue
            sf.write(path, decode_audio(sample["audio"]), TARGET_SAMPLE_RATE)

    # Its own generator, so every manifest lists the utterances in the same
    # order however many distractors each one gets.
    order = list(range(len(texts)))
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(order)

    rng = random.Random(args.seed)
    for count in distractor_counts:
        manifest_path = output_dir / f"manifest-{count}.jsonl"
        total_terms = 0
        with open(manifest_path, "w", encoding="utf-8") as manifest:
            for index in order:
                raw_text = raw_texts[index]
                text = texts[index]
                terms = spoken_terms[index]
                # Distractors come from the corpus's own rare words, so they are
                # the kind of word that gets confused for the real ones. Nothing
                # this utterance says can be drawn, or a correct transcript would
                # be scored as a false alarm.
                spoken_here = set(text.split())
                available = [word for word in rare_pool if word not in spoken_here]
                if len(available) < count:
                    raise SystemExit(
                        f"Only {len(available)} distractors available for "
                        f"{count} requested; lower --distractor-counts or "
                        f"raise --common-words."
                    )
                keyterms = terms + rng.sample(available, count)
                total_terms += len(keyterms)
                manifest.write(
                    json.dumps(
                        {
                            "audio": f"audio/{index:05d}.wav",
                            "text": raw_text,
                            "keyterms": keyterms,
                        }
                    )
                    + "\n"
                )
        print(
            f"{manifest_path}: {len(texts)} utterances, "
            f"{total_terms / len(texts):.0f} terms each",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
