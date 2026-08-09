#!/usr/bin/env python3
"""Measure what runtime key-term biasing buys, and what it costs.

Contextual biasing trades two things off against each other, so a single
accuracy number cannot tell you whether a boost setting is any good:

  * **Key-term recall** - of the key terms actually spoken, how many make it
    into the transcript. This is what biasing is for, and it goes up with the
    boost.
  * **False alarms** - key terms appearing where nobody said them. This also
    goes up with the boost, and it is the failure mode users notice, because a
    product name spliced into an unrelated sentence reads as nonsense.

Overall WER is reported alongside them to catch collateral damage: a boost high
enough to force terms in will start bending ordinary words around them. That
total is also split in two, since the two halves move in opposite directions and
a single number hides the trade:

  * **term WER** - errors on the reference words your lists asked for, plus any
    word invented out of one. This is the half biasing is meant to fix.
  * **other WER** - errors on every word the list never mentioned. This is the
    bill, and it is charged on every utterance whether a term was said or not.

The script sweeps the boost so you can see the whole curve and pick the knee,
rather than trusting a default. Passing a realistic number of terms matters as
much as the boost: false alarms scale with how many terms are live, so a list of
one term will look far cleaner than the hundred a real deployment sends. Use
``--distractors-file`` to pad the list out to a realistic size with terms that
are *not* in the audio.

Manifest format (JSON Lines), one utterance per line::

    {"audio": "clip.wav", "text": "we migrated onto Kubernetes last quarter",
     "keyterms": ["Kubernetes"]}

``audio`` is resolved relative to the manifest's own directory unless absolute.
``keyterms`` is optional per line; ``--keyterms`` adds terms to every line.

Examples
--------
Sweep the boost on the bundled example manifest::

    python scripts/eval-keyterm-biasing.py \\
        --manifest scripts/data/keyterm-eval-example.jsonl \\
        --model-path test-assets/tiny-streaming-en

Check how the same setting holds up when 200 unrelated terms are also live::

    python scripts/eval-keyterm-biasing.py --manifest my-domain.jsonl \\
        --distractors-file /usr/share/dict/words --distractor-count 200

Note on normalization: this uses a simple lower-case/strip-punctuation
normalizer rather than the Whisper English normalizer that
``scripts/eval-librispeech.py`` uses, so the absolute WER here is not
comparable with that script's. What matters for tuning is the comparison
between rows, which all share this normalizer.
"""

import argparse
import json
import random
import re
import string
import sys
from array import array
from pathlib import Path

_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(text):
    """Lower-case, strip punctuation, and collapse whitespace."""
    return " ".join(text.lower().translate(_PUNCTUATION).split())


def word_alignment(reference_words, hypothesis_words):
    """Levenshtein alignment as a list of (operation, reference, hypothesis).

    The whole matrix is kept rather than one row at a time, because the path
    matters here and not just its length: charging each error to the reference
    word it happened to is what separates the key terms from everything else.
    Utterances are sentences, so the matrix stays small.
    """
    rows = len(reference_words) + 1
    columns = len(hypothesis_words) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
    for column in range(1, columns):
        costs[0][column] = column
    for row in range(1, rows):
        reference_word = reference_words[row - 1]
        for column in range(1, columns):
            substitution = costs[row - 1][column - 1] + (
                reference_word != hypothesis_words[column - 1]
            )
            costs[row][column] = min(
                substitution, costs[row - 1][column] + 1, costs[row][column - 1] + 1
            )

    operations = []
    row, column = len(reference_words), len(hypothesis_words)
    while row > 0 or column > 0:
        # Diagonal moves are preferred among equal-cost paths, so a misheard
        # word is one substitution rather than a deletion plus an insertion,
        # which would charge the same error to two different word classes.
        if row > 0 and column > 0:
            matched = reference_words[row - 1] == hypothesis_words[column - 1]
            if costs[row][column] == costs[row - 1][column - 1] + (0 if matched else 1):
                operations.append(
                    (
                        "hit" if matched else "substitution",
                        reference_words[row - 1],
                        hypothesis_words[column - 1],
                    )
                )
                row -= 1
                column -= 1
                continue
        if row > 0 and costs[row][column] == costs[row - 1][column] + 1:
            operations.append(("deletion", reference_words[row - 1], None))
            row -= 1
            continue
        operations.append(("insertion", None, hypothesis_words[column - 1]))
        column -= 1
    operations.reverse()
    return operations


def words_in_terms(terms):
    """The individual words making up a list of key terms.

    Terms can be phrases, and an error lands on one word at a time, so the
    split is what the per-word accounting needs.
    """
    words = set()
    for term in terms:
        words.update(normalize(term).split())
    return words


def count_occurrences(normalized_text, normalized_term):
    """How many times a term appears, respecting word boundaries.

    Word boundaries matter here: without them a term like "Ceph" would be
    credited to the word "cephalopod", overstating recall.
    """
    if not normalized_term:
        return 0
    pattern = r"(?<!\S)" + re.escape(normalized_term) + r"(?!\S)"
    return len(re.findall(pattern, normalized_text))


def load_manifest(manifest_path, extra_keyterms):
    """Read the JSONL manifest, resolving audio paths and merging key terms."""
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    utterances = []
    with open(manifest_path, encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"{manifest_path}:{line_number}: invalid JSON: {error}"
                ) from error
            for required in ("audio", "text"):
                if required not in entry:
                    raise SystemExit(
                        f"{manifest_path}:{line_number}: missing '{required}'"
                    )
            audio_path = Path(entry["audio"])
            if not audio_path.is_absolute():
                audio_path = manifest_dir / audio_path
            if not audio_path.exists():
                raise SystemExit(
                    f"{manifest_path}:{line_number}: audio not found: {audio_path}"
                )
            keyterms = list(entry.get("keyterms", [])) + list(extra_keyterms)
            utterances.append(
                {
                    "audio": audio_path,
                    "text": entry["text"],
                    "keyterms": keyterms,
                }
            )
    if not utterances:
        raise SystemExit(f"{manifest_path}: no utterances found")
    return utterances


def load_distractors(path, count, spoken_terms, seed):
    """Pick `count` terms from a word list, excluding any that are spoken.

    A distractor that happens to be in the audio would be scored as a false
    alarm when the model is in fact correct, so they are filtered out.
    """
    spoken_normalized = {normalize(term) for term in spoken_terms}
    candidates = []
    with open(path, encoding="utf-8", errors="ignore") as word_file:
        for line in word_file:
            term = line.strip()
            if not term or "," in term:
                continue
            if normalize(term) in spoken_normalized:
                continue
            candidates.append(term)
    if not candidates:
        raise SystemExit(f"{path}: no usable distractor terms")
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def evaluate(transcriber, utterances, distractors, load_audio):
    """Transcribe every utterance and total up the metrics.

    The transcriber's key terms are set per utterance, which is also a workout
    for the mid-stream setter: a real caller swaps terms as context changes.
    """
    total_reference_words = 0
    total_edits = 0
    total_spoken_occurrences = 0
    total_recalled_occurrences = 0
    total_false_alarms = 0
    total_hypothesis_words = 0
    biased_reference_words = 0
    biased_errors = 0
    unbiased_reference_words = 0
    unbiased_errors = 0

    for utterance in utterances:
        transcriber.set_keyterms(list(utterance["keyterms"]) + distractors)
        audio, sample_rate = load_audio(utterance["audio"])
        transcript = transcriber.transcribe_without_streaming(audio, sample_rate)
        hypothesis = " ".join(line.text for line in transcript.lines if line.text)

        reference_normalized = normalize(utterance["text"])
        hypothesis_normalized = normalize(hypothesis)
        reference_words = reference_normalized.split()
        hypothesis_words = hypothesis_normalized.split()
        total_reference_words += len(reference_words)
        total_hypothesis_words += len(hypothesis_words)

        alignment = word_alignment(reference_words, hypothesis_words)
        total_edits += sum(1 for operation, _, _ in alignment if operation != "hit")

        # Split the errors by whether they landed on a word the list was asking
        # for. A substitution or deletion is charged to the reference word it
        # destroyed; an insertion is charged to the word that appeared out of
        # nowhere, which is how a false alarm shows up as a key-term error
        # rather than as collateral damage.
        biasing_words = words_in_terms(utterance["keyterms"]) | words_in_terms(
            distractors
        )
        for operation, reference_word, hypothesis_word in alignment:
            if operation == "insertion":
                if hypothesis_word in biasing_words:
                    biased_errors += 1
                else:
                    unbiased_errors += 1
                continue
            if reference_word in biasing_words:
                biased_reference_words += 1
                if operation != "hit":
                    biased_errors += 1
            else:
                unbiased_reference_words += 1
                if operation != "hit":
                    unbiased_errors += 1

        # Every term in the live list is checked, not just the spoken ones: the
        # distractors are what reveal over-triggering.
        for term in set(utterance["keyterms"]) | set(distractors):
            term_normalized = normalize(term)
            spoken = count_occurrences(reference_normalized, term_normalized)
            heard = count_occurrences(hypothesis_normalized, term_normalized)
            total_spoken_occurrences += spoken
            total_recalled_occurrences += min(spoken, heard)
            total_false_alarms += max(0, heard - spoken)

    word_error_rate = (
        100.0 * total_edits / total_reference_words if total_reference_words else 0.0
    )
    recall = (
        100.0 * total_recalled_occurrences / total_spoken_occurrences
        if total_spoken_occurrences
        else float("nan")
    )
    false_alarms_per_1k = (
        1000.0 * total_false_alarms / total_hypothesis_words
        if total_hypothesis_words
        else 0.0
    )
    return {
        "wer": word_error_rate,
        "recall": recall,
        "spoken": total_spoken_occurrences,
        "recalled": total_recalled_occurrences,
        "false_alarms": total_false_alarms,
        "false_alarms_per_1k": false_alarms_per_1k,
        "biased_wer": (
            100.0 * biased_errors / biased_reference_words
            if biased_reference_words
            else float("nan")
        ),
        "unbiased_wer": (
            100.0 * unbiased_errors / unbiased_reference_words
            if unbiased_reference_words
            else float("nan")
        ),
        "biased_words": biased_reference_words,
        "unbiased_words": unbiased_reference_words,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Sweep key-term biasing strength and report recall, false "
        "alarms and WER.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON Lines file of {audio, text, keyterms} entries.",
    )
    parser.add_argument(
        "--model-path",
        help="Model directory. Downloads the model for --language if omitted.",
    )
    parser.add_argument(
        "--model-arch",
        default="tiny_streaming",
        help="Model architecture (default: tiny_streaming). Biasing needs one "
        "of the streaming architectures.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language to download a model for when --model-path is omitted.",
    )
    parser.add_argument(
        "--boosts",
        default="0,2,4,6,8,12",
        help="Comma-separated boost values to sweep (default: 0,2,4,6,8,12). "
        "Zero is the unbiased baseline.",
    )
    parser.add_argument(
        "--keyterms",
        default="",
        help="Comma-separated terms to add to every utterance's list.",
    )
    parser.add_argument(
        "--distractors-file",
        help="Word list (one term per line) to pad the key-term list with terms "
        "that are not in the audio, so false alarms are measured at a "
        "realistic list size.",
    )
    parser.add_argument(
        "--distractor-count",
        type=int,
        default=100,
        help="How many distractors to draw (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the distractor sample (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N utterances of the manifest.",
    )
    parser.add_argument(
        "--audio-cache-mb",
        type=float,
        default=512.0,
        help="How much decoded audio to hold in memory across the sweep "
        "(default: 512). Clips past the budget are re-read each row.",
    )
    parser.add_argument(
        "--disable-vad",
        action="store_true",
        help="Transcribe each clip as a single segment instead of letting the "
        "VAD split it. Right for short single-utterance clips (as in "
        "eval-librispeech.py), but it will wreck the numbers on clips longer "
        "than a sentence or two, because the streaming decoder is then asked "
        "for one very long hypothesis.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from moonshine_voice import ModelArch, Transcriber, get_model_for_language
    from moonshine_voice.utils import load_wav_file

    extra_keyterms = [term.strip() for term in args.keyterms.split(",") if term.strip()]
    utterances = load_manifest(args.manifest, extra_keyterms)
    if args.limit is not None:
        utterances = utterances[: args.limit]

    distractors = []
    if args.distractors_file:
        spoken_terms = [term for u in utterances for term in u["keyterms"]]
        distractors = load_distractors(
            args.distractors_file, args.distractor_count, spoken_terms, args.seed
        )
        print(f"Padding every list with {len(distractors)} distractors", file=sys.stderr)

    try:
        boosts = [float(value) for value in args.boosts.split(",") if value.strip()]
    except ValueError as error:
        raise SystemExit(f"--boosts must be numbers: {error}") from error
    if not boosts:
        raise SystemExit("--boosts is empty")

    arch = getattr(ModelArch, args.model_arch.upper())
    model_path = args.model_path
    if model_path:
        resolved_arch = arch
    else:
        model_path, resolved_arch = get_model_for_language(args.language, arch)
    print(
        f"Model {model_path} (arch={resolved_arch}), "
        f"{len(utterances)} utterances, boosts {boosts}",
        file=sys.stderr,
    )

    # Audio is cached across the sweep so the boost is the only thing that
    # changes between rows, and so decoding dominates the runtime. Samples are
    # held as compact float arrays and the cache stops growing past a budget,
    # because a corpus-sized manifest is hours of audio and Python floats cost
    # around eight times what the samples do.
    audio_cache = {}
    cached_samples = 0
    sample_budget = int(args.audio_cache_mb * 1024 * 1024 / 4)

    def load_audio(path):
        nonlocal cached_samples
        cached = audio_cache.get(path)
        if cached is None:
            samples, sample_rate = load_wav_file(str(path))
            if cached_samples + len(samples) <= sample_budget:
                audio_cache[path] = (array("f", samples), sample_rate)
                cached_samples += len(samples)
            return samples, sample_rate
        return cached[0], cached[1]

    header = (
        f"{'boost':>7}  {'WER %':>7}  {'other WER %':>12}  {'term WER %':>11}  "
        f"{'term recall %':>14}  {'false alarms/1k':>16}"
    )
    print(header)
    print("-" * len(header))
    for boost in boosts:
        # The boost is fixed at load time, so each row needs its own
        # transcriber; the key terms themselves are set per utterance.
        options = {"keyterm_boost": boost}
        if args.disable_vad:
            options["vad_threshold"] = 0.0
            options["vad_max_segment_duration"] = 100000.0
        transcriber = Transcriber(model_path, resolved_arch, options=options)
        try:
            metrics = evaluate(transcriber, utterances, distractors, load_audio)
        finally:
            transcriber.close()
        print(
            f"{boost:>7.1f}  {metrics['wer']:>7.2f}  {metrics['unbiased_wer']:>12.2f}  "
            f"{metrics['biased_wer']:>11.2f}  {metrics['recall']:>14.1f}  "
            f"{metrics['false_alarms_per_1k']:>16.2f}"
        )
        last_metrics = metrics

    print(
        f"\n'term WER' counts only the {last_metrics['biased_words']} reference "
        f"words your lists asked for, plus any word invented out of one; 'other "
        f"WER' counts the {last_metrics['unbiased_words']} words that were none "
        f"of the list's business. Raise the boost while the first falls; stop "
        f"when the second starts to rise, because that cost is paid on every "
        f"utterance whether a term was said or not.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
