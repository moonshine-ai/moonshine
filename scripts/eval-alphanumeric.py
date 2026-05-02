"""Evaluate AlphanumericListener on the test-assets/alphanumeric dataset.

Each subfolder of ``test-assets/alphanumeric`` is named after the spoken word
in its WAV files (``a``, ``b``, ..., ``z``, ``zero``, ``one``, ..., ``nine``).
For every clip we run :class:`Transcriber.transcribe_without_streaming` (with
``MOONSHINE_FLAG_SPELLING_MODE`` so the C++ transcriber's spelling-CNN fusion
runs end-to-end), synthesize a ``LineCompleted`` event from the resulting
line(s), and feed that to a fresh :class:`AlphanumericListener`. The first
``CHARACTER`` event emitted by the listener is taken as the prediction.

Usage::

    python scripts/eval-alphanumeric.py
    python scripts/eval-alphanumeric.py --dataset-dir test-assets/alphanumeric
    python scripts/eval-alphanumeric.py --no-spelling-model
    python scripts/eval-alphanumeric.py --verbose
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from moonshine_voice import (
    Transcriber,
    get_model_for_language,
    get_spelling_model_path,
    load_wav_file,
    string_to_model_arch,
)
from moonshine_voice.alphanumeric_listener import (
    AlphanumericEvent,
    AlphanumericEventType,
    AlphanumericListener,
    AlphanumericMatcher,
)
from moonshine_voice.moonshine_api import TranscriptLine
from moonshine_voice.transcriber import LineCompleted, MOONSHINE_FLAG_SPELLING_MODE


E_SET = ("b", "c", "d", "e", "g", "p", "t", "v", "z")


def folder_name_to_expected_char(name: str, matcher: AlphanumericMatcher) -> str:
    """Resolve a folder name (e.g. ``"a"``, ``"eight"``) to its character."""
    match = matcher.classify(name)
    if not match.is_character or match.character is None:
        raise ValueError(f"Cannot resolve folder name {name!r} to a character")
    return match.character


def predict_character(
    transcriber: Transcriber,
    audio: np.ndarray,
    sample_rate: int,
    transcribe_flags: int = 0,
) -> tuple[str | None, str]:
    """Run transcription and return predictions for a single clip.

    Returns ``(predicted_char, raw_transcript_text)``:

    * ``predicted_char`` is the first ``CHARACTER`` event emitted by a
      fresh :class:`AlphanumericListener` fed with synthesized
      ``LineCompleted`` events, or ``None`` if nothing was recognised.
      When ``transcribe_flags`` includes
      ``MOONSHINE_FLAG_SPELLING_MODE`` the character is the C++ fusion
      result; otherwise it's the matcher's lexical hit.
    * ``raw_transcript_text`` is ``" | "``-joined raw STT line text
      (already the resolved character when fusion is enabled).
    """
    transcript = transcriber.transcribe_without_streaming(
        audio, sample_rate, flags=transcribe_flags,
    )

    captured: list[AlphanumericEvent] = []

    def on_event(event: AlphanumericEvent) -> None:
        captured.append(event)

    listener = AlphanumericListener(on_event)

    raw_texts: list[str] = []
    for idx, line in enumerate(transcript.lines):
        raw_texts.append(line.text)
        synth_line = TranscriptLine(
            text=line.text,
            start_time=line.start_time,
            duration=line.duration,
            line_id=idx,
            is_complete=True,
            audio_data=None,
        )
        listener(LineCompleted(line=synth_line, stream_handle=0))

    predicted: str | None = None
    for event in captured:
        if event.type is AlphanumericEventType.CHARACTER:
            predicted = event.character
            break

    return predicted, " | ".join(t for t in raw_texts if t)


def print_confusion_matrix(
    matrix: dict[str, dict[str, int]],
    labels: tuple[str, ...],
    title: str,
) -> None:
    """Pretty-print a confusion matrix with ``labels`` along both axes."""
    print()
    print(title)
    print("=" * len(title))
    print("(rows = expected, cols = predicted; '?' = unrecognised)")
    print()

    col_labels = list(labels) + ["?", "other"]
    cell_w = max(4, max(len(c) for c in col_labels) + 1)
    header = " " * 8 + "".join(f"{c:>{cell_w}}" for c in col_labels) + f"  {'tot':>5}  {'acc':>6}"
    print(header)

    for row in labels:
        row_counts = matrix.get(row, {})
        cells = []
        for col in col_labels:
            cells.append(f"{row_counts.get(col, 0):>{cell_w}}")
        total = sum(row_counts.values())
        correct = row_counts.get(row, 0)
        acc = (correct / total * 100.0) if total else 0.0
        print(
            f"  {row:>4}  "
            + "".join(cells)
            + f"  {total:>5}  {acc:>5.1f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="test-assets/alphanumeric",
        help="Directory containing one subfolder per spoken word.",
    )
    parser.add_argument(
        "--language", type=str, default="en", help="Language code for the model."
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default=None,
        help="Optional model architecture override (e.g. tiny, base).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override the model directory (requires --model-arch).",
    )
    parser.add_argument(
        "--options",
        type=str,
        default=None,
        help="Comma-separated key=value transcriber options.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-clip predictions.",
    )
    parser.add_argument(
        "--spelling-model-path",
        type=str,
        default=None,
        help=(
            "Override the spelling-CNN .ort path passed to the C++ "
            "transcriber. Defaults to ``get_spelling_model_path(language)``."
        ),
    )
    parser.add_argument(
        "--no-spelling-model",
        action="store_true",
        help=(
            "Disable the C++ spelling-CNN fusion path entirely. The "
            "matcher's lexical lookup is then the only signal used."
        ),
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    if args.model_path is not None:
        if args.model_arch is None:
            print("--model-arch is required when --model-path is set", file=sys.stderr)
            return 1
        model_path = args.model_path
        model_arch = string_to_model_arch(args.model_arch)
    else:
        wanted_arch = (
            string_to_model_arch(args.model_arch) if args.model_arch else None
        )
        model_path, model_arch = get_model_for_language(
            wanted_language=args.language, wanted_model_arch=wanted_arch
        )

    options: dict[str, str] = {"vad_threshold": "0.0"}
    if args.options:
        for option in args.options.split(","):
            key, value = option.split("=")
            options[key.strip()] = value.strip()

    spelling_model_path: str | None = None
    if not args.no_spelling_model:
        spelling_model_path = args.spelling_model_path
        if spelling_model_path is None:
            try:
                spelling_model_path = get_spelling_model_path(args.language)
            except Exception as e:
                print(
                    f"Spelling model lookup failed ({e!r}); continuing "
                    "without C++ fusion.",
                    file=sys.stderr,
                )
        if spelling_model_path is not None:
            options["spelling_model_path"] = spelling_model_path
            print(f"Spelling model: {spelling_model_path}")
        else:
            print("Spelling model: <none> (matcher-only)")
    else:
        print("Spelling model: disabled (--no-spelling-model)")

    transcribe_flags = (
        MOONSHINE_FLAG_SPELLING_MODE if spelling_model_path is not None else 0
    )

    print(f"Loading model from {model_path}")
    transcriber = Transcriber(
        model_path=model_path, model_arch=model_arch, options=options
    )

    matcher = AlphanumericMatcher()

    folders = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not folders:
        print(f"No subfolders found under {dataset_dir}", file=sys.stderr)
        return 1

    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = 0
    correct = 0
    unrecognised = 0

    e_set_set = set(E_SET)

    for folder in folders:
        try:
            expected = folder_name_to_expected_char(folder.name, matcher)
        except ValueError as e:
            print(f"Skipping {folder.name}: {e}", file=sys.stderr)
            continue

        wav_files = sorted(folder.glob("*.wav"))
        if not wav_files:
            continue

        for wav_path in wav_files:
            audio, sample_rate = load_wav_file(wav_path)
            audio_np = np.asarray(audio, dtype=np.float32)
            predicted, raw_text = predict_character(
                transcriber,
                audio_np,
                sample_rate,
                transcribe_flags=transcribe_flags,
            )

            total += 1
            per_class_total[expected] += 1

            if predicted is None:
                unrecognised += 1
            elif predicted == expected:
                correct += 1
                per_class_correct[expected] += 1

            if expected in e_set_set:
                if predicted is None:
                    col = "?"
                elif predicted in e_set_set:
                    col = predicted
                else:
                    col = "other"
                confusion[expected][col] += 1

            if args.verbose:
                marker = "OK" if predicted == expected else "  "
                pred_str = predicted if predicted is not None else "<none>"
                print(
                    f"  [{marker}] {folder.name:>5} -> {pred_str!r}"
                    f"  (raw: {raw_text!r})"
                    f"  [{wav_path.name}]",
                    flush=True
                )

    print()
    print("=" * 60)
    if total:
        print(f"Overall accuracy: {correct}/{total} = {correct / total:.2%}")
        print(
            f"Unrecognised:     {unrecognised}/{total} = "
            f"{unrecognised / total:.2%}"
        )
    print()
    print("Per-class accuracy")
    print("------------------")
    for label in sorted(per_class_total):
        n = per_class_total[label]
        c = per_class_correct[label]
        print(f"  {label!r:>5}: {c:>3}/{n:<3}  {c / n:.2%}")

    print_confusion_matrix(
        {row: dict(cols) for row, cols in confusion.items()},
        E_SET,
        title="E-set confusion matrix",
    )

    e_total = sum(per_class_total[c] for c in E_SET)
    e_correct = sum(per_class_correct[c] for c in E_SET)
    if e_total:
        print()
        print(
            f"E-set accuracy: {e_correct}/{e_total} = {e_correct / e_total:.2%}"
        )

    transcriber.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
