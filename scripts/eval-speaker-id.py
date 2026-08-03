import moonshine_voice

import numpy as np

from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

from datasets import Audio, load_dataset

import argparse
import io
import soundfile

# Starting point for v1 of speaker identification (embedding model +
# online clusterer, per-line speaker IDs):
# Average speaker confusion: 30.67%
#
# v1 with shorter segments unable to create new clusters:
# Average speaker confusion: 26.44%
#
# v2 uses the cpp-annote port of the pyannote community-1 pipeline and
# reports per-line speaker *spans*, which are scored directly against the
# reference below.

parser = argparse.ArgumentParser()
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--sample-count", type=int, default=10)
parser.add_argument("--model-arch", type=int, default=0)
parser.add_argument("--options", type=str, default=None)
args = parser.parse_args()

# Speaker identification is opt-in, and transcription itself isn't needed
# for this evaluation. The diarization models are a download rather than part
# of the library (docs/diarization-models.md), so fetch them first.
options = {
    "skip_transcription": True,
    "identify_speakers": True,
    "diarization_model_dir": moonshine_voice.get_diarization_model(),
}
if args.options is not None:
    for option in args.options.split(","):
        key, value = option.split("=")
        options[key] = value


def describe_error_rate(components):
    """The three parts of DER, plus their sum, as percentages of reference speech.

    Confusion alone answers "when we heard someone, did we credit the right
    speaker", which is what this script tracked historically. It says nothing
    about speech we dropped or invented, so a setting that simply finds less
    speech can improve confusion while making the diarization worse. Reporting
    all three keeps that trade visible.
    """
    total = components["total"]
    if not total:
        return "no reference speech"
    missed = components.get("missed detection", 0.0)
    false_alarm = components.get("false alarm", 0.0)
    confusion = components.get("confusion", 0.0)
    error_rate = (missed + false_alarm + confusion) / total
    return (
        f"DER {error_rate:.2%} "
        f"(confusion {confusion / total:.2%}, "
        f"missed {missed / total:.2%}, "
        f"false alarm {false_alarm / total:.2%})"
    )


def read_sample_audio(sample):
    """Mono float32 audio and its sample rate, decoded with soundfile.

    The dataset library's own audio decoder wants torchcodec, and so all of
    torch, to do this. Reading the bytes here keeps the evaluation dependencies
    to what scoring actually needs.
    """
    audio = sample["audio"]
    if audio.get("bytes"):
        samples, sample_rate = soundfile.read(
            io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True
        )
    else:
        samples, sample_rate = soundfile.read(
            audio["path"], dtype="float32", always_2d=True
        )
    # Telephone recordings sometimes keep each side on its own channel, and
    # diarizing one mixed stream is what the library is given in practice.
    return samples.mean(axis=1).astype(np.float32), sample_rate


ds = load_dataset("diarizers-community/callhome", "eng")
data = ds["data"].cast_column("audio", Audio(decode=False))

model_path, model_arch = moonshine_voice.get_model_for_language("en", args.model_arch)

metric = DiarizationErrorRate()

total_confusion = 0.0
for sample_index in range(args.start_index, args.start_index + args.sample_count):
    # Create a new transcriber for each sample to avoid remembering previous speaker IDs.
    transcriber = moonshine_voice.Transcriber(
        model_path=model_path, model_arch=model_arch, options=options
    )
    sample = data[sample_index]
    audio, sample_rate = read_sample_audio(sample)
    transcriber.start()
    transcriber.add_audio(audio, sample_rate)
    transcript = transcriber.stop()
    reference = Annotation()
    timestamps_start = sample["timestamps_start"]
    timestamps_end = sample["timestamps_end"]
    reference = Annotation()
    ref_unique_speakers = set()
    for i in range(len(timestamps_start)):
        speaker_index = sample['speakers'][i]
        start_time = timestamps_start[i]
        end_time = timestamps_end[i]
        ref_unique_speakers.add(speaker_index)
        reference[Segment(start_time, end_time)] = f"sample_{sample_index}_{speaker_index}"
    # The hypothesis is built from the per-line speaker spans. Span times are
    # absolute stream times already clipped to each line, so they can be used
    # directly as diarization segments.
    hypothesis = Annotation()
    hyp_unique_speakers = set()
    hyp_span_count = 0
    for line in transcript.lines:
        for span in line.speaker_spans or []:
            hyp_unique_speakers.add(span.speaker_index)
            hyp_span_count += 1
            hypothesis[Segment(span.start_time, span.start_time + span.duration)] = (
                f"sample_{sample_index}_{span.speaker_index}"
            )
    sample_metrics = metric(reference, hypothesis, detailed=True)
    confusion = sample_metrics["confusion"]
    total = sample_metrics["total"]
    print(f"Speaker confusion: {confusion / total:.2%}")
    print(f"  {describe_error_rate(sample_metrics)}")
    print(f"Reference unique speakers: {ref_unique_speakers}")
    print(f"Hypothesis unique speakers: {hyp_unique_speakers}")
    print(f"Reference line count: {len(reference)}")
    print(f"Hypothesis line count: {len(transcript.lines)}, span count: {hyp_span_count}")

confusion = metric["confusion"]
total = metric["total"]

print(f"Average speaker confusion: {confusion / total:.2%}")
# The accumulated metric indexes components but is not a dict, so copy the
# pieces out before summarising them.
print(
    "Overall "
    + describe_error_rate(
        {
            name: metric[name]
            for name in ("total", "missed detection", "false alarm", "confusion")
        }
    )
)
if args.options is not None:
    print(f"Options: {args.options}")
