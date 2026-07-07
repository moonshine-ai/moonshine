import moonshine_voice

import numpy as np

from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

from datasets import load_dataset

import argparse

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
# for this evaluation.
options = {"skip_transcription": True, "identify_speakers": True}
if args.options is not None:
    for option in args.options.split(","):
        key, value = option.split("=")
        options[key] = value

ds = load_dataset("diarizers-community/callhome", "eng")

model_path, model_arch = moonshine_voice.get_model_for_language("en", args.model_arch)

metric = DiarizationErrorRate()

total_confusion = 0.0
for sample_index in range(args.start_index, args.start_index + args.sample_count):
    # Create a new transcriber for each sample to avoid remembering previous speaker IDs.
    transcriber = moonshine_voice.Transcriber(
        model_path=model_path, model_arch=model_arch, options=options
    )
    sample = ds["data"][sample_index]
    audio = sample["audio"]["array"].astype(np.float32)
    sample_rate = 16000
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
    print(f"Reference unique speakers: {ref_unique_speakers}")
    print(f"Hypothesis unique speakers: {hyp_unique_speakers}")
    print(f"Reference line count: {len(reference)}")
    print(f"Hypothesis line count: {len(transcript.lines)}, span count: {hyp_span_count}")

confusion = metric["confusion"]
total = metric["total"]

print(f"Average speaker confusion: {confusion / total:.2%}")
