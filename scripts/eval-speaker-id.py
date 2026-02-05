import moonshine_voice

import numpy as np

from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

from datasets import load_dataset

import argparse

# Starting point for v1 of speaker identification:
# Speaker confusion: 17.37%
# Speaker confusion: 34.01%
# Speaker confusion: 20.66%
# Speaker confusion: 18.13%
# Speaker confusion: 37.92%
# Speaker confusion: 15.27%
# Speaker confusion: 44.41%
# Speaker confusion: 24.09%
# Speaker confusion: 25.11%
# Speaker confusion: 45.82%
# Average speaker confusion: 30.67%
#
# With shorter segments unable to create new clusters:
# Speaker confusion: 17.37%
# Speaker confusion: 34.01%
# Speaker confusion: 19.70%
# Speaker confusion: 25.45%
# Speaker confusion: 35.07%
# Speaker confusion: 15.78%
# Speaker confusion: 28.11%
# Speaker confusion: 19.82%
# Speaker confusion: 21.81%
# Speaker confusion: 32.78%
# Average speaker confusion: 26.44%

parser = argparse.ArgumentParser()
parser.add_argument("--sample-count", type=int, default=10)
parser.add_argument("--model-arch", type=int, default=0)
parser.add_argument("--options", type=str, default=None)
args = parser.parse_args()

options = {"skip_transcription": True}
if args.options is not None:
    for option in args.options.split(","):
        key, value = option.split("=")
        options[key] = value

ds = load_dataset("diarizers-community/callhome", "eng")

model_path, model_arch = moonshine_voice.get_model_for_language("en", args.model_arch)

transcriber = moonshine_voice.Transcriber(
    model_path=model_path, model_arch=model_arch, options=options)

metric = DiarizationErrorRate()

total_confusion = 0.0
for sample_index in range(args.sample_count):
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
    for i in range(len(timestamps_start)):
        reference[Segment(timestamps_start[i], timestamps_end[i])] = (
            f"sample_{sample_index}_{sample['speakers'][i]}"
        )
    hypothesis = Annotation()
    for line in transcript.lines:
        hypothesis[Segment(line.start_time, line.start_time + line.duration)] = (
            f"sample_{sample_index}_{line.speaker_index}"
        )
    sample_metrics = metric(reference, hypothesis, detailed=True)
    confusion = sample_metrics["confusion"]
    total = sample_metrics["total"]
    print(f"Speaker confusion: {confusion / total:.2%}")

confusion = metric["confusion"]
total = metric["total"]

print(f"Average speaker confusion: {confusion / total:.2%}")
