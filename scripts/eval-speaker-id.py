import moonshine_voice

import numpy as np

from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

from datasets import load_dataset

import argparse

# Starting point for v1 of speaker identification:
# DER: 36.96%
# DER: 57.26%
# DER: 43.60%
# DER: 54.25%
# DER: 55.97%
# DER: 35.99%
# DER: 69.83%
# DER: 49.20%
# DER: 49.95%
# DER: 73.80%
# Average DER: 52.68%

parser = argparse.ArgumentParser()
parser.add_argument("--sample-count", type=int, default=10)
parser.add_argument("--model-arch", type=int, default=0)
args = parser.parse_args()

ds = load_dataset("diarizers-community/callhome", "eng")

model_path, model_arch = moonshine_voice.get_model_for_language("en", args.model_arch)

transcriber = moonshine_voice.Transcriber(model_path=model_path, model_arch=model_arch)

total_der = 0.0
for sample_index in range(args.sample_count):
    sample = ds['data'][sample_index]
    audio = sample['audio']['array'].astype(np.float32)
    sample_rate = 16000
    transcriber.start()
    transcriber.add_audio(audio, sample_rate)
    transcript = transcriber.stop()
    reference = Annotation()
    timestamps_start = sample['timestamps_start']
    timestamps_end = sample['timestamps_end']
    reference = Annotation()
    for i in range(len(timestamps_start)):
        reference[Segment(timestamps_start[i], timestamps_end[i])] = f"sample_{sample_index}_{sample['speakers'][i]}"
    hypothesis = Annotation()
    for line in transcript.lines:
        hypothesis[Segment(line.start_time, line.start_time + line.duration)] = f"sample_{sample_index}_{line.speaker_index}"
    metric = DiarizationErrorRate()
    sample_der = metric(reference, hypothesis)
    total_der += sample_der
    print(f"DER: {sample_der:.2%}")
print(f"Average DER: {total_der / args.sample_count:.2%}")