# Benchmarks

- [Whisper Comparisons](#whisper-comparisons)

The core library includes a benchmarking tool that simulates processing live audio by loading a .wav audio file and feeding it in chunks to the model. To run it:

<!-- doc-test: skip -->
```bash
cd core
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
./benchmark
```

This will report the absolute time taken to process the audio, what percentage of the audio file's duration that is, and the average latency for a response.

The percentage is helpful because it approximates how much of a compute load the model will be on your hardware. For example, if it shows 20% then that means the speech processing will take a fifth of the compute time when running in your application, leaving 80% for the rest of your code.

The latency metric needs a bit of explanation. What most applications care about is how soon they are notified about a phrase after the user has finished talking, since this determines how fast the product can respond. As with any user interface, the time between speech ending and the app doing something determines how responsive the voice interface feels, with a goal of keeping it below 200ms. The latency figure logged here is the average time between when the library determines the user has stopped talking and the delivery of the final transcript of that phrase to the client. This is where streaming models have the most impact, since they do a lot of their work upfront, while speech is still happening, so they can usually finish very quickly.

By default the benchmark binary uses the Tiny English model and the `two_cities.wav` recording from this repository's `test-assets` folder, which is why it's run from the build directory, but you can pass in the `--model-path`, `--model-arch`, and `--wav-path` parameters to choose [a model you've downloaded](downloading-models.md) or a different recording.

You can also choose how often the transcript should be updated using the `--transcription-interval` argument. This defaults to 0.5 seconds, but the right value will depend on how fast your application needs updates. Longer intervals reduce the compute required a bit, at the cost of slower updates.

The MacBook Pro, Pixel 10a, and iPad (A16) Tiny / Small / Medium Streaming cells in the comparison table use the same latency metric, measured by [`scripts/test-mobile-latency.sh`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/test-mobile-latency.sh) (also run from [`scripts/build-all-platforms.sh`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/build-all-platforms.sh)). That script downloads the models from the CDN, feeds `two_cities.wav` in small chunks as fast as the device can process, and averages `lastTranscriptionLatencyMs` over completed lines. Re-measure and refresh the README with the command below; a cell is rewritten only when the new average differs from the published value by more than 5%:

<!-- doc-test: skip -->
```bash
./scripts/test-mobile-latency.sh --update-readme
```

That writes whatever a single run measured, which is fine for the iPad — three consecutive runs there landed within a millisecond of each other — but the Mac and the Pixel wander enough that one run is not a number worth publishing. Medium Streaming on the Mac varied between 56 and 83ms across three runs, and on the Pixel between 383 and 432ms, mostly according to how recently the machine had been busy. The published cells are medians of three runs taken with the device given time to cool between them, so expect a single run to disagree with them a little. The Linux x86 and Raspberry Pi 5 columns are measured separately and were last taken before the build-optimization fix described in the changelog, so they read pessimistically until someone with that hardware refreshes them.

## Whisper Comparisons

For platforms that support Python, you can run the [`scripts/run-benchmarks.py`](https://github.com/moonshine-ai/moonshine/blob/main/scripts/run-benchmarks.py) script which will evaluate similar metrics, with the advantage that it can also download the models so you don't need to worry about path handling.

It also evaluates equivalent Whisper models. This is a pretty opinionated benchmark that looks at the latency and total compute cost
of the two families of models in a situation that is representative of many common
real-time voice applications' requirements:

- Speech needs to be responded to as quickly as possible once a user completes a phrase.
- The phrases are of durations between a range of one to ten seconds.

These are very different requirements from bulk offline processing scenarios, where the
overall throughput of the system is more important, and so the latency on a single
segment of speech is less important than the overall throughput of the system. This
allows optimizations like batch processing.

We are not claiming that Whisper is not a great model for offline processing, but we
do want to highlight the advantages we that Moonshine offers for live speech
applications with real-time latency requirements.

The experimental setup is as follows:

- We use the two_cities.wav audio file as a test case, since it has a mix of short
  and long phrases. You can vary this by passing in your own audio file with the
  --wav_path argument.
- We use the Moonshine Tiny, Base, Tiny Streaming, Small Streaming, and Medium
  Streaming models.
- We compare these to the Whisper Tiny, Base, Small, and Large v3 models. Since the
  Moonshine Medium Streaming model achieves lower WER than Whisper Large v3 we compare
  those two, otherwise we compare each with their namesake.
- We use the Moonshine VAD segmenter to split the audio into phrases, and feed each
  phrase to Whisper for transcription.
- Response latency for both models is measured as the time between a phrase being
  identified as complete by the VAD segmenter and the transcribed text being returned.
  For Whisper this means the full transcription time, but since the Moonshine models
  are streaming we can do a lot of the work while speech is still happening, so the
  latency is much lower.
- We measure the total compute cost of the models by totalling the duration of the
  audio processing times for each model, and then expressing that as a percentage of the
  total audio duration. This is the inverse of the commonly used real-time factor (RTF)
  metric, but it reflects the compute load required for a real-time application.
- We're using faster-whisper for Whisper, since that seems to provide the best
  cross-platform performance. We're also sticking with the CPU, since most applications
  can't rely on GPU or NPU acceleration being present on all the platforms they target.
  We know there are a lot of great GPU/NPU-accelerated Whisper implementations out there,
  but these aren't portable enough to be useful for the applications we care about.
