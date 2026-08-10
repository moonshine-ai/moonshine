# Transcription

- [Getting Started](#getting-started)
- [Advanced](#advanced)
- [Transcription Event Flow](#transcription-event-flow)

## Getting Started

We have [examples](../examples.md) for most platforms so as a first step I recommend checking out what we have for the systems you're targeting.

Next, you'll need to [add the library to your project](adding-the-library.md). We aim to provide pre-built binaries for all major platforms using their native package managers. On Python this means a pip install, for Android it's a Maven package, and for MacOS and iOS we provide a Swift package through SPM.

For live speech, create a `MicTranscriber`, attach the callbacks you care about, call `load()` to fetch and open the model, then `start()` to begin listening:

```python
from moonshine_voice import MicTranscriber

mic = (
    MicTranscriber()
    .language("en")
    .on_text(lambda text: print(text, end="\r", flush=True))
    .on_line(lambda line: print(line.text))
)
mic.load()
mic.start()
```

`load()` blocks on first use while it downloads the recommended model for the language into a local cache; later launches reuse that cache and run offline. Pass `on_progress()` a handler if you want to drive a progress bar (attaching one also silences the default terminal progress bars):

```python
mic = (
    MicTranscriber()
    .language("en")
    .on_progress(lambda fraction, file: print(f"{fraction:.0%} {file}"))
    .on_text(lambda text: show_in_progress(text))
    .on_line(lambda line: append_line(line.text))
)
mic.load()
mic.start()
```

`on_text()` is called whenever the in-progress line changes; `on_line()` fires once per finished segment with the final line (see [`transcript_line_t`](../api/c-api.md#transcript_line_t) for the underlying fields). Call `stop()` when you're done listening, and `close()` (or use a context manager) to release the microphone and model.

By default the catalog's recommended model for the language is used — medium streaming for English. Most languages publish only one model. Override with `model_arch()` if you need a specific size, or point at a directory you already have with `models_from()` (see [Downloading Models](downloading-models.md)).

## Advanced

If you need to feed audio yourself — for example from a WAV file, a custom capture pipeline, or multiple streams — use the base `Transcriber` class. After [downloading the model files](downloading-models.md), place them somewhere the application can find them and pass that path in:

```python
transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
```

If the model isn't found, or if there's any other error, this will throw an exception with information about the problem. You can also check the console for logs from the core library, these are printed to `stderr` or your system's equivalent.

Now we'll create a listener that contains the app logic that you want triggered when the transcript updates, and attach it to your transcriber:

```python
class TestListener(TranscriptEventListener):
    def on_line_started(self, event):
        print(f"Line started: {event.line.text}")

    def on_line_text_changed(self, event):
        print(f"Line text changed: {event.line.text}")

    def on_line_completed(self, event):
        print(f"Line completed: {event.line.text}")

listener = TestListener()
transcriber.add_listener(listener)
```

`MicTranscriber` still supports `add_listener()` for line ids, speaker spans, word timings, and the moment a line starts; the `on_text` / `on_line` shortcuts cover the common case. Here's how to feed a `.wav` file into a `Transcriber` for testing:

```python
    audio_data, sample_rate = load_wav_file(wav_path)

    transcriber.start()

    # Loop through the audio data in chunks to simulate live streaming
    # from a microphone or other source.
    chunk_duration = 0.1
    chunk_size = int(chunk_duration * sample_rate)
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i: i + chunk_size]
        transcriber.add_audio(chunk, sample_rate)

    transcriber.stop()
```

The important things to notice here are:

- We create an array of mono audio data from a wav file, using the convenience `load_wav_file()` function that's part of the Moonshine library.
- We start the transcriber to activate its processing code.
- The loop adds audio in chunks. These chunks can be any length and any sample rate, the library takes care of all the housekeeping.
- As audio is added, the event listener you added will be called, giving information about the latest speech.

In a real application you'd be calling `add_audio()` from an audio handler that's receiving it from your source. Since the library can handle arbitrary durations and sample rates, just make sure it's mono and otherwise feed it in as-is.

The transcriber analyses the speech at a default interval of every 500ms of input. You can change this with the `update_interval` argument to the transcriber constructor (or `MicTranscriber.update_interval()`). For streaming models most of the work is done as the audio is being added, and it's automatically done at the end of a phrase, so changing this won't usually affect the workload or latency massively.

The interval is a floor rather than a fixed cadence. A pass has to cover at least as much audio as the last one took to make, up to ten intervals' worth, so a machine that cannot keep up transcribes in larger batches instead of falling further behind with every pass. Where there is processing time to spare this makes no difference and the interval governs as before.

The key takeaway is that you usually don't need to worry about the transcript data structure itself, the event system tells you when something important happens. You can manually trigger a transcript update by calling `update_transcription()` which returns a transcript object with all of the information about the current session if you do need to examine the state.

By calling `start()` and `stop()` on a transcriber (or stream) we're beginning and ending a session. Each session has one transcript document associated with it, and it is started fresh on every `start()` call, so you should make copies of any data you need from the transcript object before that.

The transcriber class also offers a simpler `transcribe_without_streaming()` method, for when you have an array of data from the past that you just want to analyse, such as a file or recording.

## Transcription Event Flow

The main communication channel between the library and your application is through events that are passed to any listener functions you have registered. There are five major event types:

- `LineStarted`. This is sent to listeners when the beginning of a new speech segment is detected. It may or may not contain any text, but since it's dispatched near the start of an utterance, that text is likely to change over time.
- `LineUpdated`. Called whenever any of the information about a line changes, including the duration, audio data, and text.
- `LineTextChanged`. Called only when the text associated with a line is updated. This is a subset of `LineUpdated` that focuses on the common need to refresh the text shown to users as often as possible to keep the experience interactive.
- `LineSpeakersChanged`. Only fired when the opt-in `identify_speakers` option is enabled. Called when the speaker spans attached to a line change. Unlike the other line events, this can fire for lines that are already complete, because the diarization algorithm keeps refining its speaker assignments as more audio arrives.
- `LineCompleted`. Sent when we detect that someone has paused speaking, and we've ended the current segment. The line data structure has the final values for the text and duration.

We offer some guarantees about these events:

- `LineStarted` is always called exactly once for any segment.
- `LineCompleted` is always called exactly once after `LineStarted` for any segment.
- `LineUpdated` and `LineTextChanged` will only ever be called after the `LineStarted` and before the `LineCompleted` events for a segment.
- Those update events are not guaranteed to be called (and in practice can be disabled by setting `update_interval` to a very large value).
- There will only be one line active at any one time for any given stream.
- Once `LineCompleted` has been called, the library will never alter that line's text, timing, or audio data again. The one exception is the line's speaker spans: when `identify_speakers` is enabled, those can be revised for recent audio (signaled by `LineSpeakersChanged`), since diarization re-clusters a sliding window of recent speech. Assignments for audio older than `diarization_cluster_window_sec` are frozen.
- If `stop()` is called on a transcriber or stream, any active lines will have `LineCompleted` called.
- Each line has a 64-bit `lineId` that is designed to be unique enough to avoid collisions.
- This `lineId` remains the same for the line over time, from the first `LineStarted` event onwards.
