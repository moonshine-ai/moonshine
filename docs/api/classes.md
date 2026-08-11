# Classes

- [Transcriber](#transcriber)
    - [`__init__()`](#transcriber-init)
    - [`transcribe_without_streaming()`](#transcriber-transcribe-without-streaming)
    - [`start()`](#transcriber-start)
    - [`stop()`](#transcriber-stop)
    - [`add_audio()`](#transcriber-add-audio)
    - [`update_transcription`](#transcriber-update-transcription)
    - [`create_stream()`](#transcriber-create-stream)
    - [`add_listener()`](#transcriber-add-listener)
    - [`remove_listener()`](#transcriber-remove-listener)
    - [`remove_all_listeners()`](#transcriber-remove-all-listeners)
    - [`set_keyterms()`](#transcriber-set-keyterms)
    - [`set_context()`](#transcriber-set-context)
- [MicTranscriber](#mictranscriber)
    - [`__init__()`](#mictranscriber-init)
    - [`language()`](#mictranscriber-language)
    - [`model_arch()`](#mictranscriber-model-arch)
    - [`models_from()`](#mictranscriber-models-from)
    - [`use_transcriber()`](#mictranscriber-use-transcriber)
    - [`update_interval()`](#mictranscriber-update-interval)
    - [`options()`](#mictranscriber-options)
    - [`spelling_model()`](#mictranscriber-spelling-model)
    - [`transcribe_flags()`](#mictranscriber-transcribe-flags)
    - [`device()`](#mictranscriber-device)
    - [`samplerate()`](#mictranscriber-samplerate)
    - [`channels()`](#mictranscriber-channels)
    - [`blocksize()`](#mictranscriber-blocksize)
    - [`on_text()`](#mictranscriber-on-text)
    - [`on_line()`](#mictranscriber-on-line)
    - [`on_error()`](#mictranscriber-on-error)
    - [`on_progress()`](#mictranscriber-on-progress)
    - [`load()`](#mictranscriber-load)
    - [`set_keyterms()`](#mictranscriber-set-keyterms)
    - [`set_context()`](#mictranscriber-set-context)
    - [`start()`](#mictranscriber-start)
    - [`mute()`](#mictranscriber-mute)
    - [`stop()`](#mictranscriber-stop)
    - [`close()`](#mictranscriber-close)
- [Stream](#stream)
- [TranscriptEventListener](#transcripteventlistener)
- [AgentFlow](#agentflow)
    - [`__init__()`](#agentflow-init)
    - [`language()`](#agentflow-language)
    - [`model_arch()`](#agentflow-model-arch)
    - [`voice()`](#agentflow-voice)
    - [`speech_options()`](#agentflow-speech-options)
    - [`models_from()`](#agentflow-models-from)
    - [`microphone()`](#agentflow-microphone)
    - [`speech()`](#agentflow-speech)
    - [`output_device()`](#agentflow-output-device)
    - [`trigger_threshold()`](#agentflow-trigger-threshold)
    - [`use_embeddings()`](#agentflow-use-embeddings)
    - [`beeps()`](#agentflow-beeps)
    - [`spell_feedback()`](#agentflow-spell-feedback)
    - [`barge_in()`](#agentflow-barge-in)
    - [`log_io()`](#agentflow-log-io)
    - [`debug()`](#agentflow-debug)
    - [`on_progress()`](#agentflow-on-progress)
    - [`on_heard()`](#agentflow-on-heard)
    - [`on_said()`](#agentflow-on-said)
    - [`on_error()`](#agentflow-on-error)
    - [`speak_with()`](#agentflow-speak-with)
    - [`use_text_to_speech()`](#agentflow-use-text-to-speech)
    - [`use_mic_transcriber()`](#agentflow-use-mic-transcriber)
    - [`listen_for()`](#agentflow-listen-for)
    - [`unregister_flow()`](#agentflow-unregister-flow)
    - [`always()`](#agentflow-always)
    - [`otherwise()`](#agentflow-otherwise)
    - [`load()`](#agentflow-load)
    - [`start_listening()`](#agentflow-start-listening)
    - [`stop_listening()`](#agentflow-stop-listening)
    - [`handle_utterance()`](#agentflow-handle-utterance)
    - [`cancel()`](#agentflow-cancel)
    - [`say()`](#agentflow-say)
    - [`close()`](#agentflow-close)
- [Dialog](#dialog)
    - [`say()`](#dialog-say)
    - [`ask()`](#dialog-ask)
    - [`confirm()`](#dialog-confirm)
    - [`choose()`](#dialog-choose)
    - [`cancel()`](#dialog-cancel)
    - [`restart()`](#dialog-restart)
    - [`replay_last_prompt()`](#dialog-replay-last-prompt)
- [TextToSpeech](#texttospeech)
    - [`__init__()`](#texttospeech-init)
    - [`language()`](#texttospeech-language)
    - [`voice()`](#texttospeech-voice)
    - [`models_from()`](#texttospeech-models-from)
    - [`cloning()`](#texttospeech-cloning)
    - [`options()`](#texttospeech-options)
    - [`output_device()`](#texttospeech-output-device)
    - [`volume()`](#texttospeech-volume)
    - [`on_progress()`](#texttospeech-on-progress)
    - [`debug()`](#texttospeech-debug)
    - [`load()`](#texttospeech-load)
    - [`clone_from()`](#texttospeech-clone-from)
    - [`start_cloning()`](#texttospeech-start-cloning)
    - [`synthesize()`](#texttospeech-synthesize)
    - [`say()`](#texttospeech-say)
    - [`stop()`](#texttospeech-stop)
    - [`wait()`](#texttospeech-wait)
    - [`is_talking()`](#texttospeech-is-talking)
    - [`close()`](#texttospeech-close)
- [GraphemeToPhonemizer](#graphemetophonemizer)
    - [`__init__()`](#graphemetophonemizer-init)
    - [`to_ipa()`](#graphemetophonemizer-to-ipa)
    - [`close()`](#graphemetophonemizer-close)


## Transcriber

Handles the speech to text pipeline.

- <a id="transcriber-init"></a>`__init__()`: Loads and initializes the transcriber.
  - `model_path`: The path to the directory holding the component model files needed for the complete flow. Note that this is a path to the **folder**, not an individual **file**. You can download and get a path to a cached version of the standard models using the [download_model()](../using/downloading-models.md) function.
  - `model_arch`: The architecture of the model to load, from the selection defined in `ModelArch`.
  - `update_interval`: By default the transcriber will periodically run text transcription as new audio data is fed, so that update events can be triggered. This value is how often the speech to text model should be run. You can set this to a large duration to suppress updates between a line starting and ending, but because the streaming models do a lot of their work before the final speech to text stage, this may not reduce overall latency by much.
  - <a id="transcriber-options"></a>`options`: Advanced configuration as a string-keyed map (Python also accepts bools and numbers). See [Options → Speech to Text](options.md#speech-to-text) for every key, including VAD, diarization, key terms, and shared keys such as `log_api_calls` and `ort_providers`.

- <a id="transcriber-transcribe-without-streaming"></a>`transcribe_without_streaming()`: A convenience function to extract text from a non-live audio source, such as a file. We optimize for streaming use cases, so you're probably better off using libraries that specialize in bulk, batched transcription if you use this a lot and have performance constraints. This will still call any registered event listeners as it processes the lines, so this can be useful to test your application using pre-recorded files, or to easily integrate offline audio sources.
  - `audio_data`: An array of 32-bit float values, representing mono PCM audio between -1.0 and 1.0, to be analyzed for speech.
  - `sample_rate`: The number of samples per second. The library uses this to convert to its working rate (16KHz) internally.
  - `flags`: Integer, a bitwise OR of flags. Currently the only supported flag is `MOONSHINE_FLAG_SPELLING_MODE`, which applies alphanumeric-spelling fusion when a spelling model was loaded.

- <a id="transcriber-start"></a>`start()`: Begins a new transcription session. You need to call this after you've created the `Transcriber` and before you add any audio.
- <a id="transcriber-stop"></a>`stop()`: Ends a transcription session. If a speech segment was still active, it's marked as complete and the appropriate event handlers are called.
- <a id="transcriber-add-audio"></a>`add_audio()`: Call this every time you have a new chunk of audio from your input, to begin processing. The size and sample rate of the audio should be whatever's natural for your source, since the library will handle all conversions.
  - `audio_data`: Array of 32-bit floats representing a mono PCM chunk of audio.
  - `sample_rate`: How many samples per second are present in the input audio. The library uses this to convert the data to its preferred rate.
- <a id="transcriber-update-transcription"></a>`update_transcription`: The transcript is usually updated periodically as audio data is added, but if you need to trigger one yourself, for example when a user presses refresh, or want access to the complete transcript, you can call this manually.
  - `flags`: Integer holding flags that are combined using bitwise or (`|`).
    - `MOONSHINE_FLAG_FORCE_UPDATE`: By default the transcriber returns a cached version of the transcript if less than 200ms of new audio has come in since the last transcription, but by setting this you can ensure that a transcription happens regardless.

- <a id="transcriber-create-stream"></a>`create_stream()`: If your application is taking audio input from multiple sources, for example a microphone and system audio, then you'll want to create multiple streams on a single transcriber to avoid loading multiple copies of the models. Each stream has its own transcript, and line events are tagged with the stream handle they came from. You don't need to worry about this if you only need to deal with a single input though, just use the `Transcriber` class's `start()`, `stop()`, etc. This function returns `Stream` class object.
  - `flags`: Integer, reserved for future expansion.
  - `update_interval`: Period in seconds between transcription updates.

- <a id="transcriber-add-listener"></a>`add_listener()`: Registers a callable object with the transcriber. This object will be called back as audio is fed in and text is extracted.
  - `listener`: This is often a subclass of `TranscriptEventListener`, but can be a plain function. It defines what code is called when a speech event happens.

- <a id="transcriber-remove-listener"></a>`remove_listener()`: Deletes a listener so that it no longer receives events.
  - `listener`: An object you previously passed into `add_listener()`.

- <a id="transcriber-remove-all-listeners"></a>`remove_all_listeners()`: Deletes all registered listeners so than none of them receive events anymore.

- <a id="transcriber-set-keyterms"></a>`set_keyterms()`: Biases the decoder towards a list of jargon, product names, or proper nouns, replacing any previous list. Takes effect on the next transcription and does not rewrite text already emitted. Pass `None` or an empty list to turn biasing off. Strength is set with the `keyterm_boost` option at load time. See [Domain Customization](../models/domain-customization.md). Only streaming architectures support this.
  - `keyterms`: Sequence of terms (for example `["Kubernetes", "Ceph"]`). Terms must not contain commas.

- <a id="transcriber-set-context"></a>`set_context()`: Picks the key terms out of a passage of text and biases towards them, replacing any previous list. For when you have context but not a list: the document on screen, the agenda, the last few messages in a thread. Same semantics as [`set_keyterms()`](#transcriber-set-keyterms) otherwise. See [Domain Customization](../models/domain-customization.md). Only streaming architectures support this.
  - `context`: The passage to read terms out of. Pass `None` or an empty string to turn biasing off.
  - `max_terms`: Most terms to take, 200 by default.

## MicTranscriber

Transcribes speech straight from the system's microphone, so you never call [`add_audio()`](classes.md#transcriber-add-audio) yourself. In Python this uses the [`sounddevice` library](https://python-sounddevice.readthedocs.io/), but in other languages the class uses the native audio API under the hood.

Construct one, configure it with chainable setters, call [`load()`](#mictranscriber-load) to fetch and open the model, then [`start()`](#mictranscriber-start) to begin listening.

```python
mic = (
    MicTranscriber()
    .on_text(lambda text: show_in_progress(text))
    .on_line(lambda line: append_line(line.text))
)
mic.load()
mic.start()
```

- <a id="mictranscriber-init"></a>`__init__()`: Constructs an unconfigured transcriber. Takes no arguments and cannot fail, so nothing is downloaded or opened until [`load()`](#mictranscriber-load).

Every setter returns the transcriber, so one can be built in a single expression, and every one has a working default. Call them before `load()`.

- <a id="mictranscriber-language"></a>`language()`: Sets the speech-to-text language. Defaults to `"en"`.
- <a id="mictranscriber-model-arch"></a>`model_arch()`: Picks a specific model size. By default the catalog's recommended model for the language is used, which is medium streaming for English. Most languages publish only one model.
- <a id="mictranscriber-models-from"></a>`models_from()`: Loads the model from a directory you supply rather than downloading it.
- <a id="mictranscriber-use-transcriber"></a>`use_transcriber()`: Reuses a [`Transcriber`](classes.md#transcriber) you already have instead of opening another. It stays yours to close.
- <a id="mictranscriber-update-interval"></a>`update_interval()`: Seconds between automatic streaming updates. Defaults to `0.5`.
- <a id="mictranscriber-options"></a>`options()`: Passes a dictionary of advanced [transcriber options](options.md#speech-to-text) straight through, for anything the setters don't cover.
- <a id="mictranscriber-spelling-model"></a>`spelling_model()`: Uses a specific alphanumeric spelling model. `load()` finds the published one for the language by itself, so this is only for when you keep your own copy. Pass `None` to go without one.
- <a id="mictranscriber-transcribe-flags"></a>`transcribe_flags()`: Sets the flags applied to every streaming update. Pass `MOONSHINE_FLAG_SPELLING_MODE` to turn on the spelling-CNN fusion that makes dictated codes and passwords accurate. Takes effect immediately when already loaded.
- <a id="mictranscriber-device"></a>`device()`: Captures from a specific input device, by index or name. Defaults to the system default.
- <a id="mictranscriber-samplerate"></a>`samplerate()`: Asks the capture device for a sample rate. Defaults to `16000`, and `start()` falls back to the device's own rate if it refuses, so this rarely needs setting.
- <a id="mictranscriber-channels"></a>`channels()`: Number of channels to capture. Defaults to `1`.
- <a id="mictranscriber-blocksize"></a>`blocksize()`: Frames per capture callback. Defaults to `1024`.

The callbacks below cover almost everything. For line ids, speaker spans, word timings, or the moment a line starts, attach a [`TranscriptEventListener`](classes.md#transcripteventlistener) with [`add_listener()`](#transcriber-add-listener) instead; listeners registered before `load()` are held and applied once the stream exists.

- <a id="mictranscriber-on-text"></a>`on_text()`: Called with the in-progress text of the line currently being spoken, each time it changes.
- <a id="mictranscriber-on-line"></a>`on_line()`: Called once per finished line, with the binding's line object (see [`transcript_line_t`](c-api.md#transcript_line_t) for the underlying fields).
- <a id="mictranscriber-on-error"></a>`on_error()`: Called when the audio or transcription pipeline raises.
- <a id="mictranscriber-on-progress"></a>`on_progress()`: Reports model download progress as a `0..1` fraction and the file being fetched. Attaching a handler also silences the default terminal progress bars.

- <a id="mictranscriber-load"></a>`load()`: Downloads the model if needed, opens it, and returns the transcriber. Blocking, since the first call may have to fetch several hundred megabytes; report progress with [`on_progress()`](#mictranscriber-on-progress). Safe to call twice.
- <a id="mictranscriber-set-keyterms"></a>`set_keyterms()`: Biases the decoder towards a list of terms while listening. Same semantics as [`Transcriber.set_keyterms()`](#transcriber-set-keyterms); call after `load()`. To start with a list instead, pass `keyterms` through [`options()`](#mictranscriber-options).
- <a id="mictranscriber-set-context"></a>`set_context()`: Picks the key terms out of a passage of text and biases towards them while listening. Same semantics as [`Transcriber.set_context()`](#transcriber-set-context); call after `load()`. To start with a passage instead, pass `context` through [`options()`](#mictranscriber-options).
- <a id="mictranscriber-start"></a>`start()`: Opens the microphone and begins transcribing. Raises if you haven't called `load()`.
- <a id="mictranscriber-mute"></a>`mute()`: Drops incoming audio without closing the microphone, so an assistant doesn't transcribe its own synthesized speech.
- <a id="mictranscriber-stop"></a>`stop()`: Stops listening and flushes any audio still in flight, so the final line is complete.
- <a id="mictranscriber-close"></a>`close()`: Releases the microphone, the stream, and the model. Also available as a context manager.

## Stream

The access point for when you need to feed multiple audio inputs into a single transcriber. Supports [`start()`](#transcriber-start), [`stop()`](#transcriber-stop), [`add_audio()`](classes.md#transcriber-add-audio), [`update_transcription()`](#transcriber-update-transcription), [`add_listener()`](#transcriber-add-listener), [`remove_listener()`](#transcriber-remove-listener), and [`remove_all_listeners()`](#transcriber-remove-all-listeners) as documented in the [`Transcriber`](classes.md#transcriber) class.

## TranscriptEventListener

A convenience class to derive from to create your own listener code. Override any or all of `on_line_started()`, `on_line_updated()`, `on_line_text_changed()`, `on_line_speakers_changed()`, `on_line_completed()`, and `on_error()`, and they'll be called back when the corresponding event occurs. Every method has a no-op default, so you only need to write the ones you care about.

## AgentFlow

A runner that drives generator-based conversational flows, and the entry point for voice interfaces. You register flow functions against trigger phrases, and the runner routes completed transcript lines either to trigger matching (when no flow is active) or to the currently suspended generator (when one is). Matching is semantic, using an embedding model that the runner downloads and loads the first time it needs one. [`load()`](#agentflow-load) opens the microphone transcriber and speech synthesizer for you, so there's no listener to wire up by hand; pass [`use_mic_transcriber()`](#agentflow-use-mic-transcriber) if you'd rather it listened to a transcriber you already have. See [Getting Started with a Conversational Agent](../using/conversational-agent.md) for usage examples.

A flow is an ordinary Python generator function that takes a [`Dialog`](classes.md#dialog) as its argument and yields prompt objects back to the runner. The runner carries out each prompt (speaking text, waiting for the user's response) and resumes the generator with the answer via `.send()`. This lets you write multi-step, branching conversations using regular Python control flow, including loops and exception handlers, without any async machinery. Trigger matching, confirmation, and option selection are all done semantically through the embedding model, so alternative phrasings will work without you needing to enumerate them.

- <a id="agentflow-init"></a>`__init__()`: Constructs an unconfigured runner. Takes no arguments — configure it with the chainable setters below, then call [`load()`](#agentflow-load).

Every setter returns the runner, so a whole voice interface can be built in one expression, and every one of them has a working default. Call them before `load()`.

- <a id="agentflow-language"></a>`language()`: Sets the language used for both recognition and speech. Defaults to `"en"`.
- <a id="agentflow-model-arch"></a>`model_arch()`: Picks a specific speech recognition model size instead of the default for the language.
- <a id="agentflow-voice"></a>`voice()`: Chooses the synthesis voice used to speak prompts.
- <a id="agentflow-speech-options"></a>`speech_options()`: Passes a dictionary of advanced [TTS options](options.md#text-to-speech) straight through to the [`TextToSpeech`](classes.md#texttospeech) synthesizer.
- <a id="agentflow-models-from"></a>`models_from()`: Reads and caches model files under the given directory instead of the default cache location.
- <a id="agentflow-microphone"></a>`microphone()`: Whether `load()` should open a microphone. Defaults to `True`. Turn it off to drive the runner from text with [`handle_utterance()`](#agentflow-handle-utterance).
- <a id="agentflow-speech"></a>`speech()`: Whether `load()` should open a speech synthesizer. Defaults to `True`. Turn it off for a silent runner: prompts are still logged and flows still advance, they just aren't spoken.
- <a id="agentflow-output-device"></a>`output_device()`: Pins playback to a specific audio output device, for machines where the host default isn't the speaker you want.
- <a id="agentflow-trigger-threshold"></a>`trigger_threshold()`: The similarity a phrase must reach to fire, between 0 and 1. Defaults to `0.7`. Raise it when triggers fire on unrelated speech, lower it when they don't fire on genuine attempts.
- <a id="agentflow-use-embeddings"></a>`use_embeddings()`: Whether to match phrases by meaning. Defaults to `True`, which downloads a small language model so "set up wifi" also fires on "I need to get online". Turn it off to fall back to case-insensitive substring matching and load no model, which is what offline tests usually want.
- <a id="agentflow-beeps"></a>`beeps()`: Whether to play the recognition cue tones. Defaults to `True`, which plays a short "got it" tone when an utterance matches and a distinct "didn't get that" tone when nothing does, so a misrecognition never ends in silence.
- <a id="agentflow-spell-feedback"></a>`spell_feedback()`: Whether to echo each character during spelled input. Defaults to `True`, speaking back `"haitch"` for `"h"` and `"deleting <character>"` for an undo, so the user hears that the right letter came off the end.
- <a id="agentflow-barge-in"></a>`barge_in()`: Whether the user can interrupt the assistant mid-prompt. Off by default, because an utterance arriving while the assistant is talking is usually the microphone hearing the speakers. Enable it only when you have reliable echo cancellation.
- <a id="agentflow-log-io"></a>`log_io()`: Logs the dialogue to stderr as `user: ...` / `assistant: ...` lines. Off by default. This is the user-facing transcript; use `debug()` for the verbose internal trace.
- <a id="agentflow-debug"></a>`debug()`: Traces every internal stage transition, with timings, to stderr.
- <a id="agentflow-on-progress"></a>`on_progress()`: Reports model download and load progress as `(fraction, name)`.
- <a id="agentflow-on-heard"></a>`on_heard()`: Reports every utterance the runner receives from the microphone, including trigger phrases and answers to prompts. Use [`otherwise()`](#agentflow-otherwise) instead for just the lines that didn't match anything.
- <a id="agentflow-on-said"></a>`on_said()`: Reports every prompt the runner speaks.
- <a id="agentflow-on-error"></a>`on_error()`: Reports errors raised by a flow or by the audio pipeline. Without a handler the runner prints them to stderr and carries on; a flow that raises is torn down either way, so one bad turn can't wedge the runner.
- <a id="agentflow-speak-with"></a>`speak_with()`: Speaks prompts with your own callable instead of the built-in synthesizer. It must block until playback finishes, since the runner resumes the flow as soon as it returns. Setting this stops `load()` creating a synthesizer.
- <a id="agentflow-use-text-to-speech"></a>`use_text_to_speech()`: Speaks with an existing [`TextToSpeech`](classes.md#texttospeech) instead of creating one.
- <a id="agentflow-use-mic-transcriber"></a>`use_mic_transcriber()`: Listens to an existing [`MicTranscriber`](classes.md#mictranscriber), or any object with the same `add_listener` / `start` / `stop` shape — a plain [`Transcriber`](classes.md#transcriber) fed from a file works, which is handy for testing a flow against recorded audio.

The runner won't close a synthesizer or transcriber it didn't create.

- <a id="agentflow-listen-for"></a>`listen_for()`: Starts a flow whenever the user says something like the trigger phrase.
  - `trigger_phrase`: A canonical phrase that is embedded once at registration time and compared against utterances via cosine similarity, so alternative phrasings of the same meaning will all start the flow.
  - `flow`: A callable that takes a [`Dialog`](classes.md#dialog) and returns a generator yielding prompts. Typically a generator function.

- <a id="agentflow-unregister-flow"></a>`unregister_flow()`: Removes a flow registered with `listen_for()`. Returns `True` if a flow was removed, `False` otherwise.
  - `trigger_phrase`: The trigger phrase used when the flow was registered.

- <a id="agentflow-always"></a>`always()`: Registers a phrase that stays live at every moment, whether or not a flow is running. "Cancel" and "start over" are built in and need no registration, but they apply only to a flow in progress, so an interface that dictates whatever it hears keeps those words when nothing is active. Registering either here opts it into being live all the time.
  - `trigger_phrase`: The canonical phrase to match, in the same way as `listen_for()`.
  - `handler`: A callable that takes the current [`Dialog`](classes.md#dialog) and returns an optional prompt to speak (or `None`). The handler can also call `d.cancel()` or `d.restart()` to abandon or reset the active flow.

- <a id="agentflow-otherwise"></a>`otherwise()`: Handles speech that matched no trigger and no waiting prompt. This is what a dictation interface hangs its text off: `on_heard()` reports every line including commands and answers, while this one reports only the lines nothing else claimed, so "delete the last sentence" starts your flow instead of being typed into the document. Registering a handler also silences the "didn't get that" cue, since unmatched speech is no longer a dead end. Nothing arrives here while a flow is running, because a flow's prompts take every line until it finishes.
  - `handler`: A callable that takes the utterance as a string.

- <a id="agentflow-load"></a>`load()`: Downloads and opens everything the runner needs — the phrase-matching model, a speech synthesizer, and a microphone transcriber — skipping any you've supplied or turned off. Blocking, since the first call may have to download models; report progress with `on_progress()`. Returns the runner.

- <a id="agentflow-start-listening"></a>`start_listening()`: Starts listening on the microphone, calling `load()` first if you haven't. Returns as soon as the microphone is live: transcript lines arrive on the audio thread and drive your flows from there, so the caller is free to sleep, run a UI, or do anything else.

- <a id="agentflow-stop-listening"></a>`stop_listening()`: Stops listening. Safe to call when already stopped.

- <a id="agentflow-handle-utterance"></a>`handle_utterance()`: Routes an utterance manually, without going through transcript events. Returns `True` if the utterance was consumed by a flow or a global handler, `False` otherwise. Useful for unit tests, or for driving the runner from input sources other than a `Transcriber`.
  - `utterance`: The string to route.

- <a id="agentflow-cancel"></a>`cancel()`: Abandons the currently running flow, if any. Returns `True` if a flow was canceled.

- <a id="agentflow-say"></a>`say()`: Speaks `text` outside any flow. Useful for welcome messages, status announcements, and error notifications that don't need a full flow registration. Blocks until playback finishes, and shares the same playback path as in-flow prompts, so mic muting and self-capture suppression still apply.
  - `text`: The string to speak.

- <a id="agentflow-close"></a>`close()`: Stops listening and releases everything the runner opened. Only closes what it created itself: a synthesizer or transcriber you passed in stays yours to close. Safe to call more than once, and safe on a runner that never loaded anything.

- `is_active`: A read-only boolean property that's `True` when a flow is currently in progress.
- `active_trigger`: A read-only property returning the trigger phrase of the active flow, or `None` if no flow is running.
- `registered_flows`: A read-only list of all registered flow trigger phrases.

## Dialog

The context object passed as the first argument to every flow function. Each method returns a prompt object that the flow `yield`s back to the runner; the runner then carries out the prompt (speaking text, waiting for input) and sends the result, if any, back into the generator via `.send()`. `Dialog` itself performs no I/O, so flows can be unit-tested by constructing a `Dialog`, calling the flow function, and driving the resulting generator manually without any audio, TTS, or event loop.

- `trigger_phrase`: The phrase that started the flow, available to the flow function as `d.trigger_phrase`.
- `state`: A `dict` for the flow's own per-conversation state, initially empty.

- <a id="dialog-say"></a>`say()`: Returns a prompt that, when yielded, speaks `text` and resumes the flow once playback has finished. The flow receives `None` from the `yield`.
  - `text`: The string for the assistant to speak.
  - `barge_in`: Reserved for future use; when supported, will allow the user to interrupt playback by speaking.

- <a id="dialog-ask"></a>`ask()`: Returns a prompt that speaks a question and resumes the flow with the user's next utterance as a string.
  - `prompt`: The string for the assistant to speak before listening.
  - `mode`: One of `FREE` (free-form natural-language input, the default), `SPELLED` (the user dictates one character at a time, terminated by "done"/"stop"/"finish", with each character spoken back as feedback and support for NATO-alphabet style words and "delete"/"undo" commands), `DIGITS` (digits-only spelled input), or `PHRASE` (a single phrase). These constants are exported from the `moonshine_voice` package.
  - `bias_terms`: Optional list of strings reserved for future use; currently ignored. For runtime ASR biasing, pass `keyterms` / call [`set_keyterms()`](#transcriber-set-keyterms) on the underlying transcriber instead.
  - `timeout`: Seconds to wait for a response before reprompting. Defaults to 8 seconds.
  - `no_input_reprompt`: Template used to reprompt the user when no input arrives within the timeout. `{prompt}` is substituted with the original prompt text. Pass `None` to skip the reprompt.
  - `max_retries`: Number of times to reprompt before raising `NoInputError` into the flow. Defaults to 2.

- <a id="dialog-confirm"></a>`confirm()`: Returns a prompt that asks a yes/no question and resumes the flow with a `bool`. Matching is semantic, so "okay", "affirmative", and "go ahead" all count as yes, and "no", "cancel", and "stop" count as no.
  - `prompt`: The yes/no question for the assistant to speak.
  - `timeout`: Seconds to wait for a response. Defaults to 6 seconds.
  - `max_retries`: Number of reprompts before raising `NoMatchError` into the flow. Defaults to 1.

- <a id="dialog-choose"></a>`choose()`: Returns a prompt that asks the user to pick from a set of named options and resumes the flow with the key of the matched option as a string. Each option key has a list of associated phrases; matching is done against the union of the key and its phrases using the embedding model.
  - `prompt`: The string for the assistant to speak.
  - `options`: A mapping of option keys to lists of associated phrases the user might say.
  - `timeout`: Seconds to wait for a response. Defaults to 8 seconds.
  - `max_retries`: Number of reprompts before raising `NoMatchError`. Defaults to 2.

- <a id="dialog-cancel"></a>`cancel()`: Raises `DialogCancelled` into the generator to abandon the active flow entirely. Typically called from a global handler registered with `AgentFlow.always()`.

- <a id="dialog-restart"></a>`restart()`: Raises `DialogRestart` into the generator to restart the active flow from the beginning. Typically called from a global handler.

- <a id="dialog-replay-last-prompt"></a>`replay_last_prompt()`: Returns a `Say` prompt that re-speaks the most recent question. Intended for global "repeat" / "say that again" handlers; returns `None` if nothing has been spoken yet.

## TextToSpeech

On-device text-to-speech using the Moonshine native stack (Kokoro, Piper, and ZipVoice vocoders plus per-language G2P assets). Invalid language tags raise `MoonshineTtsLanguageError`; missing or unknown voices raise `MoonshineTtsVoiceError`. Playback failures from `say()` raise `MoonshineAudioOutputError` with a list of output devices when enumeration succeeds.

Construct one, configure it with chainable setters, call [`load()`](#texttospeech-load) to fetch and open the voice, then [`say()`](#texttospeech-say):

```python
tts = TextToSpeech().language("en_us").voice("kokoro_af_heart")
tts.load()
tts.say("Hello from Moonshine.")
tts.wait()
```

`say()` is non-blocking and queued: each call returns immediately and utterances are played back in order by a background pipeline. A dedicated synthesis thread pre-synthesizes the next utterance while the current one is playing, minimizing the gap between consecutive utterances. Use `stop()` to cancel all pending speech, `wait()` to block until everything has been played, and `is_talking()` to poll playback state. The same API shape is available across Python, Swift, and Android (Java).

Use `list_tts_languages()`, `list_tts_voices()`, and `get_tts_voice_catalog()` to discover supported tags and voices. Asset layout and licenses are summarized in [`core/moonshine-tts/data/README.md`](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-tts/data/README.md); see also [Downloading Models](../using/downloading-models.md#text-to-speech-models).

- <a id="texttospeech-init"></a>`__init__()`: Constructs an unconfigured synthesizer. Takes no arguments and cannot fail, so nothing is downloaded or opened until [`load()`](#texttospeech-load).

Every setter returns the synthesizer, so one can be built in a single expression. Call them before `load()`.

- <a id="texttospeech-language"></a>`language()`: BCP-47-style tag for the speaking locale (for example `en_us`, `de`, `fr`). Aliases such as `en-us` are normalized by the library. Defaults to `"en"`.
- <a id="texttospeech-voice"></a>`voice()`: Catalog voice id. Prefix with `kokoro_`, `piper_`, or `zipvoice_` to choose the vocoder (for example `kokoro_af_heart`). Clears [`cloning()`](#texttospeech-cloning).
- <a id="texttospeech-models-from"></a>`models_from()`: Loads voice assets from a directory you supply rather than the default cache. Pass `download=True` to use that directory as the cache root and fetch anything missing into it; with `download=False` (the default for this setter) the directory must already be populated.
- <a id="texttospeech-cloning"></a>`cloning()`: Creates this synthesizer as a ZipVoice cloning engine. Call before `load()` so ZipVoice and clone-ASR assets are fetched up front; afterwards [`clone_from()`](#texttospeech-clone-from) / [`start_cloning()`](#texttospeech-start-cloning) stay offline. Clears [`voice()`](#texttospeech-voice).
- <a id="texttospeech-options"></a>`options()`: Escape hatch for native options the setters don't cover. See [Options → Text to Speech](options.md#text-to-speech).
- <a id="texttospeech-output-device"></a>`output_device()`: Playback device for `say()`, as a PortAudio index or a name substring. Defaults to the system default output.
- <a id="texttospeech-volume"></a>`volume()`: Playback gain applied to everything `say()` plays.
- <a id="texttospeech-on-progress"></a>`on_progress()`: Asset download progress, as a `0..1` fraction plus the file being fetched.
- <a id="texttospeech-debug"></a>`debug()`: Trace synthesis and playback to stderr.

- <a id="texttospeech-load"></a>`load()`: Downloads the voice assets if needed and prepares the synthesizer. Blocking, since the first call may have to fetch a few hundred megabytes; report progress with [`on_progress()`](#texttospeech-on-progress). Calling it again is a no-op. With [`cloning()`](#texttospeech-cloning), ZipVoice and clone ASR are both fetched here so [`clone_from()`](#texttospeech-clone-from) stays offline afterward.

- `language_tag`: Read-only property returning the language tag in use.
- `asset_root`: Read-only property returning the `pathlib.Path` directory passed to the native layer as `g2p_root` (raises if you have not called `load()` yet).
- `is_cloned`: Read-only boolean that is `True` once a voice has been cloned into this synthesizer.

- <a id="texttospeech-clone-from"></a>`clone_from()`: Clones the voice in `source` and uses it for subsequent synthesis. Requires [`cloning()`](#texttospeech-cloning) before `load()`.
  - `source`: A path to a `.wav` file, a `(pcm, sample_rate)` pair of mono float PCM, or a `VoiceClone` that has captured enough speech.
  - `transcript`: Optional transcript of the clip; when omitted, Moonshine auto-transcribes it with assets already fetched by `load()`.

- <a id="texttospeech-start-cloning"></a>`start_cloning()`: Starts capturing a reference voice from the microphone. Requires [`cloning()`](#texttospeech-cloning) before `load()`. Returns a `VoiceClone` that listens until it has heard enough usable speech.
  - `clip_duration_seconds`: Maximum capture window (default `4.0`).
  - `minimum_speech_seconds`: Minimum speech required before the clone is ready (default `2.0`).

- <a id="texttospeech-synthesize"></a>`synthesize()`: Converts `text` to mono PCM audio.
  - `text`: UTF-8 string to speak.
  - `options`: Optional per-call native options (currently only [`speed`](options.md#text-to-speech) is honored for the call duration).
  - Returns a tuple `(samples, sample_rate)` where `samples` is a list of 32-bit floats in roughly the −1.0…1.0 range and `sample_rate` is the output sample rate in Hz.

- <a id="texttospeech-say"></a>`say()`: Queues text for synthesis and playback, returning immediately. A background synthesis thread converts text to audio, then hands it to a playback thread that plays it on the selected output device. Synthesis of the next utterance overlaps with playback of the current one. Requires `pip install numpy sounddevice` on Python.
  - `text`: A string or a list of strings to speak. A list is equivalent to calling `say()` once per element in order.
  - `device`: (Python/Swift-macOS) `None` for the host default output, an integer PortAudio output device index, a decimal string index, or a case-insensitive substring of a device name. On Android, pass a `Context` (required) and optionally an `AudioDeviceInfo`.
  - `options`: Optional per-call native options (see [TTS options](options.md#text-to-speech); only `speed` is honored per call).

- <a id="texttospeech-stop"></a>`stop()`: Clears the utterance queue and stops any audio currently playing. Returns once all pending utterances are discarded and active playback has been halted. It is safe to call `say()` again afterwards.

- <a id="texttospeech-wait"></a>`wait()`: Blocks the calling thread until every queued utterance has been synthesized and played to completion. Named `waitUntilDone()` on Android.

- <a id="texttospeech-is-talking"></a>`is_talking()`: Returns `True` if utterances are still queued, being synthesized, or currently playing. Named `isTalking()` on Swift and Android.

- <a id="texttospeech-close"></a>`close()`: Stops any in-progress playback, discards pending utterances, and releases the native synthesizer handle. Called automatically when using a `with TextToSpeech() as tts:` block or on garbage collection.

## GraphemeToPhonemizer

IPA string generation without speech synthesis. Dependencies are the same CDN lexicon and ONNX bundles as TTS, but restricted to what `moonshine_get_g2p_dependencies()` reports for the language. When `download` is true, assets are placed under the package cache or `asset_root`; when false, `asset_root` must already contain those files.

- <a id="graphemetophonemizer-init"></a>`__init__()`: Creates a native G2P handle.
  - `language`: Locale tag (for example `en_us`, `ja`). Normalized the same way as for TTS.
  - `options`: Optional mapping of [G2P options](options.md#grapheme-to-phonemes) (the binding sets `g2p_root` automatically).
  - `asset_root`: Optional cache or pre-populated directory, same semantics as for `TextToSpeech`.
  - `download`: When true (default), missing G2P assets are downloaded. When false, `asset_root` is required.

- `language`: Read-only normalized tag.

- `asset_root`: Read-only `pathlib.Path` to the directory used as `g2p_root`.

- <a id="graphemetophonemizer-to-ipa"></a>`to_ipa()`: Returns a single IPA string for the input text.
  - `text`: UTF-8 surface string.
  - `options`: Optional per-call native [G2P options](options.md#grapheme-to-phonemes).

- <a id="graphemetophonemizer-close"></a>`close()`: Frees the native handle; also invoked by context manager exit and `__del__`.
