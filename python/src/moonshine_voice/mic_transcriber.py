from moonshine_voice.download import (
    ProgressCallback,
    get_diarization_model,
    get_model_for_language,
    get_spelling_model_path,
)
from moonshine_voice.transcriber import (
    LineCompleted,
    LineTextChanged,
    Error,
    Transcriber,
    TranscriptEvent,
    TranscriptLine,
    ModelArch,
)

import numpy as np
import queue
import sounddevice as sd
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union


_OLD_CONSTRUCTOR_HELP = """\
MicTranscriber() no longer takes constructor arguments. Configure it with
chainable setters and call load():

    mic = MicTranscriber().on_line(lambda line: print(line.text))
    mic.load()
    mic.start()

The old model_path argument is now .models_from(directory), and every other
argument has a setter of the same name (.model_arch(), .update_interval(),
.device(), .options(), and so on)."""


def _is_true(value: Any) -> bool:
    """Whether an option value means true, the way the C API reads it.

    Option values reach the library as strings, but callers reasonably pass a
    Python bool, so accept both.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1")


class MicTranscriber:
    """Transcribes speech from a microphone.

    Construct one, configure it with chainable setters, call :meth:`load` to
    fetch and open the model, then :meth:`start` to begin listening::

        mic = (MicTranscriber()
               .on_text(show_in_progress)
               .on_line(lambda line: append_line(line.text)))
        mic.load()
        mic.start()

    :meth:`load` blocks, since the first call may have to download a model;
    pass :meth:`on_progress` a handler to drive a progress bar. Use it as a
    context manager, or call :meth:`close` when you are done.

    :meth:`on_text` and :meth:`on_line` cover almost everything. For line ids,
    speaker spans, word timings, or the moment a line starts, attach a
    :class:`TranscriptEventListener` with :meth:`add_listener` instead.
    """

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError(_OLD_CONSTRUCTOR_HELP)

        # Deferred configuration, applied by load().
        self._language = "en"
        # None means the catalog's recommended model for the language, which is
        # medium streaming for English. Naming a specific arch that the language
        # doesn't publish is an error rather than a silent downgrade.
        self._model_arch: Optional[ModelArch] = None
        self._model_directory: Optional[Path] = None
        self._update_interval = 0.5
        self._options: dict = {}
        self._spelling_model_path: Optional[str] = None
        self._spelling_disabled = False
        self._transcribe_flags = 0
        self._progress_fn: Optional[ProgressCallback] = None
        # Listeners registered before load() has a stream to hang them on.
        self._pending_listeners: list = []
        self._owns_transcriber = False

        self.transcriber: Optional[Transcriber] = None
        self.mic_stream = None

        self._should_listen = False
        self._muted = False
        self._sd_stream = None
        self._device: Optional[Union[int, str]] = None
        self._samplerate = 16000
        self._channels = 1
        self._blocksize = 1024
        # Audio captured on the PortAudio callback is handed to a worker
        # thread through this queue. Transcription (which can block for
        # hundreds of milliseconds per update, e.g. on a Raspberry Pi) must
        # never run on the time-critical capture callback, or PortAudio
        # reports input overflows (see issue #196).
        self._audio_queue: "queue.Queue" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        # Sentinel pushed onto the queue to tell the worker to drain and exit.
        self._worker_stop = object()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def language(self, code: str) -> "MicTranscriber":
        """Set the speech-to-text language (default ``"en"``)."""
        self._language = code
        return self

    def model_arch(self, arch: ModelArch) -> "MicTranscriber":
        """Pick a specific model size.

        By default the catalog's recommended model for the language is used,
        which is medium streaming for English. Most languages publish only one
        model, so this is mainly an English-language choice.
        """
        self._model_arch = arch
        return self

    def models_from(self, directory: Union[str, Path]) -> "MicTranscriber":
        """Load the model from ``directory`` rather than downloading it."""
        self._model_directory = Path(directory)
        return self

    def use_transcriber(self, transcriber: Transcriber) -> "MicTranscriber":
        """Reuse an already-loaded transcriber rather than opening another.

        The transcriber stays yours to close.
        """
        self.transcriber = transcriber
        self._owns_transcriber = False
        return self

    def update_interval(self, seconds: float) -> "MicTranscriber":
        """Seconds between automatic streaming updates (default ``0.5``)."""
        self._update_interval = seconds
        return self

    def options(self, options: Mapping[str, Any]) -> "MicTranscriber":
        """Escape hatch for transcriber options the setters don't cover."""
        self._options.update(options)
        return self

    def spelling_model(self, path: Union[str, Path, None]) -> "MicTranscriber":
        """Use a specific alphanumeric spelling model.

        Only worth setting when you keep your own copy: :meth:`load` finds the
        published one for the language by itself. Pass ``None`` to go without
        one, which costs accuracy inside spelling mode.
        """
        self._spelling_model_path = None if path is None else str(path)
        self._spelling_disabled = path is None
        return self

    def transcribe_flags(self, flags: int) -> "MicTranscriber":
        """Set the flags applied to every streaming update.

        Pass ``MOONSHINE_FLAG_SPELLING_MODE`` to turn on the spelling-CNN
        fusion path, which is what makes dictated codes and passwords
        accurate. Takes effect immediately when already loaded.
        """
        self._transcribe_flags = int(flags)
        if self.mic_stream is not None:
            self.mic_stream.set_transcribe_flags(self._transcribe_flags)
        return self

    def device(self, device: Union[int, str, None]) -> "MicTranscriber":
        """Capture from a specific input device, by index or name."""
        self._device = device
        return self

    def samplerate(self, hz: int) -> "MicTranscriber":
        """Ask the capture device for a sample rate (default ``16000``).

        The native library resamples whatever arrives, and :meth:`start` falls
        back to the device's own rate if it refuses this one, so this rarely
        needs setting.
        """
        self._samplerate = int(hz)
        return self

    def channels(self, count: int) -> "MicTranscriber":
        """Number of channels to capture (default ``1``)."""
        self._channels = int(count)
        return self

    def blocksize(self, frames: int) -> "MicTranscriber":
        """Frames per capture callback (default ``1024``)."""
        self._blocksize = int(frames)
        return self

    def on_text(self, callback: Callable[[str], None]) -> "MicTranscriber":
        """Called with the in-progress text of the line currently being spoken."""

        def listener(event: TranscriptEvent) -> None:
            if isinstance(event, LineTextChanged):
                callback(event.line.text)

        self.add_listener(listener)
        return self

    def on_line(self, callback: Callable[[TranscriptLine], None]) -> "MicTranscriber":
        """Called once per finished line."""

        def listener(event: TranscriptEvent) -> None:
            if isinstance(event, LineCompleted):
                callback(event.line)

        self.add_listener(listener)
        return self

    def on_error(self, callback: Callable[[BaseException], None]) -> "MicTranscriber":
        """Called when the audio or transcription pipeline raises."""

        def listener(event: TranscriptEvent) -> None:
            if isinstance(event, Error):
                callback(event.error)

        self.add_listener(listener)
        return self

    def on_progress(self, callback: ProgressCallback) -> "MicTranscriber":
        """Report model download progress as ``(fraction, filename)``.

        Attaching a handler also silences the default terminal progress bars.
        """
        self._progress_fn = callback
        return self

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "MicTranscriber":
        """Download the model if needed, open it, and return self.

        Blocking. Safe to call twice; the second call does nothing.
        """
        if self.mic_stream is not None:
            return self

        if self.transcriber is None:
            options = dict(self._options)
            spelling_path = self._spelling_model_path
            if self._model_directory is not None:
                model_path: Union[str, Path] = self._model_directory
                model_arch = (
                    self._model_arch
                    if self._model_arch is not None
                    else ModelArch.MEDIUM_STREAMING
                )
            else:
                model_path, model_arch = get_model_for_language(
                    self._language,
                    self._model_arch,
                    on_progress=self._progress_fn,
                )
                if spelling_path is None and not self._spelling_disabled:
                    # The spelling model is what makes dictated codes and
                    # passwords accurate, but it is not published for every
                    # language and its absence only costs accuracy inside
                    # spelling mode, so a failure here is not fatal.
                    try:
                        spelling_path = get_spelling_model_path(self._language)
                    except Exception as e:
                        print(
                            f"MicTranscriber: no spelling model for "
                            f"{self._language}: {e}",
                            file=sys.stderr,
                        )
            if spelling_path is not None:
                options.setdefault("spelling_model_path", spelling_path)
            # Speaker identification needs its own models, which are a download
            # rather than compiled-in data. Unlike the spelling model, a failure
            # here is fatal: the caller asked for speaker IDs and would
            # otherwise get a transcriber that silently never produces them.
            if _is_true(options.get("identify_speakers")) and not options.get(
                "diarization_model_dir"
            ):
                options["diarization_model_dir"] = get_diarization_model()
            self.transcriber = Transcriber(
                str(model_path), model_arch, options=options or None
            )
            self._owns_transcriber = True

        self.mic_stream = self.transcriber.create_stream(
            self._update_interval, transcribe_flags=self._transcribe_flags
        )
        for listener in self._pending_listeners:
            self.mic_stream.add_listener(listener)
        self._pending_listeners.clear()
        return self

    def _query_device_default_samplerate(self) -> Optional[int]:
        """Return the input device's native default sample rate, or None on failure.

        Used as a fallback when the requested rate isn't natively supported by
        the capture device (common on USB mics that only do 44100/48000 Hz).
        """
        try:
            info = sd.query_devices(self._device, "input")
        except (sd.PortAudioError, OSError, ValueError) as e:
            print(f"MicTranscriber: could not query device info: {e}", file=sys.stderr)
            return None
        rate = info.get("default_samplerate") if isinstance(info, dict) else None
        try:
            rate = int(rate) if rate else None
        except (TypeError, ValueError):
            rate = None
        return rate if rate and rate > 0 else None

    def _open_input_stream(self, samplerate: int, callback) -> sd.InputStream:
        stream = sd.InputStream(
            samplerate=samplerate,
            blocksize=self._blocksize,
            device=self._device,
            channels=self._channels,
            dtype="float32",
            callback=callback,
        )
        return stream

    def _start_listening(self):
        """
        Start listening to the microphone (or specified audio device).
        Incoming audio blocks are automatically fed to self.mic_stream.add_audio().
        """

        def audio_callback(in_data, frames, time, status):
            if not self._should_listen or self._muted:
                return
            if status:
                print(f"MicTranscriber: {status}")
            if in_data is not None:
                # Flatten and convert to float32 if needed
                audio_data = in_data.astype(np.float32).flatten()
                # Hand the audio to the worker thread and return immediately.
                # The C API resamples to its internal 16 kHz, so we pass
                # whatever rate the device is actually capturing at. Queueing
                # is non-blocking, keeping this callback safe to run on the
                # time-critical PortAudio thread.
                self._audio_queue.put((audio_data, self._samplerate))

        try:
            self._sd_stream = self._open_input_stream(self._samplerate, audio_callback)
        except sd.PortAudioError as e:
            # Most commonly PaErrorCode -9997 (Invalid sample rate) when the
            # capture device doesn't natively support our requested rate.
            # Fall back to the device's default rate; the C API will resample.
            fallback = self._query_device_default_samplerate()
            if fallback is None or fallback == self._samplerate:
                raise
            print(
                f"MicTranscriber: device does not support {self._samplerate} Hz "
                f"({e}); falling back to {fallback} Hz.",
                file=sys.stderr,
            )
            self._samplerate = fallback
            self._sd_stream = self._open_input_stream(self._samplerate, audio_callback)
        self._sd_stream.start()

    def _process_audio_queue(self):
        """Drain queued audio into the stream from a dedicated worker thread.

        The blocking ``update_transcription`` that ``Stream.add_audio``
        triggers every ``update_interval`` runs here instead of on the
        PortAudio capture callback, so audio capture is never stalled by
        inference (see issue #196).

        Whenever the worker wakes it consumes everything currently waiting on
        the queue and hands it to ``add_audio`` in as few calls as possible
        (one per run of chunks sharing a sample rate). Coalescing a backlog
        this way means a single transcription pass instead of one per chunk,
        which lowers latency and avoids redundant work when the worker falls
        behind.
        """
        while True:
            item = self._audio_queue.get()
            if item is self._worker_stop:
                break
            batch = [item]
            stop_requested = False
            # Grab anything else already waiting without blocking.
            while True:
                try:
                    queued = self._audio_queue.get_nowait()
                except queue.Empty:
                    break
                if queued is self._worker_stop:
                    stop_requested = True
                    break
                batch.append(queued)
            self._add_batch(batch)
            if stop_requested:
                break

    def _add_batch(self, batch):
        """Concatenate consecutive same-sample-rate chunks and add each run once."""
        run_chunks = []
        run_rate = None
        for audio_data, sample_rate in batch:
            if run_chunks and sample_rate != run_rate:
                self._add_run(run_chunks, run_rate)
                run_chunks = []
            run_chunks.append(audio_data)
            run_rate = sample_rate
        if run_chunks:
            self._add_run(run_chunks, run_rate)

    def _add_run(self, chunks, sample_rate):
        audio_data = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        try:
            self.mic_stream.add_audio(audio_data, sample_rate)
        except Exception as e:
            print(
                f"MicTranscriber: error transcribing audio: {e}",
                file=sys.stderr,
            )

    def _start_worker(self):
        if self._worker_thread is None:
            self._worker_thread = threading.Thread(
                target=self._process_audio_queue,
                name="MicTranscriberWorker",
                daemon=True,
            )
            self._worker_thread.start()

    def _stop_worker(self):
        """Signal the worker to drain the queue and exit, then join it."""
        if self._worker_thread is not None:
            self._audio_queue.put(self._worker_stop)
            self._worker_thread.join()
            self._worker_thread = None

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def start(self) -> "MicTranscriber":
        """Open the microphone and begin transcribing."""
        if self.mic_stream is None:
            raise RuntimeError("No model loaded. Call load() before start().")
        self.mic_stream.start()
        self._start_worker()
        if self._sd_stream is None:
            self._start_listening()
        self._should_listen = True
        return self

    def mute(self, muted: bool = True) -> "MicTranscriber":
        """Drop incoming audio without closing the microphone.

        Used to stop an assistant transcribing its own synthesized speech.
        """
        self._muted = muted
        return self

    def stop(self) -> None:
        self._should_listen = False
        # Let the worker finish transcribing any queued audio, then join it
        # before flushing the stream so the final transcript is complete.
        self._stop_worker()
        if self.mic_stream is not None:
            self.mic_stream.stop()

    def close(self) -> None:
        """Release the microphone, the stream, and any transcriber we opened."""
        self._should_listen = False
        self._stop_worker()
        if self._sd_stream is not None:
            self._sd_stream.close()
            self._sd_stream = None
        if self.mic_stream is not None:
            self.mic_stream.close()
            self.mic_stream = None
        if self.transcriber is not None and self._owns_transcriber:
            self.transcriber.close()
        self.transcriber = None

    def __enter__(self) -> "MicTranscriber":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def set_transcribe_flags(self, flags: int) -> None:
        """Non-chainable alias for :meth:`transcribe_flags`.

        Lets DialogFlow flip ``MOONSHINE_FLAG_SPELLING_MODE`` on only while a
        ``SPELLED`` / ``DIGITS`` prompt is in progress.
        """
        self.transcribe_flags(flags)

    # ------------------------------------------------------------------
    # Full event interface
    #
    # on_text / on_line / on_error cover the common cases and are built on
    # these. Reach for these directly when you need line ids, speaker spans,
    # word timings, or the moment a line starts.
    # ------------------------------------------------------------------

    def add_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        """Attach a listener, either a callable or a TranscriptEventListener.

        Listeners attached before :meth:`load` are held and applied when the
        stream exists, so a builder chain can register them up front.
        """
        if self.mic_stream is None:
            self._pending_listeners.append(listener)
        else:
            self.mic_stream.add_listener(listener)

    def remove_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        if self.mic_stream is None:
            if listener in self._pending_listeners:
                self._pending_listeners.remove(listener)
        else:
            self.mic_stream.remove_listener(listener)

    def remove_all_listeners(self) -> None:
        self._pending_listeners.clear()
        if self.mic_stream is not None:
            self.mic_stream.remove_all_listeners()

    def push_listener(self, listener: Callable[[TranscriptEvent], None]) -> None:
        """Push a temporary listener, saving the current listeners on a stack."""
        self._require_stream("push_listener").push_listener(listener)

    def pop_listener(self) -> None:
        """Restore the listeners that were active before the last push."""
        self._require_stream("pop_listener").pop_listener()

    def pop_all_listeners(self) -> None:
        """Unwind the entire listener stack, restoring the original listeners."""
        self._require_stream("pop_all_listeners").pop_all_listeners()

    def _require_stream(self, what: str):
        if self.mic_stream is None:
            raise RuntimeError(f"No model loaded. Call load() before {what}().")
        return self.mic_stream


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MicTranscriber example")
    parser.add_argument(
        "--language", type=str, default="en", help="Language to use for transcription"
    )
    parser.add_argument(
        "--model-arch",
        type=int,
        default=None,
        help="Model architecture to use for transcription",
    )
    args = parser.parse_args()

    mic = MicTranscriber().language(args.language)
    if args.model_arch is not None:
        mic.model_arch(ModelArch(args.model_arch))

    if sys.stdout.isatty():
        # Rewrite the current line in place as the text firms up, then leave it
        # behind and start a new one once the line is final.
        state = {"width": 0}

        def show_partial(text: str) -> None:
            padding = max(state["width"] - len(text), 0)
            print(f"\r{text}{' ' * padding}", end="", flush=True)
            state["width"] = len(text)

        def show_final(line: TranscriptLine) -> None:
            show_partial(line.text)
            state["width"] = 0
            print()

        mic.on_text(show_partial).on_line(show_final)
    else:
        mic.on_line(lambda line: print(line.text, flush=True))

    print("Loading the model…", file=sys.stderr)
    mic.load()

    print("Listening to the microphone, press Ctrl+C to stop...", file=sys.stderr)
    with mic:
        mic.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            mic.stop()
