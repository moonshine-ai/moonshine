"""Capture of the short reference clip that zero-shot voice cloning needs."""

import sys
import threading
import time
from typing import Any, Callable, List, Optional, Sequence

from moonshine_voice.errors import MoonshineError
from moonshine_voice.moonshine_api import moonshine_extract_speech_clip


def _import_microphone_deps():
    """Import numpy and sounddevice for `VoiceClone.from_microphone`."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        raise MoonshineError(
            "VoiceClone.from_microphone() requires numpy and sounddevice "
            "(e.g. `pip install numpy sounddevice`)."
        ) from e
    return np, sd


class VoiceClone:
    """Captures the short reference clip that zero-shot voice cloning needs.

    ```python
    clone = tts.start_cloning()
    clone.on_ready(lambda: print("Got it, you can stop talking."))
    clone.from_microphone()
    tts.clone_from(clone)
    ```

    Finding a usable clip means locating a window of the recording that is
    mostly speech rather than silence or breathing. That search runs in the
    core, so the browser, Python, iOS and Android bindings all agree on what a
    good clip looks like. No model download is involved: the voice-activity
    detector is compiled into the library.
    """

    #: Sample rate of the clip handed back by :attr:`audio`.
    CLIP_SAMPLE_RATE = 16000
    #: Give up looking for a good window after this much recording.
    DEFAULT_MAX_RECORD_SECONDS = 20.0
    # How much new audio to accumulate between speech searches.
    _SEARCH_INTERVAL_SECONDS = 0.25

    def __init__(
        self,
        *,
        clip_duration_seconds: float = 4.0,
        minimum_speech_seconds: float = 2.0,
    ):
        self._clip_duration_seconds = float(clip_duration_seconds)
        self._minimum_speech_seconds = float(minimum_speech_seconds)
        self._lock = threading.RLock()
        self._recording: List[float] = []
        self._recording_sample_rate = VoiceClone.CLIP_SAMPLE_RATE
        self._samples_since_search = 0
        self._clip: Optional[List[float]] = None
        self._speech_seconds = 0.0
        self._ready_handlers: List[Callable[[], None]] = []
        self._progress_handlers: List[Callable[[float, float], None]] = []
        self._capturing = False
        self._cancelled = False

    # ---------------------------------------------------------------- setup

    def on_ready(self, handler: Callable[[], None]) -> "VoiceClone":
        """Fires once, as soon as enough speech has been captured."""
        with self._lock:
            already_ready = self._clip is not None
            if not already_ready:
                self._ready_handlers.append(handler)
        if already_ready:
            handler()
        return self

    def on_progress(self, handler: Callable[[float, float], None]) -> "VoiceClone":
        """Reports how long the caller has been recording and how much of the
        best window so far was speech, both in seconds."""
        with self._lock:
            self._progress_handlers.append(handler)
        return self

    # ---------------------------------------------------------------- state

    @property
    def is_ready(self) -> bool:
        """True once :attr:`audio` holds a usable reference clip."""
        with self._lock:
            return self._clip is not None

    @property
    def audio(self) -> Optional[List[float]]:
        """The captured clip (16 kHz mono), or None until :attr:`is_ready`."""
        with self._lock:
            return list(self._clip) if self._clip is not None else None

    @property
    def sample_rate(self) -> int:
        return VoiceClone.CLIP_SAMPLE_RATE

    @property
    def speech_seconds(self) -> float:
        """Speech found in the best window so far, in seconds."""
        with self._lock:
            return self._speech_seconds

    @property
    def recorded_seconds(self) -> float:
        with self._lock:
            if self._recording_sample_rate <= 0:
                return 0.0
            return len(self._recording) / float(self._recording_sample_rate)

    # ---------------------------------------------------------------- audio

    def add_audio(self, pcm: Sequence[float], sample_rate: int) -> None:
        """Feeds captured audio in. Call this from your own audio pipeline; the
        search for a usable window runs a few times a second rather than on
        every chunk."""
        with self._lock:
            if self._clip is not None or sample_rate <= 0:
                return
            samples = _as_float_list(pcm)
            if not samples:
                return
            if sample_rate != self._recording_sample_rate:
                # Mixed rates in one buffer would make the clip come out at the
                # wrong speed, so a change starts the recording over.
                self._recording.clear()
                self._recording_sample_rate = int(sample_rate)
                self._samples_since_search = 0
            self._recording.extend(samples)
            self._samples_since_search += len(samples)
            due = self._samples_since_search >= (
                VoiceClone._SEARCH_INTERVAL_SECONDS * sample_rate
            )
            if due:
                self._samples_since_search = 0
        if due:
            self._search()

    def from_microphone(
        self,
        max_seconds: float = DEFAULT_MAX_RECORD_SECONDS,
        *,
        device: Optional[Any] = None,
    ) -> List[float]:
        """Opens the microphone and records until there is enough speech, or
        until ``max_seconds`` have passed. Blocks, and returns the clip, which
        is also available from :attr:`audio`."""
        np, sd = _import_microphone_deps()
        with self._lock:
            if self._capturing:
                raise MoonshineError("This VoiceClone is already recording.")
            self._capturing = True
            self._cancelled = False

        sample_rate = _default_input_sample_rate(sd, device)

        def audio_callback(in_data, frames, time_info, status):
            if status:
                print(f"VoiceClone: {status}", file=sys.stderr)
            if in_data is None:
                return
            self.add_audio(in_data.astype(np.float32).flatten(), sample_rate)

        stream = sd.InputStream(
            samplerate=sample_rate,
            device=device,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        )
        deadline = time.monotonic() + float(max_seconds)
        try:
            stream.start()
            while not self.is_ready:
                with self._lock:
                    cancelled = self._cancelled
                if cancelled:
                    break
                if time.monotonic() >= deadline:
                    # Out of patience: take the best window we have, even a
                    # quiet one.
                    self._search(accept_anything=True)
                    break
                time.sleep(0.05)
        finally:
            stream.stop()
            stream.close()
            with self._lock:
                self._capturing = False

        clip = self.audio
        if clip is None:
            raise MoonshineError(
                f"No speech detected in {int(max_seconds)}s of recording. "
                "Try again somewhere quieter."
            )
        return clip

    def cancel(self) -> None:
        """Stops an in-flight :meth:`from_microphone` capture."""
        with self._lock:
            self._cancelled = True

    def reset(self) -> None:
        """Throws away everything captured so far."""
        with self._lock:
            self._recording = []
            self._samples_since_search = 0
            self._clip = None
            self._speech_seconds = 0.0

    # ------------------------------------------------------------ internals

    def _search(self, accept_anything: bool = False) -> None:
        with self._lock:
            if self._clip is not None:
                return
            samples = list(self._recording)
            rate = self._recording_sample_rate
        if not samples:
            return

        try:
            result = moonshine_extract_speech_clip(
                samples,
                rate,
                clip_duration_seconds=self._clip_duration_seconds,
                minimum_speech_seconds=(
                    0.0 if accept_anything else self._minimum_speech_seconds
                ),
            )
        except MoonshineError:
            # A search failure on one window is not worth ending the capture
            # over; the next one runs a quarter of a second later.
            return

        with self._lock:
            self._speech_seconds = result.speech_duration
            recorded = len(samples) / float(rate) if rate else 0.0
            progress_handlers = list(self._progress_handlers)
            ready_handlers: List[Callable[[], None]] = []
            if result.audio:
                self._clip = result.audio
                ready_handlers = self._ready_handlers
                self._ready_handlers = []

        for handler in progress_handlers:
            handler(recorded, result.speech_duration)
        for handler in ready_handlers:
            handler()


def _as_float_list(pcm: Any) -> List[float]:
    """Flatten whatever the caller handed us into a list of Python floats."""
    tolist = getattr(pcm, "tolist", None)
    if tolist is not None:
        flatten = getattr(pcm, "ravel", None)
        values = (flatten() if flatten is not None else pcm).tolist()
        return values if isinstance(values, list) else [float(values)]
    return [float(x) for x in pcm]


def _default_input_sample_rate(sd: Any, device: Optional[Any]) -> int:
    """The capture device's native rate. The core resamples to 16 kHz itself,
    so opening at the native rate avoids a PortAudio rate conversion."""
    try:
        info = sd.query_devices(device, "input")
        rate = info.get("default_samplerate") if isinstance(info, dict) else None
        if rate:
            return int(rate)
    except Exception as e:
        print(f"VoiceClone: could not query device info: {e}", file=sys.stderr)
    return VoiceClone.CLIP_SAMPLE_RATE
