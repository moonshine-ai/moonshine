"""Text-to-speech via the Moonshine C API."""

import queue
import sys
import threading
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from moonshine_voice.download import (
    download_tts_assets,
    ensure_tts_voice_downloaded,
    tts_asset_cache_path,
    validate_tts_language,
    validate_tts_voice_known,
)
from moonshine_voice.errors import MoonshineAudioOutputError, MoonshineError
from moonshine_voice.moonshine_api import (
    MOONSHINE_HEADER_VERSION,
    _MoonshineLib,
    moonshine_options_array,
    moonshine_text_to_speech_samples,
)


@dataclass
class _SayRequest:
    """Queued utterance for the background synthesis worker."""
    text: str
    speed: Optional[float]
    volume: Optional[float]
    device: Optional[Union[int, str]]
    options: Optional[Dict[str, Union[str, int, float, bool]]]


@dataclass
class _PlayItem:
    """Synthesized audio ready for the playback worker."""
    data: Any  # numpy float32 array
    sample_rate: int
    device: Optional[Union[int, str]]


@dataclass
class _BeepRequest:
    """Queued beep marker; routed through the say queue so it plays in
    the same order as any in-flight :meth:`TextToSpeech.say` calls
    rather than racing ahead on the play queue.

    ``kind`` selects the cached waveform:

    * ``"error"`` – descending two-tone (660 Hz → 440 Hz) used to signal
      a misrecognition or a rejected utterance.
    * ``"success"`` – ascending two-tone (523 Hz → 784 Hz) used to
      signal a recognized utterance, played just before the TTS
      response.
    """
    device: Optional[Union[int, str]]
    kind: str = "error"


_SHUTDOWN_SENTINEL = object()

# Sample rate used for synthetically generated beeps.  22.05 kHz matches
# the typical TTS output rate and is plenty for a short sine tone.
_BEEP_SAMPLE_RATE = 22050

# Cache of generated beep waveforms keyed by ``(kind, sample_rate)`` so
# we only pay the numpy cost once per TextToSpeech lifetime per beep
# kind.
_BEEP_CACHE: Dict[Tuple[str, int], Any] = {}


def _beep_envelope(np: Any, sample_rate: int, freq: float, duration_ms: int) -> Any:
    """Generate a fixed-amplitude sine tone with a 10 ms linear fade in/out.

    Peak amplitude is held to ~0.25 so beeps stay audible but don't
    startle next to normal-volume TTS.
    """
    n = int(sample_rate * duration_ms / 1000)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    wave = (0.25 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    ramp_n = min(int(sample_rate * 0.010), n // 2)
    if ramp_n > 0:
        ramp = np.linspace(0.0, 1.0, ramp_n, dtype=np.float32)
        wave[:ramp_n] *= ramp
        wave[-ramp_n:] *= ramp[::-1]
    return wave


def _generate_error_beep_samples(np: Any, sample_rate: int) -> Any:
    """Build a short, two-tone *descending* beep as float32 mono samples.

    Shape:
      * 660 Hz for 80 ms (with a 10 ms linear fade in/out to avoid clicks)
      * 30 ms silence
      * 440 Hz for 120 ms (same fade envelope)

    The descending pitch reads as "uh-oh" / negative confirmation.
    """
    gap = np.zeros(int(sample_rate * 0.030), dtype=np.float32)
    return np.concatenate([
        _beep_envelope(np, sample_rate, 660.0, 80),
        gap,
        _beep_envelope(np, sample_rate, 440.0, 120),
    ])


def _generate_success_beep_samples(np: Any, sample_rate: int) -> Any:
    """Build a short, two-tone *ascending* beep as float32 mono samples.

    Shape:
      * 523 Hz (C5) for 60 ms (10 ms fade in/out)
      * 20 ms silence
      * 784 Hz (G5) for 80 ms (same fade envelope)

    Slightly shorter overall than the error beep so the TTS reply still
    arrives promptly after the cue.  The rising pitch reads as
    "got it" / positive acknowledgement, mirroring the descending error
    beep's "didn't get it" feel.
    """
    gap = np.zeros(int(sample_rate * 0.020), dtype=np.float32)
    return np.concatenate([
        _beep_envelope(np, sample_rate, 523.0, 60),
        gap,
        _beep_envelope(np, sample_rate, 784.0, 80),
    ])


_BEEP_GENERATORS: Dict[str, Any] = {
    "error": _generate_error_beep_samples,
    "success": _generate_success_beep_samples,
}


def _get_beep_samples(np: Any, kind: str, sample_rate: int) -> Any:
    cache_key = (kind, sample_rate)
    cached = _BEEP_CACHE.get(cache_key)
    if cached is None:
        generator = _BEEP_GENERATORS.get(kind)
        if generator is None:
            raise ValueError(f"Unknown beep kind {kind!r}")
        cached = generator(np, sample_rate)
        _BEEP_CACHE[cache_key] = cached
    return cached


def _import_say_audio_deps():
    """Import numpy and sounddevice for `TextToSpeech.say`; raise MoonshineError if missing."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        raise MoonshineError(
            "TextToSpeech.say() requires numpy and sounddevice "
            "(e.g. `pip install numpy sounddevice`)."
        ) from e
    return np, sd


def _say_enumerate_output_devices(sd) -> List[Tuple[int, str]]:
    """PortAudio device indices with at least one output channel, and host API name."""
    out: List[Tuple[int, str]] = []
    try:
        devices = sd.query_devices()
    except (sd.PortAudioError, OSError, ValueError) as e:
        raise MoonshineAudioOutputError(
            f"Could not query audio devices: {e}",
            available_outputs=[],
        ) from e
    for i, d in enumerate(devices):
        try:
            n_out = int(d.get("max_output_channels", 0) or 0)
        except (TypeError, ValueError):
            n_out = 0
        if n_out <= 0:
            continue
        name = str(d.get("name", "") or "")
        out.append((i, name))
    return out


def _say_device_lines(outs: List[Tuple[int, str]]) -> List[str]:
    return [f"[{i}] {name}" for i, name in outs]


def _say_device_spec_key(device: Optional[Union[int, str]]) -> Tuple[Any, ...]:
    if device is None:
        return ("default",)
    if isinstance(device, int):
        return ("idx", device)
    s = str(device).strip()
    if not s:
        return ("default",)
    try:
        return ("idx", int(s, 10))
    except ValueError:
        return ("name", s.casefold())


# Sample rates probed when the output device rejects the synthesizer's
# native rate. 48 kHz is the modern default and is a clean 2x integer
# upsample from the 24 kHz the C core emits, so we try it first. The rest
# are common consumer-audio rates; selection prefers integer multiples
# (or divisors) of the source rate and only falls back to the nearest
# rate when no clean ratio is available.
_RESAMPLE_CANDIDATES: Tuple[int, ...] = (
    48000, 96000, 192000, 44100, 88200, 32000, 22050, 16000, 11025, 8000,
)


def _select_output_sample_rate(
    sd: Any, *, device: Optional[int], source_sr: int,
) -> Tuple[Optional[int], Optional[Exception]]:
    """Pick the best output sample rate the device will actually open.

    Returns ``(target_sr, last_error)``. ``target_sr`` is ``None`` if no
    candidate rate works (in which case ``last_error`` is the PortAudio
    error from the source-rate probe and useful for diagnostics).
    Otherwise ``target_sr`` is:

    1. ``source_sr`` if the device accepts it natively (no resampling),
    2. 48000 Hz when supported (cleanest fallback for 24 kHz input),
    3. otherwise the largest supported rate that is an integer multiple
       or divisor of ``source_sr``,
    4. otherwise the supported rate closest to ``source_sr``.
    """
    last_err: Optional[Exception] = None

    def _try(sr: int) -> bool:
        nonlocal last_err
        try:
            sd.check_output_settings(
                samplerate=sr, channels=1, dtype="float32", device=device,
            )
            return True
        except (sd.PortAudioError, OSError, ValueError) as e:
            last_err = e
            return False

    if _try(source_sr):
        return source_sr, None
    supported = [
        sr for sr in _RESAMPLE_CANDIDATES
        if sr != source_sr and _try(sr)
    ]
    if not supported:
        return None, last_err
    if 48000 in supported:
        return 48000, last_err
    multiples = [
        sr for sr in supported
        if sr % source_sr == 0 or source_sr % sr == 0
    ]
    if multiples:
        return max(multiples), last_err
    return min(supported, key=lambda sr: abs(sr - source_sr)), last_err


def _resample_linear(
    np: Any, samples: Any, source_sr: int, target_sr: int,
) -> Any:
    """Numpy-only linear-interpolation resample of mono float32 audio.

    Linear interpolation is the highest-quality resampler we can build
    without scipy; for clean integer ratios (e.g. 24 kHz -> 48 kHz) the
    new sample positions land exactly between source samples, so the
    interpolation error stays small.
    """
    if source_sr == target_sr or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n_src = int(samples.shape[0])
    n_dst = max(1, int(round(n_src * target_sr / source_sr)))
    if n_src == 1:
        return np.full(n_dst, float(samples[0]), dtype=np.float32)
    src_x = np.arange(n_src, dtype=np.float64)
    dst_x = np.linspace(0.0, n_src - 1, n_dst, dtype=np.float64)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def _say_resolve_output_index(
    spec_key: Tuple[Any, ...],
    outs: List[Tuple[int, str]],
    *,
    device_label_for_errors: str,
) -> Optional[int]:
    """Return PortAudio output device index, or None for the host default stream device."""
    lines = _say_device_lines(outs)
    if spec_key == ("default",):
        return None
    if spec_key[0] == "idx":
        idx = spec_key[1]
        valid = {i for i, _ in outs}
        if idx not in valid:
            raise MoonshineAudioOutputError(
                f"Output device index {idx} is not available or is not an output device.",
                available_outputs=lines,
            )
        return int(idx)
    needle = spec_key[1]
    for i, name in outs:
        if needle in name.casefold():
            return i
    raise MoonshineAudioOutputError(
        f"No output device name contains substring {device_label_for_errors!r} "
        "(match is case-insensitive substring).",
        available_outputs=lines,
    )


class TextToSpeech:
    """
    On-device TTS using Moonshine (Kokoro / Piper and language assets under ``g2p_root``).

    Required assets are resolved with ``moonshine_get_tts_dependencies`` and, unless you pass
    ``download=False`` and ``asset_root``, downloaded from ``https://download.moonshine.ai/tts/``.

    The ``language`` tag is validated against the native TTS catalog before download
    (`MoonshineTtsLanguageError` lists supported tags).     If ``voice`` is set, it is checked against
    voices reported as ``found`` for that language under the resolved asset root
    (`MoonshineTtsVoiceError` lists downloaded ids then catalog ids available for download).
    With ``download=True``, a voice that is catalogued but not yet on disk is downloaded automatically.
    Use a ``kokoro_`` or ``piper_`` prefix on ``voice`` to pin the vocoder (e.g. ``kokoro_af_heart``).
    Use `list_tts_languages` and
    `list_tts_voices` (``present`` / ``downloadable``) or `get_tts_voice_catalog` to discover options.
    """

    def __init__(
        self,
        language: str,
        *,
        voice: Optional[str] = None,
        options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        asset_root: Optional[Path] = None,
        download: bool = True,
        output_device: Optional[Union[int, str]] = None,
        volume: Optional[float] = None,
    ):
        self._extra_options = dict(options) if options else {}
        if voice is None:
            opt_voice = self._extra_options.get("voice")
            if isinstance(opt_voice, str):
                ov = opt_voice.strip()
                if ov:
                    voice = ov
        if voice is not None:
            vs = str(voice).strip()
            voice = vs if vs else None
        self._voice = voice
        # Voice is tracked separately in ``self._voice`` and re-added by
        # ``_c_options_for_create``; drop it from ``_extra_options`` so the catalog
        # lookup below (via ``list_tts_voices``) is not biased by a voice we are
        # still validating, and so downstream dependency resolution in
        # ``download_tts_assets`` cannot silently request a non-existent voice file.
        self._extra_options.pop("voice", None)
        if download:
            validate_root = tts_asset_cache_path(
                Path(asset_root) if asset_root is not None else None)
        else:
            if asset_root is None:
                raise MoonshineError(
                    "When download=False, asset_root must point to a directory "
                    "already populated with TTS assets (g2p_root layout)."
                )
            validate_root = Path(asset_root).resolve()
        self._language = validate_tts_language(
            language,
            voice=voice,
            options=self._extra_options,
            root_path=validate_root,
        )

        if self._voice is not None:
            validate_tts_voice_known(
                self._language,
                self._voice,
                options=self._extra_options,
                root_path=validate_root,
            )

        if download:
            self._asset_root = download_tts_assets(
                self._language,
                voice=voice,
                options=self._extra_options,
                cache_root=Path(
                    asset_root) if asset_root is not None else None,
            )
        else:
            self._asset_root = validate_root

        if self._voice is not None:
            ensure_tts_voice_downloaded(
                self._language,
                self._voice,
                self._asset_root,
                options=self._extra_options,
                download_missing=download,
                show_progress=True,
            )

        self._lib = _MoonshineLib().lib
        create_opts = self._c_options_for_create()
        opt_arr, opt_n, _keep = moonshine_options_array(create_opts)
        lang_b = self._language.encode("utf-8")
        handle = self._lib.moonshine_create_tts_synthesizer_from_files(
            lang_b,
            None,
            0,
            opt_arr,
            opt_n,
            MOONSHINE_HEADER_VERSION,
        )
        if handle < 0:
            msg = self._lib.moonshine_error_to_string(handle)
            raise MoonshineError(
                msg.decode(
                    "utf-8") if msg else f"Failed to create TTS synthesizer ({handle})"
            )
        self._handle = handle
        self._say_device_cache: Optional[Tuple[Tuple[Any, ...],
                                               Optional[int]]] = None
        # ((spec_key, source_sr), target_sr) — target_sr equals source_sr
        # when the device accepts the synthesizer rate natively; otherwise
        # it's the resample target chosen by _select_output_sample_rate.
        self._say_settings_ok: Optional[Tuple[Tuple[Tuple[Any, ...], int], int]] = None

        self._say_queue: queue.Queue = queue.Queue()
        self._play_queue: queue.Queue = queue.Queue(maxsize=1)
        self._say_stop_event = threading.Event()
        self._synth_thread: Optional[threading.Thread] = None
        self._play_thread: Optional[threading.Thread] = None
        self._say_lock = threading.Lock()
        self._output_device = output_device
        self._volume = volume
        
    def _c_options_for_create(self) -> Dict[str, Union[str, int, float, bool]]:
        merged: Dict[str, Union[str, int, float, bool]] = dict(
            self._extra_options)
        merged["g2p_root"] = str(self._asset_root)
        if self._voice is not None:
            merged["voice"] = self._voice
        return merged

    @property
    def language(self) -> str:
        return self._language

    @property
    def asset_root(self) -> Path:
        return self._asset_root

    def synthesize(
        self,
        text: str,
        *,
        speed: Optional[float] = None,
        volume: Optional[float] = None,
        options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> Tuple[List[float], int]:
        """Synthesize ``text`` to PCM float samples ``(-1..1)`` and sample rate in Hz."""

        if options is None:
            options = {}

        if speed is not None:
            options["speed"] = str(speed)

        if volume is not None:
            options["output_volume"] = str(volume)

        return moonshine_text_to_speech_samples(self._handle, text, options)

    def say(
        self,
        text: Union[str, List[str]],
        *,
        speed: Optional[float] = None,
        volume: Optional[float] = None,
        device: Optional[Union[int, str]] = None,
        options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> None:
        """
        Queue ``text`` for synthesis and playback, returning immediately.

        ``text`` may be a single string or a list of strings. A list is equivalent to calling
        ``say`` once per element in order.

        Utterances are played in order. Synthesis of the next utterance is pipelined with playback
        of the current one so there is minimal gap between consecutive utterances. Call `stop` to
        cancel all pending utterances and halt the currently-playing audio.

        Uses `synthesize` for audio generation; ``options`` is passed through unchanged.

        ``device`` may be ``None`` (host default output), a PortAudio device index, a decimal string
        index, or a substring of the device name (case-insensitive). If there are no output devices,
        the index is invalid, or no name matches, the error is raised on the calling thread before
        the utterance is enqueued.
        """
        _import_say_audio_deps()

        if self._output_device is not None:
            device = self._output_device

        if self._volume is not None:
            volume = self._volume

        texts = text if isinstance(text, list) else [text]
        for t in texts:
            self._say_queue.put(_SayRequest(
                text=t,
                speed=speed,
                volume=volume,
                device=device,
                options=options,
            ))
        self._ensure_say_workers()

    def play_error(
        self,
        *,
        device: Optional[Union[int, str]] = None,
    ) -> None:
        """Play a short two-tone *descending* "error" beep and return immediately.

        The beep is generated synthetically (no text synthesis, no
        language or voice assets needed) and queued for playback through
        the same pipeline as :meth:`say` — so if a previous ``say``
        hasn't finished speaking yet the beep plays right after it,
        rather than racing ahead.  Use :meth:`wait` / :meth:`is_talking`
        to track playback.

        Pairs with :meth:`play_success`: callers that want audible
        feedback for whether speech recognition succeeded can call
        :meth:`play_success` on a recognized utterance and
        :meth:`play_error` on an unrecognized one.

        ``device`` accepts the same values as :meth:`say` (``None`` =
        host default, a PortAudio index, a decimal string index, or a
        case-insensitive device-name substring).
        """
        _import_say_audio_deps()
        self._say_queue.put(_BeepRequest(device=device, kind="error"))
        self._ensure_say_workers()

    def play_success(
        self,
        *,
        device: Optional[Union[int, str]] = None,
    ) -> None:
        """Play a short two-tone *ascending* "success" beep and return immediately.

        Counterpart to :meth:`play_error` for positive feedback: a brief
        rising chirp confirming that the most recent utterance was
        recognized / accepted.  Same queueing rules as
        :meth:`play_error` — synthesized once, cached, and ordered
        through the say queue so it never races ahead of an in-flight
        :meth:`say`.

        ``device`` accepts the same values as :meth:`say`.
        """
        _import_say_audio_deps()
        self._say_queue.put(_BeepRequest(device=device, kind="success"))
        self._ensure_say_workers()

    def _ensure_say_workers(self) -> None:
        with self._say_lock:
            alive = (
                self._synth_thread is not None and self._synth_thread.is_alive()
                and self._play_thread is not None and self._play_thread.is_alive()
            )
            if alive:
                return
            self._say_stop_event.clear()
            st = threading.Thread(target=self._synth_worker, daemon=True)
            pt = threading.Thread(target=self._play_worker, daemon=True)
            st.start()
            pt.start()
            self._synth_thread = st
            self._play_thread = pt

    # -- synthesis thread ----------------------------------------------------

    def _synth_worker(self) -> None:
        np, _ = _import_say_audio_deps()

        while not self._say_stop_event.is_set():
            try:
                req = self._say_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if req is _SHUTDOWN_SENTINEL:
                self._say_queue.task_done()
                break

            if self._say_stop_event.is_set():
                self._say_queue.task_done()
                break

            try:
                if isinstance(req, _BeepRequest):
                    item = _PlayItem(
                        data=_get_beep_samples(np, req.kind, _BEEP_SAMPLE_RATE),
                        sample_rate=_BEEP_SAMPLE_RATE,
                        device=req.device,
                    )
                else:
                    item = self._synthesize_one(req, np)
                if not self._say_stop_event.is_set():
                    self._play_queue.put(item)
            except Exception:
                print(
                    "TextToSpeech: synthesis worker dropped an utterance:",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
            finally:
                self._say_queue.task_done()

    def _synthesize_one(self, req: _SayRequest, np: Any) -> _PlayItem:
        """Synthesize a single utterance into a _PlayItem (runs on synthesis thread)."""
        samples, sr = self.synthesize(
            req.text, speed=req.speed, volume=req.volume, options=req.options)
        data = np.asarray(samples, dtype=np.float32)
        return _PlayItem(data=data, sample_rate=int(sr), device=req.device)

    # -- playback thread -----------------------------------------------------

    def _play_worker(self) -> None:
        np, sd = _import_say_audio_deps()

        while not self._say_stop_event.is_set():
            try:
                item = self._play_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _SHUTDOWN_SENTINEL:
                self._play_queue.task_done()
                break

            if self._say_stop_event.is_set():
                self._play_queue.task_done()
                break

            try:
                self._play_one(item, sd, np)
            except Exception:
                print(
                    "TextToSpeech: playback worker failed to play an utterance:",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
            finally:
                self._play_queue.task_done()

    def _play_one(self, item: _PlayItem, sd: Any, np: Any) -> None:
        """Resolve the device and play a single synthesized utterance (runs on playback thread)."""
        spec_key = _say_device_spec_key(item.device)
        if isinstance(item.device, str):
            name_label = item.device.strip()
        elif item.device is not None:
            name_label = str(item.device)
        else:
            name_label = ""

        if self._say_device_cache is None or self._say_device_cache[0] != spec_key:
            outs = _say_enumerate_output_devices(sd)
            if not outs:
                raise MoonshineAudioOutputError(
                    "No audio output devices are available.",
                    available_outputs=[],
                )
            resolved = _say_resolve_output_index(
                spec_key,
                outs,
                device_label_for_errors=name_label or str(
                    item.device) if item.device is not None else "",
            )
            self._say_device_cache = (spec_key, resolved)
            self._say_settings_ok = None

        resolved = self._say_device_cache[1]

        if self._say_stop_event.is_set():
            return

        settings_key = (spec_key, item.sample_rate)
        cached = self._say_settings_ok
        if cached is not None and cached[0] == settings_key:
            target_sr = cached[1]
        else:
            target_sr, last_err = _select_output_sample_rate(
                sd, device=resolved, source_sr=item.sample_rate,
            )
            if target_sr is None:
                outs = _say_enumerate_output_devices(sd)
                tried = ", ".join(
                    str(sr) for sr in (item.sample_rate,) + _RESAMPLE_CANDIDATES
                )
                err_suffix = f": {last_err}" if last_err is not None else ""
                raise MoonshineAudioOutputError(
                    f"Audio output {resolved!r} rejected every probed sample rate "
                    f"({tried} Hz) for mono float32{err_suffix}.",
                    available_outputs=_say_device_lines(outs),
                ) from last_err
            self._say_settings_ok = (settings_key, target_sr)
            if target_sr != item.sample_rate:
                print(
                    f"TextToSpeech: output device does not support "
                    f"{item.sample_rate} Hz; resampling to {target_sr} Hz.",
                    file=sys.stderr,
                )

        if self._say_stop_event.is_set():
            return

        if target_sr != item.sample_rate:
            data = _resample_linear(
                np, item.data, item.sample_rate, target_sr)
        else:
            data = item.data

        try:
            sd.play(data, target_sr, device=resolved)
            while sd.get_stream().active:
                if self._say_stop_event.is_set():
                    sd.stop()
                    return
                self._say_stop_event.wait(timeout=0.05)
        except (sd.PortAudioError, OSError, ValueError) as e:
            outs = _say_enumerate_output_devices(sd)
            raise MoonshineAudioOutputError(
                f"Failed to play audio: {e}",
                available_outputs=_say_device_lines(outs),
            ) from e

    def is_talking(self) -> bool:
        """Return ``True`` if utterances are queued, being synthesized, or currently playing."""
        if not self._say_queue.empty() or not self._play_queue.empty():
            return True
        try:
            _, sd = _import_say_audio_deps()
            stream = sd.get_stream()
            if stream is not None and stream.active:
                return True
        except Exception:
            pass
        return False

    def wait(self) -> None:
        """Block until all queued utterances have been synthesized and played."""
        self._say_queue.join()
        self._play_queue.join()

    def stop(self) -> None:
        """Clear the utterance queue and stop any audio currently playing.

        Returns once all pending utterances are discarded and the active playback (if any) has
        been halted. It is safe to call `say` again afterwards.
        """
        self._say_stop_event.set()

        for q in (self._say_queue, self._play_queue):
            while True:
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break

        _, sd = _import_say_audio_deps()
        try:
            sd.stop()
        except Exception:
            pass

        for thread in (self._synth_thread, self._play_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        self._synth_thread = None
        self._play_thread = None

    def close(self) -> None:
        if getattr(self, "_say_queue", None) is not None:
            self._say_stop_event.set()

            for q in (self._say_queue, self._play_queue):
                while True:
                    try:
                        q.get_nowait()
                        q.task_done()
                    except queue.Empty:
                        break

            try:
                _, sd = _import_say_audio_deps()
                sd.stop()
            except Exception:
                pass

            for thread in (self._synth_thread, self._play_thread):
                if thread is not None and thread.is_alive():
                    thread.join(timeout=2.0)
            self._synth_thread = None
            self._play_thread = None
        self._say_device_cache = None
        self._say_settings_ok = None
        if getattr(self, "_handle", None) is not None:
            self._lib.moonshine_free_tts_synthesizer(self._handle)
            self._handle = None

    def __enter__(self) -> "TextToSpeech":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _parse_options_cli(pairs: List[str]) -> Dict[str, Union[str, int, float, bool]]:
    """Parse ``KEY=value`` pairs from repeated ``--options`` (values stay strings except booleans)."""
    out: Dict[str, Union[str, int, float, bool]] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(
                f"Invalid --options entry {item!r}; expected KEY=value (use quotes if needed)"
            )
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --options entry {item!r}; key is empty")
        v = value.strip()
        low = v.lower()
        if low == "true":
            out[key] = True
        elif low == "false":
            out[key] = False
        else:
            out[key] = v
    return out


def _write_wav_mono_pcm16(path: Path, samples: List[float], sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate_hz)
        frames = bytearray()
        for x in samples:
            if not isinstance(x, float):
                x = float(x)
            if x > 1.0:
                x = 1.0
            elif x < -1.0:
                x = -1.0
            pcm = int(round(x * 32767.0))
            if pcm > 32767:
                pcm = 32767
            elif pcm < -32768:
                pcm = -32768
            frames.extend(pcm.to_bytes(2, byteorder="little", signed=True))
        w.writeframes(bytes(frames))


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Synthesize speech with Moonshine TTS (downloads assets if needed)."
    )
    parser.add_argument(
        "--language",
        default="en_us",
        help="Language tag (e.g. en_us, de, fr) (default: %(default)s)",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice id with kokoro_ or piper_ prefix (e.g. kokoro_af_heart)",
    )
    parser.add_argument(
        "--text",
        default="Hello world!",
        help="Text to speak (default: %(default)r)",
    )
    out_or_device = parser.add_mutually_exclusive_group()
    out_or_device.add_argument(
        "--out",
        default=None,
        type=Path,
        metavar="PATH",
        help="Write mono PCM16 WAV to PATH (omit to play on default output or fall back to out.wav)",
    )
    out_or_device.add_argument(
        "--device",
        default=None,
        metavar="INDEX_OR_NAME",
        help="sounddevice output device (index or name substring)",
    )
    parser.add_argument(
        "--asset-root",
        default=None,
        metavar="PATH",
        help="Path to the asset root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--options",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra moonshine_option_t entries; repeat for multiple (e.g. --options speed=1.1)",
    )
    args = parser.parse_args()

    if args.asset_root is None:
        args.asset_root = tts_asset_cache_path(None)

    extra: Dict[str, Union[str, int, float, bool]] = {
        "g2p_root": str(args.asset_root)}
    if args.options:
        try:
            for k, v in _parse_options_cli(args.options).items():
                extra[k] = v
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(2)

    try:
        with TextToSpeech(
            args.language,
            voice=args.voice,
            options=extra,
        ) as tts:
            if args.out is not None:
                samples, sr = tts.synthesize(args.text, options=extra)
                _write_wav_mono_pcm16(args.out, samples, sr)
            else:
                dev = None
                if args.device is not None:
                    t = args.device.strip()
                    dev = t if t else None
                try:
                    tts.say(args.text, device=dev, options=extra)
                    tts.wait()
                except MoonshineError as play_err:
                    fallback_path = Path("out.wav")
                    print(
                        f"Audio playback failed ({play_err}); writing {fallback_path}",
                        file=sys.stderr,
                    )
                    samples, sr = tts.synthesize(args.text, options=extra)
                    _write_wav_mono_pcm16(fallback_path, samples, sr)
    except MoonshineError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
