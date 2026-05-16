"""Text-to-speech via the Moonshine C API."""

import queue
import sys
import threading
import time
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

    ``kind`` selects which packaged WAV to play:

    * ``"error"`` – ``assets/error.wav``, played to signal a
      misrecognition or a rejected utterance.
    * ``"success"`` – ``assets/success.wav``, played to signal a
      recognized utterance, just before the TTS response.
    """
    device: Optional[Union[int, str]]
    kind: str = "error"


_SHUTDOWN_SENTINEL = object()

# Beep WAVs shipped with the package.  The bundled clips are short,
# pre-recorded cues that already include any lead-in / fade
# the recording artist wanted; the runner just decodes and plays them
# verbatim through the same pipeline as a synthesized utterance, so
# the same device-resolution / resampling / mute path applies.
_BEEP_ASSET_FILES: Dict[str, str] = {
    "success": "success.wav",
    "error": "error.wav",
}

# Cache of decoded beep waveforms keyed by ``kind``.  Each entry is a
# ``(samples_float32, sample_rate)`` pair — the WAVs ship at 48 kHz
# and the runner converts to numpy float32 mono once per process,
# so subsequent ``play_success`` / ``play_error`` calls only pay the
# queue-handoff cost.  ``load_wav_file`` mixes stereo down to mono
# automatically, which is exactly what the playback worker expects.
_BEEP_CACHE: Dict[str, Tuple[Any, int]] = {}


def _load_beep_samples(np: Any, kind: str) -> Tuple[Any, int]:
    """Return ``(samples_float32, sample_rate)`` for the named beep.

    Loaded lazily from the packaged ``assets/<kind>.wav`` file the first
    time it's requested and cached for the lifetime of the process.
    Raises :class:`ValueError` for unknown kinds and lets file errors
    from :func:`load_wav_file` propagate (so a missing asset is loud,
    not silent — the symptom otherwise would be the same "no beep"
    issue we just spent two iterations debugging).
    """
    cached = _BEEP_CACHE.get(kind)
    if cached is not None:
        return cached
    filename = _BEEP_ASSET_FILES.get(kind)
    if filename is None:
        raise ValueError(f"Unknown beep kind {kind!r}")
    # Imported lazily to keep ``moonshine_voice.tts`` import-light
    # for callers that never invoke ``play_success`` / ``play_error``.
    from moonshine_voice.utils import get_assets_path, load_wav_file

    path = get_assets_path() / filename
    samples_list, sr = load_wav_file(path)
    samples = np.asarray(samples_list, dtype=np.float32)
    cached = (samples, int(sr))
    _BEEP_CACHE[kind] = cached
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


def _resolve_default_output_index(sd: Any) -> Optional[int]:
    """Return PortAudio's current default output device index, if any.

    Wraps ``sd.default.device`` because sounddevice exposes that as a
    custom ``_InputOutputPair`` NamedTuple, *not* a plain ``tuple`` /
    ``list``, so an ``isinstance(..., (tuple, list))`` check on it
    misses and the diagnostic flag would silently never light up.
    Falls back through several access patterns so a minor sounddevice
    API change can't suppress the "current default" annotation.
    """
    try:
        default_pair = sd.default.device
    except Exception:
        return None
    try:
        out_idx = default_pair[1]
    except (TypeError, IndexError):
        try:
            out_idx = getattr(default_pair, "output", None)
        except Exception:
            out_idx = None
        if out_idx is None:
            try:
                out_idx = int(default_pair)
            except (TypeError, ValueError):
                return None
    if out_idx is None or out_idx == -1:
        return None
    try:
        return int(out_idx)
    except (TypeError, ValueError):
        return None


def list_output_devices() -> List[str]:
    """Return human-readable output device descriptions for diagnostics.

    Each entry is formatted ``"[idx] name (hostapi: NAME)"`` so a
    caller can paste either the index or a substring of the name into
    :class:`TextToSpeech`'s ``output_device`` argument (or the
    ``--output-device`` flag of the CLI demo).  The first line of the
    returned list flags PortAudio's default output device — on
    Raspberry Pi this is often *not* the one with speakers attached
    (e.g. HDMI when the user has wired up the 3.5 mm jack), and a
    silent assistant with no errors in the logs is exactly the
    symptom of the wrong device being selected.

    Use :class:`TextToSpeech`'s ``output_device`` to pin to a
    specific one when ``None`` (= host default) doesn't reach a
    speaker.
    """
    _, sd = _import_say_audio_deps()
    lines: List[str] = []
    default_out = _resolve_default_output_index(sd)
    try:
        hostapis = list(sd.query_hostapis())
    except Exception:
        hostapis = []
    try:
        devices = sd.query_devices()
    except Exception as e:
        return [f"<sounddevice query_devices() failed: {e!r}>"]
    for i, d in enumerate(devices):
        try:
            n_out = int(d.get("max_output_channels", 0) or 0)
        except (TypeError, ValueError):
            n_out = 0
        if n_out <= 0:
            continue
        name = str(d.get("name", "") or "")
        hostapi_idx = d.get("hostapi", -1)
        hostapi_name = ""
        try:
            hostapi_name = hostapis[hostapi_idx]["name"]
        except (IndexError, KeyError, TypeError):
            pass
        try:
            sr = int(d.get("default_samplerate", 0) or 0)
        except (TypeError, ValueError):
            sr = 0
        marker = "*" if i == default_out else " "
        lines.append(
            f"{marker} [{i}] {name} (hostapi: {hostapi_name or '?'}, "
            f"channels: {n_out}, default_sr: {sr or '?'} Hz)"
        )
    if not lines:
        lines.append("<no PortAudio output devices available>")
    return lines


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

    Probes that fail with ``paDeviceUnavailable`` (-9985) are retried
    a few times with a short backoff: that error code is transient on
    exclusive-access ALSA devices (e.g. USB DACs) when a previous
    ``sd.play()`` stream hasn't fully released the device yet, and a
    single failed probe shouldn't be enough to disqualify a rate the
    device actually supports.
    """
    last_err: Optional[Exception] = None

    def _try(sr: int) -> bool:
        nonlocal last_err
        # paDeviceUnavailable on Linux/ALSA after a just-finished play
        # commonly clears within ~10-50 ms; give it up to ~200 ms total
        # before believing the device really doesn't support this rate.
        for attempt in range(5):
            try:
                sd.check_output_settings(
                    samplerate=sr, channels=1, dtype="float32", device=device,
                )
                return True
            except sd.PortAudioError as e:
                last_err = e
                # PortAudioError stores the numeric code in ``args[1]``
                # for sounddevice; -9985 == paDeviceUnavailable.
                code = e.args[1] if len(e.args) >= 2 else None
                if code == -9985 and attempt < 4:
                    time.sleep(0.05)
                    continue
                return False
            except (OSError, ValueError) as e:
                last_err = e
                return False
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
        debug: bool = False,
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
        self._debug = bool(debug)
        self._log_start: Optional[float] = None
        self._log_last: Optional[float] = None
        self._log_lock = threading.Lock()
        # Set on the first `_play_one` call to a one-line summary of
        # which PortAudio device the runner actually opened — printed
        # unconditionally so a silent setup with no errors in the
        # logs (often a wrong-default-device problem on Raspberry Pi)
        # is debuggable without needing --debug.
        self._announced_resolved_device = False

    def _announce_resolved_device(self, sd: Any, resolved: Optional[int]) -> None:
        """Print a one-liner identifying the PortAudio device we opened.

        Always prints (regardless of ``debug``): a silent runtime with
        no errors in the trace is almost always the *wrong* PortAudio
        device being selected (e.g. HDMI on a Pi when speakers are on
        the 3.5 mm jack), and the only way to spot that from the
        logs is to see *which* device the worker actually opened.

        Includes the host-API name and the default sample rate the
        device claims to support; on top of that, when the resolved
        device is the host default we list the other available
        outputs so the user can pin a specific one via
        ``TextToSpeech(output_device=...)`` (or the CLI's
        ``--output-device`` flag).
        """
        try:
            if resolved is None:
                idx = _resolve_default_output_index(sd)
                origin = "host default"
            else:
                idx = resolved
                origin = "explicit"
            info = sd.query_devices(idx) if idx is not None else None
            try:
                hostapis = list(sd.query_hostapis())
            except Exception:
                hostapis = []
            hostapi_name = ""
            if info is not None:
                try:
                    hostapi_name = hostapis[info.get("hostapi", -1)]["name"]
                except (IndexError, KeyError, TypeError):
                    pass
            name = info.get("name", "?") if info else "?"
            default_sr = info.get("default_samplerate", "?") if info else "?"
            print(
                f"TextToSpeech: opening PortAudio output [{idx}] {name!r} "
                f"({origin}, hostapi: {hostapi_name or '?'}, "
                f"default_sr: {default_sr} Hz). "
                f"If you don't hear anything, this may be the wrong "
                f"device — list alternatives with "
                f"`moonshine_voice.tts.list_output_devices()` and pin "
                f"one via `TextToSpeech(output_device=...)`.",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:
            print(
                f"TextToSpeech: could not introspect resolved output "
                f"device (resolved={resolved!r}): {e!r}",
                file=sys.stderr,
                flush=True,
            )

    def _log(self, msg: str) -> None:
        """Emit a timestamped trace line to stderr when ``debug=True``.

        Same shape as ``DialogFlow._log`` so traces from the two
        components can be read together: each line shows the wall
        time since the previous log line and since the first log
        line of this instance.  Off by default — TTS traces are
        verbose and only useful for diagnosing playback problems
        (e.g. why a beep didn't seem to play).
        """
        if not self._debug:
            return
        with self._log_lock:
            now = time.perf_counter()
            if self._log_start is None:
                self._log_start = now
                self._log_last = now
            delta_ms = (now - (self._log_last or now)) * 1000.0
            total_ms = (now - (self._log_start or now)) * 1000.0
            self._log_last = now
            print(
                f"[TextToSpeech +{delta_ms:7.1f}ms / {total_ms:8.1f}ms] {msg}",
                file=sys.stderr,
                flush=True,
            )

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
        """Play the bundled "error" beep and return immediately.

        Plays ``assets/error.wav`` queued through the same pipeline as
        :meth:`say` — so if a previous ``say`` hasn't finished speaking
        yet the beep plays right after it, rather than racing ahead.
        Use :meth:`wait` / :meth:`is_talking` to track playback.

        Pairs with :meth:`play_success`: callers that want audible
        feedback for whether speech recognition succeeded can call
        :meth:`play_success` on a recognized utterance and
        :meth:`play_error` on an unrecognized one.

        ``device`` accepts the same values as :meth:`say` (``None`` =
        constructor's ``output_device`` if set, otherwise host
        default; a PortAudio index, a decimal string index, or a
        case-insensitive device-name substring).  Honouring the
        constructor's ``output_device`` here matches :meth:`say`'s
        behaviour — without that, callers who pin a specific output
        device for speech would get their cue beeps routed to the
        host default and end up with audible speech but inaudible
        beeps.
        """
        _import_say_audio_deps()
        if device is None and self._output_device is not None:
            device = self._output_device
        self._log(f"play_error: enqueue (device={device!r})")
        self._say_queue.put(_BeepRequest(device=device, kind="error"))
        self._ensure_say_workers()

    def play_success(
        self,
        *,
        device: Optional[Union[int, str]] = None,
    ) -> None:
        """Play the bundled "success" beep and return immediately.

        Counterpart to :meth:`play_error` for positive feedback: plays
        ``assets/success.wav`` confirming that the most recent
        utterance was recognized / accepted.  Same queueing rules as
        :meth:`play_error` — decoded once, cached, and ordered
        through the say queue so it never races ahead of an in-flight
        :meth:`say`.

        ``device`` accepts the same values as :meth:`say` and falls
        back to the constructor's ``output_device`` when ``None``,
        for the same reason as :meth:`play_error`.
        """
        _import_say_audio_deps()
        if device is None and self._output_device is not None:
            device = self._output_device
        self._log(f"play_success: enqueue (device={device!r})")
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
                    self._log(
                        f"synth_worker: dequeued beep kind={req.kind!r}"
                    )
                    samples, beep_sr = _load_beep_samples(np, req.kind)
                    item = _PlayItem(
                        data=samples,
                        sample_rate=beep_sr,
                        device=req.device,
                    )
                    self._log(
                        f"synth_worker: beep ready ({req.kind!r}, "
                        f"{len(item.data)} samples @ {item.sample_rate} Hz)"
                    )
                else:
                    self._log(
                        "synth_worker: dequeued say request"
                        f" (text={(req.text or '')[:40]!r})"
                    )
                    item = self._synthesize_one(req, np)
                    self._log(
                        f"synth_worker: synth done "
                        f"({len(item.data)} samples @ {item.sample_rate} Hz)"
                    )
                if not self._say_stop_event.is_set():
                    self._log("synth_worker: handing item to play_queue")
                    self._play_queue.put(item)
                    self._log("synth_worker: item accepted by play_queue")
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

        if not self._announced_resolved_device:
            self._announced_resolved_device = True
            self._announce_resolved_device(sd, resolved)

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

        expected_duration_s = (len(data) / float(target_sr)) if target_sr else 0.0
        self._log(
            f"play_one: starting sd.play "
            f"({len(data)} samples @ {target_sr} Hz, "
            f"~{expected_duration_s * 1000.0:.1f} ms expected, "
            f"device={resolved!r})"
        )
        t_start = time.perf_counter()
        try:
            sd.play(data, target_sr, device=resolved)
            t_after_play = time.perf_counter()
            # Snapshot ``stream.active`` immediately after sd.play() so
            # we can tell, after the fact, whether the playback ever
            # actually started.  ``sd.play`` returns once the stream
            # has been opened and ``start()`` has been called, but on
            # some backends the driver takes a few ms to ramp up — for
            # very short clips (like the ~160 ms success beep) the
            # stream can transition False → True → False entirely
            # between two polls of ``get_stream().active``, which
            # makes the worker think the item was played even though
            # nothing reached the speakers.  Logging the initial /
            # final state plus elapsed time vs. expected duration
            # makes that race visible.
            try:
                initial_active = bool(sd.get_stream().active)
            except Exception:
                initial_active = False
            self._log(
                f"play_one: sd.play returned in "
                f"{(t_after_play - t_start) * 1000.0:.1f} ms; "
                f"stream.active={initial_active}"
            )
            poll_count = 0
            while sd.get_stream().active:
                poll_count += 1
                if self._say_stop_event.is_set():
                    sd.stop()
                    self._log(
                        f"play_one: stop_event set after "
                        f"{(time.perf_counter() - t_start) * 1000.0:.1f} ms; "
                        f"calling sd.stop"
                    )
                    return
                self._say_stop_event.wait(timeout=0.05)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self._log(
                f"play_one: playback done after {elapsed_ms:.1f} ms "
                f"(expected ~{expected_duration_s * 1000.0:.1f} ms, "
                f"polls={poll_count})"
            )
            # Explicitly tear down the global play stream once the
            # buffer is exhausted.  ``sd.play()`` leaves the stream
            # *inactive but open* by default, which on Linux/ALSA
            # exclusive-access devices (e.g. plain ``hw:`` USB DACs)
            # means PortAudio keeps the device claimed.  The next
            # ``sd.check_output_settings()`` then fails with
            # ``paDeviceUnavailable`` for every probed rate, even
            # rates the device just played at — and the play worker
            # raises and drops the next utterance.  Calling
            # ``sd.stop()`` here (with ``ignore_errors=True``)
            # releases the device promptly.  ``sd.play()`` will
            # transparently re-open it for the next item.
            try:
                sd.stop(ignore_errors=True)
                self._log("play_one: sd.stop after natural completion")
            except Exception as stop_err:  # pragma: no cover - defensive
                self._log(f"play_one: sd.stop after completion raised: {stop_err!r}")
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
