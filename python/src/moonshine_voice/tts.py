"""Text-to-speech via the Moonshine C API."""

import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from moonshine_voice.download import (
    download_tts_assets,
    ensure_tts_voice_downloaded,
    tts_asset_cache_path,
    validate_tts_language,
)
from moonshine_voice.errors import MoonshineAudioOutputError, MoonshineError
from moonshine_voice.moonshine_api import (
    MOONSHINE_HEADER_VERSION,
    _MoonshineLib,
    moonshine_options_array,
    moonshine_text_to_speech_samples,
)


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
    ):
        self._extra_options = dict(options) if options else {}
        if voice is not None:
            vs = str(voice).strip()
            voice = vs if vs else None
        self._voice = voice
        if download:
            validate_root = tts_asset_cache_path(Path(asset_root) if asset_root is not None else None)
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

        if download:
            self._asset_root = download_tts_assets(
                self._language,
                voice=voice,
                options=self._extra_options,
                cache_root=Path(asset_root) if asset_root is not None else None,
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
                msg.decode("utf-8") if msg else f"Failed to create TTS synthesizer ({handle})"
            )
        self._handle = handle
        self._say_device_cache: Optional[Tuple[Tuple[Any, ...], Optional[int]]] = None
        self._say_settings_ok: Optional[Tuple[Tuple[Any, ...], int]] = None

    def _c_options_for_create(self) -> Dict[str, Union[str, int, float, bool]]:
        merged: Dict[str, Union[str, int, float, bool]] = dict(self._extra_options)
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
        options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> Tuple[List[float], int]:
        """Synthesize ``text`` to PCM float samples ``(-1..1)`` and sample rate in Hz."""
        return moonshine_text_to_speech_samples(self._handle, text, options)

    def say(
        self,
        text: str,
        *,
        device: Optional[Union[int, str]] = None,
        options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> None:
        """
        Synthesize ``text`` and play it on an audio output device via ``sounddevice``.

        Uses `synthesize` for audio generation; ``options`` is passed through unchanged.

        ``device`` may be ``None`` (host default output), a PortAudio device index, a decimal string
        index, or a substring of the device name (case-insensitive). If there are no output devices,
        the index is invalid, or no name matches, raises `MoonshineAudioOutputError` with a list of
        available devices. Device resolution and successful `check_output_settings` results are
        cached per ``device`` argument and sample rate so repeated calls avoid re-querying hardware.
        """
        np, sd = _import_say_audio_deps()
        spec_key = _say_device_spec_key(device)
        if isinstance(device, str):
            name_label = device.strip()
        elif device is not None:
            name_label = str(device)
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
                device_label_for_errors=name_label or str(device) if device is not None else "",
            )
            self._say_device_cache = (spec_key, resolved)
            self._say_settings_ok = None

        resolved = self._say_device_cache[1]
        samples, sr = self.synthesize(text, options)
        sample_rate = int(sr)

        settings_key = (spec_key, sample_rate)
        if self._say_settings_ok != settings_key:
            try:
                sd.check_output_settings(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    device=resolved,
                )
            except (sd.PortAudioError, OSError, ValueError) as e:
                outs = _say_enumerate_output_devices(sd)
                raise MoonshineAudioOutputError(
                    f"Audio output cannot play {sample_rate} Hz mono float32 on the selected device: {e}",
                    available_outputs=_say_device_lines(outs),
                ) from e
            self._say_settings_ok = settings_key

        data = np.asarray(samples, dtype=np.float32)
        try:
            sd.play(data, sample_rate, device=resolved)
            sd.wait()
        except (sd.PortAudioError, OSError, ValueError) as e:
            outs = _say_enumerate_output_devices(sd)
            raise MoonshineAudioOutputError(
                f"Failed to play audio: {e}",
                available_outputs=_say_device_lines(outs),
            ) from e

    def close(self) -> None:
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

    extra: Optional[Dict[str, Union[str, int, float, bool]]] = {
        "g2p_root": str(args.asset_root),
    }
    if args.options:
        try:
            extra = _parse_options_cli(args.options)
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
