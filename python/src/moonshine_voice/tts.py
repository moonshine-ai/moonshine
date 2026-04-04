"""Text-to-speech via the Moonshine C API."""

import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from moonshine_voice.download import (
    download_tts_assets,
    normalize_moonshine_language_tag,
)
from moonshine_voice.errors import MoonshineError
from moonshine_voice.moonshine_api import (
    MOONSHINE_HEADER_VERSION,
    _MoonshineLib,
    moonshine_options_array,
    moonshine_text_to_speech_samples,
)


class TextToSpeech:
    """
    On-device TTS using Moonshine (Kokoro / Piper and language assets under ``g2p_root``).

    Required assets are resolved with ``moonshine_get_tts_dependencies`` and, unless you pass
    ``download=False`` and ``asset_root``, downloaded from ``https://download.moonshine.ai/tts/``.
    """

    def __init__(
        self,
        language: str,
        *,
        voice: Optional[str] = None,
        engine: Optional[str] = None,
        options: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        asset_root: Optional[Path] = None,
        download: bool = True,
    ):
        self._language = normalize_moonshine_language_tag(language)
        self._voice = voice
        self._engine = engine
        self._extra_options = dict(options) if options else {}

        if download:
            self._asset_root = download_tts_assets(
                self._language,
                voice=voice,
                engine=engine,
                options=self._extra_options,
                cache_root=Path(asset_root) if asset_root is not None else None,
            )
        else:
            if asset_root is None:
                raise MoonshineError(
                    "When download=False, asset_root must point to a directory "
                    "already populated with TTS assets (g2p_root layout)."
                )
            self._asset_root = Path(asset_root).resolve()

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

    def _c_options_for_create(self) -> Dict[str, Union[str, int, float, bool]]:
        merged: Dict[str, Union[str, int, float, bool]] = dict(self._extra_options)
        merged["g2p_root"] = str(self._asset_root)
        if self._voice is not None:
            merged["voice"] = self._voice
        if self._engine is not None:
            merged["vocoder_engine"] = self._engine
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

    def close(self) -> None:
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


def _parse_sounddevice_output_device(s: str) -> Union[int, str]:
    """Accept device index or a name substring for sounddevice."""
    t = s.strip()
    try:
        return int(t, 10)
    except ValueError:
        return t


def _can_play_default(sample_rate_hz: int) -> bool:
    import sounddevice as sd

    try:
        sd.check_output_settings(
            samplerate=sample_rate_hz,
            channels=1,
            dtype="float32",
            device=None,
        )
        return True
    except (sd.PortAudioError, OSError, ValueError):
        return False


def _play_mono_samples(
    samples: List[float],
    sample_rate_hz: int,
    *,
    device: Optional[Union[int, str]] = None,
) -> None:
    import numpy as np
    import sounddevice as sd

    data = np.asarray(samples, dtype=np.float32)
    sd.play(data, sample_rate_hz, device=device)
    sd.wait()


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
        "--engine",
        default=None,
        help="Vocoder: kokoro, piper, or auto (maps to vocoder_engine)",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Kokoro voice id or Piper ONNX stem",
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
        "--options",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra moonshine_option_t entries; repeat for multiple (e.g. --options speed=1.1)",
    )
    args = parser.parse_args()

    extra: Optional[Dict[str, Union[str, int, float, bool]]] = None
    if args.options:
        try:
            extra = _parse_options_cli(args.options)
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(2)

    sd_device: Optional[Union[int, str]] = None
    if args.device is not None:
        sd_device = _parse_sounddevice_output_device(args.device)

    try:
        with TextToSpeech(
            args.language,
            voice=args.voice,
            engine=args.engine,
            options=extra,
        ) as tts:
            samples, sr = tts.synthesize(args.text)

        if args.device is not None:
            _play_mono_samples(samples, sr, device=sd_device)
        elif args.out is not None:
            _write_wav_mono_pcm16(args.out, samples, sr)
        else:
            fallback_path = Path("out.wav")
            if _can_play_default(sr):
                _play_mono_samples(samples, sr, device=None)
            else:
                print(
                    "No audio output found, writing to out.wav",
                    file=sys.stderr,
                )
                _write_wav_mono_pcm16(fallback_path, samples, sr)
    except MoonshineError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
