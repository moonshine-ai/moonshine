"""Utility functions for Moonshine Voice."""

import os
import struct
from pathlib import Path
from typing import Tuple


def get_assets_path() -> Path:
    """
    Get the path to the assets directory included in the package.

    Returns:
        Path object pointing to the assets directory

    Example:
        >>> from moonshine_voice.utils import get_assets_path
        >>> assets_path = get_assets_path()
        >>> model_path = assets_path / "tiny-en"
    """
    # Get the directory where this package is installed
    package_dir = Path(__file__).parent
    assets_dir = package_dir / "assets"
    return assets_dir


def get_model_path(model_name: str = "tiny-en") -> Path:
    """
    Get the path to a specific model directory in the assets folder.

    Args:
        model_name: Name of the model directory (default: "tiny-en")

    Returns:
        Path object pointing to the model directory

    Example:
        >>> from moonshine_voice.utils import get_model_path
        >>> model_path = get_model_path("tiny-en")
        >>> transcriber = Transcriber(str(model_path))
    """
    assets_path = get_assets_path()
    model_path = assets_path / model_name
    return model_path


def load_wav_file(file_path: str | Path) -> Tuple[list[float], int]:
    """
    Load a WAV file and return audio data as float array and sample rate.

    Supports 16-bit and 24-bit PCM WAV files. Audio samples are normalized
    to the range [-1.0, 1.0].

    Args:
        file_path: Path to the WAV file

    Returns:
        Tuple of (audio_data, sample_rate) where:
        - audio_data: List of float samples in range [-1.0, 1.0]
        - sample_rate: Sample rate in Hz

    Raises:
        ValueError: If the file is not a valid WAV file
        IOError: If the file cannot be read

    Example:
        >>> from moonshine_voice.utils import load_wav_file
        >>> audio_data, sample_rate = load_wav_file("audio.wav")
        >>> print(f"Loaded {len(audio_data)} samples at {sample_rate} Hz")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise IOError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        # Read RIFF header
        riff_header = f.read(4)
        if riff_header != b"RIFF":
            raise ValueError("Not a valid RIFF file")

        # Read chunk size (we don't need it, but must read it)
        chunk_size = struct.unpack("<I", f.read(4))[0]

        # Read WAVE header
        wave_header = f.read(4)
        if wave_header != b"WAVE":
            raise ValueError("Not a valid WAVE file")

        # Find the "fmt " chunk and "data" chunk
        found_fmt = False
        found_data = False
        audio_format = 0
        num_channels = 0
        sample_rate = 0
        bits_per_sample = 0
        block_align = 0
        data_size = 0

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break

            chunk_size = struct.unpack("<I", f.read(4))[0]

            if chunk_id == b"fmt ":
                found_fmt = True
                # Read fmt chunk data
                if chunk_size < 16:
                    raise ValueError("fmt chunk too small")

                # Read audio format (1 = PCM)
                audio_format = struct.unpack("<H", f.read(2))[0]
                num_channels = struct.unpack("<H", f.read(2))[0]
                sample_rate = struct.unpack("<I", f.read(4))[0]
                byte_rate = struct.unpack("<I", f.read(4))[0]
                block_align = struct.unpack("<H", f.read(2))[0]
                bits_per_sample = struct.unpack("<H", f.read(2))[0]

                # Skip any extra fmt bytes
                if chunk_size > 16:
                    f.read(chunk_size - 16)

                if audio_format != 1:
                    raise ValueError(
                        f"Only PCM format (format=1) is supported, got format={audio_format}"
                    )

                if bits_per_sample not in (16, 24, 32):
                    raise ValueError(
                        f"Only 16-bit, 24-bit, or 32-bit PCM is supported, got {bits_per_sample}-bit"
                    )
            elif chunk_id == b"data":
                # Found data chunk
                found_data = True
                data_size = chunk_size
                break
            else:
                # Skip unknown chunks
                f.seek(chunk_size, os.SEEK_CUR)

        if not found_fmt:
            raise ValueError("No fmt chunk found in WAV file")

        if not found_data:
            raise ValueError("No data chunk found in WAV file")

        # Now we're at the data chunk
        bytes_per_sample = bits_per_sample // 8
        bytes_per_frame = bytes_per_sample * num_channels
        num_frames = data_size // bytes_per_frame

        if num_channels > 1:
            # For multi-channel audio, mix down to mono by averaging
            audio_data = []

            for _ in range(num_frames):
                channel_sum = 0.0
                for _ in range(num_channels):
                    if bits_per_sample == 16:
                        sample = struct.unpack("<h", f.read(2))[0]
                        channel_sum += sample / 32768.0
                    elif bits_per_sample == 24:
                        # 24-bit samples are stored as 3 bytes, little-endian
                        b1, b2, b3 = struct.unpack("<BBB", f.read(3))
                        # Sign extend to 32-bit
                        sample = b1 | (b2 << 8) | (b3 << 16)
                        if sample & 0x800000:  # Sign bit
                            sample |= 0xFF000000  # Sign extend
                        sample = struct.unpack(
                            "<i", struct.pack("<I", sample & 0xFFFFFFFF)
                        )[0]
                        channel_sum += sample / 8388608.0  # 2^23
                    elif bits_per_sample == 32:
                        sample = struct.unpack("<i", f.read(4))[0]
                        channel_sum += sample / 2147483648.0  # 2^31
                # Average across channels for mono output
                audio_data.append(channel_sum / num_channels)
        else:
            # Mono audio
            audio_data = []
            for _ in range(num_frames):
                if bits_per_sample == 16:
                    sample = struct.unpack("<h", f.read(2))[0]
                    audio_data.append(sample / 32768.0)
                elif bits_per_sample == 24:
                    # 24-bit samples are stored as 3 bytes, little-endian
                    b1, b2, b3 = struct.unpack("<BBB", f.read(3))
                    # Sign extend to 32-bit
                    sample = b1 | (b2 << 8) | (b3 << 16)
                    if sample & 0x800000:  # Sign bit
                        sample |= 0xFF000000  # Sign extend
                    sample = struct.unpack(
                        "<i", struct.pack("<I", sample & 0xFFFFFFFF)
                    )[0]
                    audio_data.append(sample / 8388608.0)  # 2^23
                elif bits_per_sample == 32:
                    sample = struct.unpack("<i", f.read(4))[0]
                    audio_data.append(sample / 2147483648.0)  # 2^31

    return audio_data, sample_rate
