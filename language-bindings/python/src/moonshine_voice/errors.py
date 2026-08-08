"""Error classes for Moonshine Voice."""

from typing import List, Optional


class MoonshineError(Exception):
    """Base exception for all Moonshine Voice errors."""

    def __init__(self, message: str, error_code: int = 0):
        super().__init__(message)
        self.error_code = error_code


class MoonshineUnknownError(MoonshineError):
    """Unknown error occurred."""

    def __init__(self, message: str = "Unknown error"):
        super().__init__(message, error_code=-1)


class MoonshineInvalidHandleError(MoonshineError):
    """Invalid transcriber or stream handle."""

    def __init__(self, message: str = "Invalid handle"):
        super().__init__(message, error_code=-2)


class MoonshineInvalidArgumentError(MoonshineError):
    """Invalid argument provided to function."""

    def __init__(self, message: str = "Invalid argument"):
        super().__init__(message, error_code=-3)


class MoonshineTtsLanguageError(MoonshineInvalidArgumentError):
    """Unknown or unsupported TTS language tag (see ``alternatives`` for valid tags)."""

    def __init__(
        self,
        language: str,
        alternatives: List[str],
        message: Optional[str] = None,
    ):
        alt = ", ".join(alternatives) if alternatives else "(none reported by native API)"
        msg = message or (
            f"Unknown TTS language {language!r}. "
            f"Supported language tags include: {alt}"
        )
        super().__init__(msg)
        self.language = language
        self.alternatives = list(alternatives)


class MoonshineAudioOutputError(MoonshineError):
    """Playback failed: no suitable output device, unknown device, or PortAudio cannot open it."""

    def __init__(
        self,
        message: str,
        *,
        available_outputs: Optional[List[str]] = None,
    ):
        lines = list(available_outputs or [])
        if lines:
            suffix = "\nAvailable output devices:\n  " + "\n  ".join(lines)
        else:
            suffix = "\nAvailable output devices: (none)"
        super().__init__(message + suffix, error_code=0)
        self.available_outputs = lines


class MoonshineTtsVoiceError(MoonshineInvalidArgumentError):
    """
    Requested TTS voice is not on disk for the language.

    ``alternatives`` lists voice ids already present under ``g2p_root`` (``found``).
    ``alternatives_available_for_download`` lists catalog voices that are ``missing`` locally
    (typically fetchable via the TTS asset download flow).
    """

    def __init__(
        self,
        voice: str,
        language: str,
        alternatives: List[str],
        *,
        alternatives_available_for_download: Optional[List[str]] = None,
        message: Optional[str] = None,
    ):
        downloaded = list(alternatives)
        avail = list(alternatives_available_for_download or [])
        d_txt = ", ".join(downloaded) if downloaded else "(none on disk for this language)"
        a_txt = ", ".join(avail) if avail else "(none listed by native catalog)"
        msg = message or (
            f"TTS voice {voice!r} is not available for language {language!r} "
            f"(not found under g2p_root). "
            f"Downloaded voices for this setup: {d_txt}. "
            f"Available for download: {a_txt}."
        )
        super().__init__(msg)
        self.voice = voice
        self.language = language
        self.alternatives = downloaded
        self.alternatives_available_for_download = avail


def check_error(error_code: int) -> None:
    """Check error code and raise appropriate exception if non-zero."""
    if error_code >= 0:
        return
    elif error_code == -1:
        raise MoonshineUnknownError()
    elif error_code == -2:
        raise MoonshineInvalidHandleError()
    elif error_code == -3:
        raise MoonshineInvalidArgumentError()
    else:
        raise MoonshineError(f"Unknown error code: {error_code}", error_code)
