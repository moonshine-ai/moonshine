"""Import ``sounddevice`` on demand and turn its start-up failures into MoonshineErrors.

``import sounddevice`` calls ``Pa_Initialize()`` while the module loads, so a
PortAudio problem surfaces as an exception from the import itself. Routing
every import through here keeps that off the constructors (which must not fail)
and gives callers one exception type to catch.
"""

import sys

from moonshine_voice.errors import MoonshineError

# PortAudio's paInternalError. On Windows this is what Pa_Initialize() returns
# when the Windows Audio service is stopped, which is the default on Server
# SKUs, containers, and headless VMs. With the service running PortAudio
# initializes normally even when there is no audio hardware at all.
PA_INTERNAL_ERROR = -9986

_WINDOWS_AUDIO_HINT = (
    " On Windows this usually means the Windows Audio service is not running. "
    "Start it from an elevated PowerShell with "
    "`Set-Service Audiosrv -StartupType Automatic; Start-Service Audiosrv`."
)


def import_sounddevice(purpose: str):
    """Return the ``sounddevice`` module, raising MoonshineError if it cannot be used.

    ``purpose`` names the caller for the message, e.g. ``"MicTranscriber.start()"``.
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        raise MoonshineError(
            f"{purpose} requires sounddevice (e.g. `pip install sounddevice`)."
        ) from e
    except Exception as e:
        # sounddevice.PortAudioError cannot be named here because the module
        # never finished importing, so match on the class name instead.
        if type(e).__name__ != "PortAudioError":
            raise
        code = e.args[1] if len(e.args) > 1 else None
        hint = ""
        if code == PA_INTERNAL_ERROR and sys.platform == "win32":
            hint = _WINDOWS_AUDIO_HINT
        raise MoonshineError(
            f"{purpose} could not initialize PortAudio: {e}.{hint}"
        ) from e
    return sd
