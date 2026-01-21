"""Error classes for Moonshine Voice."""


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
