/**
 * Error hierarchy mirroring the Python bindings' `errors.py`, so failing calls
 * surface as typed exceptions instead of raw negative return codes.
 *
 * The embind bridge throws `Error("moonshine:<code>:<text>")`; {@link toMoonshineError}
 * parses that back into the right subclass.
 */

/** Moonshine error codes from `core/moonshine-c-api.h`. */
export const MoonshineErrorCode = {
  NONE: 0,
  UNKNOWN: -1,
  INVALID_HANDLE: -2,
  INVALID_ARGUMENT: -3,
} as const;

/** Base class for all errors thrown by the Moonshine binding. */
export class MoonshineError extends Error {
  /** The underlying numeric error code, if known. */
  readonly code: number;

  constructor(message: string, code: number = MoonshineErrorCode.UNKNOWN) {
    super(message);
    this.name = 'MoonshineError';
    this.code = code;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class MoonshineUnknownError extends MoonshineError {
  constructor(message = 'Unknown Moonshine error') {
    super(message, MoonshineErrorCode.UNKNOWN);
    this.name = 'MoonshineUnknownError';
  }
}

export class MoonshineInvalidHandleError extends MoonshineError {
  constructor(message = 'Invalid Moonshine handle') {
    super(message, MoonshineErrorCode.INVALID_HANDLE);
    this.name = 'MoonshineInvalidHandleError';
  }
}

export class MoonshineInvalidArgumentError extends MoonshineError {
  constructor(message = 'Invalid argument') {
    super(message, MoonshineErrorCode.INVALID_ARGUMENT);
    this.name = 'MoonshineInvalidArgumentError';
  }
}

/** Raised when a network/asset download fails. */
export class MoonshineDownloadError extends MoonshineError {
  constructor(message: string) {
    super(message, MoonshineErrorCode.UNKNOWN);
    this.name = 'MoonshineDownloadError';
  }
}

/**
 * Normalizes anything thrown across the embind boundary into a
 * {@link MoonshineError}. Recognizes the `moonshine:<code>:<text>` format
 * emitted by the C++ bridge.
 */
export function toMoonshineError(err: unknown): MoonshineError {
  if (err instanceof MoonshineError) return err;

  const message =
    typeof err === 'string'
      ? err
      : err instanceof Error
        ? err.message
        : String(err);

  const match = /^moonshine:(-?\d+):(.*)$/s.exec(message);
  if (match) {
    const code = Number(match[1]);
    const text = match[2];
    switch (code) {
      case MoonshineErrorCode.INVALID_HANDLE:
        return new MoonshineInvalidHandleError(text);
      case MoonshineErrorCode.INVALID_ARGUMENT:
        return new MoonshineInvalidArgumentError(text);
      default:
        return new MoonshineError(text, code);
    }
  }
  return new MoonshineUnknownError(message);
}

/** Runs `fn`, re-throwing any embind error as a typed {@link MoonshineError}. */
export function wrapErrors<T>(fn: () => T): T {
  try {
    return fn();
  } catch (err) {
    throw toMoonshineError(err);
  }
}
