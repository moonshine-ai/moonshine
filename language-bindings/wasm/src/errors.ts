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
  BUSY: -4,
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

/**
 * Raised when a call would have competed with a streaming reply for the
 * synthesizer, e.g. {@link TextToSpeech.synthesize} part-way through a
 * {@link TextToSpeech.stream}.
 */
export class MoonshineBusyError extends MoonshineError {
  constructor(
    message = 'A streaming reply is in progress. Let it finish with endInput(), or drop it with cancelStream().',
  ) {
    super(message, MoonshineErrorCode.BUSY);
    this.name = 'MoonshineBusyError';
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
 * The Emscripten module, once loaded, so a thrown C++ exception pointer can be
 * turned back into its message. Set by {@link registerErrorModule}.
 */
let exceptionModule: NativeExceptionReader | undefined;

/** Emscripten's decoder, which reports `[type, message]` for a C++ throw. */
interface NativeExceptionReader {
  getExceptionMessage?: (ptr: number) => string | [string, string];
}

/**
 * Lets {@link toMoonshineError} decode native exceptions. Called with the
 * Emscripten module as soon as it is instantiated.
 */
export function registerErrorModule(mod: unknown): void {
  const candidate = mod as NativeExceptionReader;
  if (candidate && typeof candidate.getExceptionMessage === 'function') {
    exceptionModule = candidate;
  }
}

/**
 * Normalizes anything thrown across the embind boundary into a
 * {@link MoonshineError}. Recognizes the `moonshine:<code>:<text>` format
 * emitted by the C++ bridge.
 */
export function toMoonshineError(err: unknown): MoonshineError {
  if (err instanceof MoonshineError) return err;

  // A C++ throw arrives as a heap pointer unless we ask Emscripten to read it.
  if (typeof err === 'number' && exceptionModule?.getExceptionMessage) {
    try {
      const decoded = exceptionModule.getExceptionMessage(err);
      return toMoonshineError(Array.isArray(decoded) ? decoded[1] : decoded);
    } catch {
      return new MoonshineUnknownError(`native exception at ${err}`);
    }
  }

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
      // The core's error_to_string has no entry for this code, so `text` reads
      // "Unknown error"; the class default says what to do about it instead.
      case MoonshineErrorCode.BUSY:
        return new MoonshineBusyError();
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
