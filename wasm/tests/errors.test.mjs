// Error-handling tests: the embind bridge throws `Error("moonshine:<code>:<text>")`
// and the TS layer must map that back into the typed hierarchy the other
// bindings expose (mirrors the Python bindings' errors.py behaviour and the
// Android JNITest.testMoonshineErrorToString expectations).

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { DIST } from './helpers.mjs';

const {
  MoonshineError,
  MoonshineUnknownError,
  MoonshineInvalidHandleError,
  MoonshineInvalidArgumentError,
  MoonshineDownloadError,
  MoonshineErrorCode,
  toMoonshineError,
  wrapErrors,
} = await import(path.join(DIST, 'errors.js'));

test('error codes match the C ABI constants', () => {
  assert.equal(MoonshineErrorCode.NONE, 0);
  assert.equal(MoonshineErrorCode.UNKNOWN, -1);
  assert.equal(MoonshineErrorCode.INVALID_HANDLE, -2);
  assert.equal(MoonshineErrorCode.INVALID_ARGUMENT, -3);
});

test('the typed errors form a MoonshineError hierarchy', () => {
  for (const Ctor of [
    MoonshineUnknownError,
    MoonshineInvalidHandleError,
    MoonshineInvalidArgumentError,
    MoonshineDownloadError,
  ]) {
    const err = Ctor === MoonshineDownloadError ? new Ctor('x') : new Ctor();
    assert.ok(err instanceof MoonshineError, `${Ctor.name} extends MoonshineError`);
    assert.ok(err instanceof Error);
  }
});

test('toMoonshineError parses invalid-handle codes', () => {
  const err = toMoonshineError('moonshine:-2:bad handle');
  assert.ok(err instanceof MoonshineInvalidHandleError);
  assert.equal(err.code, MoonshineErrorCode.INVALID_HANDLE);
  assert.equal(err.message, 'bad handle');
});

test('toMoonshineError parses invalid-argument codes', () => {
  const err = toMoonshineError('moonshine:-3:bad arg');
  assert.ok(err instanceof MoonshineInvalidArgumentError);
  assert.equal(err.code, MoonshineErrorCode.INVALID_ARGUMENT);
  assert.equal(err.message, 'bad arg');
});

test('toMoonshineError keeps unrecognized codes as a generic MoonshineError', () => {
  const err = toMoonshineError('moonshine:-99:mystery');
  assert.ok(err instanceof MoonshineError);
  assert.equal(err.constructor, MoonshineError);
  assert.equal(err.code, -99);
  assert.equal(err.message, 'mystery');
});

test('toMoonshineError preserves multi-line error text', () => {
  const err = toMoonshineError('moonshine:-3:line one\nline two');
  assert.ok(err instanceof MoonshineInvalidArgumentError);
  assert.equal(err.message, 'line one\nline two');
});

test('toMoonshineError wraps non-moonshine strings as unknown', () => {
  const err = toMoonshineError('some raw failure');
  assert.ok(err instanceof MoonshineUnknownError);
  assert.equal(err.code, MoonshineErrorCode.UNKNOWN);
  assert.equal(err.message, 'some raw failure');
});

test('toMoonshineError unwraps Error instances', () => {
  const err = toMoonshineError(new Error('moonshine:-2:from Error'));
  assert.ok(err instanceof MoonshineInvalidHandleError);
  assert.equal(err.message, 'from Error');
});

test('toMoonshineError passes an existing MoonshineError through unchanged', () => {
  const original = new MoonshineInvalidArgumentError('already typed');
  assert.equal(toMoonshineError(original), original);
});

test('wrapErrors returns the value on success', () => {
  assert.equal(wrapErrors(() => 42), 42);
});

test('wrapErrors rethrows as a typed MoonshineError', () => {
  assert.throws(
    () =>
      wrapErrors(() => {
        throw new Error('moonshine:-3:nope');
      }),
    (err) =>
      err instanceof MoonshineInvalidArgumentError && err.message === 'nope',
  );
});
