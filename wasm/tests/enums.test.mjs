// Enum + name-mapping tests, mirroring the Python `model_arch_to_string` /
// `string_to_model_arch` helpers and the Swift `TranscribeStreamFlags` /
// spelling-mode bit assertions (testSpellingModeApiSurface).

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { DIST } from './helpers.mjs';

const {
  ModelArch,
  EmbeddingModelArch,
  TranscribeFlags,
  modelArchToString,
  stringToModelArch,
} = await import(path.join(DIST, 'enums.js'));

test('ModelArch values match MOONSHINE_MODEL_ARCH_* constants', () => {
  assert.equal(ModelArch.Tiny, 0);
  assert.equal(ModelArch.Base, 1);
  assert.equal(ModelArch.TinyStreaming, 2);
  assert.equal(ModelArch.BaseStreaming, 3);
  assert.equal(ModelArch.SmallStreaming, 4);
  assert.equal(ModelArch.MediumStreaming, 5);
});

test('modelArchToString round-trips through stringToModelArch', () => {
  const cases = [
    [ModelArch.Tiny, 'tiny'],
    [ModelArch.Base, 'base'],
    [ModelArch.TinyStreaming, 'tiny_streaming'],
    [ModelArch.BaseStreaming, 'base_streaming'],
    [ModelArch.SmallStreaming, 'small_streaming'],
    [ModelArch.MediumStreaming, 'medium_streaming'],
  ];
  for (const [arch, name] of cases) {
    assert.equal(modelArchToString(arch), name);
    assert.equal(stringToModelArch(name), arch);
  }
});

test('modelArchToString renders unknown values defensively', () => {
  assert.equal(modelArchToString(999), 'unknown(999)');
});

test('stringToModelArch throws on an unknown name', () => {
  assert.throws(() => stringToModelArch('gigantic'), /Unknown model arch/);
});

test('TranscribeFlags are the expected bit flags', () => {
  assert.equal(TranscribeFlags.None, 0);
  assert.equal(TranscribeFlags.ForceUpdate, 1 << 0);
  // Matches Swift's TranscribeStreamFlags.flagSpellingMode == 1 << 1.
  assert.equal(TranscribeFlags.SpellingMode, 1 << 1);
  // Flags compose bitwise.
  assert.equal(
    TranscribeFlags.ForceUpdate | TranscribeFlags.SpellingMode,
    0b11,
  );
});

test('EmbeddingModelArch exposes the Gemma 300M default', () => {
  assert.equal(EmbeddingModelArch.Gemma300M, 0);
});
