// Node smoke test for the Moonshine WASM binding.
//
// Loads the generated module and checks the low-level embind surface without
// downloading any models: module version, class registration, and the JSON
// manifest helpers that drive the AssetDownloader. Run via scripts/test-wasm.sh
// (which builds first) or `node --test tests` once dist/ is populated.

import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const dist = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist');

async function loadModule() {
  const factory = (await import(path.join(dist, 'moonshine.mjs'))).default;
  return factory({
    // Resolve the .wasm next to the .mjs regardless of cwd.
    locateFile: (file) => path.join(dist, file),
  });
}

test('module exposes a version', async () => {
  const mod = await loadModule();
  const version = mod.version();
  assert.equal(typeof version, 'number');
  assert.ok(version > 0, 'version should be a positive integer');
});

test('core STT classes are registered', async () => {
  const mod = await loadModule();
  assert.equal(typeof mod.Transcriber, 'function');
  assert.equal(typeof mod.Stream, 'function');
  assert.equal(typeof mod.IntentRecognizer, 'function');
});

test('STT dependency manifest is valid JSON with groups', async () => {
  const mod = await loadModule();
  const json = mod.sttDependencies('en', '1'); // arch 1 == Base
  const manifest = JSON.parse(json);
  assert.ok(Array.isArray(manifest.groups), 'manifest should have a groups array');
  assert.ok(manifest.groups.length > 0, 'manifest should list at least one group');
  const group = manifest.groups[0];
  assert.equal(typeof group.base_url, 'string');
  assert.ok(Array.isArray(group.files) && group.files.length > 0);
});

test('TTS surface is present when built with TTS support', async () => {
  const mod = await loadModule();
  // The default (full) build includes TTS; if a future STT-only build drops it,
  // this test documents the expectation rather than failing hard.
  if (typeof mod.TextToSpeech === 'function') {
    assert.equal(typeof mod.ttsDependencies, 'function');
    assert.equal(typeof mod.GraphemeToPhonemizer, 'function');
  }
});
