// Intent-recognition tests. The dependency-manifest check and the invalid-path
// error mirror Swift's IntentRecognizerTests / Android's IntentRecognizerTest
// (testCreateIntentRecognizer_invalidPath_throws). The full register/match path
// needs the embedding model, which is download-only, so it is opt-in via
// MOONSHINE_DOWNLOAD_TESTS=1 (matching AssetDownloaderNetworkTests).

import test from 'node:test';
import assert from 'node:assert/strict';
import { importApi, loadRawModule } from './helpers.mjs';

const mod = await loadRawModule();

test('intent dependency manifest points at the all-in-one embedding model', () => {
  const manifest = JSON.parse(mod.intentDependencies('', ''));
  assert.ok(Array.isArray(manifest.groups) && manifest.groups.length > 0);
  const group = manifest.groups[0];
  assert.match(group.base_url, /embeddinggemma/);
  // Each `files` entry is an object {name, url, size, checksum, checksum_type}.
  const names = group.files.map((f) => f.name);
  assert.ok(names.includes('tokenizer.bin'));
  // The model now ships as a single self-contained `.ort` (no `.onnx_data`).
  assert.ok(names.some((n) => n.endsWith('.ort')));
  assert.ok(!names.some((n) => n.endsWith('.onnx_data')));
  // Files carry a fully-qualified url and (when registered) a size/checksum.
  for (const file of group.files) {
    assert.equal(typeof file.name, 'string');
    assert.equal(typeof file.url, 'string');
    assert.ok(file.url.includes(file.name));
  }
});

test('constructing with no model buffer throws', () => {
  assert.throws(() => new mod.IntentRecognizer([], [], 0, 'q4'));
});

const downloadTests = process.env.MOONSHINE_DOWNLOAD_TESTS === '1';
const matchSkip = downloadTests
  ? false
  : 'set MOONSHINE_DOWNLOAD_TESTS=1 to download the embedding model and run';

test('registers phrases and finds the closest intent', { skip: matchSkip }, async () => {
  const { IntentRecognizer } = await importApi();
  const recognizer = await IntentRecognizer.load({ variant: 'q4', module: mod });
  try {
    recognizer.register(['turn on the lights', 'play some music']);
    const best = recognizer.bestIntent('turn on the lights', 0);
    assert.ok(best, 'expected a best match');
    assert.equal(best.canonicalPhrase, 'turn on the lights');

    const matches = recognizer.closestIntents('turn on the lights', 0);
    assert.ok(matches.length >= 1);
    assert.ok(matches[0].similarity >= matches[matches.length - 1].similarity);

    recognizer.unregister('play some music');
    recognizer.clear();
  } finally {
    recognizer.close();
  }
});
