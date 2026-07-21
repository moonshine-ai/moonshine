// Intent-recognition tests. The dependency-manifest check and the invalid-path
// error mirror Swift's IntentRecognizerTests / Android's IntentRecognizerTest
// (testCreateIntentRecognizer_invalidPath_throws). The full register/match path
// needs the embedding model, which is download-only, so it is opt-in via
// MOONSHINE_DOWNLOAD_TESTS=1 (matching AssetDownloaderNetworkTests).

import test from 'node:test';
import assert from 'node:assert/strict';
import { importApi, loadRawModule } from './helpers.mjs';

const mod = await loadRawModule();

test('intent dependency manifest points at the embedding model', () => {
  const manifest = JSON.parse(mod.intentDependencies('', ''));
  assert.ok(Array.isArray(manifest.groups) && manifest.groups.length > 0);
  const group = manifest.groups[0];
  assert.match(group.base_url, /embeddinggemma/);
  assert.ok(Array.isArray(group.files) && group.files.includes('tokenizer.bin'));
});

test('constructing from a non-existent model path throws', () => {
  assert.throws(() => new mod.IntentRecognizer('/no-such-intent-model', 0, ''));
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
