// Text-embedding tests. EmbeddingModel is internal — AgentFlow is the public
// way to match phrases — so these import the module directly rather than
// through the package entry point. The dependency-manifest check and the
// invalid-buffer error mirror Swift's EmbeddingModelTests / Android's
// EmbeddingModelTest (testCreateEmbeddingModel_invalidPath_throws). The full
// embed/match path needs the embedding model, which is download-only, so it is
// opt-in via MOONSHINE_DOWNLOAD_TESTS=1 (matching AssetDownloaderNetworkTests).

import test from 'node:test';
import assert from 'node:assert/strict';
import { importInternal, loadRawModule } from './helpers.mjs';

const mod = await loadRawModule();

test('embedding dependency manifest points at the all-in-one embedding model', () => {
  const manifest = JSON.parse(mod.embeddingDependencies('', ''));
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
  assert.throws(() => new mod.EmbeddingModel([], [], 0, 'q4'));
});

test('the phrase matcher falls back to substrings without a model', async () => {
  const { PhraseMatcher } = await importInternal('embedding-model.js');
  const matcher = new PhraseMatcher();
  const groups = [{ key: 'weather', phrases: ['the weather', 'forecast'] }];
  assert.equal(matcher.match("what's the weather like", groups, 0.7), 'weather');
  assert.equal(matcher.match('play some music', groups, 0.7), undefined);
});

const downloadTests = process.env.MOONSHINE_DOWNLOAD_TESTS === '1';
const matchSkip = downloadTests
  ? false
  : 'set MOONSHINE_DOWNLOAD_TESTS=1 to download the embedding model and run';

test('embeds text and scores the closest phrase', { skip: matchSkip }, async () => {
  const { EmbeddingModel, PhraseMatcher } = await importInternal('embedding-model.js');
  const model = await EmbeddingModel.load({ variant: 'q4', module: mod });
  try {
    const lights = model.calculateEmbedding('turn on the lights');
    assert.ok(lights.length > 0);
    const music = model.calculateEmbedding('play some music');
    assert.ok(model.distance(lights, lights) > model.distance(lights, music));

    const matcher = new PhraseMatcher(model);
    const phrases = ['turn on the lights', 'play some music'];
    assert.equal(matcher.matchPhrases('switch the lights on', phrases, 0.5),
      'turn on the lights');
    assert.equal(matcher.matchPhrases('what time is it', phrases, 0.9), undefined);
  } finally {
    model.close();
  }
});
