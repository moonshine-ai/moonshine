// Grapheme-to-phoneme tests, mirroring Python's test_g2p_prints_ipa and the
// G2P dependency check in Android's JNITest. The en_us G2P assets are vendored
// in-repo (core/moonshine-tts/data), so the conversion test runs without any
// network; it's skip-guarded in case that data tree is unavailable.

import test from 'node:test';
import assert from 'node:assert/strict';
import { importApi, loadRawModule, ttsAssetMapOrNull } from './helpers.mjs';

const mod = await loadRawModule();
const g2pSupported = typeof mod.GraphemeToPhonemizer === 'function';

test('g2p dependency manifest lists the en_us assets', { skip: !g2pSupported && 'no G2P support' }, () => {
  const keys = mod.g2pDependencies('en_us').split(',').filter(Boolean);
  assert.ok(keys.length > 0);
  assert.ok(keys.includes('en_us/g2p-config.json'));
  assert.ok(keys.some((k) => k.endsWith('.onnx')));
});

const keys = g2pSupported
  ? mod.g2pDependencies('en_us').split(',').filter(Boolean)
  : [];
const assets = g2pSupported ? ttsAssetMapOrNull(keys) : null;
const g2pSkip = !g2pSupported
  ? 'no G2P support'
  : assets === null
    ? 'en_us G2P assets not vendored locally'
    : false;

test('converts English text to non-empty IPA phonemes', { skip: g2pSkip }, async () => {
  const { GraphemeToPhonemizer } = await importApi();
  const g2p = await GraphemeToPhonemizer.load({
    language: 'en_us',
    assets,
    module: mod,
  });
  try {
    const phonemes = g2p.textToPhonemes('Hello world');
    assert.equal(typeof phonemes, 'string');
    assert.ok(phonemes.length > 0);
    assert.notEqual(phonemes, 'Hello world');
    // Deterministic: repeated calls agree.
    assert.equal(g2p.textToPhonemes('Hello world'), phonemes);
  } finally {
    g2p.close();
  }
});
