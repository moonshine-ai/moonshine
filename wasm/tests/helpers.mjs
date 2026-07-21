// Shared test helpers for the Moonshine WASM binding test suite.
//
// Mirrors the fixture/loader helpers the other bindings use (Swift's
// WAVLoader + TranscriberTests.getTestAssetsPath, Android's Utils.loadWav...):
// resolve the repo test-assets, decode a mono PCM16 wav into float samples, and
// load either the built ESM API (dist/index.js) or the raw embind module.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

export const DIST = path.resolve(here, '..', 'dist');
export const REPO_ROOT = path.resolve(here, '..', '..');
export const TEST_ASSETS = path.join(REPO_ROOT, 'test-assets');
export const TTS_DATA = path.join(REPO_ROOT, 'core', 'moonshine-tts', 'data');
export const TINY_EN_DIR = path.join(TEST_ASSETS, 'tiny-en');

/** Locate the sibling .wasm/.worker files regardless of the process cwd. */
export function locateFile(file) {
  return path.join(DIST, file);
}

export function fileExists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

/** Imports the built, idiomatic ESM API surface (dist/index.js). */
export function importApi() {
  return import(path.join(DIST, 'index.js'));
}

/** Loads the raw embind module directly (for low-level surface checks). */
export async function loadRawModule() {
  const factory = (await import(path.join(DIST, 'moonshine.mjs'))).default;
  return factory({ locateFile });
}

/**
 * Minimal mono PCM16 WAV decoder -> { audio: Float32Array in [-1, 1],
 * sampleRate }. Matches the tiny loaders in the Swift/Android tests.
 */
export function readWav(file) {
  const buf = fs.readFileSync(file);
  let offset = 12; // skip "RIFF"<size>"WAVE"
  let sampleRate = 16000;
  let channels = 1;
  let bitsPerSample = 16;
  let dataOffset = -1;
  let dataLength = 0;
  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === 'fmt ') {
      channels = buf.readUInt16LE(offset + 10);
      sampleRate = buf.readUInt32LE(offset + 12);
      bitsPerSample = buf.readUInt16LE(offset + 22);
    } else if (id === 'data') {
      dataOffset = offset + 8;
      dataLength = size;
      break;
    }
    offset += 8 + size + (size & 1); // chunks are word-aligned
  }
  if (dataOffset < 0) throw new Error(`No data chunk in ${file}`);
  if (bitsPerSample !== 16) {
    throw new Error(`Only PCM16 wav is supported (got ${bitsPerSample}-bit)`);
  }
  const frames = Math.floor(dataLength / 2 / channels);
  const audio = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    // Downmix to mono by taking the first channel.
    audio[i] = buf.readInt16LE(dataOffset + i * 2 * channels) / 32768;
  }
  return { audio, sampleRate };
}

export function tinyEnAvailable() {
  return (
    fileExists(path.join(TINY_EN_DIR, 'encoder_model.ort')) &&
    fileExists(path.join(TINY_EN_DIR, 'decoder_model_merged.ort')) &&
    fileExists(path.join(TINY_EN_DIR, 'tokenizer.bin'))
  );
}

/** Reads the bundled tiny-en STT model into the byte buffers `load` expects. */
export function tinyEnBytes() {
  return {
    encoder: fs.readFileSync(path.join(TINY_EN_DIR, 'encoder_model.ort')),
    decoder: fs.readFileSync(path.join(TINY_EN_DIR, 'decoder_model_merged.ort')),
    tokenizer: fs.readFileSync(path.join(TINY_EN_DIR, 'tokenizer.bin')),
  };
}

export function twoCities16kPath() {
  return path.join(TEST_ASSETS, 'two_cities_16k.wav');
}

/**
 * Builds an in-memory asset map for a comma/JSON list of canonical keys, read
 * from the local TTS data tree. Returns null if any file is missing so callers
 * can skip (the large kokoro model.onnx isn't vendored in-repo).
 */
export function ttsAssetMapOrNull(keys) {
  const map = new Map();
  for (const key of keys) {
    const file = path.join(TTS_DATA, key);
    if (!fileExists(file)) return null;
    map.set(key, fs.readFileSync(file));
  }
  return map;
}
