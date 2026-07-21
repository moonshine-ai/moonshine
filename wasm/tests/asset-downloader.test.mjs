// AssetDownloader tests with a mocked `fetch` (no network), mirroring the Swift
// AssetDownloaderTests (MockURLProtocol): manifest parsing, filename keying,
// progress reporting, and HTTP-error surfacing as a typed download error.

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { DIST } from './helpers.mjs';

const { AssetDownloader } = await import(path.join(DIST, 'asset-downloader.js'));
const { MoonshineDownloadError } = await import(path.join(DIST, 'errors.js'));

const enc = new TextEncoder();
const dec = new TextDecoder();

/**
 * Installs a mock global fetch for the duration of `fn`. `handler(url)` returns
 * a Response-like object (or throws). Records the requested URLs.
 */
async function withFetch(handler, fn) {
  const original = globalThis.fetch;
  const urls = [];
  globalThis.fetch = async (url) => {
    urls.push(String(url));
    return handler(String(url));
  };
  try {
    return await fn(urls);
  } finally {
    globalThis.fetch = original;
  }
}

/** A Response whose body is the url's basename bytes; no streaming body. */
function okResponse(url) {
  const basename = url.split(/[?#]/)[0].split('/').pop();
  const bytes = enc.encode(basename);
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    headers: { get: () => String(bytes.byteLength) },
    body: null,
    async arrayBuffer() {
      return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    },
    clone() {
      return this;
    },
  };
}

test('downloadManifest fetches every group file, keyed by basename', async () => {
  const manifest = JSON.stringify({
    groups: [
      { base_url: 'https://cdn.example/models/en', files: ['encoder.ort', 'tokenizer.bin'] },
    ],
  });
  await withFetch(okResponse, async (urls) => {
    const files = await new AssetDownloader().downloadManifest(manifest);
    assert.deepEqual([...files.keys()].sort(), ['encoder.ort', 'tokenizer.bin']);
    assert.equal(dec.decode(files.get('encoder.ort')), 'encoder.ort');
    assert.deepEqual(urls, [
      'https://cdn.example/models/en/encoder.ort',
      'https://cdn.example/models/en/tokenizer.bin',
    ]);
  });
});

test('downloadManifest normalizes slashes when joining URLs', async () => {
  const manifest = JSON.stringify({
    groups: [{ base_url: 'https://cdn.example/x/', files: ['/a.ort'] }],
  });
  await withFetch(okResponse, async (urls) => {
    await new AssetDownloader().downloadManifest(manifest);
    assert.deepEqual(urls, ['https://cdn.example/x/a.ort']);
  });
});

test('downloadFiles keys results by basename and strips query strings', async () => {
  await withFetch(okResponse, async () => {
    const files = await new AssetDownloader().downloadFiles([
      'https://cdn.example/a/model.onnx?token=abc',
    ]);
    assert.deepEqual([...files.keys()], ['model.onnx']);
    assert.equal(dec.decode(files.get('model.onnx')), 'model.onnx');
  });
});

test('a non-OK response surfaces as MoonshineDownloadError with the status', async () => {
  const handler = () => ({
    ok: false,
    status: 404,
    statusText: 'Not Found',
    headers: { get: () => null },
  });
  await withFetch(handler, async () => {
    await assert.rejects(
      () => new AssetDownloader().fetchFile('https://cdn.example/missing.ort'),
      (err) => err instanceof MoonshineDownloadError && /404/.test(err.message),
    );
  });
});

test('invalid manifest JSON throws a MoonshineDownloadError', async () => {
  await assert.rejects(
    () => new AssetDownloader().downloadManifest('{ not json'),
    (err) =>
      err instanceof MoonshineDownloadError && /parse/i.test(err.message),
  );
});

test('onProgress is invoked with (loaded, total, file)', async () => {
  const manifest = JSON.stringify({
    groups: [{ base_url: 'https://cdn.example/en', files: ['tokenizer.bin'] }],
  });
  const progress = [];
  const downloader = new AssetDownloader({
    onProgress: (loaded, total, file) => progress.push([loaded, total, file]),
  });
  await withFetch(okResponse, async () => {
    await downloader.downloadManifest(manifest);
  });
  assert.equal(progress.length, 1);
  const [loaded, total, file] = progress[0];
  assert.equal(file, 'tokenizer.bin');
  assert.equal(loaded, enc.encode('tokenizer.bin').byteLength);
  assert.equal(total, loaded);
});
