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

test('downloadManifest fetches every group file, keyed by name, using file.url', async () => {
  const manifest = JSON.stringify({
    groups: [
      {
        base_url: 'https://cdn.example/models/en',
        files: [
          {
            name: 'encoder.ort',
            url: 'https://cdn.example/models/en/encoder.ort',
            size: null,
            checksum: '',
            checksum_type: '',
          },
          {
            name: 'tokenizer.bin',
            url: 'https://cdn.example/models/en/tokenizer.bin',
            size: null,
            checksum: '',
            checksum_type: '',
          },
        ],
      },
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

test('downloadManifest falls back to joining base_url + name when url is absent', async () => {
  const manifest = JSON.stringify({
    groups: [{ base_url: 'https://cdn.example/x/', files: [{ name: '/a.ort' }] }],
  });
  await withFetch(okResponse, async (urls) => {
    await new AssetDownloader().downloadManifest(manifest);
    assert.deepEqual(urls, ['https://cdn.example/x/a.ort']);
  });
});

test('downloadManifest rejects when a downloaded file has the wrong size', async () => {
  const manifest = JSON.stringify({
    groups: [
      {
        base_url: 'https://cdn.example/en',
        files: [
          {
            name: 'tokenizer.bin',
            url: 'https://cdn.example/en/tokenizer.bin',
            size: 999999,
            checksum: '',
            checksum_type: '',
          },
        ],
      },
    ],
  });
  await withFetch(okResponse, async () => {
    await assert.rejects(
      () => new AssetDownloader().downloadManifest(manifest),
      (err) =>
        err instanceof MoonshineDownloadError && /size mismatch/i.test(err.message),
    );
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

/** Byte length of the body okResponse serves for a given filename. */
const bodySize = (name) => enc.encode(name).byteLength;

/** A manifest of one group, declaring each file's real served size. */
function sizedManifest(names, { declareSizes = true } = {}) {
  return JSON.stringify({
    groups: [
      {
        base_url: 'https://cdn.example/en',
        files: names.map((name) => ({
          name,
          url: `https://cdn.example/en/${name}`,
          ...(declareSizes ? { size: bodySize(name) } : {}),
        })),
      },
    ],
  });
}

function recordProgress() {
  const calls = [];
  const downloader = new AssetDownloader({
    onProgress: (loaded, total, file) => calls.push({ loaded, total, file }),
  });
  return { calls, downloader };
}

test('onProgress reports bytes across the whole manifest, not per file', async () => {
  const names = ['encoder.ort', 'decoder.ort', 'tokenizer.bin'];
  const { calls, downloader } = recordProgress();
  await withFetch(okResponse, () => downloader.downloadManifest(sizedManifest(names)));

  const expectedTotal = names.reduce((sum, name) => sum + bodySize(name), 0);
  assert.equal(calls.length, names.length);
  // Every report carries the same overall total, and `loaded` accumulates
  // rather than restarting at zero for each file.
  for (const call of calls) assert.equal(call.total, expectedTotal);
  assert.deepEqual(
    calls.map((c) => c.loaded),
    names.map((_, i) =>
      names.slice(0, i + 1).reduce((sum, name) => sum + bodySize(name), 0),
    ),
  );
  // The last report is a genuine 100%.
  assert.equal(calls.at(-1).loaded, expectedTotal);
  assert.equal(calls.at(-1).file, 'tokenizer.bin');
});

test('progress never goes backwards while a manifest downloads', async () => {
  const names = ['encoder.ort', 'decoder.ort', 'tokenizer.bin', 'config.json'];
  const { calls, downloader } = recordProgress();
  await withFetch(okResponse, () => downloader.downloadManifest(sizedManifest(names)));

  for (let i = 1; i < calls.length; i++) {
    assert.ok(
      calls[i].loaded >= calls[i - 1].loaded,
      `report ${i} went backwards: ${calls[i - 1].loaded} -> ${calls[i].loaded}`,
    );
    assert.ok(calls[i].loaded <= calls[i].total, 'loaded overshot the total');
  }
});

test('a manifest missing any size reports an unknown total', async () => {
  // A partial sum would understate the download and make the bar run
  // backwards, so callers are told the total is unknown instead.
  const manifest = JSON.stringify({
    groups: [
      {
        base_url: 'https://cdn.example/en',
        files: [
          { name: 'encoder.ort', url: 'https://cdn.example/en/encoder.ort', size: 11 },
          { name: 'tokenizer.bin', url: 'https://cdn.example/en/tokenizer.bin' },
        ],
      },
    ],
  });
  const { calls, downloader } = recordProgress();
  await withFetch(okResponse, () => downloader.downloadManifest(manifest));

  assert.ok(calls.length > 0);
  for (const call of calls) assert.equal(call.total, undefined);
});

test('downloadNamedFiles accumulates bytes with an unknown total', async () => {
  const { calls, downloader } = recordProgress();
  await withFetch(okResponse, () =>
    downloader.downloadNamedFiles({
      'a.ort': 'https://cdn.example/en/a.ort',
      'b.ort': 'https://cdn.example/en/b.ort',
    }),
  );

  assert.equal(calls.length, 2);
  assert.equal(calls[0].loaded, bodySize('a.ort'));
  assert.equal(calls[1].loaded, bodySize('a.ort') + bodySize('b.ort'));
  for (const call of calls) assert.equal(call.total, undefined);
});

test('fetchFile on its own still reports just that file’s bytes', async () => {
  const { calls, downloader } = recordProgress();
  await withFetch(okResponse, () =>
    downloader.fetchFile('https://cdn.example/en/tokenizer.bin'),
  );

  assert.equal(calls.length, 1);
  assert.equal(calls[0].file, 'tokenizer.bin');
  assert.equal(calls[0].loaded, bodySize('tokenizer.bin'));
  assert.equal(calls[0].total, bodySize('tokenizer.bin'));
});
