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

/**
 * Installs a mock global `caches` (Cache API) for the duration of `fn`.
 * `cache` supplies the `match` and `put` methods exercised by the downloader.
 */
async function withCaches(cache, fn) {
  const had = Object.prototype.hasOwnProperty.call(globalThis, 'caches');
  const original = globalThis.caches;
  globalThis.caches = { open: async () => cache };
  try {
    return await fn();
  } finally {
    if (had) globalThis.caches = original;
    else delete globalThis.caches;
  }
}

const CHUNKS = [
  Uint8Array.of(1, 2, 3),
  Uint8Array.of(4, 5, 6),
  Uint8Array.of(7, 8),
];
const ALL_BYTES = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);

/**
 * A real Response streaming CHUNKS one per pull. With `failAfter`, the stream
 * errors once that many chunks have been delivered instead of closing.
 */
function streamingResponse({ failAfter } = {}) {
  let i = 0;
  const stream = new ReadableStream({
    pull(controller) {
      if (failAfter !== undefined && i >= failAfter) {
        controller.error(new Error('connection reset'));
      } else if (i < CHUNKS.length) {
        controller.enqueue(CHUNKS[i++].slice());
      } else {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { 'content-length': String(ALL_BYTES.byteLength) },
  });
}

/** A cache that always misses and whose put reads one chunk, then hits quota. */
function quotaCache() {
  const puts = [];
  return {
    puts,
    async match() {
      return undefined;
    },
    async put(url, response) {
      puts.push(url);
      const reader = response.body.getReader();
      await reader.read();
      throw new DOMException('Quota exceeded', 'QuotaExceededError');
    },
  };
}

test('a cache.put QuotaExceededError does not lose the downloaded bytes', async () => {
  const cache = quotaCache();
  const { calls, downloader } = recordProgress();
  const url = 'https://cdn.example/en/encoder.ort';
  await withCaches(cache, () =>
    withFetch(
      () => streamingResponse(),
      async (urls) => {
        const bytes = await downloader.fetchFile(url);
        assert.deepEqual(bytes, ALL_BYTES);
        assert.deepEqual(urls, [url]);
      },
    ),
  );
  assert.deepEqual(cache.puts, [url]);
  assert.ok(calls.length > 0);
  assert.equal(calls.at(-1).loaded, ALL_BYTES.byteLength);
  assert.equal(calls.at(-1).total, ALL_BYTES.byteLength);
  assert.equal(calls.at(-1).file, 'encoder.ort');
});

test('a body read failure still surfaces when cache.put also fails', async () => {
  const cache = quotaCache();
  const { downloader } = recordProgress();
  await withCaches(cache, () =>
    withFetch(
      () => streamingResponse({ failAfter: 1 }),
      async (urls) => {
        await assert.rejects(
          () => downloader.fetchFile('https://cdn.example/en/encoder.ort'),
          /connection reset/,
        );
        assert.equal(urls.length, 1);
      },
    ),
  );
});

test('a cache hit returns the cached bytes without touching the network', async () => {
  const url = 'https://cdn.example/en/encoder.ort';
  const cache = {
    puts: [],
    async match(key) {
      return key === url ? new Response(ALL_BYTES.slice()) : undefined;
    },
    async put(key) {
      this.puts.push(key);
    },
  };
  const { calls, downloader } = recordProgress();
  await withCaches(cache, () =>
    withFetch(
      () => {
        throw new Error('network must not be used on a cache hit');
      },
      async (urls) => {
        assert.deepEqual(await downloader.fetchFile(url), ALL_BYTES);
        assert.deepEqual(urls, []);
      },
    ),
  );
  assert.deepEqual(cache.puts, []);
  assert.equal(calls.at(-1).loaded, ALL_BYTES.byteLength);
  assert.equal(calls.at(-1).total, ALL_BYTES.byteLength);
});

test('fetch and HTTP failures still surface when a cache is available', async () => {
  const cache = quotaCache();
  const url = 'https://cdn.example/en/encoder.ort';

  await withCaches(cache, () =>
    withFetch(
      () => {
        throw new TypeError('Failed to fetch');
      },
      () =>
        assert.rejects(
          () => new AssetDownloader().fetchFile(url),
          (err) => err instanceof TypeError && /Failed to fetch/.test(err.message),
        ),
    ),
  );

  await withCaches(cache, () =>
    withFetch(
      () => new Response('unavailable', { status: 503, statusText: 'Service Unavailable' }),
      () =>
        assert.rejects(
          () => new AssetDownloader().fetchFile(url),
          (err) => err instanceof MoonshineDownloadError && /503/.test(err.message),
        ),
    ),
  );

  // Neither failure is written to the cache.
  assert.deepEqual(cache.puts, []);
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
