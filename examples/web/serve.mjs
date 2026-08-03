// Minimal static file server for the Moonshine web examples.
//
// The multithreaded WASM build needs SharedArrayBuffer, which browsers only
// expose to cross-origin-isolated pages. That requires two response headers on
// every document/script:
//
//   Cross-Origin-Opener-Policy: same-origin
//   Cross-Origin-Embedder-Policy: require-corp
//
// A normal static host (or `python -m http.server`) won't set these, so the
// examples would fail with "SharedArrayBuffer is not defined". This server adds
// them. Usage:
//
//   node examples/web/serve.mjs [port]
//
// then open http://localhost:8080/ for the index, or go straight to one of
// /stt/, /tts/, /agent-flow/, /dictation/, /meeting-notes/.
//
// By default the examples import the published binding from the jsDelivr CDN.
// To test the locally-built binding served from /wasm/dist/index.js instead,
// append ?local=1 to the URL, e.g. http://localhost:8080/stt/?local=1.
//
// A few extra path prefixes are mounted so the examples (and the browser
// integration test) can load model files straight from the repo instead of the
// CDN, handy for offline, hermetic testing:
//
//   /wasm/...        -> <repo>/wasm/...                 (the built binding)
//   /test-assets/... -> <repo>/test-assets/...          (small STT/embedding models)
//   /tts-data/...    -> <repo>/core/moonshine-tts/data/ (the kokoro TTS assets)
//
// Every response is sent `Cache-Control: no-store` so a rebuilt binding is
// always picked up by an ordinary reload. `no-cache` would not be enough: it
// lets the browser keep the bytes and merely revalidate, and since this server
// sends no ETag or Last-Modified there is nothing to revalidate against. It
// also lets Chrome reuse its compiled-WASM code cache. `no-store` forbids
// storing the response at all, which costs a re-read and re-compile of
// moonshine.wasm on each load but makes "did my rebuild take effect?"
// answerable with a plain refresh. Requests are logged below so you can watch
// the refetch actually happen.
//
// Note that `no-store` only governs the HTTP cache. Model files are also held
// in the Cache Storage API bucket `moonshine-models-v1`, which no reload of any
// kind will clear; append ?fresh=1 to a demo URL to purge it.

import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
// Repo root, so examples can import the built binding at /wasm/dist/index.js.
const REPO_ROOT = path.resolve(ROOT, '..', '..');
const PORT = Number(process.argv[2] ?? 8080);

// URL-prefix -> on-disk directory mounts. The first matching prefix wins; the
// prefix itself is stripped and the remainder joined onto the target dir.
// Everything else falls back to the examples/web tree (ROOT).
const MOUNTS = [
  { prefix: '/wasm/', dir: path.join(REPO_ROOT, 'wasm') },
  { prefix: '/test-assets/', dir: path.join(REPO_ROOT, 'test-assets') },
  { prefix: '/tts-data/', dir: path.join(REPO_ROOT, 'core', 'moonshine-tts', 'data') },
];

/** Resolves a request pathname to an absolute file path via the mount table. */
function resolveFilePath(pathname) {
  for (const { prefix, dir } of MOUNTS) {
    if (pathname.startsWith(prefix)) {
      return path.join(dir, pathname.slice(prefix.length));
    }
  }
  return path.join(ROOT, pathname);
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.wav': 'audio/wav',
};

/** Logs one request line, so a reload that refetches is visible in the terminal. */
function log(status, pathname, bytes) {
  const time = new Date().toTimeString().slice(0, 8);
  const size = bytes === undefined ? '' : ` ${(bytes / 1024).toFixed(1)}kB`;
  console.log(`${time} ${status} ${pathname}${size}`);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  let pathname = decodeURIComponent(url.pathname);
  try {
    if (pathname === '/favicon.ico') {
      res.writeHead(204).end();
      return;
    }
    if (pathname.endsWith('/')) pathname += 'index.html';

    const filePath = path.normalize(resolveFilePath(pathname));
    // Prevent path traversal outside the repo (all mounts live under it).
    if (!filePath.startsWith(REPO_ROOT)) {
      log(403, pathname);
      res.writeHead(403).end('Forbidden');
      return;
    }

    const data = await readFile(filePath);
    const ext = path.extname(filePath);
    res.writeHead(200, {
      'Content-Type': MIME[ext] ?? 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-store',
    });
    log(200, pathname, data.byteLength);
    res.end(data);
  } catch (err) {
    log(404, pathname);
    res.writeHead(404).end(`Not found: ${err.message}`);
  }
});

server.listen(PORT, () => {
  console.log(`Moonshine web examples on http://localhost:${PORT}/`);
  console.log('  http://localhost:%d/', PORT);
  console.log('  http://localhost:%d/stt/', PORT);
  console.log('  http://localhost:%d/tts/', PORT);
  console.log('  http://localhost:%d/agent-flow/', PORT);
  console.log('  http://localhost:%d/dictation/', PORT);
  console.log('  http://localhost:%d/meeting-notes/', PORT);
  console.log('(cross-origin isolation headers enabled for threads/SIMD)');
  console.log('(append ?local=1 to load the locally-built /wasm/dist binding)');
  console.log('(append &assets=local to load models from the repo, offline)');
  console.log('(append &fresh=1 to purge the cached model files)');
  console.log('(responses are no-store, so a plain reload always refetches)');
});
