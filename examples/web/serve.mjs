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
// then open http://localhost:8080/stt/ (or /tts/, /dialog-flow/).
//
// By default the examples import the published binding from the jsDelivr CDN.
// To test the locally-built binding served from /wasm/dist/index.js instead,
// append ?local=1 to the URL, e.g. http://localhost:8080/stt/?local=1.
//
// A few extra path prefixes are mounted so the examples (and the browser
// integration test) can load model files straight from the repo instead of the
// CDN — handy for offline, hermetic testing:
//
//   /wasm/...        -> <repo>/wasm/...                 (the built binding)
//   /test-assets/... -> <repo>/test-assets/...          (small STT/intent models)
//   /tts-data/...    -> <repo>/core/moonshine-tts/data/ (the kokoro TTS assets)

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
};

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://localhost:${PORT}`);
    let pathname = decodeURIComponent(url.pathname);
    if (pathname === '/favicon.ico') {
      res.writeHead(204).end();
      return;
    }
    if (pathname.endsWith('/')) pathname += 'index.html';

    const filePath = path.normalize(resolveFilePath(pathname));
    // Prevent path traversal outside the repo (all mounts live under it).
    if (!filePath.startsWith(REPO_ROOT)) {
      res.writeHead(403).end('Forbidden');
      return;
    }

    const data = await readFile(filePath);
    const ext = path.extname(filePath);
    res.writeHead(200, {
      'Content-Type': MIME[ext] ?? 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-cache',
    });
    res.end(data);
  } catch (err) {
    res.writeHead(404).end(`Not found: ${err.message}`);
  }
});

server.listen(PORT, () => {
  console.log(`Moonshine web examples on http://localhost:${PORT}/`);
  console.log('  http://localhost:%d/stt/', PORT);
  console.log('  http://localhost:%d/tts/', PORT);
  console.log('  http://localhost:%d/dialog-flow/', PORT);
  console.log('(cross-origin isolation headers enabled for threads/SIMD)');
  console.log('(append ?local=1 to load the locally-built /wasm/dist binding)');
  console.log('(append &assets=local to load models from the repo, offline)');
});
