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

import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
// Repo root, so examples can import the built binding at /wasm/dist/index.js.
const REPO_ROOT = path.resolve(ROOT, '..', '..');
const PORT = Number(process.argv[2] ?? 8080);

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
    if (pathname.endsWith('/')) pathname += 'index.html';

    // Serve examples from examples/web and the binding from /wasm/*.
    const base = pathname.startsWith('/wasm/') ? REPO_ROOT : ROOT;
    const filePath = path.join(base, pathname);
    // Prevent path traversal outside the served roots.
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
});
