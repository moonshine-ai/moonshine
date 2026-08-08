// Tests the caching contract of the examples static server (examples/web/serve.mjs).
//
// The examples exist to exercise the locally-built binding, so "I rebuilt and
// reloaded, but am I actually running the new code?" has to have an unambiguous
// answer. That rests entirely on the server sending `Cache-Control: no-store`.
// `no-cache` looks like it would do the job but does not: it permits the browser
// to keep the bytes and merely revalidate, and this server sends no ETag or
// Last-Modified for it to revalidate against.
//
// This matters most for the multi-file module graph under /wasm/dist: the entry
// point index.js is a re-export barrel, so the code being debugged usually lives
// in a sibling like mic-transcriber.js. A cache-busting query string on the
// import URL would not reach those siblings, which is why the guarantee is
// enforced here, on every response, rather than at the call site.

import test from 'node:test';
import assert from 'node:assert/strict';
import { startExampleServer } from './browser-helpers.mjs';

test('serve.mjs caching and isolation headers', async (t) => {
  const server = await startExampleServer();
  t.after(() => server.close());

  // The barrel and a sibling it imports: both must be unstorable.
  for (const path of ['/stt/', '/wasm/dist/index.js', '/wasm/dist/mic-transcriber.js']) {
    const res = await fetch(`${server.origin}${path}`);
    assert.equal(res.status, 200, `${path} should be served`);
    assert.equal(
      res.headers.get('cache-control'),
      'no-store',
      `${path} must be no-store so a rebuild is never masked by a cached copy`,
    );
    // SharedArrayBuffer (and so the threaded build) needs cross-origin isolation.
    assert.equal(res.headers.get('cross-origin-opener-policy'), 'same-origin');
    assert.equal(res.headers.get('cross-origin-embedder-policy'), 'require-corp');
  }
});

test('serve.mjs refuses path traversal outside the repo', async (t) => {
  const server = await startExampleServer();
  t.after(() => server.close());

  const res = await fetch(`${server.origin}/wasm/../../../../etc/passwd`);
  assert.ok(res.status >= 400, 'traversal should not return a file');
});
