// The snippets behind the language tabs on the demo pages are hand-maintained
// (see the header of examples/web/assets/snippets.js), so these tests pin down
// the parts a careless edit would silently break: the line numbers the voice
// agent page highlights as a conversation runs, and the metadata every tab
// needs to render.
//
// Phase 4 of docs/design/api-regularization.md generates that file from the
// examples themselves. These checks stay useful afterwards, because generating
// a snippet does not renumber the steps that point into it.

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const SNIPPETS = path.join(REPO_ROOT, 'examples', 'web', 'assets', 'snippets.js');
const { MIC_TRANSCRIPTION, TEXT_TO_SPEECH, DIALOG_FLOW, INSTALL, withInstall } = await import(
  pathToFileURL(SNIPPETS).href
);

const ALL = [
  ['MIC_TRANSCRIPTION', MIC_TRANSCRIPTION],
  ['TEXT_TO_SPEECH', TEXT_TO_SPEECH],
  ['DIALOG_FLOW', DIALOG_FLOW],
];

test('every snippet carries what a tab needs to render', async (t) => {
  const { access } = await import('node:fs/promises');
  for (const [name, snippets] of ALL) {
    await t.test(name, async () => {
      assert.ok(snippets.length >= 3, 'expected at least three languages');
      const ids = snippets.map((s) => s.id);
      assert.equal(new Set(ids).size, ids.length, `duplicate tab ids: ${ids}`);
      assert.equal(ids[0], 'javascript', 'the page runs JavaScript, so it opens that tab');

      for (const snippet of snippets) {
        assert.ok(snippet.label, `${snippet.id} needs a tab label`);
        assert.ok(snippet.file, `${snippet.id} needs a filename caption`);
        assert.ok(snippet.code.trim(), `${snippet.id} needs code`);
        assert.ok(INSTALL[snippet.id], `${snippet.id} has no install line`);
        // A caption links to its source, so a path that no longer exists would
        // ship a 404 to the reader.
        await access(path.join(REPO_ROOT, snippet.path));
      }
    });
  }
});

test('withInstall attaches the line and hint for each language', () => {
  const tabs = withInstall(MIC_TRANSCRIPTION);
  assert.equal(tabs[0].install, 'npm i @moonshine-ai/moonshine-wasm');
  assert.equal(tabs[1].install, 'pip install moonshine-voice');
  assert.match(tabs[2].install, /^https:\/\/github\.com\//);
  assert.equal(tabs[2].installHint, 'Xcode ▸ Add Package Dependencies…');
  // npm and pip lines say where they go by themselves.
  assert.equal(tabs[0].installHint, undefined);
  // The originals are left alone, since the pages import them as constants.
  assert.equal(MIC_TRANSCRIPTION[0].install, undefined);
});

// What each step of the voice agent flow must be pointing at. The page calls
// `step('askSsid')` immediately before awaiting `d.ask(...)`, so the line it
// highlights has to be that call, in whichever language is showing.
const STEP_EXPECTATIONS = {
  askSsid: [/\bask\(/, /name of your wifi network/],
  confirmSsid: [/\bconfirm\(/, /Is that right\?/],
  startOver: [/\bsay\(/, /start over/],
  confirmApply: [/\bconfirm\(/, /Apply these changes\?/],
  done: [/\bsay\(/, /Done\. Connecting/],
  unchanged: [/\bsay\(/, /nothing changed/],
};

test('every voice agent step points at the line it claims', async (t) => {
  for (const snippet of DIALOG_FLOW) {
    await t.test(snippet.id, () => {
      const lines = snippet.code.split('\n');
      assert.deepEqual(
        Object.keys(snippet.steps).sort(),
        Object.keys(STEP_EXPECTATIONS).sort(),
        'every language needs the same set of steps, since one page drives them all',
      );

      for (const [step, patterns] of Object.entries(STEP_EXPECTATIONS)) {
        const lineNumber = snippet.steps[step];
        const line = lines[lineNumber];
        assert.ok(
          line !== undefined,
          `${snippet.id} step "${step}" points at line ${lineNumber}, past the end`,
        );
        for (const pattern of patterns) {
          assert.match(line, pattern, `${snippet.id} step "${step}" (line ${lineNumber})`);
        }
      }

      // The steps run in the order the conversation does, and no two share a
      // line, which would make the highlight ambiguous.
      const order = ['askSsid', 'confirmSsid', 'startOver', 'confirmApply', 'done', 'unchanged'];
      const numbers = order.map((step) => snippet.steps[step]);
      assert.deepEqual(numbers, [...numbers].sort((a, b) => a - b), 'steps run out of order');
      assert.equal(new Set(numbers).size, numbers.length, 'two steps share a line');
    });
  }
});
