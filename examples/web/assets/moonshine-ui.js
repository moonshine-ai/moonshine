/**
 * Shared chrome and helpers for the Moonshine web examples.
 *
 * The demo pages each show off a different part of the library, but they
 * all need the same surrounding furniture: navigation between the demos, a
 * download-progress presenter that makes the one-time model fetch feel
 * intentional, a code panel that shows the calls the page actually made, and a
 * footer that tells you where to go next.
 *
 * No build step, no dependencies: plain ES modules loaded straight from
 * serve.mjs.
 */

/** Cache the binding's AssetDownloader writes model files into. */
const MODEL_CACHE = 'moonshine-models-v1';

const GITHUB_URL = 'https://github.com/moonshine-ai/moonshine';
const SITE_URL = 'https://moonshine.ai';
const DISCORD_URL = 'https://discord.gg/27qp9zSRXF';
const NPM_PACKAGE = '@moonshine-ai/moonshine-wasm';
// Snippet captions link here. Main rather than a tag, so a link keeps working
// as the examples move around between releases.
const SOURCE_BASE = `${GITHUB_URL}/blob/main`;

// Walkthroughs of the mechanics first, then the apps built out of them.
const DEMOS = [
  { id: 'stt', href: '/stt/', label: 'Speech to Text' },
  { id: 'tts', href: '/tts/', label: 'Text to Speech' },
  { id: 'agent', href: '/agent-flow/', label: 'Voice Agent' },
  { id: 'dictation', href: '/dictation/', label: 'Dictation App' },
];

// --- Page configuration ---------------------------------------------------

/**
 * Reads the query parameters the examples share.
 *
 * `local=1` swaps the published jsDelivr binding for the one built into
 * /wasm/dist, and `assets=local` points the model downloads at the copies
 * vendored in the repo so a page can run with no network at all. The browser
 * integration test uses both.
 */
export function pageConfig() {
  const params = new URLSearchParams(location.search);
  const useLocal = params.get('local') === '1';
  return {
    params,
    useLocal,
    useLocalAssets: params.get('assets') === 'local',
    moduleUrl: useLocal
      ? '/wasm/dist/index.js'
      : `https://cdn.jsdelivr.net/npm/${NPM_PACKAGE}/dist/index.js`,
  };
}

/** Preserves the demo-wiring params when linking between pages. */
function carryParams(href, params) {
  const keep = new URLSearchParams();
  for (const name of ['local', 'assets']) {
    const value = params.get(name);
    if (value !== null) keep.set(name, value);
  }
  const query = keep.toString();
  return query ? `${href}?${query}` : href;
}

// --- Chrome ---------------------------------------------------------------

// The official mark from moonshine.ai. Its crescent is dark navy, drawn for a
// light background, so .ms-brand__mark sits it on a pale chip rather than
// recolouring artwork we do not own.
const LOGO_IMG = `
<img class="ms-brand__mark" src="/assets/moonshine-logo.png" alt="" width="28" height="28" />`;

const GITHUB_SVG = `
<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
  <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
           0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
           1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
           0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.4 7.4 0 0 1 2-.27c.68 0
           1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0
           3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01
           8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
</svg>`;

/**
 * Injects the shared top navigation and footer.
 *
 * @param {object} options
 * @param {'stt'|'tts'|'agent'|'dictation'|'home'} options.active  Which tab to mark.
 * @param {string} [options.ctaTitle]  Headline above the footer buttons.
 */
export function mountChrome({ active, ctaTitle } = {}) {
  const { params } = pageConfig();

  const nav = document.createElement('nav');
  nav.className = 'ms-nav';
  nav.innerHTML = `
    <div class="ms-shell ms-nav__inner">
      <a class="ms-brand" href="${carryParams('/', params)}">
        ${LOGO_IMG}<span>Moonshine Voice</span>
      </a>
      <div class="ms-tabs">
        ${DEMOS.map(
          (demo) => `
          <a class="ms-tab${demo.id === active ? ' is-active' : ''}"
             href="${carryParams(demo.href, params)}">${demo.label}</a>`,
        ).join('')}
      </div>
      <a class="ms-nav__github" href="${GITHUB_URL}" target="_blank" rel="noopener"
         aria-label="Moonshine on GitHub">${GITHUB_SVG}</a>
    </div>`;
  document.body.prepend(nav);

  const footer = document.createElement('footer');
  footer.className = 'ms-footer';
  footer.innerHTML = `
    <div class="ms-shell">
      <div class="ms-footer__cta">
        <h2>${ctaTitle ?? 'Build this into your own app.'}</h2>
        <a class="ms-btn ms-btn--primary" href="${GITHUB_URL}#readme" target="_blank" rel="noopener">
          Get started on GitHub
        </a>
      </div>
      <div class="ms-footer__links">
        <a href="${SITE_URL}" target="_blank" rel="noopener">moonshine.ai</a>
        <a href="${DISCORD_URL}" target="_blank" rel="noopener">Discord</a>
        <a href="https://pypi.org/project/moonshine-voice/" target="_blank" rel="noopener">Python</a>
        <a href="https://www.npmjs.com/package/${NPM_PACKAGE}" target="_blank" rel="noopener">npm</a>
        <a href="https://huggingface.co/UsefulSensors" target="_blank" rel="noopener">Hugging Face</a>
        <span>Runs on-device. MIT licensed.</span>
      </div>
    </div>`;
  document.body.append(footer);
}

const ROTATOR_HOLD_MS = 1000;
const ROTATOR_MOVE_MS = 340;

/**
 * The platforms this library ships on, for the rotating word in the footer
 * copy. Shared so the three demos cannot drift apart on the list or its order.
 */
export const PLATFORMS = ['Web', 'iOS', 'Android', 'IoT', 'Python'];

/**
 * Writes the keyframes for a reel of `count` words: hold each one still, then
 * slide to the next. The frames depend on how many words there are, so they
 * are generated rather than written out, and reused across reels of the same
 * length.
 */
function ensureReelKeyframes(count, holdMs, moveMs) {
  const name = `ms-reel-${count}-${holdMs}-${moveMs}`;
  if (document.getElementById(name)) return name;

  const stepMs = holdMs + moveMs;
  const totalMs = stepMs * count;
  const stepPct = 100 / count;
  const holdPct = (holdMs / totalMs) * 100;
  let frames = '';
  for (let i = 0; i < count; i++) {
    const from = (i * stepPct).toFixed(3);
    const until = (i * stepPct + holdPct).toFixed(3);
    frames += `${from}%,${until}%{transform:translateY(calc(var(--ms-reel-step) * -${i}))}`;
  }
  // Landing on the duplicated first word means restarting at 0% is invisible.
  frames += `100%{transform:translateY(calc(var(--ms-reel-step) * -${count}))}`;

  const style = document.createElement('style');
  style.id = name;
  style.textContent = `@keyframes ${name}{${frames}}`;
  document.head.append(style);
  return name;
}

/**
 * A word that cycles vertically through `words`, for saying "this runs
 * everywhere" without listing every platform in the sentence.
 *
 * Returns HTML, so it can be dropped into copy like a word. The space it takes
 * is the width of the longest word in the list, with each one centred in it, so
 * the sentence around it never reflows.
 */
export function rotatingWords(words, { holdMs = ROTATOR_HOLD_MS, moveMs = ROTATOR_MOVE_MS } = {}) {
  const name = ensureReelKeyframes(words.length, holdMs, moveMs);
  const spans = (list) => list.map((word) => `<span>${word}</span>`).join('');
  // Three words are on screen at once, so the wrap needs one word carried
  // above the first and two below the last. Stopping at one leaves the slot
  // under the final word empty, and the next word appears out of nowhere when
  // the animation restarts instead of already sitting there faded.
  const at = (i) => words[((i % words.length) + words.length) % words.length];
  const reel = [at(-1), ...words, at(0), at(1)];
  const label = words.length > 1
    ? `${words.slice(0, -1).join(', ')}, or ${words[words.length - 1]}`
    : words[0];

  return (
    `<span class="ms-reel" role="img" aria-label="${label}"` +
    ` style="--ms-reel-name:${name};--ms-reel-duration:${(holdMs + moveMs) * words.length}ms">` +
    // Invisible, in-flow, and holding every word stacked in one grid cell, so
    // the element is exactly as wide as the longest of them.
    `<span class="ms-reel__sizer" aria-hidden="true">${spans(words)}</span>` +
    `<span class="ms-reel__track" aria-hidden="true">${spans(reel)}</span>` +
    `</span>`
  );
}

/** Renders a proof-point strip. Each entry is `{ value, label }`. */
export function statStrip(stats) {
  return `<ul class="ms-stats">${stats
    .map(
      (stat) => `<li class="ms-stat">
        <span class="ms-stat__value">${stat.value}</span>
        <span class="ms-stat__label">${stat.label}</span>
      </li>`,
    )
    .join('')}</ul>`;
}

// --- Model loading experience --------------------------------------------

/**
 * Empties the model cache when the page was opened with ?fresh=1.
 *
 * serve.mjs sends `no-store`, so the HTTP cache never holds a stale binding.
 * Model files are different: the binding's AssetDownloader stores them through
 * the Cache Storage API, which is deliberately independent of the HTTP cache
 * and survives every flavour of reload, including Shift+Cmd+R. This is the only
 * way to force a redownload short of clearing site data by hand.
 *
 * Returns true when something was actually deleted.
 */
export async function purgeModelCacheIfRequested() {
  const { params } = pageConfig();
  if (params.get('fresh') !== '1') return false;
  try {
    if (typeof caches === 'undefined') return false;
    const deleted = await caches.delete(MODEL_CACHE);
    console.info(
      deleted
        ? `[moonshine] purged model cache "${MODEL_CACHE}" (?fresh=1)`
        : `[moonshine] model cache "${MODEL_CACHE}" was already empty (?fresh=1)`,
    );
    return deleted;
  } catch {
    return false;
  }
}

/** True when the model cache already holds something, i.e. a repeat visit. */
export async function modelsAreCached() {
  try {
    if (typeof caches === 'undefined') return false;
    const cache = await caches.open(MODEL_CACHE);
    const keys = await cache.keys();
    return keys.length > 0;
  } catch {
    return false;
  }
}

function formatBytes(bytes) {
  if (!bytes || bytes < 0) return '';
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(bytes < 100 * 1024 * 1024 ? 1 : 0)} MB`;
}

/**
 * What the status line says for the whole download. Every status on these
 * pages is a word or two, short enough that it cannot wrap and shunt the card
 * around; the numbers live on their own line inside the loading region.
 */
const DOWNLOAD_STATUS = 'Downloading…';

/**
 * Presents the one-time model download.
 *
 * The download is the longest thing that happens on any of these pages and the
 * easiest place to lose someone, so rather than a bare percentage this shows a
 * real bar plus a rotating line explaining what the wait buys, and says
 * nothing at all when the models are already cached.
 */
export class LoadingUi {
  /**
   * @param {object} options
   * @param {HTMLElement} options.status  Line for the headline state.
   * @param {HTMLElement} options.progress  The `.ms-progress` wrapper.
   * @param {HTMLElement} [options.note]  Line for the rotating explainer.
   * @param {HTMLElement} [options.detail]  Line for the percentage and bytes.
   * @param {string[]} [options.notes]  Explainer copy to cycle through.
   */
  constructor({ status, progress, note, detail, notes = [] }) {
    this.statusEl = status;
    this.progressEl = progress;
    this.barEl = progress?.querySelector('.ms-progress__bar');
    this.noteEl = note;
    this.detailEl = detail;
    this.notes = notes;
    this.noteIndex = 0;
    this.noteTimer = undefined;
    this.regionEl = progress?.closest('.ms-loading') ?? note?.closest('.ms-loading');
    this.shown = false;

    // Done up front, not on the first tick, because the region holds its space
    // from the moment the page renders whether a download happens or not.
    this.reserveNoteHeight();
    let queued = false;
    window.addEventListener('resize', () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        this.reserveNoteHeight();
      });
    });
  }

  setStatus(text, tone) {
    if (!this.statusEl) return;
    this.statusEl.textContent = text;
    if (tone) this.statusEl.dataset.tone = tone;
    else delete this.statusEl.dataset.tone;
  }

  setDetail(text) {
    if (this.detailEl) this.detailEl.textContent = text;
  }

  /**
   * Progress callback in the shape the binding's `onProgress` uses. The
   * fraction already covers the whole model, so it goes straight on the bar.
   * Filenames are deliberately not shown: they mean nothing to a visitor, and
   * a bar that only ever moves forwards reads as progress where a list of
   * `.ort` files reads as noise.
   */
  onProgress = (fraction, _file, bytes) => {
    this.show();
    const known = bytes?.total > 0;
    this.progressEl?.classList.toggle('is-indeterminate', !known);

    if (!known) {
      // No declared sizes, so any percentage would be invented.
      this.setDetail(`${formatBytes(bytes?.loaded ?? 0)} so far`);
      return;
    }
    const percent = Math.round(100 * fraction);
    if (this.barEl) this.barEl.style.width = `${percent}%`;
    this.setDetail(
      `${percent}% · ${formatBytes(bytes.loaded)} of ${formatBytes(bytes.total)}`,
    );
  };

  /** Reveals the bar and starts cycling the explainer copy. */
  show() {
    if (this.shown) return;
    this.shown = true;
    this.setStatus(DOWNLOAD_STATUS);
    if (this.regionEl) this.regionEl.classList.add('is-open');
    else if (this.progressEl) this.progressEl.hidden = false;
    if (this.noteEl && this.notes.length && !this.noteTimer) {
      this.noteEl.textContent = this.notes[0];
      this.noteTimer = setInterval(() => this.rotateNote(), 4200);
    }
  }

  /**
   * Pins the explainer to the height of its longest line, so neither the copy
   * rotating nor the region emptying out can change how much room it takes.
   */
  reserveNoteHeight() {
    const el = this.noteEl;
    if (!el || !this.notes.length) return;
    const previous = el.textContent;
    el.style.minHeight = '0px';
    let tallest = 0;
    for (const note of this.notes) {
      el.textContent = note;
      tallest = Math.max(tallest, el.scrollHeight);
    }
    el.textContent = previous;
    el.style.minHeight = `${tallest}px`;
  }

  rotateNote() {
    if (!this.noteEl) return;
    this.noteIndex = (this.noteIndex + 1) % this.notes.length;
    this.noteEl.classList.add('is-fading');
    setTimeout(() => {
      this.noteEl.textContent = this.notes[this.noteIndex];
      this.noteEl.classList.remove('is-fading');
    }, 240);
  }

  /** Blanks the bar and stops the copy rotating, leaving the space behind. */
  done() {
    this.shown = false;
    this.progressEl?.classList.remove('is-indeterminate');
    clearInterval(this.noteTimer);
    this.noteTimer = undefined;
    if (this.regionEl) this.regionEl.classList.remove('is-open');
    else if (this.progressEl) this.progressEl.hidden = true;
  }
}

/**
 * Turns an exception into something a visitor can act on. The binding throws
 * useful errors, but a bare `NotAllowedError` on the page helps nobody.
 */
export function describeError(err) {
  const message = err?.message ?? String(err);
  if (err?.name === 'NotAllowedError' || /permission denied/i.test(message)) {
    return 'Microphone access was blocked. Allow it in your browser’s address bar and try again.';
  }
  if (err?.name === 'NotFoundError') {
    return 'No microphone found. Plug one in, or use the options below instead.';
  }
  if (/SharedArrayBuffer/.test(message)) {
    return 'This browser did not get the cross-origin isolation headers the demo needs. Serve the page with examples/web/serve.mjs.';
  }
  if (/fetch|network|Failed to download/i.test(message)) {
    return `Could not download the model: ${message}`;
  }
  return message;
}

// --- Microphone level meter ----------------------------------------------

/**
 * A row of bars that follow the microphone level, so it is obvious the page is
 * hearing something before any text has been transcribed.
 */
// --- Audio input device ---------------------------------------------------

/** Where the chosen capture device is remembered across pages and reloads. */
const DEVICE_KEY = 'moonshine.audioInputDeviceId';

/**
 * The capture device the visitor picked, or null to let the browser choose.
 *
 * Worth having at all because Chrome keeps its own "default" microphone that
 * can disagree with the operating system's, and when it is wrong it does not
 * fail. It hands back a live track of digital silence. Naming a device
 * explicitly is the only reliable way around that, so the choice is remembered
 * and shared by every demo.
 */
export function preferredAudioDeviceId() {
  try {
    return localStorage.getItem(DEVICE_KEY) || null;
  } catch {
    // Storage can be blocked; fall back to the browser default.
    return null;
  }
}

/** Remembers a capture device. Pass null to go back to the browser default. */
export function setPreferredAudioDeviceId(deviceId) {
  try {
    if (deviceId) localStorage.setItem(DEVICE_KEY, deviceId);
    else localStorage.removeItem(DEVICE_KEY);
  } catch {
    // Not being able to remember is survivable; the picker still works.
  }
}

/**
 * Builds the `audio` value for `getUserMedia`, honouring the saved device.
 *
 * The saved id is checked against the current device list first: ids are only
 * stable while a device stays plugged in, and an `exact` constraint naming one
 * that has gone away throws OverconstrainedError instead of falling back. A
 * stale id is dropped rather than left to break capture on every later visit.
 */
export async function audioConstraints() {
  const deviceId = preferredAudioDeviceId();
  if (!deviceId) return true;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const present = devices.some(
      (d) => d.kind === 'audioinput' && d.deviceId === deviceId,
    );
    if (!present) {
      setPreferredAudioDeviceId(null);
      return true;
    }
  } catch {
    return true;
  }
  return { deviceId: { exact: deviceId } };
}

/** Opens a capture stream on the saved device, falling back if it refuses. */
export async function openMicrophone() {
  const constraints = await audioConstraints();
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: constraints });
  } catch (err) {
    if (constraints === true) throw err;
    // The device disappeared between the check above and now.
    setPreferredAudioDeviceId(null);
    return navigator.mediaDevices.getUserMedia({ audio: true });
  }
}

/** Human-readable name for the saved device, for status text. */
export async function preferredAudioDeviceLabel() {
  const deviceId = preferredAudioDeviceId();
  if (!deviceId) return null;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const match = devices.find(
      (d) => d.kind === 'audioinput' && d.deviceId === deviceId,
    );
    return match ? match.label || 'Selected microphone' : null;
  } catch {
    return null;
  }
}

/**
 * Turns a `<select>` into a capture-device picker backed by the shared
 * preference, and keeps it in sync as devices come and go.
 *
 * Device labels are empty until microphone permission has been granted, so
 * this can be called early and refreshed later once a stream has been opened.
 *
 * @param {HTMLSelectElement} select
 * @param {{ onChange?: (deviceId: string|null) => void }} [options]
 * @returns {{ refresh: () => Promise<void> }}
 */
export function mountDevicePicker(select, { onChange } = {}) {
  async function refresh() {
    let inputs = [];
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      inputs = devices.filter((d) => d.kind === 'audioinput');
    } catch {
      // Leave the default-only list in place.
    }
    const saved = preferredAudioDeviceId();
    select.innerHTML = '';
    const auto = document.createElement('option');
    auto.value = '';
    auto.textContent = 'Browser default';
    select.append(auto);
    for (const device of inputs) {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent =
        device.label || `Microphone ${device.deviceId.slice(0, 8)}`;
      select.append(option);
    }
    // A saved device that is no longer present falls back to the default entry.
    select.value = inputs.some((d) => d.deviceId === saved) ? saved : '';
    if (saved && select.value === '') setPreferredAudioDeviceId(null);
  }

  select.addEventListener('change', () => {
    setPreferredAudioDeviceId(select.value || null);
    onChange?.(select.value || null);
  });
  navigator.mediaDevices?.addEventListener?.('devicechange', refresh);

  void refresh();
  return { refresh };
}

export class MicMeter {
  constructor(element, barCount = 28) {
    this.el = element;
    this.bars = [];
    // Pages build a fresh meter each time capture starts, so drop any bars a
    // previous instance left behind. Appending to them instead would grow the
    // row on every restart, and only the newest set (the ones on the right)
    // would animate.
    this.el.replaceChildren();
    for (let i = 0; i < barCount; i++) {
      const bar = document.createElement('span');
      this.el.append(bar);
      this.bars.push(bar);
    }
    this.raf = undefined;
  }

  /** Taps an existing MediaStream and animates until {@link stop}. */
  start(stream, audioContext) {
    this.ctx = audioContext ?? new AudioContext();
    this.ownsCtx = !audioContext;
    // A suspended context leaves the analyser reading zeroes forever, so the
    // meter would sit flat even while the microphone is working fine.
    if (this.ctx.state === 'suspended') void this.ctx.resume();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.75;
    this.source = this.ctx.createMediaStreamSource(stream);
    this.source.connect(this.analyser);
    this.data = new Uint8Array(this.analyser.frequencyBinCount);
    this.tick();
  }

  tick = () => {
    this.analyser.getByteFrequencyData(this.data);
    const perBar = Math.floor(this.data.length / (this.bars.length * 2.5));
    this.bars.forEach((bar, i) => {
      let sum = 0;
      for (let j = 0; j < perBar; j++) sum += this.data[i * perBar + j] ?? 0;
      const level = Math.min(1, sum / perBar / 140);
      bar.style.height = `${3 + level * 33}px`;
      bar.style.opacity = String(0.35 + level * 0.65);
    });
    this.raf = requestAnimationFrame(this.tick);
  };

  stop() {
    cancelAnimationFrame(this.raf);
    this.raf = undefined;
    this.source?.disconnect();
    this.analyser?.disconnect();
    if (this.ownsCtx) void this.ctx?.close();
    this.source = this.analyser = this.ctx = undefined;
    for (const bar of this.bars) {
      bar.style.height = '3px';
      bar.style.opacity = '0.35';
    }
  }
}

// --- Code panel -----------------------------------------------------------

/**
 * Per-language keyword lists and comment markers. Only the languages the
 * snippet tabs use, and only enough of each to make a short sample readable.
 */
const LANGUAGES = {
  javascript: {
    keywords:
      /\b(await|async|const|let|var|new|return|if|else|function|for|of|in|import|from|export|class|try|catch|throw|true|false|null|undefined)\b/g,
    comment: /(\/\/.*)$/,
  },
  python: {
    keywords:
      /\b(def|class|return|if|elif|else|for|in|while|with|as|import|from|lambda|try|except|finally|raise|yield|pass|not|and|or|is|async|await|None|True|False|self)\b/g,
    comment: /(#.*)$/,
  },
  swift: {
    keywords:
      /\b(let|var|func|return|if|else|guard|for|in|while|repeat|try|await|async|throws|throw|struct|class|enum|protocol|extension|import|weak|self|public|private|internal|static|nil|true|false)\b/g,
    comment: /(\/\/.*)$/,
    // Also matches a trailing closure, so `.onText { … }` reads as the call it
    // is rather than as a property.
    call: /\.(\w+)(?=\s*[({])/g,
  },
  java: {
    keywords:
      /\b(new|return|if|else|for|while|do|class|interface|extends|implements|public|private|protected|static|final|void|try|catch|finally|throw|throws|import|package|this|int|long|float|double|boolean|char|String|var|null|true|false)\b/g,
    comment: /(\/\/.*)$/,
  },
};

/** A string literal in any of the four languages above. */
const STRING_LITERAL = /(['"`])(?:\\.|(?!\1)[^\\])*\1/g;
/** Matches the placeholders parked in a private-use block by `stashStrings`. */
const STASHED_STRING = /\uE000([\uE100-\uE1ff])/g;

/** Just enough highlighting to make a short snippet readable. */
function highlight(line, language = 'javascript') {
  const spec = LANGUAGES[language] ?? LANGUAGES.javascript;

  const escaped = line
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Strings come out first, stashed behind placeholders so no later pass can
  // reach inside one: not the keyword pass, which would otherwise colour prose,
  // and not the comment scan, for which the `//` in a URL is not a comment. The
  // placeholders contain no digits or word characters, so the number and
  // keyword passes step over them.
  const strings = [];
  const stashed = escaped.replace(STRING_LITERAL, (match) => {
    strings.push(match);
    return `\uE000${String.fromCharCode(0xe100 + strings.length - 1)}`;
  });
  const restore = (text, wrap) =>
    text.replace(STASHED_STRING, (_, mark) => {
      const literal = strings[mark.charCodeAt(0) - 0xe100];
      return wrap ? `<span class="tok-str">${literal}</span>` : literal;
    });

  const comment = stashed.match(spec.comment);
  const code = comment ? stashed.slice(0, comment.index) : stashed;

  const painted = code
    .replace(spec.keywords, '<span class="tok-key">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="tok-num">$1</span>')
    .replace(spec.call ?? /\.(\w+)(?=\()/g, '.<span class="tok-fn">$1</span>');

  // A string inside a comment is restored unwrapped, so the whole comment stays
  // one colour rather than growing a green island.
  return comment
    ? `${restore(painted, true)}<span class="tok-com">${restore(comment[1], false)}</span>`
    : restore(painted, true);
}

/** Renders one snippet as `<span class="ms-line">` blocks. */
function codeLines(code, language) {
  // Joined with nothing: each line is its own block element, so a newline
  // between them would double-space the listing inside <pre>.
  return code
    .replace(/\n+$/, '')
    .split('\n')
    .map(
      (line, i) =>
        `<span class="ms-line" data-line="${i}">${highlight(line, language) || '&nbsp;'}</span>`,
    )
    .join('');
}

/**
 * Builds a code panel. Every line is wrapped in its own element so a page can
 * highlight the step it is currently executing.
 *
 * Pass `code` for a single snippet, or `tabs` for the same thing in several
 * languages. Tabbed panels stack every snippet in one grid cell and hide the
 * inactive ones, so the panel is always as tall as the longest and switching
 * language moves nothing on the page.
 *
 * @param {object} options
 * @param {string} [options.code]  The snippet to display.
 * @param {string} [options.language]  Language of `code`, for highlighting.
 * @param {string} [options.file]  Caption shown in the header.
 * @param {boolean|string} [options.install]  The install line, or true for npm,
 *   or false for none. Ignored when `tabs` is given; see `tabs[].install`.
 * @param {Array<object>} [options.tabs]  One entry per language, each with
 *   `id`, `label`, `code`, and optionally `file`, `install` and `installHint`.
 * @param {string} [options.path]  Repository-relative path of the file this
 *   came from. Turns the caption into a link to it on GitHub. Per-tab as
 *   `tabs[].path`.
 * @param {[number, number]} [options.lines]  Inclusive line range within
 *   `path`, added to the link as a `#L12-L34` anchor. Per-tab as
 *   `tabs[].lines`.
 * @param {string} [options.active]  Id of the tab to open on. Defaults to the
 *   first.
 * @param {(pane: object, index: number) => void} [options.onTab]  Called when
 *   the reader switches language, and once on creation.
 */
export function codePanel({
  code,
  language = 'javascript',
  file = 'index.html',
  install = true,
  path,
  lines,
  tabs,
  active,
  onTab,
}) {
  const panes = tabs ?? [{ id: language, label: language, file, path, lines, code, install }];
  const initial = Math.max(
    panes.findIndex((pane) => pane.id === active),
    0,
  );
  const installLine = (pane) =>
    pane.install === true ? `npm i ${NPM_PACKAGE}` : pane.install || '';
  const hasInstall = panes.some((pane) => installLine(pane));

  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div class="ms-code">
      <div class="ms-code__head${tabs ? ' ms-code__head--tabbed' : ''}">
        ${
          tabs
            ? `<span class="ms-code__tabs" role="tablist">${panes
                .map(
                  (pane, i) =>
                    `<button type="button" class="ms-lang-tab${i === initial ? ' is-active' : ''}"
                       role="tab" aria-selected="${i === initial}" data-tab="${i}">${pane.label}</button>`,
                )
                .join('')}</span>`
            : ''
        }
        <a class="ms-code__file" data-file target="_blank" rel="noopener"></a>
        <span class="ms-code__actions">
          <button type="button" class="ms-btn ms-btn--quiet ms-small" data-copy>Copy</button>
        </span>
      </div>
      <div class="ms-code__panes">
        ${panes
          .map(
            (pane, i) =>
              `<pre class="ms-code__pane${i === initial ? ' is-active' : ''}" data-pane="${i}"
                 role="tabpanel"><code>${codeLines(pane.code, pane.id)}</code></pre>`,
          )
          .join('')}
      </div>
    </div>
    ${
      hasInstall
        ? `<div class="ms-install">
             <code data-install></code>
             <span class="ms-install__hint" data-install-hint></span>
             <button type="button" class="ms-btn ms-btn--quiet ms-small" data-copy-install>Copy</button>
           </div>`
        : ''
    }`;

  let current = initial;
  const fileEl = wrap.querySelector('[data-file]');
  const installEl = wrap.querySelector('[data-install]');
  const hintEl = wrap.querySelector('[data-install-hint]');

  const show = (index) => {
    current = index;
    for (const [i, pane] of wrap.querySelectorAll('[data-pane]').entries()) {
      pane.classList.toggle('is-active', i === index);
    }
    for (const [i, tab] of wrap.querySelectorAll('[data-tab]').entries()) {
      tab.classList.toggle('is-active', i === index);
      tab.setAttribute('aria-selected', String(i === index));
    }
    if (fileEl) {
      const { file: name, path: source, lines: range } = panes[index];
      fileEl.textContent = name ?? '';
      // Without a path there is nothing to point at, so it stays a plain
      // caption rather than becoming a link that goes nowhere.
      if (source) {
        const anchor = range ? `#L${range[0]}-L${range[1]}` : '';
        fileEl.href = `${SOURCE_BASE}/${source}${anchor}`;
        fileEl.title = range ? `${source} lines ${range[0]}-${range[1]}` : source;
      } else {
        fileEl.removeAttribute('href');
        fileEl.removeAttribute('title');
      }
    }
    if (installEl) installEl.textContent = installLine(panes[index]);
    if (hintEl) hintEl.textContent = panes[index].installHint ?? '';
    onTab?.(panes[index], index);
  };
  show(initial);

  for (const tab of wrap.querySelectorAll('[data-tab]')) {
    tab.addEventListener('click', () => show(Number(tab.dataset.tab)));
  }

  const copyButton = wrap.querySelector('[data-copy]');
  copyButton?.addEventListener('click', () =>
    copyToClipboard(panes[current].code, copyButton),
  );
  const installButton = wrap.querySelector('[data-copy-install]');
  installButton?.addEventListener('click', () =>
    copyToClipboard(installLine(panes[current]), installButton),
  );

  return wrap;
}

async function copyToClipboard(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const original = button.textContent;
    button.textContent = 'Copied';
    setTimeout(() => {
      button.textContent = original;
    }, 1400);
  } catch {
    button.textContent = 'Press ⌘C';
  }
}

/**
 * Walks a code panel as a flow runs: marks one line as running and everything
 * before it as done. Pass `null` to clear.
 */
export function markCodeStep(root, lineIndex) {
  // Every pane is cleared, but only the open one is marked. Clearing just the
  // open one would leave the language the reader switched away from lit up on
  // whatever line it had reached, waiting to reappear on the way back.
  for (const line of root.querySelectorAll('.ms-line')) {
    line.classList.remove('is-running', 'is-done');
  }
  if (lineIndex == null) return;
  const lines = root.querySelectorAll('.ms-code__pane.is-active .ms-line');
  lines.forEach((line, i) => {
    line.classList.toggle('is-running', i === lineIndex);
    line.classList.toggle('is-done', i < lineIndex);
  });
}

// --- Audio helpers --------------------------------------------------------

/** Wraps mono Float32 PCM as a 16-bit WAV blob. */
export function pcmToWavBlob(audio, sampleRate) {
  const frames = audio.length;
  const buffer = new ArrayBuffer(44 + frames * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + frames * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, frames * 2, true);
  for (let i = 0; i < frames; i++) {
    const s = Math.max(-1, Math.min(1, audio[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/** Decodes an audio file and downmixes/resamples it to mono 16 kHz. */
export async function decodeAudioFile(file) {
  const bytes = file instanceof ArrayBuffer ? file : await file.arrayBuffer();
  const decoded = await new OfflineAudioContext(1, 1, 16000).decodeAudioData(bytes);
  const frames = Math.max(1, Math.round(decoded.duration * 16000));
  const offline = new OfflineAudioContext(1, frames, 16000);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return { audio: rendered.getChannelData(0), sampleRate: 16000 };
}

export { GITHUB_URL, SITE_URL, DISCORD_URL, NPM_PACKAGE, formatBytes };
