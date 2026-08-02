/**
 * Reading an editable document out of the page, and writing it back out as
 * plain text, Markdown, or a Word file.
 *
 * The dictation app keeps its document in a `contenteditable` element, which is
 * how it can offer bold, italic, underline and strike-through with no editor
 * library. Everything here works from one intermediate form: a list of
 * paragraphs, each a list of runs carrying the text and which marks are on. Read
 * the document once, then hand the same paragraphs to whichever writer the
 * reader asked for, so the three exports can never disagree about what the
 * document said.
 *
 * The Word writer emits a real `.docx`, which is a zip of a few XML parts. That
 * needs a zip writer, and the one below stores rather than compresses: a
 * dictated document is small, and `deflate` would be an order of magnitude more
 * code than the rest of this file. Word, Pages, LibreOffice and Google Docs all
 * open a stored zip without complaint.
 *
 * Kept separate from the page so the iOS and Android ports have an obvious
 * thing to mirror.
 */

/** A stretch of text with its marks. */
/** @typedef {{ text: string, bold?: boolean, italic?: boolean, underline?: boolean, strike?: boolean }} Run */

const BLOCK_TAGS = new Set(['DIV', 'P', 'LI', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'BLOCKQUOTE']);

/** Tags that turn a mark on, whatever the browser felt like emitting. */
const MARK_TAGS = {
  B: 'bold',
  STRONG: 'bold',
  I: 'italic',
  EM: 'italic',
  U: 'underline',
  INS: 'underline',
  S: 'strike',
  STRIKE: 'strike',
  DEL: 'strike',
};

/** Inline styles that mean the same thing, since execCommand may use either. */
function marksFromStyle(style) {
  const marks = {};
  if (!style) return marks;
  const weight = style.fontWeight;
  if (weight === 'bold' || Number(weight) >= 600) marks.bold = true;
  if (style.fontStyle === 'italic') marks.italic = true;
  const decoration = `${style.textDecoration} ${style.textDecorationLine}`;
  if (decoration.includes('underline')) marks.underline = true;
  if (decoration.includes('line-through')) marks.strike = true;
  return marks;
}

/**
 * Reads an editable element as paragraphs of runs.
 *
 * @param {HTMLElement} root
 * @param {{ skip?: (el: Element) => boolean }} [options] `skip` hides an
 *   element and its contents, which is how the in-progress phrase is kept out
 *   of an export.
 * @returns {Run[][]} One entry per paragraph. An empty document is `[]`.
 */
export function readDocument(root, { skip } = {}) {
  const paragraphs = [[]];

  const pushRun = (text, marks) => {
    if (!text) return;
    const runs = paragraphs[paragraphs.length - 1];
    const last = runs[runs.length - 1];
    // Merge with the previous run when the marks match, so a document that has
    // been edited into a dozen nested spans still exports as flat sentences.
    if (last && sameMarks(last, marks)) last.text += text;
    else runs.push({ text, ...marks });
  };

  const breakParagraph = () => paragraphs.push([]);

  const walk = (node, marks) => {
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        pushRun(child.data.replace(/\n/g, ' '), marks);
        continue;
      }
      if (child.nodeType !== Node.ELEMENT_NODE) continue;
      if (skip?.(child)) continue;
      if (child.tagName === 'BR') {
        breakParagraph();
        continue;
      }
      const isBlock = BLOCK_TAGS.has(child.tagName);
      // A block that follows content starts a paragraph. The first one does
      // not, or every document would open with a blank line.
      if (isBlock && paragraphs[paragraphs.length - 1].length > 0) breakParagraph();
      const inherited = { ...marks, ...markFor(child) };
      walk(child, inherited);
    }
  };

  walk(root, {});
  // Trailing empties come from the <br> browsers park at the end of a line.
  while (paragraphs.length && paragraphs[paragraphs.length - 1].length === 0) paragraphs.pop();
  return paragraphs;
}

function markFor(element) {
  const named = MARK_TAGS[element.tagName];
  const marks = named ? { [named]: true } : {};
  return { ...marks, ...marksFromStyle(element.style) };
}

function sameMarks(a, b) {
  return (
    Boolean(a.bold) === Boolean(b.bold) &&
    Boolean(a.italic) === Boolean(b.italic) &&
    Boolean(a.underline) === Boolean(b.underline) &&
    Boolean(a.strike) === Boolean(b.strike)
  );
}

/** The document as plain text, one line per paragraph. */
export function toPlainText(paragraphs) {
  return paragraphs.map((runs) => runs.map((run) => run.text).join('')).join('\n');
}

/**
 * The document as Markdown.
 *
 * Marks are wrapped tightly: the whitespace at the edges of a run stays outside
 * the delimiters, because `** bold **` is not bold in any Markdown dialect.
 * Underline has no Markdown of its own, so it keeps the HTML tag, which every
 * renderer that allows inline HTML passes through.
 */
export function toMarkdown(paragraphs) {
  const wrap = (run) => {
    const [, before, body, after] = /^(\s*)([\s\S]*?)(\s*)$/.exec(run.text);
    if (!body) return run.text;
    let out = escapeMarkdown(body);
    if (run.bold) out = `**${out}**`;
    if (run.italic) out = `*${out}*`;
    if (run.strike) out = `~~${out}~~`;
    if (run.underline) out = `<u>${out}</u>`;
    return `${before}${out}${after}`;
  };
  // Blank line between paragraphs, which is the only line break Markdown
  // renders without trailing-space trickery.
  return `${paragraphs.map((runs) => runs.map(wrap).join('')).join('\n\n')}\n`;
}

/** Escapes the characters that would otherwise become Markdown syntax. */
function escapeMarkdown(text) {
  return text.replace(/([\\`*_[\]<>])/g, '\\$1');
}

// --- Word ------------------------------------------------------------------

const CONTENT_TYPES = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>`;

const ROOT_RELS = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>`;

const W_NAMESPACE = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main';

function escapeXml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function documentXml(paragraphs) {
  const runXml = (run) => {
    const marks =
      (run.bold ? '<w:b/>' : '') +
      (run.italic ? '<w:i/>' : '') +
      (run.underline ? '<w:u w:val="single"/>' : '') +
      (run.strike ? '<w:strike/>' : '');
    const properties = marks ? `<w:rPr>${marks}</w:rPr>` : '';
    // xml:space, or Word eats the spaces between runs.
    return `<w:r>${properties}<w:t xml:space="preserve">${escapeXml(run.text)}</w:t></w:r>`;
  };
  const body = paragraphs.map((runs) => `<w:p>${runs.map(runXml).join('')}</w:p>`).join('');
  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="${W_NAMESPACE}"><w:body>${body || '<w:p/>'}</w:body></w:document>`;
}

/** The document as a `.docx` blob, ready to hand to a download link. */
export function toDocx(paragraphs) {
  const bytes = zipStored([
    ['[Content_Types].xml', CONTENT_TYPES],
    ['_rels/.rels', ROOT_RELS],
    ['word/document.xml', documentXml(paragraphs)],
  ]);
  return new Blob([bytes], {
    type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  });
}

// --- Zip -------------------------------------------------------------------

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let bit = 0; bit < 8; bit++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/** MS-DOS date and time fields, which is what a zip entry records. */
function dosStamp(when) {
  const time = (when.getHours() << 11) | (when.getMinutes() << 5) | (when.getSeconds() >> 1);
  const date =
    ((when.getFullYear() - 1980) << 9) | ((when.getMonth() + 1) << 5) | when.getDate();
  return { time, date };
}

/**
 * A zip archive of UTF-8 text entries, stored uncompressed.
 *
 * @param {Array<[string, string]>} entries `[path, text]` pairs.
 * @returns {Uint8Array}
 */
export function zipStored(entries) {
  const encoder = new TextEncoder();
  const { time, date } = dosStamp(new Date());
  const chunks = [];
  const directory = [];
  let offset = 0;

  for (const [path, text] of entries) {
    const name = encoder.encode(path);
    const data = encoder.encode(text);
    const crc = crc32(data);

    const header = new DataView(new ArrayBuffer(30));
    header.setUint32(0, 0x04034b50, true);
    header.setUint16(4, 20, true); // version needed to extract
    header.setUint16(8, 0, true); // stored, not deflated
    header.setUint16(10, time, true);
    header.setUint16(12, date, true);
    header.setUint32(14, crc, true);
    header.setUint32(18, data.length, true);
    header.setUint32(22, data.length, true);
    header.setUint16(26, name.length, true);
    chunks.push(new Uint8Array(header.buffer), name, data);

    const entry = new DataView(new ArrayBuffer(46));
    entry.setUint32(0, 0x02014b50, true);
    entry.setUint16(4, 20, true); // version made by
    entry.setUint16(6, 20, true); // version needed
    entry.setUint16(10, 0, true); // stored
    entry.setUint16(12, time, true);
    entry.setUint16(14, date, true);
    entry.setUint32(16, crc, true);
    entry.setUint32(20, data.length, true);
    entry.setUint32(24, data.length, true);
    entry.setUint16(28, name.length, true);
    entry.setUint32(42, offset, true);
    directory.push(new Uint8Array(entry.buffer), name);

    offset += 30 + name.length + data.length;
  }

  const directorySize = directory.reduce((total, part) => total + part.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, entries.length, true);
  end.setUint16(10, entries.length, true);
  end.setUint32(12, directorySize, true);
  end.setUint32(16, offset, true);

  const parts = [...chunks, ...directory, new Uint8Array(end.buffer)];
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}
