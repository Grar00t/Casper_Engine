'use strict';

/* ══════════════════════════════════════════════════════════════════
   stem.js — Deterministic morphological normalisation.

   Written from scratch. No trained model, no lookup corpus, no
   downloaded weights. Pure rule algebra over Unicode codepoints.

   Purpose: make "computing", "computed", "computes" collapse to the
   same term so BM25 IDF statistics are not fragmented, and do the
   same for Arabic clitics (ال، وال، ها، ون...) which otherwise make
   every inflected form look like an unrelated token.
   ══════════════════════════════════════════════════════════════════ */

/* ── Arabic orthographic normalisation ──────────────────────────── */
const AR_DIACRITICS = /[\u064B-\u065F\u0670\u0640]/g; // tashkeel + tatweel

function normalizeArabic(w) {
  return w
    .replace(AR_DIACRITICS, '')
    .replace(/[\u0622\u0623\u0625]/g, '\u0627') // آ أ إ → ا
    .replace(/\u0649/g, '\u064A')               // ى → ي
    .replace(/\u0629/g, '\u0647');              // ة → ه
}

/* Light-stemming: strip definite-article/conjunction prefixes then
   plural/possessive suffixes. Order matters (longest first). */
const AR_PREFIXES = ['\u0648\u0627\u0644', '\u0628\u0627\u0644', '\u0643\u0627\u0644',
                     '\u0641\u0627\u0644', '\u0644\u0644', '\u0627\u0644', '\u0648'];
const AR_SUFFIXES = ['\u0627\u062A\u0647', '\u0627\u062A', '\u0648\u0646', '\u064A\u0646',
                     '\u0647\u0627', '\u0647\u0645', '\u0643\u0645', '\u0646\u0627',
                     '\u064A\u0647', '\u0648\u0627', '\u0627\u0646', '\u0647'];

function stemArabic(word) {
  let w = normalizeArabic(word);
  if (w.length <= 3) return w;
  for (const p of AR_PREFIXES) {
    if (w.startsWith(p) && w.length - p.length >= 3) { w = w.slice(p.length); break; }
  }
  for (const s of AR_SUFFIXES) {
    if (w.endsWith(s) && w.length - s.length >= 3) { w = w.slice(0, -s.length); break; }
  }
  return w;
}

/* ── English suffix reduction (Porter step-1/2 subset) ───────────── */
function stemEnglish(word) {
  let w = word;
  if (w.length <= 3) return w;

  // Nominalisations first — they carry the root.
  const heavy = [
    ['ization', 'ize'], ['ational', 'ate'], ['fulness', 'ful'],
    ['iveness', 'ive'], ['ousness', 'ous'], ['biliti', 'ble'],
    ['ation', 'ate'], ['alism', 'al'], ['aliti', 'al'],
    ['iviti', 'ive'], ['ement', ''], ['ment', ''], ['ness', ''],
    ['tional', 'tion'], ['ence', ''], ['ance', ''], ['able', ''], ['ible', ''],
  ];
  for (const [suf, rep] of heavy) {
    if (w.endsWith(suf) && w.length - suf.length >= 3) return w.slice(0, -suf.length) + rep;
  }

  // Plurals.
  if (w.endsWith('ies') && w.length > 4) return w.slice(0, -3) + 'i';
  if (w.endsWith('sses')) return w.slice(0, -2);
  if (w.endsWith('es') && w.length > 4 && !/[aeiou]es$/.test(w)) w = w.slice(0, -2);
  else if (w.endsWith('s') && !w.endsWith('ss') && !w.endsWith('us') && w.length > 3) w = w.slice(0, -1);

  // Verb inflections.
  if (w.endsWith('ing') && w.length > 5) {
    w = w.slice(0, -3);
    if (/([^aeiou])\1$/.test(w)) w = w.slice(0, -1); // running → run
    else if (!/[aeiou]/.test(w.slice(-2))) w += 'e';  // 'writ' → 'write'
  } else if (w.endsWith('ed') && w.length > 4) {
    w = w.slice(0, -2);
    if (/([^aeiou])\1$/.test(w)) w = w.slice(0, -1);
  }
  if (w.endsWith('ly') && w.length > 4) w = w.slice(0, -2);
  if (w.endsWith('y') && w.length > 3) w = w.slice(0, -1) + 'i';

  return w;
}

/* ── Dispatch by script ─────────────────────────────────────────── */
function stem(word) {
  if (!word) return '';
  if (/[\u0600-\u06FF]/.test(word)) return stemArabic(word);
  // CJK has no inflection to strip — identity keeps the token intact.
  if (/[\u4E00-\u9FFF]/.test(word)) return word;
  return stemEnglish(word);
}

/* Character n-grams — a script-agnostic similarity backstop used when
   stemming still misses a morphological relation. */
function charNgrams(word, n = 3) {
  const w = `\u0002${word}\u0003`;
  if (w.length <= n) return [w];
  const out = [];
  for (let i = 0; i <= w.length - n; i++) out.push(w.slice(i, i + n));
  return out;
}

module.exports = { stem, stemArabic, stemEnglish, normalizeArabic, charNgrams };
