'use strict';

/* ══════════════════════════════════════════════════════════════════
   relevance.js — TF-IDF + cosine similarity
   Supports: Arabic · English · Chinese (basic)
   No neural networks. No hallucination. Pure math.
   ══════════════════════════════════════════════════════════════════ */

const { stem } = require('./stem');

const STOPWORDS = new Set([
  // English
  'the','a','an','is','are','was','were','of','to','in','on','for','and','or',
  'but','with','this','that','it','as','be','by','at','from','has','have','had',
  'not','no','so','if','its','than','then','into','about','which','when','there',
  'their','they','we','our','you','your','he','she','his','her','been','will',
  'do','does','did','can','could','would','should','may','might','shall',
  // Arabic
  'من','الى','إلى','على','في','عن','مع','هذا','هذه','ذلك','التي','الذي',
  'كان','هو','هي','ما','لا','ان','إن','او','أو','ثم','قد','كل','أن','وأن',
  'بعض','غير','حتى','عند','بين','منذ','حول','خلال','ضمن','هنا','هناك',
  'أيضا','أيضاً','جدا','جداً','جميع','مثل','فقط','وقد','وكان','وهو','وهي',
  'وان','والتي','والذي','ولا','وما','وأن','إذ','إذا','لذا','لذلك','لأن',
  // Chinese common function words
  '的','了','在','是','我','有','和','就','不','都','一','上','也','这','到',
  '他','她','它','们','个','说','要','去','会','着','没','与','从','以','被',
]);

/* ── Language detection (char frequency heuristic) ── */
function detectLang(text) {
  if (!text) return 'en';
  const ar = (text.match(/[\u0600-\u06FF]/g) || []).length;
  const zh = (text.match(/[\u4E00-\u9FFF]/g) || []).length;
  const tot = text.replace(/\s/g,'').length || 1;
  if (zh / tot > 0.15) return 'zh';
  if (ar / tot > 0.15) return 'ar';
  return 'en';
}

/* ── Tokenize — Unicode-aware, strips Arabic diacritics ── */
function tokenize(text) {
  if (!text) return [];
  return text
    .toLowerCase()
    .normalize('NFKC')
    .replace(/[\u064B-\u065F\u0670\u0640]/g, '') // strip tashkeel + tatweel
    .match(/[\p{L}\p{N}]+/gu) || [];
}

/* ── Tokenize + morphological reduction + stopword removal ──
   This is the token stream every statistical stage consumes. Stemming
   here (rather than per-callsite) guarantees the BM25 index, the
   TextRank graph and the MMR redundancy check all share one vocabulary
   — otherwise their scores are not comparable. */
function stemTokens(text) {
  const out = [];
  for (const t of tokenize(text)) {
    if (t.length < 2 || STOPWORDS.has(t)) continue;
    const s = stem(t);
    if (s && s.length >= 2 && !STOPWORDS.has(s)) out.push(s);
  }
  return out;
}

function termFrequencies(tokens) {
  const tf = new Map();
  for (const t of tokens) {
    if (STOPWORDS.has(t) || t.length < 2) continue;
    tf.set(t, (tf.get(t) || 0) + 1);
  }
  return tf;
}

function cosineSim(tfA, tfB) {
  let dot = 0, magA = 0, magB = 0;
  for (const [term, fa] of tfA) {
    magA += fa * fa;
    if (tfB.has(term)) dot += fa * tfB.get(term);
  }
  for (const fb of tfB.values()) magB += fb * fb;
  if (!magA || !magB) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function rankByRelevance(query, items, textExtractor) {
  const qTf = termFrequencies(tokenize(query));
  return items
    .map(item => ({
      ...item,
      relevanceScore: cosineSim(qTf, termFrequencies(tokenize(textExtractor(item)))),
    }))
    .sort((a, b) => b.relevanceScore - a.relevanceScore);
}

/* ── Junk sentence patterns — navigation, footers, social, ads ── */
const JUNK_RE = [
  /^(الرئيس|الرئيسية|تسجيل|اتصل بنا|سياسة|حقوق|للأعلى|للأسفل|شارك|إغلاق)/,
  /جميع الحقوق محفوظة/,
  /powered by/i,
  /^(Home|Login|Sign in|Privacy|Contact|All rights reserved|Back to top|Cookie)/i,
  /^(Facebook|Twitter|Instagram|YouTube|تويتر|فيسبوك|يوتيوب)[\s|$]/i,
  /^\s*[\d\W]{0,6}\s*$/,
  /شاهد أيض|مواضيع ذات صل/,
  /اشترك في النشرة|Subscribe to/i,
  /أضف تعليق|Add comment/i,
];

function isJunk(s) {
  if (!s) return true;
  const trimmed = s.trim();
  if (trimmed.length < 20 || trimmed.length > 500) return true;
  return JUNK_RE.some(p => p.test(trimmed));
}

/* ── Near-duplicate removal (cosine threshold) ── */
function dedup(sentences, threshold = 0.82) {
  const kept = [];
  const keptTfs = [];
  for (const s of sentences) {
    const tf = termFrequencies(tokenize(s));
    const isDup = keptTfs.some(k => cosineSim(tf, k) >= threshold);
    if (!isDup) { kept.push(s); keptTfs.push(tf); }
  }
  return kept;
}

function splitSentences(text) {
  if (!text) return [];
  const raw = text
    .split(/(?<=[.!?؟。])\s+|\n{2,}|\n(?=[A-Z\u0600-\u06FF])/)
    .map(s => s.replace(/\s+/g, ' ').trim())
    .filter(s => !isJunk(s));
  return dedup(raw);
}

function extractTopSentences(query, longText, n = 3) {
  const sentences = splitSentences(longText);
  if (!sentences.length) return [];
  return rankByRelevance(query, sentences.map(s => ({ text: s })), i => i.text)
    .slice(0, n)
    .filter(r => r.relevanceScore > 0.01);
}

module.exports = {
  tokenize, stemTokens, termFrequencies, cosineSim, rankByRelevance,
  splitSentences, extractTopSentences, detectLang, dedup, isJunk,
};
