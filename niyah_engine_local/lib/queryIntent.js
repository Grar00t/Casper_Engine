'use strict';

/* ══════════════════════════════════════════════════════════════════
   queryIntent.js — Deterministic question typology.

   A retrieval engine that treats "who founded X" and "how do I
   install X" identically will answer both with whatever paragraph
   scored highest. Knowing the expected *answer shape* lets the
   composer prefer sentences that actually contain that shape: a
   person name, a numeral, a date, an ordered procedure.

   Pure pattern algebra over the query string. Arabic · English · CJK.
   ══════════════════════════════════════════════════════════════════ */

const { detectLang } = require('./relevance');

const PATTERNS = [
  { type: 'definition', re: [/^\s*(what|whats|what's)\s+(is|are|was|were)\b/i, /\bdefine\b/i, /\bmeaning of\b/i,
      /^\s*(ما|ماهو|ما هو|ما هي|ماهي|وش|شو)\b/, /\bتعريف\b/, /\bمعنى\b/, /^(什么是|定义)/] },
  { type: 'procedure',  re: [/^\s*how\s+(do|can|to|does)\b/i, /\bsteps? to\b/i, /\btutorial\b/i, /\binstall\b/i,
      /^\s*(كيف|كيفية)\b/, /\bخطوات\b/, /\bطريقة\b/, /^(如何|怎么|怎样)/] },
  { type: 'causal',     re: [/^\s*why\b/i, /\breason(s)? (for|why)\b/i, /\bcause(s)? of\b/i,
      /^\s*(لماذا|ليش|لم)\b/, /\bسبب\b/, /^(为什么|为何)/] },
  { type: 'temporal',   re: [/^\s*when\b/i, /\bwhat (year|date|time)\b/i, /\bhow long\b/i,
      /^\s*(متى|متي)\b/, /\bتاريخ\b/, /^(什么时候|何时)/] },
  { type: 'person',     re: [/^\s*who\b/i, /\bfounder\b/i, /\binvented by\b/i, /\bceo of\b/i,
      /^\s*(من هو|من هي|من)\b/, /\bمؤسس\b/, /^(谁|是谁)/] },
  { type: 'location',   re: [/^\s*where\b/i, /\blocated\b/i, /\bcapital of\b/i,
      /^\s*(أين|اين|وين)\b/, /\bموقع\b/, /^(哪里|在哪)/] },
  { type: 'quantity',   re: [/^\s*how (many|much|old|far|big|tall|fast)\b/i, /\bpopulation of\b/i, /\bprice of\b/i,
      /^\s*(كم)\b/, /\bعدد\b/, /\bسعر\b/, /^(多少|几)/] },
  { type: 'comparison', re: [/\b(vs|versus)\b/i, /\b(difference|compare|better)\b.*\b(between|than|or)\b/i,
      /\b(الفرق|مقارنة|أفضل من)\b/, /(区别|比较|哪个更)/] },
  { type: 'enumeration',re: [/^\s*(list|name|what are the)\b/i, /\b(types|kinds|examples|benefits|advantages|features)\s+of\b/i,
      /^\s*(اذكر|اسرد)\b/, /\b(أنواع|انواع|أمثلة|فوائد|ميزات|مزايا)\b/, /(有哪些|列出)/] },
  { type: 'boolean',    re: [/^\s*(is|are|does|do|did|can|will|should|has|have)\b/i,
      /^\s*(هل|أهل)\b/, /^(是否|能不能)/] },
];

/* Answer-shape detectors — used by the composer to score candidate
   sentences against the *expected* form of the answer. */
const SHAPE = {
  definition: /\b(is|are|was|were|refers to|means|defined as|known as)\b|\b(هو|هي|يُعرف|تعني|يعني|عبارة عن|يُقصد)\b|是指|称为/i,
  procedure:  /\b(first|then|next|finally|step|click|run|open|select|install|type)\b|\b(أولا|أولاً|ثم|بعد ذلك|اضغط|افتح|شغّل|قم ب)\b|首先|然后|接着/i,
  causal:     /\b(because|due to|caused by|as a result|therefore|since|leads to)\b|\b(لأن|بسبب|نتيجة|مما يؤدي|ولذلك)\b|因为|由于|导致/i,
  temporal:   /\b(in|on|since|until)\s+\d{3,4}\b|\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b|\b\d{4}\b|\b(عام|سنة|في\s+\d{4})\b|\d+年/i,
  person:     /\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b|\b(founded by|created by|invented by|led by)\b|\b(مؤسس|أسسه|أنشأه|بقيادة)\b/,
  location:   /\b(in|at|near|located|based)\s+[A-Z][a-z]+/,
  quantity:   /\d[\d,.\u0660-\u0669]*\s*(%|percent|million|billion|thousand|kg|km|m|cm|mb|gb|usd|\$|€|بالمئة|مليون|مليار|ألف)?/i,
  comparison: /\b(whereas|while|unlike|compared to|in contrast|however|but)\b|\b(بينما|بخلاف|مقارنة ب|في المقابل|لكن)\b|而|相比/i,
  enumeration:/(^|\s)(\d+[.)]|[-•*])\s|\b(include|includes|such as|following)\b|\b(تشمل|منها|مثل|التالية)\b|包括|例如/i,
  boolean:    /\b(yes|no|can|cannot|is not|does not|it is|they are)\b|\b(نعم|لا|يمكن|لا يمكن|ليس)\b/i,
};

/* Interrogatives carry no topical information — dropping them keeps
   IDF focused on the subject of the question. */
const INTERROGATIVES = new Set([
  'what','whats','which','who','whom','whose','when','where','why','how','is','are','was','were',
  'do','does','did','can','could','will','would','should','list','name','define','tell','explain',
  'ما','ماهو','ماهي','من','متى','اين','أين','لماذا','كيف','كم','هل','اذكر','عرف','تعريف','وش','شو',
  '什么','谁','哪里','为什么','如何','怎么','多少',
]);

/**
 * @param {string} query
 * @returns {{type:string, lang:string, confidence:number, shape:RegExp|null,
 *            focusTerms:string[], isMultiPart:boolean}}
 */
function classifyIntent(query) {
  const q = (query || '').trim();
  const lang = detectLang(q);

  const hits = [];
  for (const { type, re } of PATTERNS) {
    for (const r of re) {
      if (r.test(q)) { hits.push(type); break; }
    }
  }

  // Specific types win over the very broad `boolean` prefix match.
  const ordered = hits.filter(h => h !== 'boolean');
  const type = ordered[0] || hits[0] || 'open';

  // Focus terms = the query minus its interrogative scaffolding.
  const focusTerms = (q.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [])
    .filter(w => !INTERROGATIVES.has(w) && w.length > 1);

  return {
    type,
    lang,
    // More independent pattern families agreeing ⇒ higher certainty.
    confidence: hits.length === 0 ? 0.3 : Math.min(0.5 + 0.25 * hits.length, 1),
    shape: SHAPE[type] || null,
    focusTerms,
    isMultiPart: /\band\b|\bو\s|\?.+\?/.test(q) && q.length > 40,
  };
}

module.exports = { classifyIntent, SHAPE, INTERROGATIVES };
