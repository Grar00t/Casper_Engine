'use strict';

/* ══════════════════════════════════════════════════════════════════
   composer.js — Deterministic answer construction.

   HONESTY CONTRACT — read before editing:
   Every proposition in the output is a verbatim span from a fetched
   source. This module may only:
     · choose which spans appear and in what order
     · merge spans that corroborate each other, unioning citations
     · insert discourse markers ("Additionally", "However", "ثم")
     · number ordered steps
   It may NOT paraphrase, interpolate, complete, or infer new facts.
   That constraint is what makes the citation [n] next to a sentence
   a checkable claim rather than decoration.
   ══════════════════════════════════════════════════════════════════ */

const { overlapSim } = require('./textrank');

/* Discourse markers per language. Additive / contrastive / sequential. */
const MARKERS = {
  en: { add: ['Additionally', 'Furthermore', 'Also'], contrast: ['However', 'By contrast'],
        seq: ['First', 'Then', 'Next', 'After that', 'Finally'], corroborate: 'Multiple sources agree' },
  ar: { add: ['وأيضاً', 'كذلك', 'إضافةً إلى ذلك'], contrast: ['لكن', 'في المقابل'],
        seq: ['أولاً', 'ثم', 'بعد ذلك', 'وبعدها', 'وأخيراً'], corroborate: 'تتفق عدة مصادر' },
  zh: { add: ['此外', '同时', '另外'], contrast: ['然而', '相比之下'],
        seq: ['首先', '然后', '接着', '之后', '最后'], corroborate: '多个来源一致' },
};

const CONTRAST_RE = /^\s*(however|but|although|though|whereas|unlike|in contrast|on the other hand|لكن|لكنّ|بينما|غير أنّ|إلا أن|然而|但是)/i;
const LEADING_MARKER_RE = /^\s*(additionally|furthermore|also|moreover|however|but|then|next|finally|first(ly)?|second(ly)?|third(ly)?|in addition|وأيضا|وأيضاً|كذلك|أولا|أولاً|ثم|بعد ذلك|وأخيرا|وأخيراً|此外|首先|然后|最后)[\s,،:—-]+/i;

const CORROBORATION_THRESHOLD = 0.34;

/* ── Sentence hygiene ───────────────────────────────────────────── */
function cleanSpan(text) {
  let s = text.replace(/\s+/g, ' ').trim();
  // A span lifted mid-paragraph often starts with a marker that now
  // points at nothing. Strip it so our own marker reads correctly.
  s = s.replace(LEADING_MARKER_RE, '');
  // Drop an unmatched opening bracket left by the extraction window.
  if ((s.match(/\(/g) || []).length > (s.match(/\)/g) || []).length) s = s.replace(/\([^)]*$/, '').trim();
  s = s.replace(/[,،;:]\s*$/, '');
  if (s && !/[.!?؟。]$/.test(s)) s += '.';
  return s.charAt(0).toUpperCase() === s.charAt(0) ? s : s.charAt(0).toUpperCase() + s.slice(1);
}

function citeList(nums) {
  return [...new Set(nums)].sort((a, b) => a - b).map(n => `[${n}]`).join('');
}

/* ── Merge spans that say the same thing from different sources ──
   Two independent pages asserting the same fact is the strongest
   signal an extractive engine has. Collapsing them into one line
   with two citations reports that agreement instead of padding the
   answer with a near-duplicate. */
function mergeCorroborating(items) {
  const groups = [];
  for (const it of items) {
    let placed = false;
    for (const g of groups) {
      if (g.sourceIndices.has(it.sourceIndex)) continue; // same page ≠ corroboration
      if (overlapSim(it.tokens, g.lead.tokens) >= CORROBORATION_THRESHOLD) {
        g.members.push(it);
        g.sourceIndices.add(it.sourceIndex);
        // Keep the longer span — it carries more of the shared claim.
        if (it.text.length > g.lead.text.length) g.lead = it;
        placed = true;
        break;
      }
    }
    if (!placed) groups.push({ lead: it, members: [it], sourceIndices: new Set([it.sourceIndex]) });
  }
  return groups;
}

/* ── Ordering strategy per intent ───────────────────────────────── */
function orderGroups(groups, intent) {
  if (intent.type === 'procedure' || intent.type === 'enumeration') {
    // Preserve the order the source presented — a procedure's meaning
    // is its sequence, so relevance must not reshuffle it.
    return groups.slice().sort((a, b) => {
      if (a.lead.sourceIndex !== b.lead.sourceIndex) return a.lead.sourceIndex - b.lead.sourceIndex;
      return a.lead.position - b.lead.position;
    });
  }
  // Otherwise: strongest shape-matching, best-supported span leads.
  return groups.slice().sort((a, b) => {
    const aScore = a.lead.score + (a.members.length - 1) * 0.15 + (a.lead.shapeMatch ? 0.2 : 0);
    const bScore = b.lead.score + (b.members.length - 1) * 0.15 + (b.lead.shapeMatch ? 0.2 : 0);
    return bScore - aScore;
  });
}

/**
 * @param {object} args
 * @param {string} args.query
 * @param {object} args.intent - from classifyIntent
 * @param {Array} args.selected - [{text, tokens, score, sourceIndex, sourceTitle,
 *                                  sourceUrl, position, shapeMatch, centrality, novelty}]
 * @returns {{answer:string, citations:Array, structure:object}}
 */
function compose({ query, intent, selected }) {
  const lang = intent.lang in MARKERS ? intent.lang : 'en';
  const M = MARKERS[lang];

  const groups = orderGroups(mergeCorroborating(selected), intent);

  /* Citation numbering follows presentation order, so [1] is always
     the first source the reader actually encounters. */
  const citationMap = new Map();
  const citations = [];
  const numFor = (item) => {
    if (!citationMap.has(item.sourceIndex)) {
      const n = citations.length + 1;
      citationMap.set(item.sourceIndex, n);
      citations.push({ n, title: item.sourceTitle, url: item.sourceUrl });
    }
    return citationMap.get(item.sourceIndex);
  };

  const ordered = intent.type === 'procedure' || intent.type === 'enumeration';
  const lines = [];

  groups.forEach((g, i) => {
    const span = cleanSpan(g.lead.text);
    const nums = g.members.map(numFor);
    const cites = citeList(nums);

    if (ordered) {
      lines.push(`${i + 1}. ${span} ${cites}`);
      return;
    }

    let prefix = '';
    if (i > 0) {
      if (g.members.length > 1) prefix = `${M.corroborate}: `;
      else if (CONTRAST_RE.test(g.lead.text)) prefix = `${M.contrast[i % M.contrast.length]}, `;
      else prefix = `${M.add[(i - 1) % M.add.length]}, `;
      // Lowercase the joint so the marker reads as one sentence,
      // except where the span opens with a proper noun or acronym.
      const first = span.split(' ')[0];
      const isProper = /^[A-Z]{2,}$/.test(first) || /^[A-Z][a-z]+$/.test(first) === false;
      if (!isProper && lang === 'en') {
        return lines.push(`${prefix}${span.charAt(0).toLowerCase()}${span.slice(1)} ${cites}`);
      }
    }
    lines.push(`${prefix}${span} ${cites}`);
  });

  return {
    answer: lines.join(ordered ? '\n' : '\n\n'),
    citations,
    structure: {
      intent: intent.type,
      lang,
      ordered,
      spans: groups.length,
      corroboratedSpans: groups.filter(g => g.members.length > 1).length,
      independentSources: citations.length,
    },
  };
}

module.exports = { compose, cleanSpan, mergeCorroborating, CORROBORATION_THRESHOLD };
