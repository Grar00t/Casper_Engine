'use strict';

/* ══════════════════════════════════════════════════════════════════
   reasoner.js — Extractive synthesis, ensemble-scored.

   PIPELINE
     1. Segment every fetched page into candidate spans.
     2. Index the spans with Okapi BM25 → lexical evidence, IDF-weighted.
     3. Run topic-sensitive TextRank over the span graph → consensus
        centrality across the whole corpus.
     4. Score the expected answer *shape* for the detected intent.
     5. Blend those into one relevance figure, then select with MMR so
        each chosen span adds information the previous ones lacked.
     6. Compose deterministically, merging corroborating spans.
     7. Calibrate confidence from measured quantities only.

   There is no language model here, and no downloaded weights of any
   kind. Every number below is computed from the retrieved text at
   request time. Nothing is generated: the answer is a re-ordering of
   spans that exist verbatim in the cited pages.
   ══════════════════════════════════════════════════════════════════ */

const { splitSentences, stemTokens, detectLang } = require('./relevance');
const { BM25Index } = require('./bm25');
const { textRank } = require('./textrank');
const { mmrSelect } = require('./mmr');
const { classifyIntent } = require('./queryIntent');
const { compose } = require('./composer');

/* Blend weights. Chosen so no single signal can carry a span alone:
   lexical match must be corroborated by either corpus centrality or
   the expected answer shape. */
const W_BM25       = 0.50;
const W_CENTRALITY = 0.28;
const W_SHAPE      = 0.14;
const W_POSITION   = 0.08;

const MIN_RELEVANCE = 0.06;
const MAX_SPANS_PER_SOURCE = 3;

/** Earlier spans in a page are likelier to be the thesis, not detail. */
function positionPrior(position, total) {
  if (total <= 1) return 1;
  return 1 - Math.min(position / Math.max(total, 12), 1) * 0.6;
}

/**
 * Confidence from measurable evidence — never a constant.
 * Each component is in 0..1 and is reported in the breakdown so the
 * figure can be audited rather than trusted.
 */
function calibrateConfidence({ focusCoverage, retrieval, corroboration, centrality, shapeAgreement, sourceCount }) {
  const diversity = Math.min(sourceCount / 3, 1); // saturates at 3 independent pages
  const raw =
      0.28 * focusCoverage      // did we actually address the query terms?
    + 0.22 * retrieval          // how strong was the lexical evidence?
    + 0.20 * corroboration      // do independent pages agree?
    + 0.14 * centrality         // is this the corpus consensus?
    + 0.10 * shapeAgreement     // does it have the shape the question asks for?
    + 0.06 * diversity;

  // A single source cannot exceed 0.6 no matter how well it scores:
  // one page agreeing with itself is not evidence.
  const ceiling = sourceCount <= 1 ? 0.6 : 1;
  return Math.round(Math.min(raw, ceiling) * 100) / 100;
}

function noResultMessage(lang) {
  if (lang === 'ar') return 'لم يُعثر على نص ذي صلة كافية بالسؤال في المصادر المسترجعة. أعد صياغة السؤال.';
  if (lang === 'zh') return '在检索到的来源中未找到足够相关的文本。请尝试重新表述您的问题。';
  return 'No sufficiently relevant text found in the fetched sources. Try rephrasing.';
}

/**
 * @param {string} query
 * @param {Array<{title,url,text}>} sourcesWithText
 * @returns {{answer,citations,confidence,intent,structure,confidenceBreakdown,spansConsidered}}
 */
function synthesize(query, sourcesWithText, opts = {}) {
  const maxSpans = opts.maxSentencesTotal || 5;
  const lambda   = opts.mmrLambda ?? 0.72;

  const intent = classifyIntent(query);
  const queryTokens = stemTokens(query);
  // Focus terms drive scoring; interrogatives ("what", "كيف") are
  // stripped so they cannot inflate a match.
  const focusTokens = stemTokens(intent.focusTerms.join(' '));
  const scoringTokens = focusTokens.length ? focusTokens : queryTokens;

  /* ── 1. Segment ── */
  const spans = [];
  sourcesWithText.forEach((src, sourceIndex) => {
    const sentences = splitSentences(src.text);
    sentences.forEach((text, position) => {
      const tokens = stemTokens(text);
      if (tokens.length < 3) return;
      spans.push({
        text, tokens, sourceIndex, position,
        totalInSource: sentences.length,
        sourceTitle: src.title, sourceUrl: src.url,
      });
    });
  });

  if (!spans.length) {
    return {
      answer: noResultMessage(intent.lang), citations: [], confidence: 0,
      intent: intent.type, structure: null, confidenceBreakdown: null, spansConsidered: 0,
    };
  }

  /* ── 2. BM25 lexical evidence ── */
  const index = new BM25Index();
  for (const s of spans) index.add(s.tokens, s);
  const bm25Raw = spans.map((s, i) => index.scoreDoc(scoringTokens, index.docs[i]));
  const bm25Max = Math.max(...bm25Raw) || 1;
  const bm25 = bm25Raw.map(v => v / bm25Max);

  /* ── 3. Topic-sensitive centrality ── */
  const centrality = textRank(spans.map(s => s.tokens), bm25);

  /* ── 4-5. Blend + MMR selection ── */
  const shapeRe = intent.shape;
  const candidates = spans.map((s, i) => {
    const shapeMatch = shapeRe ? shapeRe.test(s.text) : false;
    const relevance =
        W_BM25 * bm25[i]
      + W_CENTRALITY * centrality[i]
      + W_SHAPE * (shapeMatch ? 1 : 0)
      + W_POSITION * positionPrior(s.position, s.totalInSource);
    return { ...s, tokens: s.tokens, bm25: bm25[i], centrality: centrality[i], shapeMatch, relevance };
  });

  const viable = candidates.filter(c => c.relevance >= MIN_RELEVANCE && c.bm25 > 0);
  if (!viable.length) {
    return {
      answer: noResultMessage(intent.lang), citations: [], confidence: 0,
      intent: intent.type, structure: null, confidenceBreakdown: null, spansConsidered: spans.length,
    };
  }

  // Cap per-source contribution so one verbose page cannot own the
  // whole answer and fake corroboration with itself.
  const quotaPenalty = (cand, chosen) => {
    const used = chosen.filter(c => c.sourceIndex === cand.sourceIndex).length;
    return used >= MAX_SPANS_PER_SOURCE ? 1 : 0;
  };

  const picks = mmrSelect(viable, maxSpans, lambda, quotaPenalty);
  const selected = picks.map(p => ({ ...viable[p.index], score: viable[p.index].relevance, novelty: p.novelty }));

  /* ── 6. Compose ── */
  const { answer, citations, structure } = compose({ query, intent, selected });

  /* ── 7. Calibrate ── */
  const answerTokens = new Set(stemTokens(answer));
  const focusCoverage = scoringTokens.length
    ? [...new Set(scoringTokens)].filter(t => answerTokens.has(t)).length / new Set(scoringTokens).size
    : 0;
  const retrieval     = selected.reduce((s, c) => s + c.bm25, 0) / selected.length;
  const centralityAvg = selected.reduce((s, c) => s + c.centrality, 0) / selected.length;
  const shapeAgreement = selected.filter(c => c.shapeMatch).length / selected.length;
  const corroboration = structure.spans > 0 ? structure.corroboratedSpans / structure.spans : 0;

  const breakdown = {
    focusCoverage: Math.round(focusCoverage * 100) / 100,
    retrieval: Math.round(retrieval * 100) / 100,
    corroboration: Math.round(corroboration * 100) / 100,
    centrality: Math.round(centralityAvg * 100) / 100,
    shapeAgreement: Math.round(shapeAgreement * 100) / 100,
    independentSources: structure.independentSources,
  };

  const confidence = calibrateConfidence({
    focusCoverage, retrieval, corroboration,
    centrality: centralityAvg, shapeAgreement,
    sourceCount: structure.independentSources,
  });

  return {
    answer, citations, confidence,
    intent: intent.type,
    intentConfidence: Math.round(intent.confidence * 100) / 100,
    structure,
    confidenceBreakdown: breakdown,
    spansConsidered: spans.length,
  };
}

module.exports = { synthesize, calibrateConfidence, positionPrior };
