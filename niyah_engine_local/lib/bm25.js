'use strict';

/* ══════════════════════════════════════════════════════════════════
   bm25.js — Okapi BM25 ranking, implemented from the formula.

   score(D,Q) = Σ_{t∈Q} IDF(t) · ( f(t,D)·(k1+1) )
                              / ( f(t,D) + k1·(1 - b + b·|D|/avgdl) )

   IDF(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

   Why this replaces raw TF cosine: cosine over term counts treats a
   term appearing in every document as informative. BM25 discounts it
   through IDF, saturates repeated terms through k1, and corrects for
   document length through b. That is the difference between "matches
   many words" and "matches the *rare* words that identify the topic".
   ══════════════════════════════════════════════════════════════════ */

const K1 = 1.5;
const B  = 0.75;

class BM25Index {
  constructor(opts = {}) {
    this.k1 = opts.k1 ?? K1;
    this.b  = opts.b ?? B;
    this.docs = [];          // [{ id, tf:Map, len, meta }]
    this.df = new Map();     // term → document frequency
    this.totalLen = 0;
  }

  /**
   * @param {string[]} tokens - already tokenized + stemmed
   * @param {any} meta - arbitrary payload returned with scores
   */
  add(tokens, meta = null) {
    const tf = new Map();
    for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
    for (const t of tf.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
    const id = this.docs.length;
    this.docs.push({ id, tf, len: tokens.length, meta });
    this.totalLen += tokens.length;
    return id;
  }

  get size() { return this.docs.length; }
  get avgdl() { return this.docs.length ? this.totalLen / this.docs.length : 0; }

  idf(term) {
    const n = this.df.get(term) || 0;
    const N = this.docs.length;
    // +1 inside ln keeps IDF strictly positive, so a term present in
    // every document contributes ~0 instead of going negative.
    return Math.log(1 + (N - n + 0.5) / (n + 0.5));
  }

  scoreDoc(queryTokens, doc) {
    const avgdl = this.avgdl || 1;
    let score = 0;
    const seen = new Set();
    for (const t of queryTokens) {
      if (seen.has(t)) continue; // query term weight counted once
      seen.add(t);
      const f = doc.tf.get(t);
      if (!f) continue;
      const denom = f + this.k1 * (1 - this.b + this.b * (doc.len / avgdl));
      score += this.idf(t) * (f * (this.k1 + 1)) / denom;
    }
    return score;
  }

  /** @returns {{id,score,meta}[]} sorted desc */
  search(queryTokens) {
    return this.docs
      .map(d => ({ id: d.id, score: this.scoreDoc(queryTokens, d), meta: d.meta }))
      .sort((a, b) => b.score - a.score);
  }

  /** Fraction of distinct query terms present in a document. */
  coverage(queryTokens, docId) {
    const doc = this.docs[docId];
    if (!doc) return 0;
    const uniq = [...new Set(queryTokens)];
    if (!uniq.length) return 0;
    let hit = 0;
    for (const t of uniq) if (doc.tf.has(t)) hit++;
    return hit / uniq.length;
  }

  /**
   * Normalise raw BM25 to 0..1 for blending with other signals.
   * Divides by the theoretical max: every query term matching with
   * saturated frequency. Keeps the ordering, bounds the range.
   */
  maxPossible(queryTokens) {
    const uniq = [...new Set(queryTokens)];
    return uniq.reduce((s, t) => s + this.idf(t) * (this.k1 + 1) / 1, 0) || 1;
  }
}

module.exports = { BM25Index };
