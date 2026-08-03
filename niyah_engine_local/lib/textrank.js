'use strict';

/* ══════════════════════════════════════════════════════════════════
   textrank.js — Graph centrality over a sentence similarity matrix.

   PageRank power iteration:
       PR(i) = (1-d)/N + d · Σ_{j∈In(i)} w(j,i)·PR(j) / Σ_k w(j,k)

   Sentence similarity (Mihalcea & Tarau):
       sim(A,B) = |A ∩ B| / (log|A| + log|B|)

   Why: BM25 answers "does this sentence match the query". TextRank
   answers "is this sentence central to what the sources collectively
   say". A sentence that many other sentences echo is the consensus
   of the corpus — that is what makes an answer feel authoritative
   instead of cherry-picked. No weights, just eigenvector iteration.
   ══════════════════════════════════════════════════════════════════ */

const DAMPING = 0.85;
const MAX_ITER = 60;
const EPSILON = 1e-6;

function overlapSim(tokensA, tokensB) {
  if (tokensA.length < 2 || tokensB.length < 2) return 0;
  const setB = new Set(tokensB);
  let common = 0;
  for (const t of new Set(tokensA)) if (setB.has(t)) common++;
  if (!common) return 0;
  const denom = Math.log(tokensA.length) + Math.log(tokensB.length);
  return denom > 0 ? common / denom : 0;
}

/**
 * @param {string[][]} tokenSets - one token array per sentence
 * @param {number[]} [prior] - optional bias (e.g. query relevance).
 *        Turns plain PageRank into a personalised/topic-sensitive one.
 * @returns {number[]} centrality per sentence, normalised to max=1
 */
function textRank(tokenSets, prior = null) {
  const N = tokenSets.length;
  if (N === 0) return [];
  if (N === 1) return [1];

  // Symmetric weight matrix.
  const W = Array.from({ length: N }, () => new Float64Array(N));
  const rowSum = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const s = overlapSim(tokenSets[i], tokenSets[j]);
      if (s > 0) { W[i][j] = s; W[j][i] = s; }
    }
  }
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let j = 0; j < N; j++) s += W[i][j];
    rowSum[i] = s;
  }

  // Personalisation vector.
  let p;
  if (prior && prior.length === N) {
    const tot = prior.reduce((a, b) => a + Math.max(0, b), 0);
    p = tot > 0 ? prior.map(v => Math.max(0, v) / tot) : new Array(N).fill(1 / N);
  } else {
    p = new Array(N).fill(1 / N);
  }

  let pr = new Float64Array(N).fill(1 / N);
  for (let iter = 0; iter < MAX_ITER; iter++) {
    const next = new Float64Array(N);
    let dangling = 0;
    for (let j = 0; j < N; j++) if (rowSum[j] === 0) dangling += pr[j];

    for (let i = 0; i < N; i++) {
      let acc = 0;
      for (let j = 0; j < N; j++) {
        if (rowSum[j] > 0 && W[j][i] > 0) acc += (W[j][i] / rowSum[j]) * pr[j];
      }
      // Dangling mass is redistributed by the prior, not uniformly,
      // so isolated sentences cannot dilute the topic focus.
      next[i] = (1 - DAMPING) * p[i] + DAMPING * (acc + dangling * p[i]);
    }

    let delta = 0;
    for (let i = 0; i < N; i++) delta += Math.abs(next[i] - pr[i]);
    pr = next;
    if (delta < EPSILON) break;
  }

  const max = Math.max(...pr) || 1;
  return Array.from(pr, v => v / max);
}

module.exports = { textRank, overlapSim };
