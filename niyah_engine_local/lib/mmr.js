'use strict';

/* ══════════════════════════════════════════════════════════════════
   mmr.js — Maximal Marginal Relevance (Carbonell & Goldstein).

       MMR = argmax_{i ∈ R\S} [ λ·rel(i) − (1−λ)·max_{j∈S} sim(i,j) ]

   Why: the previous selector took the top-N scoring sentences. The
   top-N of a web corpus are usually five restatements of the same
   fact, because every page copies the same paragraph. MMR penalises
   a candidate by its similarity to what is already selected, so each
   added sentence must contribute *new* information to earn its slot.
   ══════════════════════════════════════════════════════════════════ */

const { overlapSim } = require('./textrank');

/**
 * @param {Array<{tokens:string[], relevance:number}>} candidates
 * @param {number} k - how many to select
 * @param {number} lambda - 1 = pure relevance, 0 = pure diversity
 * @param {function} [penalty] - (candidate, selected[]) => extra 0..1 cost
 * @returns {Array<{index:number, mmr:number, novelty:number}>}
 */
function mmrSelect(candidates, k = 5, lambda = 0.72, penalty = null) {
  const chosen = [];
  const remaining = candidates.map((c, i) => i);
  if (!candidates.length) return [];

  const maxRel = Math.max(...candidates.map(c => c.relevance)) || 1;

  while (chosen.length < k && remaining.length) {
    let best = -1, bestScore = -Infinity, bestRedundancy = 0;

    for (const idx of remaining) {
      const cand = candidates[idx];
      const rel = cand.relevance / maxRel;

      let redundancy = 0;
      for (const s of chosen) {
        const sim = overlapSim(cand.tokens, candidates[s.index].tokens);
        if (sim > redundancy) redundancy = sim;
      }
      // overlapSim is unbounded above 1 for long sentences; clamp so
      // λ stays an interpretable mix ratio.
      redundancy = Math.min(redundancy, 1);

      let score = lambda * rel - (1 - lambda) * redundancy;
      if (penalty) score -= penalty(cand, chosen.map(s => candidates[s.index]));

      if (score > bestScore) {
        bestScore = score; best = idx; bestRedundancy = redundancy;
      }
    }

    if (best < 0) break;
    chosen.push({ index: best, mmr: bestScore, novelty: 1 - bestRedundancy });
    remaining.splice(remaining.indexOf(best), 1);
  }

  return chosen;
}

module.exports = { mmrSelect };
