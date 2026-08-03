'use strict';

const { SearchProvider } = require('./searchProvider');
const { MemoryStore } = require('./memory');
const { stemTokens } = require('./relevance');
const { BM25Index } = require('./bm25');
const { classifyIntent } = require('./queryIntent');
const { synthesize } = require('./reasoner');

/* Every terminal state is named, so callers never have to guess
   whether an empty answer means "nothing matched" or "the network
   failed". Silent degradation is what made the old engine feel fake. */
const STATUS = {
  OK: 'ok',
  MEMORY_HIT: 'ok_from_memory',
  EMPTY_QUERY: 'empty_query',
  SEARCH_UNAVAILABLE: 'search_unavailable',
  NO_RESULTS: 'no_search_results',
  FETCH_FAILED: 'page_fetch_failed',
  NO_RELEVANT_TEXT: 'no_relevant_text',
};

class NiyahEngine {
  constructor(opts = {}) {
    this.search = new SearchProvider({
      searxngBaseUrl: opts.searxngBaseUrl,
      braveApiKey: opts.braveApiKey,
    });
    this.memory = new MemoryStore(opts.memoryDbPath);
    this.searchResultCount = opts.searchResultCount || 8;
    this.pagesToFetch = opts.pagesToFetch || 4;
  }

  get backendName() {
    if (this.search.searxngBaseUrl) return 'searxng';
    if (this.search.braveApiKey) return 'brave';
    return 'duckduckgo_html';
  }

  async ask(query, opts = {}) {
    const start = Date.now();
    const t = () => Date.now() - start;
    query = (query || '').trim();

    const base = () => ({ citations: [], confidence: 0, fromMemory: false, tookMs: t() });

    if (!query) {
      return { ...base(), status: STATUS.EMPTY_QUERY, answer: 'Empty query.', trace: [] };
    }

    const trace = [];

    /* ── 1. Analyse the question ── */
    const intent = classifyIntent(query);
    const tokens = stemTokens(query);
    trace.push({
      step: 'ANALYSE',
      ms: t(),
      detail: `intent=${intent.type} · lang=${intent.lang} · ${tokens.length} stemmed terms`,
      data: intent.focusTerms.slice(0, 8).join(' · ') || tokens.slice(0, 8).join(' · '),
    });

    /* ── 2. Associative memory ── */
    if (!opts.forceFresh) {
      const cached = this.memory.recall(query);
      if (cached) {
        trace.push({
          step: 'MEMORY HIT', ms: t(),
          detail: `cosine similarity = ${cached.similarity.toFixed(3)} ≥ 0.55`,
          data: `matched: "${cached.query.substring(0, 50)}"`,
        });
        return {
          ...base(), status: STATUS.MEMORY_HIT,
          answer: cached.answer, citations: cached.sources,
          confidence: cached.confidence, fromMemory: true,
          memorySimilarity: cached.similarity, intent: intent.type, trace,
        };
      }
      trace.push({
        step: 'MEMORY MISS', ms: t(),
        detail: 'no cached entry with similarity ≥ 0.55',
        data: `${this.memory._allRows().length} entries checked`,
      });
    }

    /* ── 3. Retrieve ── */
    let rawResults;
    try {
      rawResults = await this.search.search(query, this.searchResultCount);
    } catch (err) {
      trace.push({ step: 'SEARCH FAILED', ms: t(), detail: err.message, data: this.backendName });
      return {
        ...base(), status: STATUS.SEARCH_UNAVAILABLE,
        answer: `Search backend (${this.backendName}) is unreachable: ${err.message}`,
        error: err.message, intent: intent.type, trace,
      };
    }

    trace.push({
      step: 'SEARCH', ms: t(),
      detail: `backend=${this.backendName} · q="${query.substring(0, 40)}"`,
      data: `${rawResults.length} results returned`,
    });

    if (rawResults.length === 0) {
      return {
        ...base(), status: STATUS.NO_RESULTS,
        answer: `The ${this.backendName} backend returned 0 results for this query.`,
        intent: intent.type, trace,
      };
    }

    /* ── 4. Rank candidates with BM25 over title+snippet ──
       IDF is computed across the result set itself, so a word that
       appears in every snippet stops being a discriminator. */
    const candIndex = new BM25Index();
    rawResults.forEach(r => candIndex.add(stemTokens(`${r.title} ${r.snippet}`), r));
    const scoringTokens = intent.focusTerms.length ? stemTokens(intent.focusTerms.join(' ')) : tokens;
    const ranked = candIndex.search(scoringTokens);
    const topCandidates = ranked.slice(0, this.pagesToFetch).map(r => ({ ...r.meta, bm25: r.score }));

    trace.push({
      step: 'BM25 RANK', ms: t(),
      detail: `Okapi BM25 (k1=1.5, b=0.75) over ${rawResults.length} snippets`,
      data: `top score = ${(ranked[0]?.score || 0).toFixed(3)} · fetching top ${topCandidates.length}`,
    });

    /* ── 5. Fetch full documents ── */
    const sourcesWithText = [];
    const fetchFailures = [];
    for (const candidate of topCandidates) {
      try {
        const text = await this.search.fetchPageText(candidate.url);
        if (text && text.length > 200) {
          sourcesWithText.push({ title: candidate.title, url: candidate.url, text });
        } else if (candidate.snippet && candidate.snippet.length > 40) {
          // Snippet-only is weaker evidence, but it is still real text
          // from the index — recorded so confidence reflects it.
          sourcesWithText.push({ title: candidate.title, url: candidate.url, text: candidate.snippet, snippetOnly: true });
        } else {
          fetchFailures.push({ url: candidate.url, reason: 'too little text' });
        }
      } catch (err) {
        if (candidate.snippet && candidate.snippet.length > 40) {
          sourcesWithText.push({ title: candidate.title, url: candidate.url, text: candidate.snippet, snippetOnly: true });
        }
        fetchFailures.push({ url: candidate.url, reason: err.message });
      }
    }

    trace.push({
      step: 'FETCH PAGES', ms: t(),
      detail: 'HTTP GET · strip markup · extract body text',
      data: `${sourcesWithText.length}/${topCandidates.length} usable` +
            (fetchFailures.length ? ` · ${fetchFailures.length} failed` : ''),
    });

    if (sourcesWithText.length === 0) {
      return {
        ...base(), status: STATUS.FETCH_FAILED,
        answer: 'Results were found but no page yielded extractable text.',
        fetchFailures, intent: intent.type, trace,
      };
    }

    /* ── 6. Extractive synthesis ── */
    const result = synthesize(query, sourcesWithText);

    if (result.citations.length === 0 || result.confidence === 0) {
      trace.push({
        step: 'SYNTHESIS EMPTY', ms: t(),
        detail: `${result.spansConsidered} spans scored, none passed the relevance floor`,
        data: `intent=${result.intent}`,
      });
      return {
        ...base(), status: STATUS.NO_RELEVANT_TEXT,
        answer: result.answer, intent: result.intent,
        spansConsidered: result.spansConsidered, trace,
      };
    }

    this.memory.store(query, result.answer, result.citations, result.confidence);

    const tookMs = t();
    trace.push({
      step: 'RANK · CENTRALITY · MMR', ms: tookMs,
      detail: `${result.spansConsidered} spans → TextRank centrality → MMR (λ=0.72) → ${result.structure.spans} kept`,
      data: `${result.structure.independentSources} sources · ` +
            `${result.structure.corroboratedSpans} corroborated · confidence=${result.confidence}`,
    });

    console.log(`[niyah] ${tookMs}ms intent=${result.intent} conf=${result.confidence} sources=${result.structure.independentSources}`);

    return {
      status: STATUS.OK,
      answer: result.answer,
      citations: result.citations,
      confidence: result.confidence,
      confidenceBreakdown: result.confidenceBreakdown,
      intent: result.intent,
      intentConfidence: result.intentConfidence,
      structure: result.structure,
      spansConsidered: result.spansConsidered,
      fetchFailures: fetchFailures.length ? fetchFailures : undefined,
      fromMemory: false,
      tookMs,
      trace,
    };
  }

  recentContext(n = 5) { return this.memory.recentContext(n); }
  close() { this.memory.close(); }
}

module.exports = { NiyahEngine, STATUS };
