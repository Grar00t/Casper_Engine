'use strict';
/**
 * Casper Agent Server — NIYAH sovereign engine
 * ---------------------------------------------------------------
 * No cloud provider. No API key. No pretrained weights. No subscription.
 * Reasoning is Okapi BM25 + TextRank centrality + MMR selection over
 * documents fetched at request time, composed extractively.
 *
 * Endpoints:
 *   GET    /health                  — service + engine capability report
 *   GET    /fetch?url=              — fetch raw page text
 *   GET    /summarize?query=&text=  — extractive summary of supplied text
 *   POST   /api/v1/ask              — full pipeline (search → rank → compose)
 *   POST   /api/v1/niyah/ask        — same pipeline + C11 symbolic audit
 *   GET    /api/v1/niyah/health
 *   GET    /api/v1/niyah/stats
 *   GET    /api/v1/niyah/context
 *   DELETE /api/v1/niyah/memory
 */

const express = require('express');
const niyahRouter = require('./routes/niyah');
const { NiyahEngine, STATUS } = require('./lib/niyahEngine');

const app = express();
app.disable('x-powered-by');

const ENGINE_ID = 'niyah-math-v5';

/* Maps a named engine outcome onto an HTTP status. A failed retrieval
   is reported as a failure, not as a 200 with an apology in the body. */
const STATUS_HTTP = {
  [STATUS.OK]: 200,
  [STATUS.MEMORY_HIT]: 200,
  [STATUS.EMPTY_QUERY]: 400,
  [STATUS.SEARCH_UNAVAILABLE]: 503,
  [STATUS.NO_RESULTS]: 404,
  [STATUS.FETCH_FAILED]: 502,
  [STATUS.NO_RELEVANT_TEXT]: 422,
};

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,DELETE,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

app.use((req, res, next) => {
  express.json({ limit: '1mb' })(req, res, (err) => {
    if (err) return res.status(400).json({ error: 'Invalid JSON', details: err.message });
    next();
  });
});

// ── Shared engine instance ───────────────────────────────────────────────────
const engine = new NiyahEngine({
  searxngBaseUrl: process.env.SEARXNG_BASE_URL,
  braveApiKey: process.env.BRAVE_API_KEY,
  memoryDbPath: process.env.NIYAH_MEMORY_DB || undefined,
});

// ── Health ───────────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    ok: true,
    service: 'casper-agent',
    engine: ENGINE_ID,
    sovereign: true,
    /* Declared explicitly so no client has to infer it from a missing
       field, and so a regression that adds a cloud call is visible. */
    externalInference: false,
    pretrainedWeights: false,
    apiKeysRequired: [],
    reasoning: ['okapi-bm25', 'textrank-centrality', 'mmr-selection', 'extractive-composition'],
    searchBackend: engine.backendName,
    memoryBackend: engine.memory.useSqlite ? 'sqlite' : 'json',
    time: new Date().toISOString(),
  });
});

// ── Raw page fetch ───────────────────────────────────────────────────────────
app.get('/fetch', async (req, res) => {
  const targetUrl = req.query.url;
  if (!targetUrl) return res.status(400).json({ ok: false, error: 'url param required' });
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    const resp = await fetch(targetUrl, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,ar;q=0.8',
      },
    });
    clearTimeout(timer);
    const html = await resp.text();
    const text = stripHtml(html);
    res.json({
      ok: true, final_url: targetUrl, status: resp.status,
      html_bytes: html.length, text_chars: text.length,
      sample: text.substring(0, 1000),
    });
  } catch (err) {
    res.status(502).json({ ok: false, error: err.message });
  }
});

// ── Extractive summary of caller-supplied text ───────────────────────────────
app.get('/summarize', (req, res) => {
  const { text, query, n } = req.query;
  if (!text || !query) return res.status(400).json({ ok: false, error: 'text and query required' });
  const { extractTopSentences } = require('./lib/relevance');
  const count = Math.min(parseInt(n, 10) || 3, 8);
  const sentences = extractTopSentences(query, text, count);
  if (!sentences.length) {
    return res.status(422).json({ ok: false, error: 'no sentence scored above the relevance floor' });
  }
  res.json({
    ok: true,
    summary: sentences.map(s => s.text).join(' '),
    sentences: sentences.length,
    scores: sentences.map(s => Math.round(s.relevanceScore * 1000) / 1000),
  });
});

// ── Primary pipeline ─────────────────────────────────────────────────────────
app.post('/api/v1/ask', async (req, res) => {
  const { query, forceFresh } = req.body || {};
  if (!query || typeof query !== 'string') {
    return res.status(400).json({ error: 'query required', engine: ENGINE_ID });
  }
  const start = Date.now();

  try {
    const result = await engine.ask(query, { forceFresh: Boolean(forceFresh) });
    const code = STATUS_HTTP[result.status] ?? 200;
    return res.status(code).json({
      ...result,
      engine: ENGINE_ID,
      mode: result.fromMemory ? 'memory' : 'extractive',
      externalInference: false,
      tookMs: Date.now() - start,
    });
  } catch (err) {
    console.error('[ask] error:', err);
    return res.status(500).json({
      error: err.message, engine: ENGINE_ID, status: 'internal_error',
      tookMs: Date.now() - start,
    });
  }
});

// ── NIYAH router (adds the C11 symbolic audit) ───────────────────────────────
app.use('/api/v1/niyah', niyahRouter);

// ── Root ─────────────────────────────────────────────────────────────────────
app.get('/', (req, res) => res.json({
  service: 'Casper Agent + NIYAH Engine',
  version: '5.0',
  engine: ENGINE_ID,
  sovereign: 'no cloud inference · no pretrained weights · no API keys',
  endpoints: {
    health: 'GET /health',
    ask: 'POST /api/v1/ask { "query": "...", "forceFresh": false }',
    ask_audited: 'POST /api/v1/niyah/ask { "query": "..." }',
    niyah_health: 'GET /api/v1/niyah/health',
    stats: 'GET /api/v1/niyah/stats',
  },
}));

// ── Start ────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const server = app.listen(PORT, () => {
  console.log(`Casper Agent (${ENGINE_ID}) listening on port ${PORT}`);
  console.log(`  health: http://localhost:${PORT}/health`);
  console.log(`  ask:    POST http://localhost:${PORT}/api/v1/ask`);
  console.log(`  search backend: ${engine.backendName} · external inference: none`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') console.error(`Port ${PORT} already in use`);
  else console.error('Server error:', err);
  process.exit(1);
});

['SIGINT', 'SIGTERM'].forEach(sig => process.on(sig, () => {
  server.close(() => { engine.close(); process.exit(0); });
}));

process.on('uncaughtException', (err) => console.error('Uncaught:', err.message));
process.on('unhandledRejection', (r) => console.error('Unhandled:', r));

// ── Helpers ──────────────────────────────────────────────────────────────────
function stripHtml(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&quot;/g, '"')
    .replace(/\s+/g, ' ').trim();
}
