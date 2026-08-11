/**
 * Zeitgeist Worker — Retrieval Frequency Tracker & Enhanced Query
 * ===============================================================
 *
 * Intercepts every Vectorize query, logs which pieces were returned,
 * updates frequency tallies, computes zeitgeist scores, and returns
 * enhanced results with metadata.
 *
 * Endpoints:
 *   POST /query          — Enhanced Vectorize query (tracks retrieval)
 *   POST /track          — Manual tracking event
 *   GET  /hot            — Top 10 hottest pieces
 *   GET  /dormant        — Most dormant pieces
 *   GET  /seismic        — Recent anti-pattern break events
 *   GET  /stats          — Aggregate statistics
 *   GET  /health         — Health check
 */

// ── D1 Schema Initialization ───────────────────────────────────
const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS pieces (
  piece_id TEXT PRIMARY KEY,
  title TEXT DEFAULT '',
  directory TEXT DEFAULT '',
  retrieval_count INTEGER DEFAULT 0,
  first_retrieved TEXT,
  last_retrieved TEXT,
  retrieval_velocity REAL DEFAULT 0,
  zeitgeist_score REAL DEFAULT 0,
  novelty_score REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS retrieval_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  piece_id TEXT NOT NULL,
  retrieved_at TEXT NOT NULL,
  retrieved_by TEXT DEFAULT 'system',
  context TEXT DEFAULT '',
  mode TEXT DEFAULT 'GOSSIP',
  score REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS npc_reference_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  npc_name TEXT NOT NULL,
  piece_id TEXT NOT NULL,
  referenced_at TEXT NOT NULL,
  room TEXT DEFAULT '',
  mode TEXT DEFAULT '',
  reaction TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS propagation_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  piece_id TEXT NOT NULL,
  source_npc TEXT NOT NULL,
  room TEXT NOT NULL,
  propagated_at TEXT NOT NULL,
  chain_id TEXT,
  depth INTEGER DEFAULT 0,
  reactor_npc TEXT DEFAULT '',
  reaction TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS behavioral_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  npc_name TEXT NOT NULL,
  piece_id TEXT NOT NULL,
  detected_at TEXT NOT NULL,
  delta REAL,
  severity TEXT DEFAULT 'NORMAL',
  description TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_pieces_zeitgeist ON pieces(zeitgeist_score DESC);
CREATE INDEX IF NOT EXISTS idx_pieces_last_retrieved ON pieces(last_retrieved);
CREATE INDEX IF NOT EXISTS idx_log_piece ON retrieval_log(piece_id);
CREATE INDEX IF NOT EXISTS idx_log_time ON retrieval_log(retrieved_at);
CREATE INDEX IF NOT EXISTS idx_npc_ref ON npc_reference_log(npc_name);
`;

// ── Helpers ────────────────────────────────────────────────────
function nowISO() {
  return new Date().toISOString();
}

function hoursAgo(ts) {
  if (!ts) return Infinity;
  return (Date.now() - new Date(ts).getTime()) / 3600000;
}

function computeZeitgeistScore(count, lastRetrieved, velocity) {
  const recencyBoost = Math.exp(-hoursAgo(lastRetrieved) / 168); // ~1 week half-life
  const freqComponent = Math.log1p(count + 1);
  const novelty = Math.max(0.01, 1.0 / (1.0 + count * 0.1));
  return (freqComponent * recencyBoost + velocity * 10) * (0.3 + 0.7 * novelty);
}

async function ensurePiece(db, pieceId, title = '', directory = '') {
  await db.prepare(
    `INSERT OR IGNORE INTO pieces (piece_id, title, directory, first_retrieved)
     VALUES (?, ?, ?, ?)`
  ).bind(pieceId, title, directory, nowISO()).run();

  if (title || directory) {
    await db.prepare(
      `UPDATE pieces SET title = COALESCE(NULLIF(?, ''), title),
             directory = COALESCE(NULLIF(?, ''), directory)
       WHERE piece_id = ?`
    ).bind(title, directory, pieceId).run();
  }
}

async function computeVelocity(db, pieceId) {
  const cutoff = new Date(Date.now() - 86400000).toISOString();
  const result = await db.prepare(
    `SELECT COUNT(*) as cnt FROM retrieval_log WHERE piece_id = ? AND retrieved_at > ?`
  ).bind(pieceId, cutoff).first();
  return (result?.cnt || 0) / 24.0;
}

async function trackRetrieval(db, pieceId, meta = {}) {
  const { title = '', directory = '', retrievedBy = 'system', context = '', mode = 'GOSSIP', score = 0 } = meta;
  const now = nowISO();

  await ensurePiece(db, pieceId, title, directory);

  const piece = await db.prepare(
    `SELECT retrieval_count FROM pieces WHERE piece_id = ?`
  ).bind(pieceId).first();
  const count = piece?.retrieval_count || 0;

  const velocity = await computeVelocity(db, pieceId);
  const novelty = Math.max(0.01, 1.0 / (1.0 + count * 0.1));
  const zeitgeist = computeZeitgeistScore(count + 1, now, velocity);

  await db.prepare(
    `UPDATE pieces SET
       retrieval_count = retrieval_count + 1,
       last_retrieved = ?,
       retrieval_velocity = ?,
       novelty_score = ?,
       zeitgeist_score = ?
     WHERE piece_id = ?`
  ).bind(now, velocity, novelty, zeitgeist, pieceId).run();

  await db.prepare(
    `INSERT INTO retrieval_log (piece_id, retrieved_at, retrieved_by, context, mode, score)
     VALUES (?, ?, ?, ?, ?, ?)`
  ).bind(pieceId, now, retrievedBy, context.slice(0, 500), mode, score).run();

  return { count: count + 1, velocity, novelty, zeitgeist };
}

// ── CORS Headers ───────────────────────────────────────────────
const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS },
  });
}

// ── Main Handler ───────────────────────────────────────────────
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    // CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, { headers: CORS });
    }

    // Initialize DB schema on first request
    try {
      await env.ZEITGEIST_DB.exec(SCHEMA_SQL);
    } catch (e) {
      // D1 binding might not be configured in dev — that's OK
      console.log('D1 not available, running in stateless mode');
    }

    // ── Routes ─────────────────────────────────────────────────

    // Health check
    if (path === '/health') {
      return json({ status: 'ok', timestamp: nowISO(), service: 'zeitgeist-worker' });
    }

    // Enhanced query — proxies Vectorize and tracks retrieval
    if (path === '/query' && method === 'POST') {
      return handleQuery(request, env, ctx);
    }

    // Manual tracking event
    if (path === '/track' && method === 'POST') {
      return handleTrack(request, env);
    }

    // Hot pieces
    if (path === '/hot') {
      const limit = parseInt(url.searchParams.get('limit') || '10');
      const results = await env.ZEITGEIST_DB.prepare(
        `SELECT * FROM pieces WHERE retrieval_count > 0 ORDER BY zeitgeist_score DESC LIMIT ?`
      ).bind(limit).all();
      return json({ hot: results.results });
    }

    // Dormant pieces
    if (path === '/dormant') {
      const minAge = parseInt(url.searchParams.get('min_age_days') || '14');
      const limit = parseInt(url.searchParams.get('limit') || '5');
      const cutoff = new Date(Date.now() - minAge * 86400000).toISOString();
      const results = await env.ZEITGEIST_DB.prepare(
        `SELECT * FROM pieces WHERE last_retrieved < ? OR last_retrieved IS NULL ORDER BY last_retrieved ASC LIMIT ?`
      ).bind(cutoff, limit).all();
      return json({ dormant: results.results });
    }

    // Seismic events
    if (path === '/seismic') {
      const limit = parseInt(url.searchParams.get('limit') || '10');
      const results = await env.ZEITGEIST_DB.prepare(
        `SELECT rl.*, p.title, p.directory FROM retrieval_log rl
         JOIN pieces p ON rl.piece_id = p.piece_id
         WHERE rl.mode = 'SEISMIC'
         ORDER BY rl.retrieved_at DESC LIMIT ?`
      ).bind(limit).all();
      return json({ seismic_events: results.results });
    }

    // Propagation chains
    if (path === '/propagation') {
      const limit = parseInt(url.searchParams.get('limit') || '10');
      const results = await env.ZEITGEIST_DB.prepare(
        `SELECT * FROM propagation_log ORDER BY propagated_at DESC LIMIT ?`
      ).bind(limit).all();
      return json({ chains: results.results });
    }

    // Behavioral events
    if (path === '/behavior') {
      const npc = url.searchParams.get('npc');
      const limit = parseInt(url.searchParams.get('limit') || '20');
      let results;
      if (npc) {
        results = await env.ZEITGEIST_DB.prepare(
          `SELECT * FROM behavioral_events WHERE npc_name = ? ORDER BY detected_at DESC LIMIT ?`
        ).bind(npc, limit).all();
      } else {
        results = await env.ZEITGEIST_DB.prepare(
          `SELECT * FROM behavioral_events ORDER BY detected_at DESC LIMIT ?`
        ).bind(limit).all();
      }
      return json({ events: results.results });
    }

    // Stats
    if (path === '/stats') {
      const total = await env.ZEITGEIST_DB.prepare(
        `SELECT COUNT(*) as cnt FROM pieces`
      ).first();
      const retrieved = await env.ZEITGEIST_DB.prepare(
        `SELECT COUNT(*) as cnt FROM pieces WHERE retrieval_count > 0`
      ).first();
      const totalRet = await env.ZEITGEIST_DB.prepare(
        `SELECT COALESCE(SUM(retrieval_count), 0) as cnt FROM pieces`
      ).first();
      const avgVel = await env.ZEITGEIST_DB.prepare(
        `SELECT COALESCE(AVG(retrieval_velocity), 0) as cnt FROM pieces`
      ).first();
      return json({
        total_pieces: total?.cnt || 0,
        pieces_retrieved: retrieved?.cnt || 0,
        pieces_never_retrieved: (total?.cnt || 0) - (retrieved?.cnt || 0),
        total_retrievals: totalRet?.cnt || 0,
        avg_velocity: avgVel?.cnt || 0,
      });
    }

    // NPC reference logging
    if (path === '/npc-reference' && method === 'POST') {
      return handleNpcReference(request, env);
    }

    // 404
    return json({ error: 'Not found', path }, 404);
  },
};

// ── Route Handlers ─────────────────────────────────────────────

async function handleQuery(request, env, ctx) {
  const body = await request.json();
  const { vector, topK = 10, retrieved_by = 'system', context = '', mode = 'GOSSIP' } = body;

  if (!vector || !Array.isArray(vector)) {
    return json({ error: 'Missing or invalid "vector" in request body' }, 400);
  }

  // Query Vectorize
  const queryResult = await env.AI_WRITINGS_INDEX.query({
    vector,
    topK,
    returnMetadata: 'all',
  });

  const matches = queryResult.matches || [];

  // Track retrieval for each returned piece (fire and forget)
  if (env.ZEITGEIST_DB) {
    ctx.waitUntil((async () => {
      for (const match of matches) {
        const meta = match.metadata || {};
        const pieceId = meta.path || match.id;
        await trackRetrieval(env.ZEITGEIST_DB, pieceId, {
          title: meta.title || '',
          directory: meta.directory || '',
          retrievedBy,
          context,
          mode,
          score: match.score || 0,
        });
      }
    })());
  }

  // Return enhanced results
  return json({
    query: body.query || '',
    count: matches.length,
    results: matches.map(m => ({
      score: m.score,
      id: m.id,
      ...(m.metadata || {}),
    })),
    tracked: env.ZEITGEIST_DB ? true : false,
    timestamp: nowISO(),
  });
}

async function handleTrack(request, env) {
  const body = await request.json();
  const { piece_id, title, directory, retrieved_by, context, mode, score } = body;

  if (!piece_id) {
    return json({ error: 'Missing piece_id' }, 400);
  }

  const result = await trackRetrieval(env.ZEITGEIST_DB, piece_id, {
    title, directory, retrievedBy: retrieved_by, context, mode, score,
  });

  return json({ ok: true, piece_id, metrics: result });
}

async function handleNpcReference(request, env) {
  const body = await request.json();
  const { npc_name, piece_id, room = '', mode = '', reaction = '', title = '', directory = '' } = body;

  if (!npc_name || !piece_id) {
    return json({ error: 'Missing npc_name or piece_id' }, 400);
  }

  await ensurePiece(env.ZEITGEIST_DB, piece_id, title, directory);
  const now = nowISO();

  await env.ZEITGEIST_DB.prepare(
    `INSERT INTO npc_reference_log (npc_name, piece_id, referenced_at, room, mode, reaction)
     VALUES (?, ?, ?, ?, ?, ?)`
  ).bind(npc_name, piece_id, now, room, mode, reaction).run();

  return json({ ok: true, npc_name, piece_id, timestamp: now });
}
