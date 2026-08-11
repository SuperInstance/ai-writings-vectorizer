/**
 * Zeitgeist Dashboard API
 * =======================
 * Serves the static dashboard and proxies API calls to the zeitgeist-worker.
 * Can also read directly from D1 if co-located.
 */

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

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS });
    }

    // API proxy — forward to zeitgeist-worker
    if (url.pathname.startsWith('/api/')) {
      const apiUrl = (env.WORKER_URL || 'http://localhost:8787') + url.pathname.replace('/api', '') + url.search;
      try {
        const resp = await fetch(apiUrl, {
          method: request.method,
          headers: { 'Content-Type': 'application/json' },
          body: request.method === 'POST' ? await request.text() : undefined,
        });
        const data = await resp.json();
        return json(data, resp.status);
      } catch (e) {
        return json({ error: e.message, hint: 'Is the zeitgeist-worker running?' }, 502);
      }
    }

    // Serve static assets
    return env.ASSETS.fetch(request);
  },
};
