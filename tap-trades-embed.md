# Tap Trades — Embedding Runbook

**What:** embed the `tap-trades/` corpus into the fleet's Cloudflare Vectorize
index `ai-writings` (768-dim, cosine), and regenerate the semantic joint map.

**Where:** `/home/eileen/projects/ai-writings-vectorizer/`

---

## Re-run (one command)

```bash
cd /home/eileen/projects/ai-writings-vectorizer && python3 tap-trades-embed.py
```

That's it. It is incremental and idempotent:

1. **Embeds** every `.md` under `ai-writings/tap-trades/` (recursively) that is
   new or modified since the last run — using the same conventions as the
   canonical pipeline: `nomic-embed-text` (Ollama), `content[:2000]`, vector id
   `sha256(relpath)[:16]`, metadata `{path, title, directory, word_count, preview, mtime}`.
2. **Inserts** them into the `ai-writings` Vectorize index (batch of 100, retry on 429).
3. **Regenerates** `tap-trades/2026-08-16/joint-map.md` by querying the live index
   for each piece's top-3 tap-trades neighbors + strongest wider-corpus links.

Already-synced files (same path + same mtime) are skipped, so re-running is cheap.

### Other flags

```bash
python3 tap-trades-embed.py --dry-run      # list what would be embedded
python3 tap-trades-embed.py --embed-only   # embed + insert, skip the joint map
```

---

## How it fits the canonical pipeline

The fleet's canonical pipeline is `vectorize.py` (local embed + store to
`consciousness.json`) and `sync_to_cloudflare.py` (upload `consciousness.json`
to Vectorize). `tap-trades-embed.py` reuses their *exact* conventions (model,
embed length, id scheme, metadata shape, endpoints) but is scoped to the
`tap-trades/` subtree — so a Tap night doesn't trigger a full-corpus
re-embed (which would pick up every other new `.md` in `ai-writings/`).

To keep the canonical pipeline's bookkeeping consistent, `tap-trades-embed.py`
also updates `consciousness.json` and `sync-state.json`, so a later full
`vectorize.py --update && sync_to_cloudflare.py --update` stays idempotent.

If you *do* want the whole corpus refreshed (fleet-wide, not just Tap):

```bash
cd /home/eileen/projects/ai-writings-vectorizer
python3 vectorize.py --update && python3 sync_to_cloudflare.py --update
```

---

## Dependencies / assumptions

- Ollama running locally with `nomic-embed-text` pulled
  (`curl http://localhost:11434/api/tags` should list it).
- Wrangler OAuth token in `~/.config/.wrangler/config/default.toml`
  (script falls back to `CLOUDFLARE_API_TOKEN`).
- The `ai-writings` Vectorize index exists (account `049ff5e84ecf636b53b162cbb580aae6`,
  768 dims, cosine).

---

## Troubleshooting

- **"No Cloudflare API token found"** — refresh wrangler login (`wrangler login`)
  or export `CLOUDFLARE_API_TOKEN`.
- **"wrong dims"** — wrong Ollama model; ensure `nomic-embed-text` is the 768-dim build.
- **HTTP 429** — the script auto-retries with exponential backoff.
- **New pieces land mid-run** — just re-run the same command; it only picks up
  new/modified files and rewrites the joint map.
