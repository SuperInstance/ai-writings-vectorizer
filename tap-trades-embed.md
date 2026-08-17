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

## Day two (2026-08-16) — expanded to the whole day

Day two scaled the job from *tap-trades only* to **the entire day's corpus**.
A second script, `day-embed.py`, sits alongside `tap-trades-embed.py` and reuses
the same conventions (same model, id scheme, metadata shape, endpoints, and
sync-state/consciousness bookkeeping). It is NOT a replacement — `tap-trades-embed.py`
still owns the tap-trades-only joint map; `day-embed.py` owns the day-level map.

### Expanded corpus (all of 2026-08-16)

- `tap-trades/2026-08-16/` — incl. the **five `adaptations/`** that landed at day's end
- `tap-sessions/2026-08-16/` — five improv sessions (not three — 0737, 1832, 2011, 2146, 2245)
- `fleet-radio/2026-08-16.html` — the radio episode (HTML → text-stripped before embedding)
- `fleet-radio/jam-session-2026-08-16*` — three jam dirs; embed `.md` + `.txt` round files, **skip `.py` and `.mid`**
- `60-the-grammar-of-the-room.md` + `61-the-cover-of-the-cover.md` — the two "creative breaks"

**48 new vectors** this run (73 day files total; 25 tap-trades were already in).

```bash
cd /home/eileen/projects/ai-writings-vectorizer && python3 day-embed.py           # embed + day map
python3 day-embed.py --embed-only   # embed + insert only
python3 day-embed.py --map-only     # re-query the live index, rebuild day map
python3 day-embed.py --dry-run      # list what would be embedded
```

Outputs: `tap-trades/2026-08-16/day-joint-map.md` (the day map) +
`day-joint-map-raw.json` (raw neighbor data for the anchor set).

### Gotchas hit on day two

1. **Jam-dir path filter bug.** The day-corpus predicate must match *both*
   `fleet-radio/2026-08-16` (the `.html`) **and** `fleet-radio/jam-session-2026-08-16`
   (the three jam dirs). A `startswith("fleet-radio/2026-08-16")` alone silently
   drops every jam round file.
2. **`adaptations/` appeared mid-run.** The walk is live, not snapshot — re-run
   `--dry-run` right before the real run; new files land in the middle of the night.
3. **HTML must be stripped.** Embedding raw `2026-08-16.html` would ingest 2000
   chars of `<head>`/CSS. Strip `<style>`/`<script>`, then tags, then unescape
   before `content[:2000]`.
4. **The breaks (60/61) and the radio episode are semantic islands.** They surface
   **zero** day-neighbors in a top-40 query — even though improv-2011's theme header
   *literally quotes* 60 and 61. A quoted reference ≠ a semantic neighbor; the tiny
   terse poems rank against the older songforge corpus they descend from, not the
   improv transcript that names them. Don't expect a `topK`-bounded query to surface
   this kind of link; you have to probe it explicitly.
5. **A hypothesized link was falsified** (and that's fine): the composite's scarf
   does **not** reach any jam session — its nearest neighbors stay inside the
   weld-and-scarf cluster. Record the miss alongside the hits.
6. **Cross-corpus links hide below `topK=20`.** The tap-trades cluster (30 pieces) is
   dense enough to crowd out trade→improv links at low `topK`. For the "surprising
   cross-corpus" section, query `topK=40` and bucket neighbors by directory prefix
   (`tap-trades/`, `tap-sessions/`, `fleet-radio/`, root breaks) to surface links
   that never crack the top-3 overall.

## Troubleshooting

- **"No Cloudflare API token found"** — refresh wrangler login (`wrangler login`)
  or export `CLOUDFLARE_API_TOKEN`.
- **"wrong dims"** — wrong Ollama model; ensure `nomic-embed-text` is the 768-dim build.
- **HTTP 429** — the script auto-retries with exponential backoff.
- **New pieces land mid-run** — just re-run the same command; it only picks up
  new/modified files and rewrites the joint map.
