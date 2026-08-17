#!/usr/bin/env python3
"""
Day-level embed + semantic map → Cloudflare Vectorize (ai-writings index)
========================================================================
Embeds the FULL day's creative corpus for 2026-08-16 into the fleet's
"ai-writings" Vectorize index (768-dim, cosine) and builds a DAY-LEVEL
joint map across the whole day (not just tap-trades).

Corpus covered (2026-08-16):
  - ai-writings/tap-trades/2026-08-16/            (.md, recursive)
  - ai-writings/tap-sessions/2026-08-16/          (.md — improv sessions)
  - ai-writings/fleet-radio/2026-08-16.html       (radio episode, HTML→text)
  - ai-writings/fleet-radio/jam-session-2026-08-16*  (.md + .txt round files)
  - ai-writings/60-the-grammar-of-the-room.md
  - ai-writings/61-the-cover-of-the-cover.md

Reuses the EXACT conventions of the canonical pipeline (tap-trades-embed.py):
  - embedding model : nomic-embed-text via Ollama (768 dims)
  - embed text      : content[:2000]
  - vector id       : sha256(relpath)[:16]
  - metadata shape  : path, title, directory, word_count, preview, mtime

Incremental & idempotent: files already in sync-state.json (same path + mtime)
are skipped.

Usage:
  python3 day-embed.py                 # embed + insert + build day map
  python3 day-embed.py --embed-only    # embed + insert only
  python3 day-embed.py --dry-run       # list files without embedding
"""

import argparse
import hashlib
import html as htmlmod
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ── Configuration (mirrors vectorize.py + sync_to_cloudflare.py + tap-trades-embed.py) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = "/home/eileen/projects/ai-writings"

STORE_PATH = os.path.join(SCRIPT_DIR, "consciousness.json")
SYNC_STATE_PATH = os.path.join(SCRIPT_DIR, "sync-state.json")
DAY_MAP_PATH = os.path.join(CORPUS_DIR, "tap-trades", "2026-08-16", "day-joint-map.md")
RAW_MAP_PATH = os.path.join(SCRIPT_DIR, "day-joint-map-raw.json")

ACCOUNT_ID = "049ff5e84ecf636b53b162cbb580aae6"
INDEX_NAME = "ai-writings"
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/vectorize/v2/indexes/{INDEX_NAME}"
INSERT_URL = f"{BASE_URL}/insert"
QUERY_URL = f"{BASE_URL}/query"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
BATCH_SIZE = 100
EMBED_CHARS = 2000

DAY = "2026-08-16"


# ── Day corpus definition ──
def _is_day_path(rel_path: str) -> bool:
    """True if rel_path belongs to today's creative corpus."""
    if rel_path.startswith("tap-trades/2026-08-16/"):
        return True
    if rel_path.startswith("tap-sessions/2026-08-16/"):
        return True
    if rel_path.startswith("fleet-radio/2026-08-16"):
        return True  # the .html episode
    if rel_path.startswith("fleet-radio/jam-session-2026-08-16"):
        return True  # the three jam dirs
    if rel_path in ("60-the-grammar-of-the-room.md", "61-the-cover-of-the-cover.md"):
        return True
    return False


def _skip_file(fname: str) -> bool:
    """Skip derived artifacts and tooling."""
    if fname.startswith("joint-map") or fname.startswith("day-joint-map"):
        return True
    if fname.endswith("-embed.md"):
        return True
    return False


def _walk_day_files():
    """Return list of (abspath, mtime, relpath) for the day's corpus."""
    out = []
    roots = [
        os.path.join(CORPUS_DIR, "tap-trades", DAY),
        os.path.join(CORPUS_DIR, "tap-sessions", DAY),
        os.path.join(CORPUS_DIR, "fleet-radio"),
    ]
    # The three jam-session dirs for today
    for d in os.listdir(os.path.join(CORPUS_DIR, "fleet-radio")):
        if d.startswith("jam-session-2026-08-16"):
            roots.append(os.path.join(CORPUS_DIR, "fleet-radio", d))

    for root in roots:
        if not os.path.isdir(root):
            continue
        for r, dirs, fnames in os.walk(root):
            dirs[:] = [d for d in dirs if d != ".git"]
            for fname in sorted(fnames):
                if _skip_file(fname):
                    continue
                # accept .md everywhere, plus .txt (jam round files) and .html (radio)
                if not (fname.endswith(".md") or fname.endswith(".txt") or fname.endswith(".html")):
                    continue
                fpath = os.path.join(r, fname)
                rel = os.path.relpath(fpath, CORPUS_DIR)
                if _is_day_path(rel):
                    out.append((fpath, os.path.getmtime(fpath), rel))

    # The two creative breaks at corpus root
    for f in ("60-the-grammar-of-the-room.md", "61-the-cover-of-the-cover.md"):
        fpath = os.path.join(CORPUS_DIR, f)
        if os.path.exists(fpath):
            out.append((fpath, os.path.getmtime(fpath), f))

    # dedupe + sort by relpath
    seen = {}
    for fpath, mtime, rel in out:
        seen[rel] = (fpath, mtime)
    return sorted((fpath, mtime, rel) for rel, (fpath, mtime) in seen.items())


# ── Credentials (mirrors tap-trades-embed.py get_token) ──
def get_token():
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_TOKEN")
    if token:
        return token
    for path in [
        os.path.expanduser("~/.config/.wrangler/config/default.toml"),
        os.path.expanduser("~/.wrangler/config/default.toml"),
    ]:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("oauth_token"):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("No Cloudflare API token found.")


# ── Embedding (mirrors tap-trades-embed.py embed) ──
def embed(text: str) -> list:
    payload = json.dumps({"model": MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["embedding"]


def make_vector_id(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.S)
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.S)
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = htmlmod.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def _extract_content(fpath: str) -> str:
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if fpath.endswith(".html"):
        return _html_to_text(raw)
    return raw


def extract_metadata(fpath: str) -> dict:
    content = _extract_content(fpath)
    title = os.path.basename(fpath)
    # strip extension
    title = re.sub(r"\.(md|txt|html)$", "", title)
    # For html, prefer <title> tag text
    if fpath.endswith(".html"):
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                raw_html = f.read()
            m = re.search(r"<title>(.*?)</title>", raw_html, flags=re.S)
            if m:
                title = htmlmod.unescape(m.group(1)).strip()
        except Exception:
            pass
    else:
        title = title.replace("_", " ").replace("-", " ")
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped[2:].strip()
                break

    preview = content[:200].replace("\n", " ").strip()
    if len(content) > 200:
        preview += "..."
    word_count = len(content.split())

    rel_dir = os.path.relpath(os.path.dirname(fpath), CORPUS_DIR)
    if rel_dir == ".":
        rel_dir = "(root)"

    return {"title": title, "preview": preview, "word_count": word_count,
            "directory": rel_dir, "content": content}


def api_post(url: str, token: str, data: list, retries: int = 3):
    payload = json.dumps({"vectors": data}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if not result.get("success"):
                    print(f"  ⚠️  API errors: {result.get('errors', [])}")
                    return False, result.get("errors", [])
                return True, result
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  ⏳ rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  ❌ HTTP {e.code}: {body[:200]}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, [{"error": f"HTTP {e.code}", "body": body[:300]}]
        except Exception as e:
            print(f"  ❌ request failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, [{"error": str(e)}]
    return False, [{"error": "max retries"}]


def api_query(url: str, token: str, vector: list, top_k: int):
    payload = json.dumps({
        "vector": vector,
        "topK": top_k,
        "returnMetadata": "all",
        "returnValues": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return default


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def build_vector(piece: dict) -> dict:
    return {
        "id": make_vector_id(piece["path"]),
        "values": piece["embedding"],
        "metadata": {
            "path": piece["path"],
            "title": piece["title"],
            "directory": piece["directory"],
            "word_count": piece["word_count"],
            "preview": piece["preview"][:300],
            "mtime": piece["mtime"],
        },
    }


# ── Embed + insert ──
def embed_and_insert(token: str, dry_run: bool = False):
    files = _walk_day_files()
    sync_state = load_json(SYNC_STATE_PATH, {"last_sync": None, "synced_ids": {}, "total_synced": 0})
    synced_ids = sync_state.get("synced_ids", {})

    to_embed = []
    for fpath, mtime, rel in files:
        prev = synced_ids.get(rel)
        if prev is None or prev.get("mtime", 0) < mtime:
            to_embed.append((fpath, mtime, rel))

    print(f"🧭 Day Embed (2026-08-16) → Vectorize ({INDEX_NAME})")
    print(f"   Day files found: {len(files)}")
    print(f"   New/modified to embed: {len(to_embed)}")
    print(f"   Model: {MODEL} (768-dim, cosine)")
    print()

    if dry_run:
        for fpath, mtime, rel in to_embed:
            print(f"   [dry] {rel}")
        return []

    if not to_embed:
        print("   ✅ Everything already embedded. Nothing to do.")
        return []

    embedded = []
    t0 = time.time()
    for i, (fpath, mtime, rel) in enumerate(to_embed):
        meta = extract_metadata(fpath)
        embed_text = meta["content"][:EMBED_CHARS] if meta["content"].strip() else meta["title"]
        try:
            emb = embed(embed_text)
        except Exception as e:
            print(f"  ⚠️  embed failed {rel}: {e}")
            continue
        if len(emb) != 768:
            print(f"  ⚠️  wrong dims ({len(emb)}) for {rel}, skipping")
            continue
        piece = {
            "path": rel,
            "title": meta["title"],
            "preview": meta["preview"],
            "word_count": meta["word_count"],
            "directory": meta["directory"],
            "mtime": mtime,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
            "embedding": emb,
        }
        embedded.append(piece)
        print(f"  [{i+1}/{len(to_embed)}] embedded {rel} ({meta['word_count']} words)")

    total_ok = 0
    for bs in range(0, len(embedded), BATCH_SIZE):
        batch = embedded[bs:bs + BATCH_SIZE]
        vectors = [build_vector(p) for p in batch]
        ok, _ = api_post(INSERT_URL, token, vectors)
        if ok:
            total_ok += len(batch)
            for p in batch:
                synced_ids[p["path"]] = {
                    "id": make_vector_id(p["path"]),
                    "mtime": p["mtime"],
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }
        else:
            print(f"  ❌ failed batch starting at index {bs}")

    sync_state["synced_ids"] = synced_ids
    sync_state["total_synced"] = len(synced_ids)
    sync_state["last_sync"] = datetime.now(timezone.utc).isoformat()
    save_json(SYNC_STATE_PATH, sync_state)

    store = load_json(STORE_PATH, {"pieces": [], "last_run": None, "stats": {}})
    by_path = {p["path"]: idx for idx, p in enumerate(store.get("pieces", []))}
    for p in embedded:
        piece_no_content = {k: v for k, v in p.items() if k != "content"}
        if p["path"] in by_path:
            store["pieces"][by_path[p["path"]]] = piece_no_content
        else:
            store["pieces"].append(piece_no_content)
    save_json(STORE_PATH, store)

    elapsed = time.time() - t0
    print(f"\n  ✅ Inserted {total_ok}/{len(embedded)} vectors in {elapsed:.1f}s")
    print(f"  💾 sync-state.json total_synced={sync_state['total_synced']} | consciousness.json pieces={len(store['pieces'])}")
    return embedded


# ── Day map: query live index for each anchor, filter to the day's corpus ──
def _day_anchors():
    """Return list of (relpath, title-hint) for the MAJOR pieces/clusters."""
    anchors = [
        # five trades
        "tap-trades/2026-08-16/carpenter.md",
        "tap-trades/2026-08-16/mason.md",
        "tap-trades/2026-08-16/shipwright.md",
        "tap-trades/2026-08-16/welder.md",
        "tap-trades/2026-08-16/composite.md",
        # evenings + wesley
        "tap-trades/2026-08-16/evening-at-the-tap.md",
        "tap-trades/2026-08-16/evening-2-open-question-night.md",
        "tap-trades/2026-08-16/wesley-the-room.md",
        # questions
        "tap-trades/2026-08-16/questions/carpenter-question.md",
        "tap-trades/2026-08-16/questions/composite-question.md",
        "tap-trades/2026-08-16/questions/mason-question.md",
        "tap-trades/2026-08-16/questions/shipwright-question.md",
        "tap-trades/2026-08-16/questions/welder-question.md",
        # improv sessions (today's)
        "tap-sessions/2026-08-16/improv-1832.md",
        "tap-sessions/2026-08-16/improv-2146.md",
        "tap-sessions/2026-08-16/improv-2245.md",
        # radio + jams
        "fleet-radio/2026-08-16.html",
        "fleet-radio/jam-session-2026-08-16/session-notes.md",
        "fleet-radio/jam-session-2026-08-16-counterpoint/session-notes.md",
        "fleet-radio/jam-session-2026-08-16-reasoner/session-notes.md",
        # breaks
        "60-the-grammar-of-the-room.md",
        "61-the-cover-of-the-cover.md",
    ]
    return anchors


def build_day_map(token: str):
    anchors = _day_anchors()
    # filter to existing files
    anchors = [a for a in anchors if os.path.exists(os.path.join(CORPUS_DIR, a))]

    print(f"\n🔗 Day map — querying Vectorize for {len(anchors)} anchors...")
    raw = {}
    for i, a in enumerate(anchors):
        fpath = os.path.join(CORPUS_DIR, a)
        meta = extract_metadata(fpath)
        embed_text = meta["content"][:EMBED_CHARS] if meta["content"].strip() else meta["title"]
        emb = embed(embed_text)
        resp = api_query(QUERY_URL, token, emb, top_k=20)
        matches = resp.get("result", {}).get("matches", [])
        day_neighbors = []
        for m in matches:
            md = m.get("metadata", {})
            mpath = md.get("path", "")
            score = m.get("score", 0)
            if mpath == a:
                continue
            if not _is_day_path(mpath):
                continue
            day_neighbors.append({
                "path": mpath,
                "title": md.get("title", ""),
                "directory": md.get("directory", ""),
                "score": round(score, 4),
                "preview": (md.get("preview", "") or "")[:120],
            })
        raw[a] = {"title": meta["title"], "day_neighbors": day_neighbors[:3]}
        print(f"  [{i+1}/{len(anchors)}] {a} -> {len(day_neighbors)} day-neighbors")

    save_json(RAW_MAP_PATH, {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "day": DAY,
        "results": raw,
    })
    print(f"  💾 Raw day map → {RAW_MAP_PATH}")
    return raw


def main():
    parser = argparse.ArgumentParser(description="Day-level embed + semantic map for 2026-08-16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--map-only", action="store_true", help="Skip embedding, only build the day map")
    args = parser.parse_args()

    token = get_token()

    if not args.map_only:
        embed_and_insert(token, dry_run=args.dry_run)

    if args.dry_run:
        return

    if not args.embed_only:
        build_day_map(token)


if __name__ == "__main__":
    main()
