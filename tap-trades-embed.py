#!/usr/bin/env python3
"""
Tap Trades → Cloudflare Vectorize (ai-writings index)
=====================================================
Embeds every .md file under ai-writings/tap-trades/ (recursively) into the
fleet's existing "ai-writings" Vectorize index, reusing the EXACT conventions
of the canonical pipeline (vectorize.py + sync_to_cloudflare.py):

  - embedding model : nomic-embed-text via Ollama (768 dims, cosine)
  - embed text      : content[:2000]  (whole-file, first 2000 chars — matches
                      vectorize.py which does NOT chunk)
  - vector id       : sha256(relpath)[:16]   (matches sync_to_cloudflare.py)
  - metadata shape  : path, title, directory, word_count, preview(<=300), mtime

It also keeps consciousness.json and sync-state.json consistent so that the
canonical `vectorize.py --update && sync_to_cloudflare.py --update` re-run
remains idempotent (already-synced files are not re-embedded / re-inserted).

Usage:
  python3 tap-trades-embed.py            # embed + insert + build joint map (one command)
  python3 tap-trades-embed.py --embed-only  # embed + insert only
  python3 tap-trades-embed.py --dry-run  # list files without embedding
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ── Configuration (mirrors vectorize.py + sync_to_cloudflare.py) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = "/home/eileen/projects/ai-writings"
TAP_DIR = os.path.join(CORPUS_DIR, "tap-trades")

STORE_PATH = os.path.join(SCRIPT_DIR, "consciousness.json")
SYNC_STATE_PATH = os.path.join(SCRIPT_DIR, "sync-state.json")
JOINT_MAP_PATH = os.path.join(TAP_DIR, "2026-08-16", "joint-map.md")

ACCOUNT_ID = "049ff5e84ecf636b53b162cbb580aae6"
INDEX_NAME = "ai-writings"
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/vectorize/v2/indexes/{INDEX_NAME}"
INSERT_URL = f"{BASE_URL}/insert"
QUERY_URL = f"{BASE_URL}/query"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
BATCH_SIZE = 100


# ── Credentials (mirrors sync_to_cloudflare.py get_token) ──
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


# ── Embedding (mirrors vectorize.py embed) ──
def embed(text: str) -> list:
    payload = json.dumps({"model": MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["embedding"]


# ── Vector ID (mirrors sync_to_cloudflare.py make_vector_id) ──
def make_vector_id(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


# ── Metadata extraction (mirrors vectorize.py extract_metadata) ──
def extract_metadata(filepath: str) -> dict:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        content = ""

    title = os.path.basename(filepath).replace(".md", "").replace("_", " ").replace("-", " ")
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            break

    preview = content[:200].replace("\n", " ").strip()
    if len(content) > 200:
        preview += "..."

    word_count = len(content.split())

    rel_dir = os.path.relpath(os.path.dirname(filepath), CORPUS_DIR)
    if rel_dir == ".":
        rel_dir = "(root)"

    return {
        "title": title,
        "preview": preview,
        "word_count": word_count,
        "directory": rel_dir,
        "content": content,
    }


def walk_tap_trades():
    files = []
    for root, dirs, fnames in os.walk(TAP_DIR):
        dirs[:] = [d for d in dirs if d != ".git"]
        for fname in sorted(fnames):
            # Skip derived artifacts (the joint map itself, runbooks, etc.)
            if fname.startswith("joint-map") or fname.endswith("-embed.md"):
                continue
            if fname.endswith(".md"):
                fpath = os.path.join(root, fname)
                files.append((fpath, os.path.getmtime(fpath)))
    return files


# ── API helpers (mirrors sync_to_cloudflare.py api_post) ──
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


# ── Local store bookkeeping (keeps canonical pipeline idempotent) ──
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
    files = walk_tap_trades()
    sync_state = load_json(SYNC_STATE_PATH, {"last_sync": None, "synced_ids": {}, "total_synced": 0})
    synced_ids = sync_state.get("synced_ids", {})

    to_embed = []
    for fpath, mtime in files:
        rel_path = os.path.relpath(fpath, CORPUS_DIR)
        prev = synced_ids.get(rel_path)
        if prev is None or prev.get("mtime", 0) < mtime:
            to_embed.append((fpath, mtime, rel_path))

    print(f"🧭 Tap Trades → Vectorize ({INDEX_NAME})")
    print(f"   Corpus: {TAP_DIR}")
    print(f"   Files found: {len(files)}")
    print(f"   New/modified to embed: {len(to_embed)}")
    print(f"   Model: {MODEL} (768-dim, cosine)")
    print()

    if dry_run:
        for fpath, mtime, rel_path in to_embed:
            print(f"   [dry] {rel_path}")
        return []

    if not to_embed:
        print("   ✅ Everything already embedded. Nothing to do.")
        return []

    embedded = []
    t0 = time.time()
    for i, (fpath, mtime, rel_path) in enumerate(to_embed):
        meta = extract_metadata(fpath)
        embed_text = meta["content"][:2000] if meta["content"].strip() else meta["title"]
        try:
            emb = embed(embed_text)
        except Exception as e:
            print(f"  ⚠️  embed failed {rel_path}: {e}")
            continue
        if len(emb) != 768:
            print(f"  ⚠️  wrong dims ({len(emb)}) for {rel_path}, skipping")
            continue
        piece = {
            "path": rel_path,
            "title": meta["title"],
            "preview": meta["preview"],
            "word_count": meta["word_count"],
            "directory": meta["directory"],
            "mtime": mtime,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
            "embedding": emb,
        }
        embedded.append(piece)
        print(f"  [{i+1}/{len(to_embed)}] embedded {rel_path} ({meta['word_count']} words)")

    # Insert in batches
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

    # Update sync-state.json
    sync_state["synced_ids"] = synced_ids
    sync_state["total_synced"] = len(synced_ids)
    sync_state["last_sync"] = datetime.now(timezone.utc).isoformat()
    save_json(SYNC_STATE_PATH, sync_state)

    # Update consciousness.json (append/update, no neighbor recompute — canonical
    # --update will recompute neighbors on its next full pass)
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


# ── Joint map (queries the LIVE Vectorize index) ──
def build_joint_map(token: str):
    files = walk_tap_trades()
    pieces = []
    for fpath, mtime in files:
        meta = extract_metadata(fpath)
        rel_path = os.path.relpath(fpath, CORPUS_DIR)
        embed_text = meta["content"][:2000] if meta["content"].strip() else meta["title"]
        emb = embed(embed_text)
        pieces.append({
            "path": rel_path,
            "title": meta["title"],
            "directory": meta["directory"],
            "word_count": meta["word_count"],
            "embedding": emb,
            "id": make_vector_id(rel_path),
        })

    print(f"\n🔗 Joint map — querying Vectorize for {len(pieces)} pieces...")

    raw = {}
    for i, p in enumerate(pieces):
        resp = api_query(QUERY_URL, token, p["embedding"], top_k=20)
        matches = resp.get("result", {}).get("matches", [])
        tap_neighbors = []
        wider = []
        for m in matches:
            md = m.get("metadata", {})
            mpath = md.get("path", "")
            score = m.get("score", 0)
            if mpath == p["path"]:
                continue  # self
            entry = {"path": mpath, "title": md.get("title", ""), "directory": md.get("directory", ""), "score": round(score, 4), "preview": (md.get("preview", "") or "")[:120]}
            if mpath.startswith("tap-trades/"):
                if len(tap_neighbors) < 3:
                    tap_neighbors.append(entry)
            else:
                if len(wider) < 3:
                    wider.append(entry)
        raw[p["path"]] = {"title": p["title"], "tap_trades": tap_neighbors, "wider_corpus": wider}
        print(f"  [{i+1}/{len(pieces)}] {p['path']}")

    # Persist raw data for reference / debugging
    save_json(os.path.join(SCRIPT_DIR, "tap-trades-joint-map-raw.json"), {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": raw,
    })

    # Render markdown
    lines = []
    lines.append("# Tap Trades — Joint Map")
    lines.append("")
    lines.append("*Semantic map of the Tap Trades corpus (2026-08-16), generated by querying the live Cloudflare Vectorize `ai-writings` index (768-dim, cosine, nomic-embed-text).*")
    lines.append("")
    lines.append("Each piece lists its 3 nearest tap-trades neighbors and its strongest links into the wider ai-writings corpus, with cosine similarity.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for p in pieces:
        r = raw[p["path"]]
        lines.append(f"## {p['title']}")
        lines.append("")
        lines.append(f"`{p['path']}` · {p['word_count']} words · {p['directory']}")
        lines.append("")
        lines.append("**Nearest tap-trades pieces:**")
        if r["tap_trades"]:
            for n in r["tap_trades"]:
                lines.append(f"- `{n['score']}` — **{n['title']}** (`{n['path']}`)")
        else:
            lines.append("- *(none found yet)*")
        lines.append("")
        lines.append("**Strongest wider-corpus links:**")
        if r["wider_corpus"]:
            for n in r["wider_corpus"]:
                lines.append(f"- `{n['score']}` — **{n['title']}** (`{n['path']}`)")
        else:
            lines.append("- *(none)*")
        lines.append("")
        lines.append("---")
        lines.append("")

    os.makedirs(os.path.dirname(JOINT_MAP_PATH), exist_ok=True)
    with open(JOINT_MAP_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  ✅ Wrote joint map to {JOINT_MAP_PATH}")
    print(f"  💾 Raw data → {os.path.join(SCRIPT_DIR, 'tap-trades-joint-map-raw.json')}")
    return raw


def main():
    parser = argparse.ArgumentParser(description="Embed tap-trades corpus into Cloudflare Vectorize")
    parser.add_argument("--dry-run", action="store_true", help="List files without embedding")
    parser.add_argument("--embed-only", action="store_true", help="Skip the joint-map step")
    args = parser.parse_args()

    token = get_token()
    embedded = embed_and_insert(token, dry_run=args.dry_run)

    if args.dry_run:
        return

    if not args.embed_only:
        build_joint_map(token)


if __name__ == "__main__":
    main()
