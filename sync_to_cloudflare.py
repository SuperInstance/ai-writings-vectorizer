#!/usr/bin/env python3
"""
Sync Local Embeddings → Cloudflare Vectorize
=============================================
Uploads all embeddings from consciousness.json to Cloudflare Vectorize
in batches. Supports incremental sync (only new/modified since last run).

Usage:
  python3 sync_to_cloudflare.py             # Full sync (all vectors)
  python3 sync_to_cloudflare.py --update    # Incremental sync (only new/modified)
  python3 sync_to_cloudflare.py --status    # Show sync state
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(SCRIPT_DIR, "consciousness.json")
SYNC_STATE_PATH = os.path.join(SCRIPT_DIR, "sync-state.json")

ACCOUNT_ID = "049ff5e84ecf636b53b162cbb580aae6"
INDEX_NAME = "ai-writings"

# Vectorize API endpoints
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/vectorize/v2/indexes/{INDEX_NAME}"
INSERT_URL = f"{BASE_URL}/insert"
DELETE_URL = f"{BASE_URL}/delete-by-ids"
INFO_URL = BASE_URL

BATCH_SIZE = 100  # max vectors per request

# ── Credentials ────────────────────────────────────────────────
def get_token():
    """Get Cloudflare API token from wrangler OAuth config."""
    # Try env first
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_TOKEN")
    if token:
        return token

    # Try wrangler config
    config_paths = [
        os.path.expanduser("~/.config/.wrangler/config/default.toml"),
        os.path.expanduser("~/.wrangler/config/default.toml"),
    ]
    for path in config_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("oauth_token"):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        return token

    raise RuntimeError("No Cloudflare API token found. Set CLOUDFLARE_API_TOKEN or configure wrangler.")


# ── Sync State ─────────────────────────────────────────────────
def load_sync_state() -> dict:
    """Load sync state or return empty."""
    if os.path.exists(SYNC_STATE_PATH):
        try:
            with open(SYNC_STATE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "last_sync": None,
        "synced_ids": {},  # path -> {id, mtime, synced_at}
        "total_synced": 0,
    }


def save_sync_state(state: dict):
    """Save sync state."""
    state["last_sync"] = datetime.now(timezone.utc).isoformat()
    with open(SYNC_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


# ── Vector ID Management ───────────────────────────────────────
def make_vector_id(path: str) -> str:
    """Create a stable, deterministic vector ID from a file path.

    Vectorize IDs must be <= 64 bytes, alphanumeric + hyphens/underscores.
    We use SHA-256 hash truncated to 16 hex chars for guaranteed uniqueness
    within 64 bytes.
    """
    import hashlib
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


# ── API Calls ──────────────────────────────────────────────────
def api_post(url: str, token: str, data: list, retries: int = 3):
    """POST to Cloudflare API with retry logic."""
    payload = json.dumps({"vectors": data}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if not result.get("success"):
                    errors = result.get("errors", [])
                    print(f"  ⚠️  API returned errors: {errors}")
                    return False, errors
                return True, result
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  ❌ HTTP {e.code}: {body[:200]}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, [{"error": f"HTTP {e.code}", "body": body[:500]}]
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, [{"error": str(e)}]

    return False, [{"error": "Max retries exceeded"}]


def api_get(url: str, token: str):
    """GET from Cloudflare API."""
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Build Vector Payload ───────────────────────────────────────
def build_vector(piece: dict) -> dict:
    """Build a Vectorize vector payload from a store piece."""
    vid = make_vector_id(piece["path"])
    return {
        "id": vid,
        "values": piece["embedding"],
        "metadata": {
            "path": piece["path"],
            "title": piece.get("title", ""),
            "directory": piece.get("directory", ""),
            "word_count": piece.get("word_count", 0),
            "preview": piece.get("preview", "")[:300],  # cap metadata size
            "mtime": piece.get("mtime", 0),
        },
    }


# ── Sync Operations ────────────────────────────────────────────
def sync_full(token: str, verbose: bool = True):
    """Upload all vectors to Cloudflare Vectorize."""
    store = json.load(open(STORE_PATH, "r", encoding="utf-8"))
    pieces = store.get("pieces", [])

    if verbose:
        print(f"☁️  Cloudflare Vectorize — Full Sync")
        print(f"   Index: {INDEX_NAME}")
        print(f"   Total vectors: {len(pieces)}")
        print(f"   Batch size: {BATCH_SIZE}")
        print()

    if not pieces:
        print("   ❌ No pieces to sync.")
        return

    state = load_sync_state()
    total_uploaded = 0
    total_failed = 0
    t0 = time.time()

    for batch_start in range(0, len(pieces), BATCH_SIZE):
        batch = pieces[batch_start : batch_start + BATCH_SIZE]
        vectors = [build_vector(p) for p in batch]

        success, result = api_post(INSERT_URL, token, vectors)
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(pieces) + BATCH_SIZE - 1) // BATCH_SIZE

        if success:
            total_uploaded += len(batch)
            # Track synced IDs
            for p in batch:
                vid = make_vector_id(p["path"])
                state["synced_ids"][p["path"]] = {
                    "id": vid,
                    "mtime": p.get("mtime", 0),
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }

            if verbose and (batch_num % 10 == 0 or batch_num == total_batches):
                elapsed = time.time() - t0
                rate = total_uploaded / max(elapsed, 0.1)
                eta = (len(pieces) - total_uploaded) / max(rate, 0.1)
                print(f"  [{batch_num}/{total_batches}] {total_uploaded}/{len(pieces)} uploaded | "
                      f"{rate:.0f} vec/s | ETA: {eta:.0f}s")
        else:
            total_failed += len(batch)
            print(f"  [{batch_num}/{total_batches}] ❌ Failed batch")

        # Small delay to be nice to the API
        if batch_start > 0 and batch_start % (BATCH_SIZE * 10) == 0:
            time.sleep(0.5)

    state["total_synced"] = total_uploaded
    save_sync_state(state)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  ✅ Uploaded {total_uploaded} vectors in {elapsed:.1f}s")
        if total_failed:
            print(f"  ⚠️  {total_failed} vectors failed")
        print(f"  💾 Sync state saved to {SYNC_STATE_PATH}")


def sync_update(token: str, verbose: bool = True):
    """Only upload new/modified vectors since last sync."""
    store = json.load(open(STORE_PATH, "r", encoding="utf-8"))
    pieces = store.get("pieces", [])
    state = load_sync_state()
    synced_ids = state.get("synced_ids", {})

    # Determine what needs syncing
    to_upload = []
    for p in pieces:
        path = p["path"]
        mtime = p.get("mtime", 0)
        if path not in synced_ids:
            to_upload.append(p)
        elif synced_ids[path].get("mtime", 0) < mtime:
            to_upload.append(p)

    # Check for deleted files (in sync state but not in store)
    current_paths = {p["path"] for p in pieces}
    to_delete = [
        synced_ids[p]["id"]
        for p in synced_ids
        if p not in current_paths
    ]

    if verbose:
        print(f"☁️  Cloudflare Vectorize — Incremental Sync")
        print(f"   Index: {INDEX_NAME}")
        print(f"   Total in store: {len(pieces)}")
        print(f"   New/modified: {len(to_upload)}")
        print(f"   Deleted (to remove): {len(to_delete)}")

    if not to_upload and not to_delete:
        if verbose:
            print("   ✅ Everything up to date.")
        state["last_sync"] = datetime.now(timezone.utc).isoformat()
        save_sync_state(state)
        return

    total_uploaded = 0
    total_deleted = 0
    t0 = time.time()

    # Upload in batches
    for batch_start in range(0, len(to_upload), BATCH_SIZE):
        batch = to_upload[batch_start : batch_start + BATCH_SIZE]
        vectors = [build_vector(p) for p in batch]

        success, result = api_post(INSERT_URL, token, vectors)
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(to_upload) + BATCH_SIZE - 1) // BATCH_SIZE

        if success:
            total_uploaded += len(batch)
            for p in batch:
                vid = make_vector_id(p["path"])
                state["synced_ids"][p["path"]] = {
                    "id": vid,
                    "mtime": p.get("mtime", 0),
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }
            if verbose and (batch_num % 5 == 0 or batch_num == total_batches):
                print(f"  [upload {batch_num}/{total_batches}] {total_uploaded}/{len(to_upload)} done")
        else:
            print(f"  [upload {batch_num}/{total_batches}] ❌ Failed")

    # Delete in batches
    for batch_start in range(0, len(to_delete), BATCH_SIZE):
        batch_ids = to_delete[batch_start : batch_start + BATCH_SIZE]
        payload = [{"id": vid} for vid in batch_ids]
        success, result = api_post(DELETE_URL, token, payload)
        if success:
            total_deleted += len(batch_ids)

    # Clean up sync state for deleted files
    deleted_paths = [p for p in synced_ids if p not in current_paths]
    for p in deleted_paths:
        del state["synced_ids"][p]

    state["total_synced"] = len(state.get("synced_ids", {}))
    save_sync_state(state)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  ✅ Uploaded {total_uploaded} | Deleted {total_deleted} in {elapsed:.1f}s")
        print(f"  💾 Sync state saved to {SYNC_STATE_PATH}")


def show_status(token: str):
    """Show current sync status."""
    state = load_sync_state()
    store = json.load(open(STORE_PATH, "r", encoding="utf-8"))
    pieces = store.get("pieces", [])

    print(f"📊 Cloudflare Vectorize Sync Status")
    print(f"   Index: {INDEX_NAME}")
    print(f"   Last sync: {state.get('last_sync', 'never')}")
    print(f"   Local pieces: {len(pieces)}")
    print(f"   Synced IDs: {state.get('total_synced', 0)}")

    synced_paths = set(state.get("synced_ids", {}).keys())
    local_paths = {p["path"] for p in pieces}
    unsynced = local_paths - synced_paths
    deleted = synced_paths - local_paths

    print(f"   Unsynced (new): {len(unsynced)}")
    print(f"   Missing locally (deleted): {len(deleted)}")

    # Get remote index info
    try:
        info = api_get(INFO_URL, token)
        result = info.get("result", {})
        print(f"\n   Remote index:")
        print(f"     Dimensions: {result.get('config', {}).get('dimensions', '?')}")
        print(f"     Metric: {result.get('config', {}).get('metric', '?')}")
        print(f"     Created: {result.get('created_on', '?')}")
    except Exception as e:
        print(f"\n   ⚠️  Could not fetch remote index info: {e}")


# ── CLI ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Sync local embeddings to Cloudflare Vectorize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--update", action="store_true", help="Incremental sync (only new/modified)")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    args = parser.parse_args()

    try:
        token = get_token()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if args.status:
        show_status(token)
    elif args.update:
        sync_update(token)
    else:
        sync_full(token)


if __name__ == "__main__":
    main()
