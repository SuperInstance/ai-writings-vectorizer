#!/usr/bin/env python3
"""
The Collective Consciousness — Vectorized Index of ai-writings
================================================================
Walks the entire ai-writings corpus, embeds every .md file into
768-dimensional space via Ollama nomic-embed-text, and stores the
results as a local JSON file with similarity indexes.

Usage:
  python3 vectorize.py                    # Full rebuild
  python3 vectorize.py --update           # Only new/modified files
  python3 vectorize.py --query "search"   # Query top 10 matches
  python3 vectorize.py --visualize        # 2D projection (t-SNE/PCA)
  python3 vectorize.py --stats            # Corpus statistics
"""

import argparse
import json
import os
import sys
import time
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import urllib.request

# ── Configuration ──────────────────────────────────────────────
CORPUS_DIR = "/home/eileen/projects/ai-writings"
STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "consciousness.json")
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
TOP_K = 10  # neighbors to compute per piece


# ── Ollama Embedding ───────────────────────────────────────────
def embed(text: str) -> list[float]:
    """Get a 768-dim embedding from Ollama nomic-embed-text."""
    payload = json.dumps({"model": MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["embedding"]


# ── Corpus Walker ──────────────────────────────────────────────
def walk_corpus(corpus_dir: str):
    """Yield (filepath, mtime) for every .md file, excluding .git."""
    for root, dirs, files in os.walk(corpus_dir):
        # Skip .git
        if ".git" in root:
            continue
        dirs[:] = [d for d in dirs if d != ".git"]
        for fname in sorted(files):
            if fname.endswith(".md"):
                fpath = os.path.join(root, fname)
                mtime = os.path.getmtime(fpath)
                yield fpath, mtime


def extract_metadata(filepath: str, corpus_dir: str) -> dict:
    """Extract title, preview, word count, directory from a file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        content = ""

    # Title: first # heading, or filename
    title = os.path.basename(filepath).replace(".md", "").replace("_", " ").replace("-", " ")
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            break

    # Preview: first 200 chars of actual content (skip title line)
    preview = content[:200].replace("\n", " ").strip()
    if len(content) > 200:
        preview += "..."

    # Word count
    word_count = len(content.split())

    # Directory relative to corpus
    rel_dir = os.path.relpath(os.path.dirname(filepath), corpus_dir)
    if rel_dir == ".":
        rel_dir = "(root)"

    return {
        "title": title,
        "preview": preview,
        "word_count": word_count,
        "directory": rel_dir,
    }


# ── Similarity Computation ─────────────────────────────────────
def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute full cosine similarity matrix."""
    # Normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normalized = embeddings / norms
    return normalized @ normalized.T


def compute_neighbors(sim_matrix: np.ndarray, top_k: int = TOP_K):
    """For each piece, find top_k most similar (excluding self)."""
    neighbors = []
    n = sim_matrix.shape[0]
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        top_indices = np.argsort(sims)[::-1][:top_k]
        neighbor_list = []
        for idx in top_indices:
            neighbor_list.append({
                "index": int(idx),
                "similarity": round(float(sims[idx]), 4),
            })
        neighbors.append(neighbor_list)
    return neighbors


# ── Store Management ───────────────────────────────────────────
def load_store() -> dict:
    """Load existing store or return empty structure."""
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "r") as f:
                return json.loads(f.read())
        except (json.JSONDecodeError, IOError):
            pass
    return {"pieces": [], "last_run": None, "stats": {}}


def save_store(store: dict):
    """Save store to disk."""
    store["last_run"] = datetime.now(timezone.utc).isoformat()
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)


def compute_stats(store: dict) -> dict:
    """Compute corpus statistics."""
    pieces = store.get("pieces", [])
    if not pieces:
        return {}

    dirs = {}
    total_words = 0
    for p in pieces:
        d = p.get("directory", "unknown")
        dirs[d] = dirs.get(d, 0) + 1
        total_words += p.get("word_count", 0)

    return {
        "total_pieces": len(pieces),
        "total_words": total_words,
        "avg_words": round(total_words / len(pieces), 1) if pieces else 0,
        "directories": len(dirs),
        "pieces_by_directory": dict(sorted(dirs.items(), key=lambda x: -x[1])),
        "dimensions": 768,
        "model": MODEL,
    }


# ── Main Operations ────────────────────────────────────────────
def op_full_rebuild(verbose=True):
    """Embed every .md file in the corpus from scratch."""
    store = {"pieces": [], "last_run": None, "stats": {}}
    files = list(walk_corpus(CORPUS_DIR))
    total = len(files)

    if verbose:
        print(f"🧠 Collective Consciousness — Full Rebuild")
        print(f"   Corpus: {CORPUS_DIR}")
        print(f"   Files to embed: {total}")
        print(f"   Model: {MODEL} (768-dim)")
        print(f"   Estimated time: ~{total * 0.3 / 60:.1f} minutes")
        print()

    embeddings_list = []
    t0 = time.time()

    for i, (fpath, mtime) in enumerate(files):
        meta = extract_metadata(fpath, CORPUS_DIR)

        # Get embedding
        try:
            content = open(fpath, "r", encoding="utf-8", errors="replace").read()
            embed_text = content[:2000] if content.strip() else (meta["title"] or os.path.basename(fpath))
            emb = embed(embed_text)
            if len(emb) != 768:
                print(f"  ⚠️  Wrong dims ({len(emb)}) for {fpath}, skipping")
                continue
        except Exception as e:
            print(f"  ⚠️  Failed to embed {fpath}: {e}")
            continue

        piece = {
            "path": os.path.relpath(fpath, CORPUS_DIR),
            "title": meta["title"],
            "preview": meta["preview"],
            "word_count": meta["word_count"],
            "directory": meta["directory"],
            "mtime": mtime,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
            "embedding": emb,
        }
        embeddings_list.append(emb)
        store["pieces"].append(piece)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  [{i+1}/{total}] {rate:.1f} files/s | ETA: {eta/60:.1f}m | Last: {meta['title'][:50]}")

    if verbose:
        elapsed = time.time() - t0
        print(f"\n  ✅ Embedded {len(store['pieces'])} pieces in {elapsed/60:.1f} minutes")

    # Compute similarity
    if embeddings_list:
        if verbose:
            print("  🔗 Computing similarity matrix...")
        emb_array = np.array(embeddings_list)
        sim_matrix = cosine_similarity_matrix(emb_array)
        neighbors = compute_neighbors(sim_matrix)

        for i, piece in enumerate(store["pieces"]):
            piece["neighbors"] = neighbors[i]

    store["stats"] = compute_stats(store)
    save_store(store)

    if verbose:
        print(f"  💾 Saved to {STORE_PATH}")
        print(f"  📊 {store['stats']['total_pieces']} pieces | {store['stats']['total_words']:,} words | {store['stats']['directories']} directories")

    return store


def op_update(verbose=True):
    """Only embed new or modified files."""
    store = load_store()
    existing_paths = {p["path"]: p for p in store.get("pieces", [])}

    files = list(walk_corpus(CORPUS_DIR))
    total = len(files)

    # Find new/modified files
    to_embed = []
    for fpath, mtime in files:
        rel_path = os.path.relpath(fpath, CORPUS_DIR)
        if rel_path not in existing_paths:
            to_embed.append((fpath, mtime, "new"))
        elif existing_paths[rel_path].get("mtime", 0) < mtime:
            to_embed.append((fpath, mtime, "modified"))

    # Find deleted files
    current_paths = {os.path.relpath(fp, CORPUS_DIR) for fp, _ in files}
    deleted = [p for p in existing_paths if p not in current_paths]

    if verbose:
        print(f"🧠 Collective Consciousness — Update Mode")
        print(f"   Total files: {total}")
        print(f"   New: {sum(1 for _,_,s in to_embed if s=='new')}")
        print(f"   Modified: {sum(1 for _,_,s in to_embed if s=='modified')}")
        print(f"   Deleted: {len(deleted)}")

    if not to_embed and not deleted:
        if verbose:
            print("   ✅ Everything up to date.")
        return store

    # Remove deleted
    if deleted:
        store["pieces"] = [p for p in store["pieces"] if p["path"] not in set(deleted)]
        if verbose:
            print(f"   🗑️  Removed {len(deleted)} deleted files")

    # Embed new/modified
    if to_embed:
        if verbose:
            print(f"   Embedding {len(to_embed)} files...")

        t0 = time.time()
        for i, (fpath, mtime, status) in enumerate(to_embed):
            meta = extract_metadata(fpath, CORPUS_DIR)
            try:
                content = open(fpath, "r", encoding="utf-8", errors="replace").read()
                embed_text = content[:2000] if content.strip() else (meta["title"] or os.path.basename(fpath))
                emb = embed(embed_text)
                if len(emb) != 768:
                    print(f"  ⚠️  Wrong dims ({len(emb)}) for {fpath}, skipping")
                    continue
            except Exception as e:
                print(f"  ⚠️  Failed: {fpath}: {e}")
                continue

            rel_path = os.path.relpath(fpath, CORPUS_DIR)
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

            # Update or append
            found = False
            for j, existing in enumerate(store["pieces"]):
                if existing["path"] == rel_path:
                    store["pieces"][j] = piece
                    found = True
                    break
            if not found:
                store["pieces"].append(piece)

            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1}/{len(to_embed)}] {meta['title'][:50]}")

        if verbose:
            print(f"   ✅ Embedded {len(to_embed)} files in {(time.time()-t0)/60:.1f}m")

    # Recompute all similarities (cheap relative to embedding)
    if store["pieces"]:
        if verbose:
            print("   🔗 Recomputing similarity matrix...")
        embeddings_list = [p["embedding"] for p in store["pieces"]]
        emb_array = np.array(embeddings_list)
        sim_matrix = cosine_similarity_matrix(emb_array)
        neighbors = compute_neighbors(sim_matrix)

        for i, piece in enumerate(store["pieces"]):
            piece["neighbors"] = neighbors[i]

    store["stats"] = compute_stats(store)
    save_store(store)

    if verbose:
        print(f"   💾 Updated {STORE_PATH}")
        print(f"   📊 {store['stats']['total_pieces']} pieces total")

    return store


def op_query(query: str, top_k: int = TOP_K):
    """Query the corpus for similar pieces."""
    store = load_store()
    if not store.get("pieces"):
        print("❌ No embeddings found. Run a full rebuild first.")
        return

    print(f"🔍 Query: \"{query}\"")
    print(f"   Embedding query...")

    query_emb = np.array(embed(query))
    pieces = store["pieces"]
    emb_matrix = np.array([p["embedding"] for p in pieces])

    # Cosine similarity
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    matrix_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1
    normalized = emb_matrix / matrix_norms
    sims = normalized @ query_norm

    top_indices = np.argsort(sims)[::-1][:top_k]

    print(f"\n   Top {top_k} matches:\n")
    for rank, idx in enumerate(top_indices, 1):
        p = pieces[idx]
        sim = sims[idx]
        print(f"   {rank:2d}. [{sim:.4f}] {p['title']}")
        print(f"       📁 {p['directory']} | {p['word_count']} words")
        print(f"       📄 {p['path']}")
        print(f"       \"{p['preview'][:120]}...\"")
        print()


def op_visualize():
    """Create a 2D projection of the corpus."""
    store = load_store()
    if not store.get("pieces"):
        print("❌ No embeddings found. Run a full rebuild first.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pieces = store["pieces"]
    embeddings = np.array([p["embedding"] for p in pieces])
    dirs = [p["directory"] for p in pieces]

    print(f"🎨 Visualizing {len(pieces)} pieces in {len(set(dirs))} directories...")

    # Try t-SNE first, fall back to PCA
    try:
        from sklearn.manifold import TSNE
        print("   Using t-SNE projection...")
        # PCA first to 50 dims, then t-SNE to 2 (standard pipeline for large datasets)
        pca = PCA(n_components=min(50, embeddings.shape[0], embeddings.shape[1]))
        reduced = pca.fit_transform(embeddings)
        tsne = TSNE(n_components=2, perplexity=min(30, len(pieces) - 1), random_state=42, max_iter=1000)
        coords = tsne.fit_transform(reduced)
        method = "t-SNE"
    except Exception as e:
        print(f"   t-SNE failed ({e}), falling back to PCA...")
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        method = "PCA"

    # Assign colors by directory
    unique_dirs = sorted(set(dirs))
    # Use a large colormap
    n_dirs = len(unique_dirs)
    if n_dirs <= 10:
        cmap = plt.cm.tab10
    elif n_dirs <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar

    dir_to_idx = {d: i for i, d in enumerate(unique_dirs)}
    colors = [dir_to_idx[d] for d in dirs]

    fig, ax = plt.subplots(1, 1, figsize=(24, 18))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap=cmap,
                         s=15, alpha=0.6, edgecolors="none")

    ax.set_title(f"The Collective Consciousness — {method} Projection\n"
                 f"{len(pieces)} pieces across {n_dirs} directories | 768 dimensions → 2",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel(f"{method} dimension 1", fontsize=12)
    ax.set_ylabel(f"{method} dimension 2", fontsize=12)

    # Legend with directory names (show top 20 most populous)
    from collections import Counter
    dir_counts = Counter(dirs)
    top_dirs = [d for d, _ in dir_counts.most_common(20)]

    # Create proxy artists for legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for d in top_dirs:
        idx = dir_to_idx[d]
        normalized_idx = idx / max(n_dirs - 1, 1)
        color = cmap(normalized_idx)
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=color, markersize=8,
                                       label=f"{d} ({dir_counts[d]})"))

    ax.legend(handles=legend_elements, loc="best", fontsize=8, framealpha=0.8,
              title="Directories", title_fontsize=9)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "consciousness_map.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n   ✅ Saved visualization to {output_path}")
    print(f"   Method: {method} | Points: {len(pieces)} | Directories: {n_dirs}")


def op_stats():
    """Print corpus statistics."""
    store = load_store()
    stats = store.get("stats", {})
    if not stats:
        stats = compute_stats(store)

    print("🧠 Collective Consciousness — Corpus Statistics")
    print("=" * 55)
    print(f"  Total pieces:     {stats.get('total_pieces', 0):,}")
    print(f"  Total words:      {stats.get('total_words', 0):,}")
    print(f"  Average words:    {stats.get('avg_words', 0):,}")
    print(f"  Directories:      {stats.get('directories', 0)}")
    print(f"  Dimensions:       {stats.get('dimensions', 768)}")
    print(f"  Model:            {stats.get('model', MODEL)}")
    print(f"  Last run:         {store.get('last_run', 'never')}")
    print()
    print("  Pieces by directory:")
    for d, count in sorted(stats.get("pieces_by_directory", {}).items(), key=lambda x: -x[1])[:20]:
        bar = "█" * min(count // 5, 40)
        print(f"    {d:40s} {count:5d}  {bar}")
    remaining = len(stats.get("pieces_by_directory", {})) - 20
    if remaining > 0:
        print(f"    ... and {remaining} more directories")


# ── CLI ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="The Collective Consciousness — Vectorized ai-writings corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 vectorize.py                          # Full rebuild
  python3 vectorize.py --update                 # Only new/modified files  
  python3 vectorize.py --query "the stick the dog the storm"
  python3 vectorize.py --visualize
  python3 vectorize.py --stats
        """
    )
    parser.add_argument("--update", action="store_true", help="Only embed new/modified files")
    parser.add_argument("--query", type=str, help="Search the corpus")
    parser.add_argument("--visualize", action="store_true", help="Generate 2D projection")
    parser.add_argument("--stats", action="store_true", help="Print corpus statistics")
    parser.add_argument("--top", type=int, default=TOP_K, help=f"Number of results for query (default: {TOP_K})")

    args = parser.parse_args()

    if args.stats:
        op_stats()
    elif args.query:
        op_query(args.query, args.top)
    elif args.visualize:
        op_visualize()
    elif args.update:
        op_update()
    else:
        op_full_rebuild()


if __name__ == "__main__":
    main()
