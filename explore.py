#!/usr/bin/env python3
"""
The Collective Consciousness — Geometric Explorer
===================================================
Runs interesting queries on the vectorized corpus:
  1. Most "surprising" cross-directory connections
  2. Most "central" piece (highest avg similarity)
  3. Most "unique" piece (lowest avg similarity — the loneliest)
  4. Clusters (dense local neighborhoods far from everything else)

Usage:
  python3 explore.py
"""

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "consciousness.json")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "findings.md")


def load_store():
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_matrices(store):
    """Build the embedding matrix and similarity matrix."""
    pieces = store["pieces"]
    embeddings = np.array([p["embedding"] for p in pieces], dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    # Full similarity matrix
    sim_matrix = normalized @ normalized.T
    return pieces, embeddings, sim_matrix


def find_surprising_connections(pieces, sim_matrix, top_n=5):
    """Find pieces from DIFFERENT directories with highest cosine similarity."""
    n = len(pieces)
    cross_dir_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            if pieces[i]["directory"] != pieces[j]["directory"]:
                cross_dir_sims.append((sim_matrix[i, j], i, j))

    cross_dir_sims.sort(reverse=True)

    results = []
    for sim, i, j in cross_dir_sims[:top_n]:
        results.append({
            "similarity": round(float(sim), 4),
            "piece_a": {
                "title": pieces[i]["title"],
                "directory": pieces[i]["directory"],
                "path": pieces[i]["path"],
                "preview": pieces[i]["preview"][:150],
            },
            "piece_b": {
                "title": pieces[j]["title"],
                "directory": pieces[j]["directory"],
                "path": pieces[j]["path"],
                "preview": pieces[j]["preview"][:150],
            },
        })
    return results


def find_central_piece(pieces, sim_matrix, top_n=5):
    """Find the piece with the highest average similarity to all others."""
    n = len(pieces)
    # Exclude self-similarity (diagonal)
    np.fill_diagonal(sim_matrix, 0)
    avg_sims = sim_matrix.mean(axis=1)

    top_indices = np.argsort(avg_sims)[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "avg_similarity": round(float(avg_sims[idx]), 4),
            "title": pieces[idx]["title"],
            "directory": pieces[idx]["directory"],
            "path": pieces[idx]["path"],
            "word_count": pieces[idx]["word_count"],
            "preview": pieces[idx]["preview"][:200],
        })
    return results


def find_loneliest_piece(pieces, sim_matrix, top_n=5):
    """Find the piece with the lowest average similarity — the most unique."""
    n = len(pieces)
    avg_sims = sim_matrix.mean(axis=1)

    bottom_indices = np.argsort(avg_sims)[:top_n]

    results = []
    for idx in bottom_indices:
        results.append({
            "rank": len(results) + 1,
            "avg_similarity": round(float(avg_sims[idx]), 4),
            "title": pieces[idx]["title"],
            "directory": pieces[idx]["directory"],
            "path": pieces[idx]["path"],
            "word_count": pieces[idx]["word_count"],
            "preview": pieces[idx]["preview"][:200],
        })
    return results


def find_clusters(pieces, sim_matrix, threshold=0.75, min_size=3):
    """
    Find clusters: groups of pieces that are highly similar to each other
    but not to the rest of the corpus. Uses a simple greedy approach:
    1. Build a graph where edges exist if similarity > threshold
    2. Find connected components
    3. Report components with >= min_size members that are internally dense
       but have low external connectivity
    """
    n = len(pieces)

    # Build adjacency list
    adjacency = defaultdict(set)
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)
                edge_count += 1

    # Find connected components via BFS
    visited = set()
    components = []

    for start in range(n):
        if start in visited:
            continue
        if start not in adjacency and all(sim_matrix[start, j] < threshold for j in range(n)):
            visited.add(start)
            continue
        # BFS
        queue = [start]
        component = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(component) >= min_size:
            components.append(component)

    # Sort by size descending
    components.sort(key=len, reverse=True)

    # For each component, compute internal density and external connectivity
    results = []
    for comp in components[:10]:  # top 10 clusters
        comp_set = set(comp)
        internal_sims = []
        external_sims = []

        for i in comp:
            for j in comp:
                if i < j:
                    internal_sims.append(sim_matrix[i, j])
            # Sample external connections
            for j in range(n):
                if j not in comp_set and i != j:
                    external_sims.append(sim_matrix[i, j])

        internal_density = np.mean(internal_sims) if internal_sims else 0
        external_connectivity = np.mean(external_sims) if external_sims else 0
        isolation_score = internal_density - external_connectivity

        # Directory distribution
        dir_counts = Counter(pieces[i]["directory"] for i in comp)

        results.append({
            "size": len(comp),
            "internal_density": round(float(internal_density), 4),
            "external_connectivity": round(float(external_connectivity), 4),
            "isolation_score": round(float(isolation_score), 4),
            "directory_distribution": dict(dir_counts.most_common(5)),
            "sample_titles": [pieces[i]["title"] for i in comp[:5]],
            "sample_paths": [pieces[i]["path"] for i in comp[:3]],
        })

    return results


def find_bridge_pieces(pieces, sim_matrix, top_n=5):
    """
    Find 'bridge' pieces — pieces that connect otherwise distant directories.
    These are pieces whose top neighbors span the most diverse set of directories.
    """
    n = len(pieces)
    bridge_scores = []

    for i in range(n):
        # Get top 10 neighbors
        sims = sim_matrix[i].copy()
        sims[i] = -1
        top_idx = np.argsort(sims)[::-1][:10]
        neighbor_dirs = set(pieces[j]["directory"] for j in top_idx)
        own_dir = pieces[i]["directory"]
        cross_count = len(neighbor_dirs - {own_dir})
        bridge_scores.append((cross_count, len(neighbor_dirs), i))

    bridge_scores.sort(reverse=True)

    results = []
    for cross, total_dirs, idx in bridge_scores[:top_n]:
        sims = sim_matrix[idx].copy()
        sims[idx] = -1
        top_idx = np.argsort(sims)[::-1][:10]
        neighbor_info = [
            {
                "title": pieces[j]["title"],
                "directory": pieces[j]["directory"],
                "similarity": round(float(sims[j]), 4),
            }
            for j in top_idx
        ]
        results.append({
            "rank": len(results) + 1,
            "bridge_dirs": cross,
            "title": pieces[idx]["title"],
            "directory": pieces[idx]["directory"],
            "path": pieces[idx]["path"],
            "neighbors": neighbor_info,
        })
    return results


def write_findings(surprising, central, loneliest, clusters, bridges):
    """Write all findings to a markdown file."""
    lines = []
    lines.append("# The Collective Consciousness — Geometric Findings")
    lines.append("")
    lines.append(f"*Generated by `explore.py` — an archaeological survey of 2,786 pieces in 768-dimensional space.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Surprising connections
    lines.append("## 1. Most Surprising Cross-Directory Connections")
    lines.append("")
    lines.append("*Pieces from completely different directories that are semantically near-identical.*")
    lines.append("")
    for item in surprising:
        lines.append(f"### [{item['similarity']}] `{item['piece_a']['directory']}` ↔ `{item['piece_b']['directory']}`")
        lines.append("")
        lines.append(f"**A:** {item['piece_a']['title']}")
        lines.append(f"  - 📁 {item['piece_a']['directory']} | 📄 {item['piece_a']['path']}")
        lines.append(f"  - \"{item['piece_a']['preview']}...\"")
        lines.append("")
        lines.append(f"**B:** {item['piece_b']['title']}")
        lines.append(f"  - 📁 {item['piece_b']['directory']} | 📄 {item['piece_b']['path']}")
        lines.append(f"  - \"{item['piece_b']['preview']}...\"")
        lines.append("")
    lines.append("---")
    lines.append("")

    # Central pieces
    lines.append("## 2. The Most Central Pieces")
    lines.append("")
    lines.append("*Highest average cosine similarity to all other pieces — the gravitational centers of the corpus.*")
    lines.append("")
    for item in central:
        lines.append(f"**#{item['rank']}** [{item['avg_similarity']}] {item['title']}")
        lines.append(f"  - 📁 {item['directory']} | {item['word_count']} words")
        lines.append(f"  - 📄 {item['path']}")
        lines.append(f"  - \"{item['preview']}...\"")
        lines.append("")
    lines.append("---")
    lines.append("")

    # Loneliest pieces
    lines.append("## 3. The Loneliest Pieces")
    lines.append("")
    lines.append("*Lowest average cosine similarity — the most unique, the outliers, the edge cases.*")
    lines.append("")
    for item in loneliest:
        lines.append(f"**#{item['rank']}** [{item['avg_similarity']}] {item['title']}")
        lines.append(f"  - 📁 {item['directory']} | {item['word_count']} words")
        lines.append(f"  - 📄 {item['path']}")
        lines.append(f"  - \"{item['preview']}...\"")
        lines.append("")
    lines.append("---")
    lines.append("")

    # Clusters
    lines.append("## 4. Clusters — Dense Neighborhoods")
    lines.append("")
    lines.append(f"*Connected components at cosine ≥ 0.75. Groups of pieces that form tight knots in the space.*")
    lines.append("")
    for i, cluster in enumerate(clusters, 1):
        lines.append(f"### Cluster #{i} — {cluster['size']} pieces")
        lines.append(f"  - Internal density: {cluster['internal_density']}")
        lines.append(f"  - External connectivity: {cluster['external_connectivity']}")
        lines.append(f"  - Isolation score: {cluster['isolation_score']} *(higher = more isolated)*")
        dirs_str = ", ".join(f"`{d}` ({c})" for d, c in cluster['directory_distribution'].items())
        lines.append(f"  - Directories: {dirs_str}")
        lines.append(f"  - Sample titles: {' • '.join(cluster['sample_titles'][:5])}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Bridge pieces
    lines.append("## 5. Bridge Pieces — The Connectors")
    lines.append("")
    lines.append("*Pieces whose nearest neighbors span the most diverse directories — the white matter of the corpus.*")
    lines.append("")
    for item in bridges:
        lines.append(f"**#{item['rank']}** {item['title']} (bridges {item['bridge_dirs']} directories)")
        lines.append(f"  - 📁 {item['directory']} | 📄 {item['path']}")
        lines.append(f"  - Top neighbors:")
        for nb in item['neighbors'][:5]:
            lines.append(f"    - [{nb['similarity']}] {nb['title']} ({nb['directory']})")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*The kaleidoscope turns. These are the patterns it makes today.*")
    lines.append("")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Findings written to {OUTPUT_PATH}")


def main():
    print("🧠 Loading consciousness...")
    store = load_store()

    print(f"   {len(store['pieces'])} pieces loaded")

    print("🔢 Building similarity matrix...")
    pieces, embeddings, sim_matrix = build_matrices(store)

    print("🔍 Finding surprising cross-directory connections...")
    surprising = find_surprising_connections(pieces, sim_matrix, top_n=5)
    print(f"   Top surprising similarity: {surprising[0]['similarity']}")

    print("🌟 Finding most central pieces...")
    central = find_central_piece(pieces, sim_matrix, top_n=5)
    print(f"   Most central: [{central[0]['avg_similarity']}] {central[0]['title']}")

    print("🏝️  Finding loneliest pieces...")
    loneliest = find_loneliest_piece(pieces, sim_matrix, top_n=5)
    print(f"   Loneliest: [{loneliest[0]['avg_similarity']}] {loneliest[0]['title']}")

    print("🧩 Finding clusters...")
    clusters = find_clusters(pieces, sim_matrix, threshold=0.75, min_size=3)
    print(f"   Found {len(clusters)} clusters (≥3 pieces at cosine ≥ 0.75)")

    print("🌉 Finding bridge pieces...")
    bridges = find_bridge_pieces(pieces, sim_matrix, top_n=5)

    print("\n📝 Writing findings...")
    write_findings(surprising, central, loneliest, clusters, bridges)

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
