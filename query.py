#!/usr/bin/env python3
"""
Query Cloudflare Vectorize — Semantic Search
=============================================
Embeds a query locally via nomic-embed-text, then searches
Cloudflare Vectorize for the closest matches in the ai-writings corpus.

Usage:
  python3 query.py "the dog narrator governs reality"
  python3 query.py "loss and memory in winter" --top 5
  python3 query.py "what is consciousness?" --json
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ── Configuration ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ACCOUNT_ID = "049ff5e84ecf636b53b162cbb580aae6"
INDEX_NAME = "ai-writings"
QUERY_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/vectorize/v2/indexes/{INDEX_NAME}/query"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

# ANSI colors for terminal output
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ── Credentials ────────────────────────────────────────────────
def get_token():
    """Get Cloudflare API token from wrangler OAuth config or env."""
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_TOKEN")
    if token:
        return token

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

    raise RuntimeError("No Cloudflare API token found.")


# ── Local Embedding ────────────────────────────────────────────
def embed_query(text: str) -> list[float]:
    """Embed a query string using local Ollama nomic-embed-text."""
    payload = json.dumps({"model": MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["embedding"]


# ── Vectorize Query ────────────────────────────────────────────
def query_vectorize(token: str, query_embedding: list[float], top_k: int = 10) -> dict:
    """Query Cloudflare Vectorize with an embedding vector."""
    payload = json.dumps({
        "vector": query_embedding,
        "topK": top_k,
        "returnMetadata": "all",
        "returnValues": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        QUERY_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Display ────────────────────────────────────────────────────
def format_score(score: float) -> str:
    """Format similarity score with color coding."""
    if score >= 0.75:
        return f"{GREEN}{score:.4f}{RESET}"
    elif score >= 0.5:
        return f"{YELLOW}{score:.4f}{RESET}"
    else:
        return f"{DIM}{score:.4f}{RESET}"


def display_results(query: str, results: list, json_output: bool = False):
    """Display query results in a readable format."""
    if json_output:
        print(json.dumps({
            "query": query,
            "count": len(results),
            "results": [
                {
                    "score": r.get("score", 0),
                    "id": r.get("id", ""),
                    **r.get("metadata", {}),
                }
                for r in results
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False))
        return

    print(f"\n{BOLD}🔍 Query:{RESET} \"{query}\"")
    print(f"{DIM}{'─' * 70}{RESET}\n")

    if not results:
        print(f"  {DIM}No results found.{RESET}")
        return

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        score = r.get("score", 0)
        title = meta.get("title", "Untitled")
        path = meta.get("path", "?")
        directory = meta.get("directory", "?")
        word_count = meta.get("word_count", 0)
        preview = meta.get("preview", "")

        score_str = format_score(score)

        print(f"  {BOLD}{i:2d}.{RESET} [{score_str}] {BOLD}{title}{RESET}")
        print(f"      {DIM}📁{RESET} {directory}  {DIM}📝{RESET} {word_count:,} words  {DIM}📄{RESET} {path}")
        if preview:
            # Truncate and clean preview
            preview = preview.replace("\n", " ").strip()
            if len(preview) > 150:
                preview = preview[:150] + "..."
            print(f"      {DIM}\"{preview}\"{RESET}")
        print()


# ── CLI ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Semantic search of ai-writings via Cloudflare Vectorize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 query.py "the dog narrator governs reality"
  python3 query.py "loss and memory in winter" --top 5
  python3 query.py "what is consciousness?" --json
        """,
    )
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        token = get_token()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"{DIM}Embedding query locally via {MODEL}...{RESET}", file=sys.stderr)
    query_emb = embed_query(args.query)

    print(f"{DIM}Querying Cloudflare Vectorize ({INDEX_NAME})...{RESET}", file=sys.stderr)
    response = query_vectorize(token, query_emb, args.top)

    if not response.get("success"):
        print(f"❌ Query failed: {response.get('errors', [])}")
        sys.exit(1)

    results = response.get("result", {}).get("matches", [])
    display_results(args.query, results, json_output=args.json)


if __name__ == "__main__":
    main()
