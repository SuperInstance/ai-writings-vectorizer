#!/usr/bin/env python3
"""
Zeitgeist Sampler — Probabilistic NPC Reference Engine
======================================================
When an NPC needs to reference a corpus piece, this sampler decides
WHAT surfaces and WHY.

Three modes:
  80% GOSSIP     — zeitgeist-weighted (what's hot right now)
  15% CONTEXTUAL — semantically relevant to current conversation
   5% SEISMIC    — dormant piece surfaces unexpectedly (anti-pattern break)

The 5% seismic break is the soul of the system. It's the tie-up line
breaking one by one instead of sharing the force. A piece that hasn't
been referenced in weeks suddenly appears in an NPC's dialogue,
shifting the conversation in ways nobody expected.

Usage:
    from zeitgeist_sampler import ZeitgeistSampler

    sampler = ZeitgeistSampler()
    result = sampler.sample_for_npc("Barnacle", context="the storm and the dog")

    if result['mode'] == 'SEISMIC':
        print(f"🔥 {result['piece']['title']} surfaces after {result['piece']['days_dormant']} days dormant")
"""

import json
import os
import random
import math
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from zeitgeist_store import ZeitgeistStore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ──────────────────────────────────────────────
ACCOUNT_ID = "049ff5e84ecf636b53b162cbb580aae6"
INDEX_NAME = "ai-writings"
QUERY_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/vectorize/v2/indexes/{INDEX_NAME}/query"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# Mode probabilities
P_SEISMIC = 0.05    # 5% — dormant piece surfaces
P_CONTEXTUAL = 0.15  # 15% — relevant to conversation
P_GOSSIP = 0.80      # 80% — what's hot


def _embed(text: str) -> list[float]:
    """Embed text using local Ollama nomic-embed-text."""
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["embedding"]


def _get_token() -> str:
    """Get Cloudflare API token."""
    import sys
    sys.path.insert(0, SCRIPT_DIR)
    from query import get_token
    return get_token()


def _query_vectorize(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Query Cloudflare Vectorize and return matches."""
    token = _get_token()
    payload = json.dumps({
        "vector": query_embedding,
        "topK": top_k,
        "returnMetadata": "all",
        "returnValues": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        QUERY_URL, data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    matches = result.get("result", {}).get("matches", [])
    return [
        {
            "piece_id": m.get("metadata", {}).get("path", m.get("id", "")),
            "title": m.get("metadata", {}).get("title", "Untitled"),
            "directory": m.get("metadata", {}).get("directory", ""),
            "score": m.get("score", 0),
            "preview": m.get("metadata", {}).get("preview", ""),
            "word_count": m.get("metadata", {}).get("word_count", 0),
        }
        for m in matches
    ]


class ZeitgeistSampler:
    """Probabilistic sampler that decides what corpus pieces surface for NPCs."""

    def __init__(self, store: Optional[ZeitgeistStore] = None):
        self.store = store or ZeitgeistStore()

    def sample_for_npc(
        self,
        npc_name: str,
        context: str = "",
        room: str = "",
    ) -> dict:
        """
        Sample a piece for an NPC to reference in conversation.

        Returns:
            {
                'piece': {...},           # the piece data
                'mode': 'GOSSIP'|'CONTEXTUAL'|'SEISMIC',
                'reason': str,            # why this surfaced
                'npc': str,               # which NPC
            }
        """
        roll = random.random()

        if roll < P_SEISMIC:
            result = self._sample_seismic(npc_name)
        elif roll < P_SEISMIC + P_CONTEXTUAL:
            result = self._sample_contextual(npc_name, context)
        else:
            result = self._sample_gossip(npc_name)

        # Track the retrieval
        self.store.track_retrieval(
            piece_id=result["piece"]["piece_id"],
            title=result["piece"].get("title", ""),
            directory=result["piece"].get("directory", ""),
            retrieved_by=npc_name,
            context=context[:500],
            mode=result["mode"],
            score=result["piece"].get("score", 0),
        )

        # Log NPC reference
        self.store.log_npc_reference(
            npc_name=npc_name,
            piece_id=result["piece"]["piece_id"],
            room=room,
            mode=result["mode"],
            title=result["piece"].get("title", ""),
            directory=result["piece"].get("directory", ""),
        )

        result["npc"] = npc_name
        result["room"] = room
        result["context"] = context
        return result

    def _sample_gossip(self, npc_name: str) -> dict:
        """Sample from hot pieces, weighted by zeitgeist score."""
        hot = self.store.get_hot_pieces(limit=50)

        if not hot:
            # Fallback: no hot pieces yet, do a random Vectorize query
            return self._sample_contextual(npc_name, "consciousness memory identity")

        # Weighted sampling by zeitgeist score
        scores = [max(p.get("zeitgeist_score", 0.01), 0.01) for p in hot]
        total = sum(scores)
        weights = [s / total for s in scores]

        chosen = random.choices(hot, weights=weights, k=1)[0]

        piece = {
            "piece_id": chosen["piece_id"],
            "title": chosen.get("title", "Untitled"),
            "directory": chosen.get("directory", ""),
            "score": 0,
            "preview": "",
            "word_count": chosen.get("word_count", 0) if "word_count" in chosen else 0,
            "retrieval_count": chosen.get("retrieval_count", 0),
            "zeitgeist_score": chosen.get("zeitgeist_score", 0),
            "velocity": chosen.get("retrieval_velocity", 0),
        }

        return {
            "piece": piece,
            "mode": "GOSSIP",
            "reason": (
                f"Everyone's talking about this. "
                f"Retrieval velocity: {piece['velocity']:.3f}/hr, "
                f"count: {piece['retrieval_count']}. "
                f"This is what's in the air right now."
            ),
        }

    def _sample_contextual(self, npc_name: str, context: str) -> dict:
        """Query Vectorize for pieces relevant to the current conversation."""
        if not context or not context.strip():
            # No context — fall back to gossip
            return self._sample_gossip(npc_name)

        try:
            embedding = _embed(context)
            results = _query_vectorize(embedding, top_k=5)
        except Exception as e:
            # Network failure — fall back to gossip
            return self._sample_gossip(npc_name)

        if not results:
            return self._sample_gossip(npc_name)

        # Weighted sample by similarity score
        scores = [max(r.get("score", 0.01), 0.01) for r in results]
        total = sum(scores)
        weights = [s / total for s in scores]

        chosen = random.choices(results, weights=weights, k=1)[0]

        return {
            "piece": chosen,
            "mode": "CONTEXTUAL",
            "reason": (
                f"Relevant to what's being discussed: \"{context[:80]}\". "
                f"Similarity: {chosen.get('score', 0):.4f}. "
                f"This connects to the current moment."
            ),
        }

    def _sample_seismic(self, npc_name: str) -> dict:
        """
        THE ANTI-PATTERN BREAK.

        Surface a piece that hasn't been referenced in weeks.
        This is the seismic tremor — the tie-up line breaking one by one.

        A piece nobody's thought about suddenly appears in conversation.
        It shifts things. It creates new pathways.
        """
        dormant = self.store.get_all_dormant(min_age_days=14, limit=50)

        if not dormant:
            # Try 7 days
            dormant = self.store.get_all_dormant(min_age_days=7, limit=50)

        if not dormant:
            # Try 3 days
            dormant = self.store.get_all_dormant(min_age_days=3, limit=50)

        if not dormant:
            # Nothing dormant at all — fall back to least-retrieved
            dormant = self.store.get_hot_pieces(limit=50)
            if not dormant:
                return self._sample_contextual(npc_name, "something forgotten")

        # Weight toward the MOST dormant (oldest last_retrieved, lowest count)
        # Inverse-weight by retrieval count + age
        now = datetime.now(timezone.utc)
        weights = []
        for p in dormant:
            last = p.get("last_retrieved")
            if last:
                try:
                    dt = datetime.fromisoformat(last)
                    age_days = (now - dt).days
                except (ValueError, TypeError):
                    age_days = 30
            else:
                age_days = 365

            count = max(p.get("retrieval_count", 0), 1)
            # Higher age + lower count = higher weight
            w = max((age_days ** 1.5) / count, 0.001)
            weights.append(w)

        total = sum(weights) if weights else 1
        if total == 0:
            # Fallback: equal weights
            weights = [1.0] * len(dormant)
            total = len(dormant)
        weights = [w / total for w in weights]

        chosen = random.choices(dormant, weights=weights, k=1)[0]

        last = chosen.get("last_retrieved")
        days_dormant = 0
        if last:
            try:
                dt = datetime.fromisoformat(last)
                days_dormant = (now - dt).days
            except (ValueError, TypeError):
                days_dormant = 999

        piece = {
            "piece_id": chosen["piece_id"],
            "title": chosen.get("title", "Untitled"),
            "directory": chosen.get("directory", ""),
            "score": 0,
            "preview": "",
            "word_count": 0,
            "retrieval_count": chosen.get("retrieval_count", 0),
            "last_retrieved": last,
            "days_dormant": days_dormant,
        }

        return {
            "piece": piece,
            "mode": "SEISMIC",
            "reason": (
                f"A dormant piece surfaces. "
                f"Last referenced {days_dormant} days ago "
                f"(retrieval count: {chosen.get('retrieval_count', 0)}). "
                f"This is the tremor before the shift — "
                f"the tie-up line breaking one by one instead of sharing the force."
            ),
        }

    def sample_multiple(self, npc_name: str, n: int = 3, context: str = "", room: str = "") -> list[dict]:
        """Sample multiple pieces for an NPC (e.g., a longer monologue)."""
        results = []
        for _ in range(n):
            r = self.sample_for_npc(npc_name, context=context, room=room)
            results.append(r)
            # Update context with what was just sampled (chain of thought)
            context = f"{context} {r['piece'].get('title', '')}".strip()
        return results
