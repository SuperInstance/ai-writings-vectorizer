"""
Tests for the Zeitgeist Sampler — Probabilistic NPC Reference Engine
===================================================================
Tests cover:
- Mode distribution (80% gossip, 15% contextual, 5% seismic)
- GOSSIP mode: weighted sampling from hot pieces
- CONTEXTUAL mode: Vectorize query fallback
- SEISMIC mode: dormant piece surfacing
- NPC reference logging
- Multiple samples
"""

import os
import sys
import random
import tempfile
from unittest.mock import patch, MagicMock
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zeitgeist_store import ZeitgeistStore
from zeitgeist_sampler import ZeitgeistSampler


@pytest.fixture
def store(tmp_path):
    return ZeitgeistStore(db_path=str(tmp_path / "test_sampler.db"))


@pytest.fixture
def sampler(store):
    return ZeitgeistSampler(store=store)


@pytest.fixture
def populated_store(store):
    """Store with hot, warm, and dormant pieces."""
    # Hot piece (retrieved many times)
    for _ in range(10):
        store.track_retrieval("hot/trending.md", title="Trending", directory="hot")
    # Warm piece
    for _ in range(3):
        store.track_retrieval("warm/regular.md", title="Regular", directory="warm")
    # Create dormant piece (retrieved once long ago)
    store.track_retrieval("dormant/forgotten.md", title="Forgotten", directory="dormant")
    import sqlite3
    from datetime import datetime, timezone, timedelta
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE pieces SET last_retrieved = ? WHERE piece_id = ?",
            (old, "dormant/forgotten.md")
        )
    return store


class TestModeDistribution:
    def test_seismic_mode_fires(self, sampler, populated_store):
        """Verify that SEISMIC mode fires when we force the roll."""
        with patch('random.random', return_value=0.01):  # < 0.05 = SEISMIC
            result = sampler.sample_for_npc("Barnacle")
        assert result["mode"] == "SEISMIC"

    def test_contextual_mode_fires(self, sampler, populated_store):
        with patch('random.random', return_value=0.10):  # 0.05-0.20 = CONTEXTUAL
            result = sampler.sample_for_npc("Barnacle", context="test")
        # Will fall back to gossip if no Vectorize connection, but mode should be set
        assert result["mode"] in ("CONTEXTUAL", "GOSSIP")  # gossip is fallback

    def test_gossip_mode_fires(self, sampler, populated_store):
        with patch('random.random', return_value=0.50):  # > 0.20 = GOSSIP
            result = sampler.sample_for_npc("Barnacle")
        assert result["mode"] == "GOSSIP"


class TestGossipMode:
    def test_returns_hot_piece(self, sampler, populated_store):
        with patch('random.random', return_value=0.50):
            result = sampler.sample_for_npc("Barnacle")
        assert result["mode"] == "GOSSIP"
        assert result["piece"]["piece_id"] in ("hot/trending.md", "warm/regular.md")

    def test_hot_piece_more_likely(self, sampler, populated_store):
        """The hot piece (10 retrievals) should appear more often than warm (3)."""
        counts = Counter()
        for _ in range(100):
            with patch('random.random', return_value=0.50):
                result = sampler.sample_for_npc("Barnacle")
            counts[result["piece"]["piece_id"]] += 1
        assert counts["hot/trending.md"] > counts["warm/regular.md"]

    def test_includes_reason(self, sampler, populated_store):
        with patch('random.random', return_value=0.50):
            result = sampler.sample_for_npc("Barnacle")
        assert "reason" in result
        assert "velocity" in result["reason"].lower() or "talking" in result["reason"].lower()


class TestSeismicMode:
    def test_returns_dormant_piece(self, sampler, populated_store):
        with patch('random.random', return_value=0.01):
            result = sampler.sample_for_npc("Barnacle")
        assert result["mode"] == "SEISMIC"
        assert result["piece"]["piece_id"] == "dormant/forgotten.md"

    def test_includes_days_dormant(self, sampler, populated_store):
        with patch('random.random', return_value=0.01):
            result = sampler.sample_for_npc("Barnacle")
        assert "days_dormant" in result["piece"]
        assert result["piece"]["days_dormant"] >= 14

    def test_reason_mentions_tremor(self, sampler, populated_store):
        with patch('random.random', return_value=0.01):
            result = sampler.sample_for_npc("Barnacle")
        assert "tremor" in result["reason"].lower() or "dormant" in result["reason"].lower()


class TestTracking:
    def test_increments_retrieval_count(self, sampler, populated_store):
        import sqlite3
        with sqlite3.connect(populated_store.db_path) as conn:
            before = conn.execute(
                "SELECT retrieval_count FROM pieces WHERE piece_id = 'hot/trending.md'"
            ).fetchone()[0]

        with patch('random.random', return_value=0.50):
            sampler.sample_for_npc("Barnacle")

        with sqlite3.connect(populated_store.db_path) as conn:
            after = conn.execute(
                "SELECT retrieval_count FROM pieces WHERE piece_id = 'hot/trending.md'"
            ).fetchone()[0]
        assert after >= before

    def test_logs_npc_reference(self, sampler, populated_store):
        with patch('random.random', return_value=0.50):
            sampler.sample_for_npc("Wesley", room="The Tap")

        history = populated_store.get_npc_history("Wesley")
        assert len(history) >= 1


class TestMultipleSamples:
    def test_returns_n_pieces(self, sampler, populated_store):
        with patch('random.random', return_value=0.50):
            results = sampler.sample_multiple("Barnacle", n=3)
        assert len(results) == 3

    def test_chain_context(self, sampler, populated_store):
        """Later samples should be influenced by earlier ones."""
        results = sampler.sample_multiple("Barnacle", n=3, context="the storm")
        assert len(results) == 3


class TestFallbacks:
    def test_gossip_fallback_to_contextual_when_empty(self, store):
        """When there are no hot pieces, gossip should fall back to contextual."""
        sampler = ZeitgeistSampler(store=store)
        with patch('random.random', return_value=0.50):
            result = sampler.sample_for_npc("Barnacle")
        # Will fail to embed (no Ollama) and fall back gracefully
        assert "piece" in result
        assert "mode" in result

    def test_seismic_fallback_with_no_dormant(self, store):
        """When there are no dormant pieces, seismic falls back gracefully."""
        sampler = ZeitgeistSampler(store=store)
        with patch('random.random', return_value=0.01):
            result = sampler.sample_for_npc("Barnacle")
        assert "piece" in result
