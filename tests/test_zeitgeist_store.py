"""
Tests for the Zeitgeist Store — Retrieval Frequency Tracker
==========================================================
Tests cover:
- DB initialization
- track_retrieval (increments, timestamps, velocity)
- get_hot_pieces, get_dormant_pieces
- get_seismic_events
- NPC reference logging
- Propagation logging
- Behavioral event logging
- Stats aggregation
"""

import os
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zeitgeist_store import ZeitgeistStore


@pytest.fixture
def store(tmp_path):
    """Fresh ZeitgeistStore with temp DB."""
    db_path = str(tmp_path / "test_zeitgeist.db")
    return ZeitgeistStore(db_path=db_path)


class TestInit:
    def test_creates_db(self, store):
        assert os.path.exists(store.db_path)

    def test_tables_exist(self, store):
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "pieces" in tables
        assert "retrieval_log" in tables
        assert "npc_reference_log" in tables
        assert "propagation_log" in tables
        assert "behavioral_events" in tables


class TestTrackRetrieval:
    def test_basic_track(self, store):
        store.track_retrieval("test/piece.md", title="Test", directory="test")
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM pieces WHERE piece_id = ?", ("test/piece.md",)
            ).fetchone()
        assert row is not None
        assert row[3] == 1  # retrieval_count

    def test_increments_count(self, store):
        store.track_retrieval("test/piece.md")
        store.track_retrieval("test/piece.md")
        store.track_retrieval("test/piece.md")
        stats = store.get_piece_stats()
        assert stats["total_retrievals"] == 3

    def test_logs_retrieval(self, store):
        store.track_retrieval("test/piece.md", retrieved_by="Barnacle", mode="GOSSIP")
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM retrieval_log WHERE piece_id = ?", ("test/piece.md",)
            ).fetchall()
        assert len(rows) == 1

    def test_updates_metadata(self, store):
        store.track_retrieval("test/piece.md", title="My Title", directory="essays")
        hot = store.get_hot_pieces(limit=1)
        assert len(hot) == 1
        assert hot[0]["title"] == "My Title"
        assert hot[0]["directory"] == "essays"

    def test_sets_timestamps(self, store):
        store.track_retrieval("test/piece.md")
        hot = store.get_hot_pieces(limit=1)
        assert hot[0]["first_retrieved"] is not None
        assert hot[0]["last_retrieved"] is not None

    def test_zeitgeist_score_positive(self, store):
        store.track_retrieval("test/piece.md")
        store.track_retrieval("test/piece.md")
        hot = store.get_hot_pieces(limit=1)
        assert hot[0]["zeitgeist_score"] > 0

    def test_velocity_stays_low_for_rare_pieces(self, store):
        store.track_retrieval("rare.md")
        hot = store.get_hot_pieces(limit=1)
        assert hot[0]["retrieval_velocity"] <= 1.0  # at most 1 per hour


class TestHotPieces:
    def test_empty_returns_empty(self, store):
        assert store.get_hot_pieces() == []

    def test_orders_by_zeitgeist(self, store):
        # Piece A: retrieved many times
        for _ in range(10):
            store.track_retrieval("hot/piece-a.md", title="A")
            time.sleep(0.01)
        # Piece B: retrieved once
        store.track_retrieval("cold/piece-b.md", title="B")

        hot = store.get_hot_pieces(limit=2)
        assert hot[0]["piece_id"] == "hot/piece-a.md"
        assert hot[0]["retrieval_count"] == 10

    def test_respects_limit(self, store):
        for i in range(15):
            store.track_retrieval(f"piece-{i}.md")
        hot = store.get_hot_pieces(limit=5)
        assert len(hot) == 5


class TestDormantPieces:
    def test_no_dormant_initially(self, store):
        dormant = store.get_dormant_pieces(min_age_days=14)
        assert dormant == []

    def test_finds_old_pieces(self, store):
        # Insert a piece with old timestamp
        store.track_retrieval("old/piece.md")
        import sqlite3
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with sqlite3.connect(store.db_path) as conn:
            conn.execute(
                "UPDATE pieces SET last_retrieved = ? WHERE piece_id = ?",
                (old_time, "old/piece.md")
            )
        dormant = store.get_dormant_pieces(min_age_days=14, limit=5)
        assert len(dormant) >= 1
        assert dormant[0]["piece_id"] == "old/piece.md"

    def test_get_all_dormant(self, store):
        store.track_retrieval("a.md")
        store.track_retrieval("b.md")
        import sqlite3
        old_time = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        with sqlite3.connect(store.db_path) as conn:
            conn.execute("UPDATE pieces SET last_retrieved = ?", (old_time,))
        dormant = store.get_all_dormant(min_age_days=14, limit=50)
        assert len(dormant) == 2


class TestSeismicEvents:
    def test_logs_seismic(self, store):
        store.track_retrieval("seismic.md", mode="SEISMIC")
        events = store.get_seismic_events()
        assert len(events) == 1
        assert events[0]["mode"] == "SEISMIC"

    def test_orders_by_recency(self, store):
        store.track_retrieval("old.md", mode="SEISMIC")
        time.sleep(0.05)
        store.track_retrieval("new.md", mode="SEISMIC")
        events = store.get_seismic_events(limit=2)
        assert events[0]["piece_id"] == "new.md"


class TestNpcReference:
    def test_logs_reference(self, store):
        store.log_npc_reference("Barnacle", "test.md", room="The Tap", mode="GOSSIP")
        history = store.get_npc_history("Barnacle")
        assert len(history) == 1
        assert history[0]["npc_name"] == "Barnacle"

    def test_multiple_npcs(self, store):
        store.log_npc_reference("Barnacle", "a.md")
        store.log_npc_reference("Flash", "b.md")
        store.log_npc_reference("Barnacle", "c.md")
        barnacle = store.get_npc_history("Barnacle")
        assert len(barnacle) == 2


class TestPropagation:
    def test_logs_propagation(self, store):
        store.log_propagation("test.md", "Barnacle", "The Tap", "chain1", depth=0)
        store.log_propagation("test.md", "Barnacle", "The Tap", "chain1", depth=1,
                              reactor_npc="Flash", reaction="AGREE")
        chains = store.get_propagation_chains()
        assert len(chains) == 2

    def test_chain_filtering(self, store):
        store.log_propagation("a.md", "Barnacle", "The Tap", "chain1")
        store.log_propagation("b.md", "Flash", "The Tap", "chain2")
        chains = store.get_propagation_chains()
        assert len(chains) == 2


class TestBehavioralEvents:
    def test_logs_event(self, store):
        store.log_behavioral_event("Barnacle", "test.md", 0.8, "SEISMIC", "Big shift")
        events = store.get_behavioral_events()
        assert len(events) == 1
        assert events[0]["severity"] == "SEISMIC"

    def test_filters_by_npc(self, store):
        store.log_behavioral_event("Barnacle", "a.md", 0.5, "SHIFT", "drift")
        store.log_behavioral_event("Flash", "b.md", 0.3, "NOTABLE", "slight")
        barnacle = store.get_behavioral_events(npc_name="Barnacle")
        assert len(barnacle) == 1
        assert barnacle[0]["npc_name"] == "Barnacle"


class TestStats:
    def test_empty_stats(self, store):
        stats = store.get_piece_stats()
        assert stats["total_pieces"] == 0
        assert stats["total_retrievals"] == 0

    def test_with_data(self, store):
        store.track_retrieval("a.md")
        store.track_retrieval("b.md")
        store.track_retrieval("a.md")
        stats = store.get_piece_stats()
        assert stats["total_pieces"] == 2
        assert stats["pieces_retrieved"] == 2
        assert stats["total_retrievals"] == 3


class TestRecomputeScores:
    def test_recompute(self, store):
        store.track_retrieval("a.md")
        store.track_retrieval("a.md")
        store.track_retrieval("b.md")
        store.recompute_all_scores()
        hot = store.get_hot_pieces(limit=2)
        assert len(hot) == 2
        assert hot[0]["retrieval_count"] == 2
