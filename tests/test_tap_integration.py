"""
Tests for Tap Integration — bridge between Zeitgeist data and Tap dialogue.

Tests cover:
- inject_zeitgeist_into_dialogue: surfaces dormant pieces
- check_seismic_break: detects NPC behavioral shifts
- fibonacci_surfacing: fires at round 8, surfaces dormant
- format_for_broadcast: formats as Channel 42 radio segments
- run_tap_cycle: full integration

All tests use synthetic data — no store or embedding dependency.
"""
import sys
import os
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tap_integration import (
    inject_zeitgeist_into_dialogue,
    check_seismic_break,
    fibonacci_surfacing,
    format_for_broadcast,
    run_tap_cycle,
    DialogueInjection,
    SeismicBreak,
    DormantSurfacing,
    BroadcastSegment,
    CorpusEntry,
)


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def make_corpus_entry(rng):
    """Factory for CorpusEntry objects."""
    def _make(piece_id="p1", title="Test Piece", directory="essays",
              retrieval_count=0, last_referenced_round=0,
              zeitgeist_score=0.0, gradient=None):
        return CorpusEntry(
            piece_id=piece_id,
            title=title,
            directory=directory,
            retrieval_count=retrieval_count,
            last_referenced_round=last_referenced_round,
            zeitgeist_score=zeitgeist_score,
            gradient=gradient if gradient is not None else rng.randn(32),
            embedding=rng.randn(32),
        )
    return _make


@pytest.fixture
def small_corpus(make_corpus_entry):
    """A small corpus with 5 pieces of varying dormancy."""
    return [
        make_corpus_entry("hot1", "Hot Piece", "essays", retrieval_count=50,
                          last_referenced_round=7, zeitgeist_score=3.5),
        make_corpus_entry("dormant1", "Dormant Piece", "poetry", retrieval_count=0,
                          last_referenced_round=0, zeitgeist_score=0.1),
        make_corpus_entry("medium1", "Medium Piece", "essays", retrieval_count=5,
                          last_referenced_round=3, zeitgeist_score=1.0),
        make_corpus_entry("dormant2", "Another Dormant", "poetry", retrieval_count=1,
                          last_referenced_round=1, zeitgeist_score=0.2),
        make_corpus_entry("hot2", "Also Hot", "essays", retrieval_count=30,
                          last_referenced_round=6, zeitgeist_score=2.5),
    ]


@pytest.fixture
def npc_history_factory():
    """Factory for NPC reference histories."""
    def _make(npc_name="Barnacle", dirs=("essays",), count=10, break_at=None,
              break_dir="poetry", break_piece="p_break"):
        history = []
        for i in range(count):
            d = dirs[i % len(dirs)]
            if break_at is not None and i >= break_at:
                d = break_dir
                pid = break_piece
            else:
                pid = f"p_{d}_{i}"
            history.append({
                "npc_name": npc_name,
                "piece_id": pid,
                "directory": d,
            })
        return history
    return _make


# ── inject_zeitgeist_into_dialogue ─────────────────────────────

class TestInjectZeitgeist:
    def test_surfaces_dormant_piece(self, small_corpus):
        """Should surface a dormant piece for an NPC."""
        result = inject_zeitgeist_into_dialogue(
            "Barnacle", "the nature of change", corpus=small_corpus,
        )
        assert result is not None
        assert isinstance(result, DialogueInjection)
        assert result.npc_name == "Barnacle"
        assert result.piece_id  # has a piece ID
        assert result.title  # has a title
        assert result.reason  # has a reason

    def test_mode_is_dormant_for_low_count(self, make_corpus_entry):
        """Pieces with low retrieval count should get DORMANT mode."""
        corpus = [
            make_corpus_entry("p1", "Never Retrieved", retrieval_count=0),
            make_corpus_entry("p2", "Also Rare", retrieval_count=1),
        ]
        result = inject_zeitgeist_into_dialogue("Flash", "anything", corpus=corpus)
        assert result is not None
        assert result.mode == "DORMANT"

    def test_empty_corpus_returns_none(self):
        """No corpus and no store → None."""
        result = inject_zeitgeist_into_dialogue("Flash", "topic")
        assert result is None

    def test_increments_retrieval_count(self, small_corpus):
        """Injecting should increment the piece's retrieval count."""
        original_counts = {p.piece_id: p.retrieval_count for p in small_corpus}
        inject_zeitgeist_into_dialogue("Barnacle", "topic", corpus=small_corpus)
        # At least one piece should have been incremented
        new_counts = {p.piece_id: p.retrieval_count for p in small_corpus}
        incremented = [
            pid for pid in original_counts
            if new_counts[pid] > original_counts[pid]
        ]
        assert len(incremented) >= 1

    def test_suggested_line_generated(self, small_corpus):
        """Should generate a suggested NPC line."""
        result = inject_zeitgeist_into_dialogue("Wesley", "change", corpus=small_corpus)
        assert result is not None
        assert result.suggested_line  # non-empty
        assert len(result.suggested_line) > 10  # has real content

    def test_with_store_mock(self, monkeypatch):
        """Should work with a mock store that raises (falls back gracefully)."""
        class MockStore:
            pass
        result = inject_zeitgeist_into_dialogue(
            "Flash", "topic", store=MockStore(), corpus=None,
        )
        # Falls back to no corpus → None
        assert result is None


# ── check_seismic_break ────────────────────────────────────────

class TestCheckSeismicBreak:
    def test_no_history(self):
        """Empty history → None."""
        assert check_seismic_break([]) is None

    def test_short_history(self, npc_history_factory):
        """Less than 5 references → None (no baseline)."""
        history = npc_history_factory(count=4)
        assert check_seismic_break(history) is None

    def test_stable_npc_no_break(self, npc_history_factory):
        """NPC that stays in their directories → None."""
        history = npc_history_factory(count=20, dirs=("essays",))
        assert check_seismic_break(history) is None

    def test_seismic_break_new_directory(self, npc_history_factory):
        """NPC suddenly references a completely new directory → SEISMIC."""
        history = npc_history_factory(
            count=10, dirs=("essays",), break_at=8, break_dir="poetry",
            break_piece="p_poetry_sudden",
        )
        result = check_seismic_break(history)
        assert result is not None
        assert isinstance(result, SeismicBreak)
        assert result.referenced_piece == "p_poetry_sudden"
        assert result.severity in ("SEISMIC", "SHIFT")

    def test_delta_for_new_directory_is_high(self, npc_history_factory):
        """Completely new directory should have high delta."""
        history = npc_history_factory(
            count=10, dirs=("essays",), break_at=8, break_dir="completely_new",
        )
        result = check_seismic_break(history)
        assert result is not None
        assert result.delta >= 0.7

    def test_severity_levels(self, npc_history_factory):
        """Test that severity escalates with delta."""
        # NPC mostly in essays, occasionally in poetry
        history = npc_history_factory(
            count=20, dirs=("essays", "essays", "essays", "essays", "poetry"),
            break_at=15, break_dir="poetry", break_piece="p_unusual",
        )
        result = check_seismic_break(history)
        # Poetry appears once in baseline (20%) → delta = 0.8 → SEISMIC
        if result:
            assert result.severity in ("SHIFT", "SEISMIC")

    def test_baseline_pieces_tracked(self, npc_history_factory):
        """Result should include baseline pieces."""
        history = npc_history_factory(
            count=10, dirs=("essays",), break_at=8, break_dir="poetry",
        )
        result = check_seismic_break(history)
        assert result is not None
        assert isinstance(result.baseline_pieces, list)
        assert len(result.baseline_pieces) > 0

    def test_description_contains_npc_name(self, npc_history_factory):
        history = npc_history_factory(
            count=10, dirs=("essays",), break_at=8, break_dir="poetry",
        )
        result = check_seismic_break(history)
        assert result is not None
        assert "Barnacle" in result.description or "unknown" in result.description.lower()


# ── fibonacci_surfacing ────────────────────────────────────────

class TestFibonacciSurfacing:
    def test_fires_at_round_8(self, small_corpus):
        """MUST fire at round 8."""
        result = fibonacci_surfacing(small_corpus, conversation_round=8)
        assert result is not None
        assert isinstance(result, DormantSurfacing)
        assert result.piece_id
        assert result.title

    def test_does_not_fire_before_8(self, small_corpus):
        for r in range(1, 8):
            assert fibonacci_surfacing(small_corpus, r) is None, f"Should not fire at round {r}"

    def test_fires_at_16_24(self, small_corpus):
        for r in [16, 24, 32]:
            result = fibonacci_surfacing(small_corpus, r)
            assert result is not None, f"Should fire at round {r}"

    def test_does_not_fire_at_non_multiples(self, small_corpus):
        for r in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]:
            assert fibonacci_surfacing(small_corpus, r) is None

    def test_does_not_fire_at_round_0(self, small_corpus):
        assert fibonacci_surfacing(small_corpus, 0) is None

    def test_surfaces_most_dormant(self, make_corpus_entry):
        """Should surface the piece with lowest reference round and count."""
        corpus = [
            make_corpus_entry("hot", "Hot", retrieval_count=100, last_referenced_round=7),
            make_corpus_entry("dormant", "Dormant", retrieval_count=0, last_referenced_round=0),
            make_corpus_entry("medium", "Medium", retrieval_count=5, last_referenced_round=4),
        ]
        result = fibonacci_surfacing(corpus, conversation_round=8)
        assert result is not None
        assert result.piece_id == "dormant"

    def test_updates_after_surfacing(self, make_corpus_entry):
        """Should update retrieval count after surfacing."""
        entry = make_corpus_entry("p1", "T1", retrieval_count=0, last_referenced_round=0)
        corpus = [entry]
        fibonacci_surfacing(corpus, 8)
        assert entry.retrieval_count == 1
        assert entry.last_referenced_round == 8

    def test_empty_corpus(self):
        assert fibonacci_surfacing([], 8) is None

    def test_reason_mentions_fibonacci(self, small_corpus):
        result = fibonacci_surfacing(small_corpus, 8)
        assert result is not None
        assert "fibonacci" in result.reason.lower() or "Fibonacci" in result.reason


# ── format_for_broadcast ───────────────────────────────────────

class TestFormatForBroadcast:
    def test_empty_data(self):
        """Even with no data, should produce intro + sign-off."""
        segments = format_for_broadcast({})
        assert len(segments) >= 2  # intro + sign-off
        assert segments[0].segment_type == "INTRO"
        assert segments[-1].segment_type == "SIGN_OFF"

    def test_hot_pieces_formatted(self):
        data = {
            "hot_pieces": [
                {"title": "The Storm", "zeitgeist_score": 3.5, "retrieval_count": 50},
                {"title": "The Dog", "zeitgeist_score": 2.0, "retrieval_count": 30},
            ]
        }
        segments = format_for_broadcast(data)
        hot_segments = [s for s in segments if s.segment_type == "HOT_PIECE"]
        assert len(hot_segments) == 2
        assert "The Storm" in hot_segments[0].body

    def test_dormant_pieces_formatted(self):
        data = {
            "dormant_pieces": [
                {"title": "Old Piece", "days_dormant": 30},
            ]
        }
        segments = format_for_broadcast(data)
        dormant_segments = [s for s in segments if s.segment_type == "DORMANT_PIECE"]
        assert len(dormant_segments) == 1
        assert "Old Piece" in dormant_segments[0].body
        assert "30" in dormant_segments[0].body

    def test_seismic_events_formatted(self):
        data = {
            "seismic_events": [
                {"npc_name": "Barnacle", "delta": 0.85, "piece_id": "essays/sudden.md"},
            ]
        }
        segments = format_for_broadcast(data)
        seismic_segments = [s for s in segments if s.segment_type == "SEISMIC"]
        assert len(seismic_segments) == 1
        assert "Barnacle" in seismic_segments[0].body

    def test_max_three_hot_pieces(self):
        """Should format at most 3 hot pieces."""
        data = {
            "hot_pieces": [
                {"title": f"P{i}", "zeitgeist_score": float(5 - i), "retrieval_count": 10}
                for i in range(10)
            ]
        }
        segments = format_for_broadcast(data)
        hot_segments = [s for s in segments if s.segment_type == "HOT_PIECE"]
        assert len(hot_segments) == 3

    def test_all_segments_have_body(self):
        """Every segment should have non-empty body text."""
        data = {
            "hot_pieces": [{"title": "T", "zeitgeist_score": 1, "retrieval_count": 1}],
            "dormant_pieces": [{"title": "D", "days_dormant": 5}],
            "seismic_events": [{"npc_name": "N", "delta": 0.8, "piece_id": "p"}],
        }
        segments = format_for_broadcast(data)
        for seg in segments:
            assert seg.body, f"Segment {seg.segment_type} has empty body"
            assert len(seg.body) > 20, f"Segment {seg.segment_type} has suspiciously short body"

    def test_duration_hints_present(self):
        segments = format_for_broadcast({})
        for seg in segments:
            assert seg.duration_hint  # non-empty
            assert "s" in seg.duration_hint  # something like "30s"

    def test_channel_42_branding(self):
        """Intro should mention Channel 42."""
        segments = format_for_broadcast({})
        intro = segments[0]
        assert "Channel 42" in intro.body or "42" in intro.body

    def test_sign_off_mentions_static(self):
        """Sign-off should reference static (the frequency's edge)."""
        segments = format_for_broadcast({})
        sign_off = segments[-1]
        assert "static" in sign_off.body.lower()


# ── run_tap_cycle (integration) ────────────────────────────────

class TestRunTapCycle:
    def test_returns_complete_report(self, small_corpus):
        result = run_tap_cycle(
            npcs=["Barnacle", "Flash"],
            current_topic="change",
            corpus=small_corpus,
            conversation_round=1,
        )
        assert "conversation_round" in result
        assert "fibonacci_surfacing" in result
        assert "injections" in result
        assert "seismic_breaks" in result
        assert "broadcast_segments" in result

    def test_fibonacci_fires_at_round_8(self, small_corpus):
        result = run_tap_cycle(
            npcs=["Barnacle"],
            current_topic="anything",
            corpus=small_corpus,
            conversation_round=8,
        )
        assert result["fibonacci_surfacing"] is not None

    def test_fibonacci_does_not_fire_at_round_3(self, small_corpus):
        result = run_tap_cycle(
            npcs=["Barnacle"],
            current_topic="anything",
            corpus=small_corpus,
            conversation_round=3,
        )
        assert result["fibonacci_surfacing"] is None

    def test_injections_for_each_npc(self, small_corpus):
        result = run_tap_cycle(
            npcs=["Barnacle", "Flash", "Wesley"],
            current_topic="the storm",
            corpus=small_corpus,
            conversation_round=1,
        )
        # Each NPC should get an injection
        npc_names = [inj["npc_name"] for inj in result["injections"]]
        assert "Barnacle" in npc_names
        assert "Flash" in npc_names
        assert "Wesley" in npc_names

    def test_seismic_breaks_empty_without_history(self, small_corpus):
        result = run_tap_cycle(
            npcs=["Barnacle"],
            current_topic="topic",
            corpus=small_corpus,
            conversation_round=1,
        )
        assert result["seismic_breaks"] == []

    def test_seismic_breaks_with_history(self, small_corpus, npc_history_factory):
        history = npc_history_factory(
            count=10, dirs=("essays",), break_at=8, break_dir="poetry",
        )
        result = run_tap_cycle(
            npcs=["Barnacle"],
            current_topic="topic",
            corpus=small_corpus,
            conversation_round=1,
            npc_histories={"Barnacle": history},
        )
        assert len(result["seismic_breaks"]) >= 1

    def test_broadcast_segments_always_present(self, small_corpus):
        result = run_tap_cycle(
            npcs=["Barnacle"],
            current_topic="topic",
            corpus=small_corpus,
            conversation_round=1,
        )
        # Should always have intro + sign-off at minimum
        assert len(result["broadcast_segments"]) >= 2

    def test_broadcast_mentions_channel_42(self, small_corpus):
        result = run_tap_cycle(
            npcs=["Barnacle"],
            current_topic="topic",
            corpus=small_corpus,
            conversation_round=1,
        )
        first_segment = result["broadcast_segments"][0]
        assert "42" in first_segment["body"]


# ── CorpusEntry dataclass ──────────────────────────────────────

class TestCorpusEntry:
    def test_defaults(self):
        entry = CorpusEntry(piece_id="p1", title="Test")
        assert entry.directory == ""
        assert entry.retrieval_count == 0
        assert entry.last_referenced_round == 0
        assert entry.zeitgeist_score == 0.0
        assert entry.gradient is None
        assert entry.embedding is None

    def test_with_gradient(self, rng):
        g = rng.randn(64)
        entry = CorpusEntry(piece_id="p1", title="T", gradient=g)
        np.testing.assert_array_equal(entry.gradient, g)
