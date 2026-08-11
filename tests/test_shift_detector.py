"""
Tests for the Behavioral Shift Detector
=======================================
Tests cover:
- Profile building from NPC history
- Delta computation (directory novelty, piece novelty, recency)
- Severity classification (NORMAL, NOTABLE, SHIFT, SEISMIC)
- Event logging
- Fleet overview
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zeitgeist_store import ZeitgeistStore
from shift_detector import BehavioralShiftDetector


@pytest.fixture
def store(tmp_path):
    return ZeitgeistStore(db_path=str(tmp_path / "test_shift.db"))


@pytest.fixture
def detector(store):
    return BehavioralShiftDetector(store=store)


@pytest.fixture
def established_npc(store):
    """NPC with established patterns in specific directories."""
    for _ in range(10):
        store.log_npc_reference("Barnacle", "essays/the-storm.md")
        store.track_retrieval("essays/the-storm.md", title="The Storm", directory="essays")
    for _ in range(5):
        store.log_npc_reference("Barnacle", "essays/memory.md")
        store.track_retrieval("essays/memory.md", title="Memory", directory="essays")
    for _ in range(3):
        store.log_npc_reference("Barnacle", "poems/rain.md")
        store.track_retrieval("poems/rain.md", title="Rain", directory="poems")
    return store


class TestProfileBuilding:
    def test_empty_profile(self, detector, store):
        profile = detector._get_npc_profile("UnknownNPC")
        assert profile["reference_count"] == 0
        assert not profile["established"]

    def test_established_profile(self, detector, established_npc):
        profile = detector._get_npc_profile("Barnacle")
        assert profile["reference_count"] == 18
        assert profile["established"]
        assert "essays" in profile["typical_directories"]

    def test_directory_distribution(self, detector, established_npc):
        profile = detector._get_npc_profile("Barnacle")
        # Barnacle references essays 15/18 times = 83%
        dir_prob = profile["typical_directories"]["essays"] / profile["reference_count"]
        assert dir_prob > 0.8

    def test_recent_pieces(self, detector, established_npc):
        profile = detector._get_npc_profile("Barnacle")
        assert len(profile["recent_pieces"]) >= 5


class TestDetectShift:
    def test_normal_for_established_directory(self, detector, established_npc):
        """Barnacle referencing essays = NORMAL."""
        result = detector.detect_shift("Barnacle", "essays/new-piece.md", piece_directory="essays")
        assert result["severity"] == "NORMAL"
        assert result["delta"] < 0.3

    def test_not_established_npc(self, detector, store):
        """NPC with no history should always be NORMAL."""
        result = detector.detect_shift("UnknownNPC", "anything.md", piece_directory="unknown")
        assert result["severity"] == "NORMAL"
        assert result["delta"] == 0.0

    def test_directory_novelty_detected(self, detector, established_npc):
        """Barnacle referencing a completely new directory should be high delta."""
        result = detector.detect_shift(
            "Barnacle", "cookbook/recipe.md", piece_directory="cookbook"
        )
        # Barnacle has never been to "cookbook" → high dir novelty
        assert result["dir_novelty"] == 1.0
        assert result["delta"] > 0.3

    def test_returns_profile_info(self, detector, established_npc):
        result = detector.detect_shift("Barnacle", "essays/test.md", piece_directory="essays")
        assert "profile" in result
        assert result["profile"]["reference_count"] == 18

    def test_logs_significant_events(self, detector, established_npc):
        """Events above NORMAL should be logged."""
        detector.detect_shift("Barnacle", "alien/portal.md", piece_directory="alien")
        events = detector.get_npc_shifts("Barnacle")
        assert len(events) >= 1

    def test_does_not_log_normal(self, detector, established_npc):
        """NORMAL events should not be logged."""
        detector.detect_shift("Barnacle", "essays/regular.md", piece_directory="essays")
        events = detector.get_npc_shifts("Barnacle", min_severity="NOTABLE")
        assert len(events) == 0


class TestSeverityClassification:
    def test_normal_threshold(self, detector, store):
        """Delta < 0.3 = NORMAL."""
        store.log_npc_reference("TestNPC", "a.md")
        store.track_retrieval("a.md", title="A", directory="essays")
        store.log_npc_reference("TestNPC", "a.md")
        store.track_retrieval("a.md", title="A", directory="essays")
        store.log_npc_reference("TestNPC", "a.md")
        store.track_retrieval("a.md", title="A", directory="essays")
        store.log_npc_reference("TestNPC", "a.md")
        store.track_retrieval("a.md", title="A", directory="essays")
        store.log_npc_reference("TestNPC", "a.md")
        store.track_retrieval("a.md", title="A", directory="essays")

        result = detector.detect_shift("TestNPC", "essays/new.md", piece_directory="essays")
        assert result["severity"] == "NORMAL"

    def test_description_contains_npc_name(self, detector, established_npc):
        result = detector.detect_shift("Barnacle", "essays/test.md", piece_directory="essays")
        assert "Barnacle" in result["description"]


class TestFleetOverview:
    def test_empty_overview(self, detector, store):
        overview = detector.get_fleet_overview()
        assert overview == []

    def test_with_events(self, detector, established_npc):
        detector.detect_shift("Barnacle", "alien/x.md", piece_directory="alien")
        detector.detect_shift("Barnacle", "alien/y.md", piece_directory="alien")
        overview = detector.get_fleet_overview()
        assert len(overview) >= 1
        assert overview[0]["npc"] == "Barnacle"
        assert overview[0]["events"] >= 2


class TestGetShifts:
    def test_filter_by_severity(self, detector, established_npc):
        detector.detect_shift("Barnacle", "alien/a.md", piece_directory="alien")
        all_events = detector.get_npc_shifts("Barnacle", min_severity="NORMAL")
        notable_only = detector.get_npc_shifts("Barnacle", min_severity="NOTABLE")
        assert len(all_events) >= len(notable_only)
