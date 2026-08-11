"""
Tests for the Gossip Protocol — Propagation Engine
==================================================
Tests cover:
- propagate() basic flow
- Reaction types (AGREE, DISAGREE, IGNORE, DEFLECT)
- DISAGREE generates contrast
- Propagation chain logging
- Fleet exclusion (source doesn't react to themselves)
- Room dynamics
"""

import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zeitgeist_store import ZeitgeistStore
from zeitgeist_sampler import ZeitgeistSampler
from gossip_protocol import GossipProtocol, BASE_REACTION_PROBS


@pytest.fixture
def store(tmp_path):
    return ZeitgeistStore(db_path=str(tmp_path / "test_gossip.db"))


@pytest.fixture
def populated_store(store):
    """Add pieces so sampling has data."""
    for _ in range(5):
        store.track_retrieval("essays/storm.md", title="The Storm", directory="essays")
        store.track_retrieval("poems/rain.md", title="Rain", directory="poems")
    return store


@pytest.fixture
def gossip(store, populated_store):
    sampler = ZeitgeistSampler(store=store)
    return GossipProtocol(store=store, sampler=sampler, fleet=["Barnacle", "Flash", "Hermes", "Wesley"])


class TestPropagate:
    def test_returns_propagation_tree(self, gossip, populated_store):
        tree = gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        assert "chain_id" in tree
        assert "source" in tree
        assert "reactions" in tree
        assert tree["source"]["npc"] == "Barnacle"

    def test_source_excluded_from_reactions(self, gossip, populated_store):
        tree = gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        reacting_npcs = {r["npc"] for r in tree["reactions"]}
        assert "Barnacle" not in reacting_npcs

    def test_all_fleet_members_react(self, gossip, populated_store):
        tree = gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        reacting_npcs = {r["npc"] for r in tree["reactions"]}
        assert reacting_npcs == {"Flash", "Hermes", "Wesley"}

    def test_counts_are_correct(self, gossip, populated_store):
        tree = gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        total = tree["total_agreed"] + tree["total_disagreed"] + tree["total_ignored"] + tree["total_deflected"]
        assert total == len(tree["reactions"])

    def test_total_reached_includes_source(self, gossip, populated_store):
        tree = gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        assert tree["total_reached"] == 1 + len(tree["reactions"])


class TestReactions:
    def test_all_reactions_valid(self, gossip, populated_store):
        tree = gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        valid = {"AGREE", "DISAGREE", "IGNORE", "DEFLECT"}
        for r in tree["reactions"]:
            assert r["reaction"] in valid

    def test_reaction_has_description(self, gossip, populated_store):
        tree = gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        for r in tree["reactions"]:
            assert r["description"]
            assert len(r["description"]) > 10

    def test_base_probs_sum_to_one(self):
        total = sum(BASE_REACTION_PROBS.values())
        assert abs(total - 1.0) < 0.01


class TestDisagree:
    def test_disagree_generates_contrast(self, gossip, populated_store):
        """When an NPC disagrees, they should pull a contrasting piece."""
        # Force DISAGREE
        import gossip_protocol as gp
        original = gp.GossipProtocol._roll_reaction
        def force_disagree(self, npc, piece, room):
            return {"npc": npc, "reaction": "DISAGREE", "description": "forced", "timestamp": "2025-01-01T00:00:00Z"}
        gp.GossipProtocol._roll_reaction = force_disagree

        try:
            tree = gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        finally:
            gp.GossipProtocol._roll_reaction = original

        assert len(tree["contrasts"]) > 0
        assert tree["contrasts"][0]["contrast_piece_id"]

    def test_contrast_has_description(self, gossip, populated_store):
        import gossip_protocol as gp
        original = gp.GossipProtocol._roll_reaction
        def force_disagree(self, npc, piece, room):
            return {"npc": npc, "reaction": "DISAGREE", "description": "forced", "timestamp": "2025-01-01T00:00:00Z"}
        gp.GossipProtocol._roll_reaction = force_disagree

        try:
            tree = gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        finally:
            gp.GossipProtocol._roll_reaction = original

        assert tree["contrasts"][0]["description"]
        assert "disagrees" in tree["contrasts"][0]["description"].lower()


class TestPropagationLogging:
    def test_logs_to_store(self, gossip, populated_store):
        gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        chains = gossip.store.get_propagation_chains()
        assert len(chains) >= 4  # 1 source + 3 reactors

    def test_chain_id_consistent(self, gossip, populated_store):
        gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        chains = gossip.store.get_propagation_chains()
        ids = {c["chain_id"] for c in chains}
        assert len(ids) == 1  # all same chain


class TestRoomDynamics:
    def test_get_room_dynamics(self, gossip, populated_store):
        gossip.propagate("essays/storm.md", "Barnacle", "The Tap", max_depth=1)
        dynamics = gossip.get_room_dynamics("The Tap")
        assert len(dynamics) >= 4

    def test_empty_room(self, gossip, populated_store):
        dynamics = gossip.get_room_dynamics("Empty Room")
        assert dynamics == []
