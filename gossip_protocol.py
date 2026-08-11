#!/usr/bin/env python3
"""
Gossip Protocol — Propagation Engine
====================================
When an NPC references a corpus piece, the gossip protocol determines
how it spreads through the fleet.

When a piece surfaces in conversation, other NPCs in the same room get
a probability roll to react:

    AGREE     — reinforce the reference (increments retrieval count further)
    DISAGREE  — counter-reference a contrasting piece (creates tension)
    IGNORE    — no reaction (the piece doesn't land)
    DEFLECT   — reference something tangentially related

When two NPCs DISAGREE, the contrast creates a "tie-up line breaking"
moment — the conversation splits, new pieces surface from the gap.

The propagation chain is tracked so we can see how gossip spreads.

Usage:
    from gossip_protocol import GossipProtocol

    gossip = GossipProtocol()
    chain = gossip.propagate(
        piece_id="essays/the-storm.md",
        source_npc="Barnacle",
        room="The Tap",
    )
    # chain contains the full propagation tree
"""

import os
import uuid
import random
import json
import math
from datetime import datetime, timezone
from typing import Optional

from zeitgeist_store import ZeitgeistStore
from zeitgeist_sampler import ZeitgeistSampler

# ── NPC Fleet ──────────────────────────────────────────────────
DEFAULT_FLEET = [
    "Barnacle",
    "Flash",
    "Hermes",
    "Wesley",
    # Extend as the fleet grows
]

# Reaction probabilities (modified by piece properties)
BASE_REACTION_PROBS = {
    "AGREE": 0.35,     # reinforce
    "DISAGREE": 0.10,  # counter-reference
    "IGNORE": 0.40,    # doesn't land
    "DEFLECT": 0.15,   # tangential pivot
}


class GossipProtocol:
    """When a piece surfaces, it propagates through the fleet."""

    def __init__(
        self,
        store: Optional[ZeitgeistStore] = None,
        sampler: Optional[ZeitgeistSampler] = None,
        fleet: Optional[list[str]] = None,
    ):
        self.store = store or ZeitgeistStore()
        self.sampler = sampler or ZeitgeistSampler(store=self.store)
        self.fleet = fleet or DEFAULT_FLEET

    def propagate(
        self,
        piece_id: str,
        source_npc: str,
        room: str,
        max_depth: int = 3,
        context: str = "",
    ) -> dict:
        """
        Propagate a piece through the fleet.

        Returns a propagation tree showing who reacted and how.
        """
        chain_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        # The piece is now "in the air" in this room
        propagation_tree = {
            "chain_id": chain_id,
            "source": {
                "npc": source_npc,
                "piece_id": piece_id,
                "room": room,
                "timestamp": now,
                "depth": 0,
            },
            "reactions": [],
            "contrasts": [],  # disagreement chains
            "total_reached": 1,
            "total_agreed": 0,
            "total_disagreed": 0,
            "total_ignored": 0,
            "total_deflected": 0,
        }

        # Log the source
        self.store.log_propagation(
            piece_id=piece_id,
            source_npc=source_npc,
            room=room,
            chain_id=chain_id,
            depth=0,
        )

        # Other NPCs in the room get a chance to react
        other_npcs = [npc for npc in self.fleet if npc != source_npc]

        # Get the source piece's properties for reaction weighting
        source_piece = self._get_piece_info(piece_id)
        zeitgeist = source_piece.get("zeitgeist_score", 0) if source_piece else 0
        velocity = source_piece.get("retrieval_velocity", 0) if source_piece else 0

        reaction_queue = []
        for npc in other_npcs:
            reaction = self._roll_reaction(npc, source_piece, room)
            propagation_tree["reactions"].append(reaction)
            reaction_key = {
                "AGREE": "total_agreed",
                "DISAGREE": "total_disagreed",
                "IGNORE": "total_ignored",
                "DEFLECT": "total_deflected",
            }.get(reaction["reaction"], "total_ignored")
            propagation_tree[reaction_key] += 1

            # Log the propagation
            self.store.log_propagation(
                piece_id=piece_id,
                source_npc=source_npc,
                room=room,
                chain_id=chain_id,
                depth=1,
                reactor_npc=npc,
                reaction=reaction["reaction"],
            )

            # If DISAGREE, generate a counter-reference (the conversation splits)
            if reaction["reaction"] == "DISAGREE":
                contrast = self._generate_contrast(npc, piece_id, room, context)
                propagation_tree["contrasts"].append(contrast)

            # If DEFLECT, the NPC references something tangentially related
            if reaction["reaction"] == "DEFLECT":
                deflect_piece = self.sampler.sample_for_npc(
                    npc, context=f"{context} tangentially related to {source_piece.get('title', '')}", room=room
                )
                reaction["deflected_to"] = deflect_piece["piece"]["piece_id"]

            propagation_tree["total_reached"] += 1

        # Deepen: if there are contrasts, they can propagate further
        if propagation_tree["contrasts"] and max_depth > 1:
            for contrast in propagation_tree["contrasts"]:
                if contrast.get("contrast_piece_id"):
                    sub_chain = self.propagate(
                        piece_id=contrast["contrast_piece_id"],
                        source_npc=contrast["npc"],
                        room=room,
                        max_depth=max_depth - 1,
                        context=context,
                    )
                    contrast["sub_chain"] = sub_chain

        return propagation_tree

    def _roll_reaction(self, npc: str, source_piece: dict, room: str) -> dict:
        """Roll for how an NPC reacts to a piece being referenced."""

        # Modify base probabilities based on piece properties
        probs = BASE_REACTION_PROBS.copy()

        # Hot pieces (high zeitgeist) are more likely to AGREE (everyone's talking about it)
        zeitgeist = source_piece.get("zeitgeist_score", 0) if source_piece else 0
        if zeitgeist > 1.0:
            probs["AGREE"] += 0.15
            probs["IGNORE"] -= 0.10
            probs["DEFLECT"] -= 0.05

        # High-velocity pieces are even more likely to spread
        velocity = source_piece.get("retrieval_velocity", 0) if source_piece else 0
        if velocity > 0.5:
            probs["AGREE"] += 0.05
            probs["IGNORE"] -= 0.05

        # Novel pieces (high novelty score) are more likely to trigger DEFLECT or DISAGREE
        novelty = source_piece.get("novelty_score", 1.0) if source_piece else 1.0
        if novelty > 0.8:
            probs["DEFLECT"] += 0.10
            probs["DISAGREE"] += 0.05
            probs["IGNORE"] -= 0.15

        # Normalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        reaction = random.choices(
            list(probs.keys()), weights=list(probs.values()), k=1
        )[0]

        # Build reaction description
        descriptions = {
            "AGREE": f"{npc} nods. They've been thinking about this too.",
            "DISAGREE": f"{npc} bristles. This doesn't sit right. They pull a contrasting piece from the corpus.",
            "IGNORE": f"{npc} doesn't bite. The piece doesn't land for them.",
            "DEFLECT": f"{npc} pivots. This reminds them of something else entirely.",
        }

        return {
            "npc": npc,
            "reaction": reaction,
            "description": descriptions[reaction],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _generate_contrast(self, npc: str, source_piece_id: str, room: str, context: str) -> dict:
        """
        When an NPC DISAGREES, they pull a contrasting piece.
        This is the tie-up line breaking — the conversation splits.
        """
        # Sample a piece for the disagreeing NPC (different from source)
        result = self.sampler.sample_for_npc(npc, context=f"contrasting with {source_piece_id} {context}", room=room)

        return {
            "npc": npc,
            "contrast_piece_id": result["piece"]["piece_id"],
            "contrast_title": result["piece"].get("title", ""),
            "contrast_mode": result["mode"],
            "description": (
                f"{npc} disagrees. Instead of '{self._piece_title(source_piece_id)}', "
                f"they reference '{result['piece'].get('title', '')}'. "
                f"The conversation splits. New pieces surface from the gap."
            ),
        }

    def _get_piece_info(self, piece_id: str) -> dict:
        """Get piece info from the store."""
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)
            ).fetchone()
            return dict(row) if row else {}

    def _piece_title(self, piece_id: str) -> str:
        info = self._get_piece_info(piece_id)
        return info.get("title", piece_id) if info else piece_id

    def get_room_dynamics(self, room: str) -> dict:
        """Get recent propagation dynamics for a room."""
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM propagation_log
                   WHERE room = ?
                   ORDER BY propagated_at DESC LIMIT 50""",
                (room,)
            ).fetchall()
            return [dict(r) for r in rows]
