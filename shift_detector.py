#!/usr/bin/env python3
"""
Behavioral Shift Detector — Changepoint Detection for NPC References
====================================================================
Tracks each NPC's retrieval patterns over time and detects when an NPC
suddenly references pieces outside their usual neighborhood.

This is Casey's vision: "you hear glimpses of a seismic change in subtle
behavior changes of otherwise predictable people." This is the tie-up line
breaking one by one.

Each NPC has a baseline profile — their typical pieces, topics, directories.
When they reference something FAR from their baseline, that's a shift.

Severity:
    NORMAL    (Δ < 0.3) — the NPC is in their comfort zone
    NOTABLE   (0.3 ≤ Δ < 0.5) — slight drift, might be interesting
    SHIFT     (0.5 ≤ Δ < 0.7) — significant shift, the NPC is changing
    SEISMIC   (Δ ≥ 0.7) — a tie-up line breaking. The NPC has stepped outside themselves.

Usage:
    from shift_detector import BehavioralShiftDetector

    detector = BehavioralShiftDetector()
    detector.update_profile("Barnacle", "essays/the-storm.md", directory="essays")
    shift = detector.detect_shift("Barnacle", "essays/the-storm.md")

    if shift['severity'] == 'SEISMIC':
        print(f"🔥 BEHAVIORAL SHIFT: {shift['description']}")
"""

import os
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict, Counter

from zeitgeist_store import ZeitgeistStore


class BehavioralShiftDetector:
    """Detects when NPCs reference pieces outside their usual neighborhood."""

    def __init__(self, store: Optional[ZeitgeistStore] = None):
        self.store = store or ZeitgeistStore()

    def update_profile(self, npc_name: str, referenced_piece: str, directory: str = ""):
        """
        Track what each NPC normally references.
        Called every time an NPC references a piece.
        """
        # The profile is implicit in the npc_reference_log table.
        # Profiles are built from historical reference logs via _get_npc_profile.
        pass

    def _get_npc_profile(self, npc_name: str) -> dict:
        """
        Build a profile of what an NPC typically references.
        Returns their typical directories, typical pieces, and reference count.
        """
        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get all references by this NPC
            rows = conn.execute(
                """SELECT piece_id, referenced_at FROM npc_reference_log
                   WHERE npc_name = ? ORDER BY referenced_at""",
                (npc_name,)
            ).fetchall()

            if not rows:
                return {
                    "reference_count": 0,
                    "typical_directories": {},
                    "typical_pieces": {},
                    "recent_pieces": [],
                    "established": False,
                }

            # Get directory for each referenced piece
            piece_dirs = {}
            for r in rows:
                pid = r["piece_id"]
                pinfo = conn.execute(
                    "SELECT directory, title FROM pieces WHERE piece_id = ?",
                    (pid,)
                ).fetchone()
                if pinfo:
                    piece_dirs[pid] = {"directory": pinfo["directory"], "title": pinfo["title"]}

            # Directory distribution
            dir_counts = Counter()
            piece_counts = Counter()
            for r in rows:
                pid = r["piece_id"]
                d = piece_dirs.get(pid, {}).get("directory", "unknown")
                dir_counts[d] += 1
                piece_counts[pid] += 1

            # Recent pieces (last 20% of references)
            recent_n = max(5, len(rows) // 5)
            recent_pieces = [r["piece_id"] for r in rows[-recent_n:]]

            return {
                "reference_count": len(rows),
                "typical_directories": dict(dir_counts.most_common(10)),
                "typical_pieces": dict(piece_counts.most_common(20)),
                "recent_pieces": recent_pieces,
                "established": len(rows) >= 5,  # need at least 5 refs for a baseline
            }

    def detect_shift(
        self,
        npc_name: str,
        referenced_piece: str,
        piece_directory: str = "",
        piece_embedding: list[float] = None,
    ) -> dict:
        """
        Is this piece outside the NPC's usual neighborhood?

        Computes Δ (delta) between this piece and the NPC's typical pieces.
        Returns severity classification.

        Δ < 0.3: NORMAL — comfort zone
        0.3-0.5: NOTABLE — slight drift
        0.5-0.7: SHIFT — significant change
        ≥ 0.7: SEISMIC — tie-up line breaking
        """
        profile = self._get_npc_profile(npc_name)

        # If NPC has no history, everything is normal (no baseline to compare)
        if not profile["established"]:
            return {
                "npc_name": npc_name,
                "piece_id": referenced_piece,
                "delta": 0.0,
                "severity": "NORMAL",
                "description": f"{npc_name} has no established baseline yet ({profile['reference_count']} refs).",
                "profile": profile,
            }

        # --- Directory-based shift ---
        # How far is this piece's directory from the NPC's typical directories?
        if not piece_directory:
            # Try to get it from the store
            with sqlite3.connect(self.store.db_path) as conn:
                row = conn.execute(
                    "SELECT directory FROM pieces WHERE piece_id = ?",
                    (referenced_piece,)
                ).fetchone()
                if row:
                    piece_directory = row[0] or "unknown"
                else:
                    piece_directory = "unknown"

        dir_counts = profile["typical_directories"]
        total_refs = sum(dir_counts.values())
        dir_prob = dir_counts.get(piece_directory, 0) / max(total_refs, 1)
        dir_novelty = 1.0 - dir_prob  # 0 = very typical, 1 = never been there

        # --- Piece-level shift ---
        # How many DISTINCT pieces has this NPC referenced?
        # If they reference many different pieces, new ones are less surprising.
        distinct_pieces = len(profile["typical_pieces"])
        has_referenced = referenced_piece in profile["typical_pieces"]
        if has_referenced:
            piece_prob = profile["typical_pieces"][referenced_piece] / max(total_refs, 1)
            piece_novelty = 1.0 - piece_prob
        else:
            piece_novelty = 1.0 / (1.0 + distinct_pieces * 0.3)

        # --- Recency shift ---
        # Is this piece in the NPC's recent references?
        recency_factor = 0.0
        if referenced_piece in profile["recent_pieces"]:
            recency_factor = -0.2  # they've been here recently, less novel

        # --- Composite delta ---
        # Directory novelty is the primary signal (70%),
        # piece novelty scaled by exploration breadth (20%),
        # recency adjustment (10%)
        delta = (dir_novelty * 0.7 + piece_novelty * 0.2 + abs(recency_factor) * 0.1)
        delta = min(delta, 1.0)  # clamp

        # Severity classification
        if delta < 0.3:
            severity = "NORMAL"
            description = (
                f"{npc_name} is in their comfort zone. "
                f"'{piece_directory}' is familiar territory. "
                f"No behavioral shift detected."
            )
        elif delta < 0.5:
            severity = "NOTABLE"
            description = (
                f"{npc_name} is drifting slightly. "
                f"They don't usually reference '{piece_directory}' "
                f"({dir_counts.get(piece_directory, 0)} times vs {total_refs} total). "
                f"Might be interesting to watch."
            )
        elif delta < 0.7:
            severity = "SHIFT"
            description = (
                f"{npc_name} is shifting. "
                f"They're referencing '{piece_directory}' — "
                f"territory they rarely visit "
                f"(probability: {dir_prob:.1%}). "
                f"The tie-up line is under strain."
            )
        else:
            severity = "SEISMIC"
            description = (
                f"{npc_name} has stepped outside themselves. "
                f"'{piece_directory}' is completely outside their usual neighborhood "
                f"(probability: {dir_prob:.1%}, delta: {delta:.3f}). "
                f"This is the tie-up line breaking one by one. "
                f"A behavioral shift is happening."
            )

        # Log the event if notable or worse
        if severity != "NORMAL":
            self.store.log_behavioral_event(
                npc_name=npc_name,
                piece_id=referenced_piece,
                delta=delta,
                severity=severity,
                description=description,
            )

        return {
            "npc_name": npc_name,
            "piece_id": referenced_piece,
            "piece_directory": piece_directory,
            "delta": round(delta, 4),
            "dir_novelty": round(dir_novelty, 4),
            "piece_novelty": round(piece_novelty, 4),
            "severity": severity,
            "description": description,
            "profile": {
                "reference_count": profile["reference_count"],
                "typical_directories": profile["typical_directories"],
            },
        }

    def get_npc_shifts(self, npc_name: str = None, min_severity: str = "NOTABLE", limit: int = 20) -> list[dict]:
        """Get behavioral shift events for an NPC (or all NPCs)."""
        severity_order = {"NORMAL": 0, "NOTABLE": 1, "SHIFT": 2, "SEISMIC": 3}
        min_level = severity_order.get(min_severity, 1)

        events = self.store.get_behavioral_events(npc_name=npc_name, limit=limit * 3)
        filtered = [e for e in events if severity_order.get(e.get("severity", "NORMAL"), 0) >= min_level]
        return filtered[:limit]

    def get_fleet_overview(self) -> list[dict]:
        """Get a behavioral overview of all NPCs in the fleet."""
        all_events = self.store.get_behavioral_events(limit=500)
        npc_summary = defaultdict(lambda: {"events": 0, "seismic": 0, "shifts": 0, "notable": 0})

        for e in all_events:
            npc = e.get("npc_name", "unknown")
            npc_summary[npc]["events"] += 1
            sev = e.get("severity", "NORMAL")
            if sev == "SEISMIC":
                npc_summary[npc]["seismic"] += 1
            elif sev == "SHIFT":
                npc_summary[npc]["shifts"] += 1
            elif sev == "NOTABLE":
                npc_summary[npc]["notable"] += 1

        result = []
        for npc, counts in sorted(npc_summary.items()):
            result.append({
                "npc": npc,
                **counts,
                "instability": counts["seismic"] * 3 + counts["shifts"] * 2 + counts["notable"],
            })
        return result
