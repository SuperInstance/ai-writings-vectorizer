#!/usr/bin/env python3
"""
Zeitgeist Store — Retrieval Frequency Tracker
=============================================
Tracks how often each piece in the corpus gets retrieved.
Stores retrieval counts, velocity, and composite zeitgeist scores.

Backed by a local SQLite database for persistence and fast queries.
The Cloudflare Worker mirrors this data to D1 for the dashboard.

Schema:
    pieces(
        piece_id TEXT PRIMARY KEY,      -- file path
        title TEXT,
        directory TEXT,
        retrieval_count INTEGER DEFAULT 0,
        first_retrieved TEXT,            -- ISO timestamp
        last_retrieved TEXT,             -- ISO timestamp
        retrieval_velocity REAL DEFAULT 0,
        zeitgeist_score REAL DEFAULT 0,
        novelty_score REAL DEFAULT 1.0
    )

    retrieval_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        piece_id TEXT,
        retrieved_at TEXT,               -- ISO timestamp
        retrieved_by TEXT,               -- NPC name or 'system'
        context TEXT,                    -- what was the conversation about?
        mode TEXT,                       -- 'GOSSIP', 'CONTEXTUAL', 'SEISMIC'
        score REAL,                      -- similarity score if applicable
        FOREIGN KEY (piece_id) REFERENCES pieces(piece_id)
    )

    npc_reference_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        npc_name TEXT,
        piece_id TEXT,
        referenced_at TEXT,
        room TEXT,                       -- which room/channel
        mode TEXT,
        reaction TEXT,                   -- AGREE, DISAGREE, IGNORE, DEFLECT
        FOREIGN KEY (piece_id) REFERENCES pieces(piece_id)
    )

Usage:
    store = ZeitgeistStore()
    store.track_retrieval("essays/the-storm.md", retrieved_by="Barnacle", mode="GOSSIP")
    hot = store.get_hot_pieces(limit=10)
    dormant = store.get_dormant_pieces(min_age_days=14, limit=5)
"""

import json
import os
import sqlite3
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Optional
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "zeitgeist.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _hours_ago(ts: Optional[str]) -> float:
    """How many hours ago was this timestamp?"""
    dt = _parse_iso(ts)
    if not dt:
        return float("inf")
    now = datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 3600


class ZeitgeistStore:
    """SQLite-backed retrieval frequency tracker for the corpus."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pieces (
                    piece_id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    directory TEXT DEFAULT '',
                    retrieval_count INTEGER DEFAULT 0,
                    first_retrieved TEXT,
                    last_retrieved TEXT,
                    retrieval_velocity REAL DEFAULT 0,
                    zeitgeist_score REAL DEFAULT 0,
                    novelty_score REAL DEFAULT 1.0
                );

                CREATE TABLE IF NOT EXISTS retrieval_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    piece_id TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    retrieved_by TEXT DEFAULT 'system',
                    context TEXT DEFAULT '',
                    mode TEXT DEFAULT 'GOSSIP',
                    score REAL DEFAULT 0,
                    FOREIGN KEY (piece_id) REFERENCES pieces(piece_id)
                );

                CREATE TABLE IF NOT EXISTS npc_reference_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    npc_name TEXT NOT NULL,
                    piece_id TEXT NOT NULL,
                    referenced_at TEXT NOT NULL,
                    room TEXT DEFAULT '',
                    mode TEXT DEFAULT '',
                    reaction TEXT DEFAULT '',
                    FOREIGN KEY (piece_id) REFERENCES pieces(piece_id)
                );

                CREATE TABLE IF NOT EXISTS propagation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    piece_id TEXT NOT NULL,
                    source_npc TEXT NOT NULL,
                    room TEXT NOT NULL,
                    propagated_at TEXT NOT NULL,
                    chain_id TEXT,
                    depth INTEGER DEFAULT 0,
                    reactor_npc TEXT DEFAULT '',
                    reaction TEXT DEFAULT '',
                    FOREIGN KEY (piece_id) REFERENCES pieces(piece_id)
                );

                CREATE TABLE IF NOT EXISTS behavioral_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    npc_name TEXT NOT NULL,
                    piece_id TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    delta REAL,
                    severity TEXT DEFAULT 'NORMAL',
                    description TEXT DEFAULT '',
                    FOREIGN KEY (piece_id) REFERENCES pieces(piece_id)
                );

                CREATE INDEX IF NOT EXISTS idx_pieces_zeitgeist ON pieces(zeitgeist_score DESC);
                CREATE INDEX IF NOT EXISTS idx_pieces_last_retrieved ON pieces(last_retrieved);
                CREATE INDEX IF NOT EXISTS idx_log_piece ON retrieval_log(piece_id);
                CREATE INDEX IF NOT EXISTS idx_log_time ON retrieval_log(retrieved_at);
                CREATE INDEX IF NOT EXISTS idx_npc_ref ON npc_reference_log(npc_name);
                CREATE INDEX IF NOT EXISTS idx_prop_chain ON propagation_log(chain_id);
                CREATE INDEX IF NOT EXISTS idx_behavior_npc ON behavioral_events(npc_name);
            """)

    def _ensure_piece(self, conn, piece_id: str, title: str = "", directory: str = ""):
        """Insert a piece row if it doesn't exist."""
        conn.execute(
            """INSERT OR IGNORE INTO pieces (piece_id, title, directory, first_retrieved)
               VALUES (?, ?, ?, ?)""",
            (piece_id, title, directory, _now_iso())
        )
        if title or directory:
            conn.execute(
                """UPDATE pieces SET title = COALESCE(NULLIF(?, ''), title),
                       directory = COALESCE(NULLIF(?, ''), directory)
                   WHERE piece_id = ?""",
                (title, directory, piece_id)
            )

    # ── Core: Track a retrieval ────────────────────────────────

    def track_retrieval(
        self,
        piece_id: str,
        title: str = "",
        directory: str = "",
        retrieved_by: str = "system",
        context: str = "",
        mode: str = "GOSSIP",
        score: float = 0.0,
    ):
        """Record that a piece was retrieved. Updates all metrics."""
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_piece(conn, piece_id, title, directory)

            # Get previous state for velocity calculation
            row = conn.execute(
                "SELECT retrieval_count, last_retrieved FROM pieces WHERE piece_id = ?",
                (piece_id,)
            ).fetchone()
            prev_count = row[0] if row else 0
            prev_last = row[1] if row else None

            # Calculate velocity (retrievals per hour over last 24h)
            velocity = self._compute_velocity(conn, piece_id)

            # Calculate novelty score: starts at 1.0, decays with retrieval count
            novelty = max(0.01, 1.0 / (1.0 + prev_count * 0.1))

            # Zeitgeist score: composite of frequency, recency, novelty, and inverse-novelty
            recency_boost = math.exp(-_hours_ago(now) / 168)  # half-life ~1 week
            freq_component = math.log1p(prev_count + 1)
            zeitgeist = (freq_component * recency_boost + velocity * 10) * (0.3 + 0.7 * novelty)

            # Update piece
            conn.execute(
                """UPDATE pieces SET
                       retrieval_count = retrieval_count + 1,
                       last_retrieved = ?,
                       retrieval_velocity = ?,
                       novelty_score = ?,
                       zeitgeist_score = ?
                   WHERE piece_id = ?""",
                (now, velocity, novelty, zeitgeist, piece_id)
            )

            # Log the retrieval
            conn.execute(
                """INSERT INTO retrieval_log
                   (piece_id, retrieved_at, retrieved_by, context, mode, score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (piece_id, now, retrieved_by, context[:500], mode, score)
            )

    def _compute_velocity(self, conn, piece_id: str) -> float:
        """Compute retrieval velocity: retrievals per hour over the last 24 hours."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        row = conn.execute(
            """SELECT COUNT(*) FROM retrieval_log
               WHERE piece_id = ? AND retrieved_at > ?""",
            (piece_id, cutoff)
        ).fetchone()
        count = row[0] if row else 0
        return count / 24.0

    # ── Recompute all scores (batch job) ───────────────────────

    def recompute_all_scores(self):
        """Recompute zeitgeist scores for all pieces. Run periodically."""
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            pieces = conn.execute(
                "SELECT piece_id, retrieval_count, last_retrieved FROM pieces"
            ).fetchall()

            for piece_id, count, last_ret in pieces:
                velocity = self._compute_velocity(conn, piece_id)
                novelty = max(0.01, 1.0 / (1.0 + count * 0.1))
                recency_boost = math.exp(-_hours_ago(last_ret) / 168)
                freq_component = math.log1p(count + 1)
                zeitgeist = (freq_component * recency_boost + velocity * 10) * (0.3 + 0.7 * novelty)

                conn.execute(
                    """UPDATE pieces SET
                           retrieval_velocity = ?,
                           novelty_score = ?,
                           zeitgeist_score = ?
                       WHERE piece_id = ?""",
                    (velocity, novelty, zeitgeist, piece_id)
                )

    # ── Queries ────────────────────────────────────────────────

    def get_hot_pieces(self, limit: int = 10) -> list[dict]:
        """Get the pieces with the highest zeitgeist scores."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM pieces
                   WHERE retrieval_count > 0
                   ORDER BY zeitgeist_score DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_dormant_pieces(self, min_age_days: int = 14, limit: int = 5) -> list[dict]:
        """Get pieces that haven't been retrieved in a while."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM pieces
                   WHERE last_retrieved < ? OR last_retrieved IS NULL
                   ORDER BY last_retrieved ASC NULLS FIRST LIMIT ?""",
                (cutoff, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_dormant(self, min_age_days: int = 14, limit: int = 50) -> list[dict]:
        """Get all dormant pieces for sampling."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM pieces
                   WHERE last_retrieved < ? OR last_retrieved IS NULL
                   ORDER BY retrieval_count ASC LIMIT ?""",
                (cutoff, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_seismic_events(self, limit: int = 10) -> list[dict]:
        """Get recent anti-pattern break (SEISMIC) events."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT rl.*, p.title, p.directory
                   FROM retrieval_log rl
                   JOIN pieces p ON rl.piece_id = p.piece_id
                   WHERE rl.mode = 'SEISMIC'
                   ORDER BY rl.retrieved_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_piece_stats(self) -> dict:
        """Get aggregate statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM pieces").fetchone()[0]
            retrieved = conn.execute(
                "SELECT COUNT(*) FROM pieces WHERE retrieval_count > 0"
            ).fetchone()[0]
            total_retrievals = conn.execute(
                "SELECT COALESCE(SUM(retrieval_count), 0) FROM pieces"
            ).fetchone()[0]
            avg_velocity = conn.execute(
                "SELECT COALESCE(AVG(retrieval_velocity), 0) FROM pieces"
            ).fetchone()[0]

            return {
                "total_pieces": total,
                "pieces_retrieved": retrieved,
                "pieces_never_retrieved": total - retrieved,
                "total_retrievals": total_retrievals,
                "avg_velocity": round(avg_velocity, 4),
            }

    # ── NPC Reference Tracking ─────────────────────────────────

    def log_npc_reference(
        self,
        npc_name: str,
        piece_id: str,
        room: str = "",
        mode: str = "",
        reaction: str = "",
        title: str = "",
        directory: str = "",
    ):
        """Log that an NPC referenced a piece."""
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_piece(conn, piece_id, title, directory)
            conn.execute(
                """INSERT INTO npc_reference_log
                   (npc_name, piece_id, referenced_at, room, mode, reaction)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (npc_name, piece_id, now, room, mode, reaction)
            )

    def log_propagation(
        self,
        piece_id: str,
        source_npc: str,
        room: str,
        chain_id: str,
        depth: int = 0,
        reactor_npc: str = "",
        reaction: str = "",
    ):
        """Log a gossip propagation event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO propagation_log
                   (piece_id, source_npc, room, propagated_at, chain_id, depth, reactor_npc, reaction)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (piece_id, source_npc, room, _now_iso(), chain_id, depth, reactor_npc, reaction)
            )

    def log_behavioral_event(
        self,
        npc_name: str,
        piece_id: str,
        delta: float,
        severity: str,
        description: str,
    ):
        """Log a behavioral shift detection event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO behavioral_events
                   (npc_name, piece_id, detected_at, delta, severity, description)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (npc_name, piece_id, _now_iso(), delta, severity, description)
            )

    def get_propagation_chains(self, limit: int = 10) -> list[dict]:
        """Get recent propagation chains."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM propagation_log
                   ORDER BY propagated_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_npc_history(self, npc_name: str, limit: int = 50) -> list[dict]:
        """Get reference history for a specific NPC."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT nrl.*, p.title, p.directory
                   FROM npc_reference_log nrl
                   JOIN pieces p ON nrl.piece_id = p.piece_id
                   WHERE nrl.npc_name = ?
                   ORDER BY nrl.referenced_at DESC LIMIT ?""",
                (npc_name, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_behavioral_events(self, npc_name: str = None, limit: int = 20) -> list[dict]:
        """Get behavioral shift events, optionally filtered by NPC."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if npc_name:
                rows = conn.execute(
                    """SELECT * FROM behavioral_events
                       WHERE npc_name = ?
                       ORDER BY detected_at DESC LIMIT ?""",
                    (npc_name, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM behavioral_events
                       ORDER BY detected_at DESC LIMIT ?""",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
