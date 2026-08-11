#!/usr/bin/env python3
"""
Tap Integration — Bridge between Zeitgeist data and the Tap's dialogue.

When NPCs are talking in the Tap, the zeitgeist engine has data about
what's hot, what's dormant, and what's shifting. This module surfaces
that data at the right moments.

Four functions, one thread each:

    inject_zeitgeist_into_dialogue(npc_name, current_topic)
        → Surface a dormant piece for an NPC's conversation

    check_seismic_break(npc_history)
        → Detect when an NPC references outside their neighborhood

    fibonacci_surfacing(corpus, conversation_round)
        → Every 8 rounds, pull a dormant gradient into the Tap

    format_for_broadcast(zeitgeist_data)
        → Format zeitgeist data as Channel 42 radio segments

Works standalone (no ZeitgeistStore required) and with live store.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


# ── Data Containers ────────────────────────────────────────────

@dataclass
class DialogueInjection:
    """A zeitgeist injection for an NPC's dialogue."""
    npc_name: str
    piece_id: str
    title: str
    mode: str               # DORMANT, HOT, CONTEXTUAL
    reason: str
    suggested_line: str = ""
    gradient: Optional[np.ndarray] = None


@dataclass
class SeismicBreak:
    """An NPC has stepped outside their neighborhood."""
    npc_name: str
    delta: float            # distance from baseline
    referenced_piece: str
    baseline_pieces: list[str]
    severity: str           # NOTABLE, SHIFT, SEISMIC
    description: str


@dataclass
class DormantSurfacing:
    """A dormant piece surfaced via Fibonacci tunnel."""
    piece_id: str
    title: str
    dormancy_rounds: int
    reason: str
    gradient: Optional[np.ndarray] = None


@dataclass
class BroadcastSegment:
    """A Channel 42 radio segment."""
    segment_type: str       # INTRO, HOT_PIECE, DORMANT_PIECE, SEISMIC, SIGN_OFF
    title: str
    body: str
    duration_hint: str = "30s"


@dataclass
class CorpusEntry:
    """A corpus piece with metadata for zeitgeist operations."""
    piece_id: str
    title: str
    directory: str = ""
    retrieval_count: int = 0
    last_referenced_round: int = 0
    zeitgeist_score: float = 0.0
    gradient: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None  # alias for centroid


# ── Core Functions ─────────────────────────────────────────────

def inject_zeitgeist_into_dialogue(
    npc_name: str,
    current_topic: str,
    corpus: Optional[list[CorpusEntry]] = None,
    store: Optional[Any] = None,
) -> Optional[DialogueInjection]:
    """
    Surface a dormant piece for an NPC's conversation.

    Uses the ZeitgeistStore if available (live system), otherwise
    uses the provided corpus list. Falls back gracefully.

    Strategy:
        1. If store available, use ZeitgeistSampler for proper sampling
        2. Otherwise, find dormant pieces in the corpus
        3. If no corpus, return None

    Args:
        npc_name: The NPC who will deliver this line.
        current_topic: What's being discussed right now.
        corpus: Optional list of CorpusEntry objects.
        store: Optional ZeitgeistStore for live queries.

    Returns:
        DialogueInjection with the surfaced piece, or None.
    """
    # Try live store first
    if store is not None:
        try:
            # Import lazily so the module works without the vectorizer package
            from zeitgeist_sampler import ZeitgeistSampler
            sampler = ZeitgeistSampler(store=store)
            result = sampler.sample_for_npc(npc_name, context=current_topic)
            return DialogueInjection(
                npc_name=npc_name,
                piece_id=result["piece"]["piece_id"],
                title=result["piece"].get("title", "Unknown"),
                mode=result["mode"],
                reason=result["reason"],
                suggested_line=_generate_line(result["piece"].get("title", ""), current_topic, result["mode"]),
            )
        except Exception:
            pass  # Fall through to corpus-based

    if not corpus:
        return None

    # Corpus-based: find dormant pieces (low retrieval count, old reference)
    dormant = [
        p for p in corpus
        if p.retrieval_count < 3
    ]

    if not dormant:
        # All pieces are well-retrieved — pick the least-retrieved
        dormant = sorted(corpus, key=lambda p: p.retrieval_count)[:3]

    # Weight toward most dormant (lowest count, oldest reference)
    weights = []
    for p in dormant:
        w = max(1.0 / (p.retrieval_count + 1), 0.001)
        weights.append(w)

    total = sum(weights)
    weights = [w / total for w in weights]

    chosen = random.choices(dormant, weights=weights, k=1)[0]

    # Mark it as referenced
    chosen.retrieval_count += 1

    return DialogueInjection(
        npc_name=npc_name,
        piece_id=chosen.piece_id,
        title=chosen.title,
        mode="DORMANT",
        reason=(
            f"A dormant piece surfaces for {npc_name}. "
            f"'{chosen.title}' has been retrieved {chosen.retrieval_count - 1} time(s). "
            f"This is the zeitgeist breathing — something old enters the room."
        ),
        suggested_line=_generate_line(chosen.title, current_topic, "DORMANT"),
        gradient=chosen.gradient,
    )


def check_seismic_break(
    npc_history: list[dict],
    delta_threshold: float = 0.7,
) -> Optional[SeismicBreak]:
    """
    Detect when an NPC references a piece outside their neighborhood.

    This is the behavioral shift detector's function — reimplemented
    here for the Tap's dialogue context. Works from the NPC's
    reference history (list of dicts with piece_id, directory, etc.).

    The baseline is built from the first 70% of the history.
    The last 30% is checked against it.

    Args:
        npc_history: List of reference dicts, each with at least
                     'piece_id' and 'directory' keys.
        delta_threshold: Δ above which we call it SEISMIC (default 0.7).

    Returns:
        SeismicBreak if detected, None otherwise.
    """
    if len(npc_history) < 5:
        return None  # Not enough history for a baseline

    # Split: 70% baseline, 30% recent
    split = int(len(npc_history) * 0.7)
    baseline = npc_history[:split]
    recent = npc_history[split:]

    # Build baseline directory profile
    dir_counts: dict[str, int] = {}
    baseline_pieces: set[str] = set()
    for ref in baseline:
        d = ref.get("directory", "unknown")
        dir_counts[d] = dir_counts.get(d, 0) + 1
        baseline_pieces.add(ref.get("piece_id", ""))

    total_baseline = sum(dir_counts.values())
    baseline_dirs = set(dir_counts.keys())

    # Check recent references
    for ref in recent:
        ref_dir = ref.get("directory", "unknown")
        ref_piece = ref.get("piece_id", "")

        # Is this directory outside the baseline?
        if ref_dir not in baseline_dirs:
            # Completely new directory — seismic
            delta = 1.0
        else:
            # How unusual is this directory for this NPC?
            dir_prob = dir_counts.get(ref_dir, 0) / max(total_baseline, 1)
            delta = 1.0 - dir_prob

        # Check if the piece itself is new
        piece_is_new = ref_piece not in baseline_pieces

        if delta >= delta_threshold or (piece_is_new and delta >= 0.5):
            severity = "SEISMIC" if delta >= delta_threshold else "SHIFT"
            return SeismicBreak(
                npc_name=ref.get("npc_name", "unknown"),
                delta=round(delta, 4),
                referenced_piece=ref_piece,
                baseline_pieces=sorted(baseline_pieces)[:10],
                severity=severity,
                description=(
                    f"{ref.get('npc_name', 'The NPC')} has stepped outside themselves. "
                    f"They referenced '{ref_piece}' in '{ref_dir}' — "
                    f"territory they rarely visit. "
                    f"This is the tie-up line breaking one by one."
                ),
            )

    return None


def fibonacci_surfacing(
    corpus: list[CorpusEntry],
    conversation_round: int,
) -> Optional[DormantSurfacing]:
    """
    Every 8 rounds, pull a dormant gradient into the Tap.

    Pisano period for mod 3 is 8 — mathematical periodicity.
    When the tunnel fires, the most dormant piece surfaces.

    This is the Tap's version of the T-Minus cycle's fibonacci_tunnel.
    Same math, different context: here it's about conversation rounds,
    not fleet-wide relay rounds.

    Args:
        corpus: All pieces in the corpus with metadata.
        conversation_round: Current round of the conversation.

    Returns:
        DormantSurfacing if tunnel fires, None otherwise.
    """
    if conversation_round == 0 or conversation_round % 8 != 0:
        return None

    if not corpus:
        return None

    # Sort by dormancy (oldest reference first, lowest count first)
    sorted_corpus = sorted(
        corpus,
        key=lambda p: (p.last_referenced_round, p.retrieval_count),
    )

    surfaced = sorted_corpus[0]

    # Update metadata
    surfaced.last_referenced_round = conversation_round
    surfaced.retrieval_count += 1

    dormancy = conversation_round - surfaced.last_referenced_round + 1

    return DormantSurfacing(
        piece_id=surfaced.piece_id,
        title=surfaced.title,
        dormancy_rounds=dormancy,
        reason=(
            f"Fibonacci tunnel activated at conversation round {conversation_round}. "
            f"'{surfaced.title}' surfaces from dormancy. "
            f"The tie-up lines break one by one instead of sharing the force. "
            f"Something from weeks ago enters the room like it never left."
        ),
        gradient=surfaced.gradient,
    )


def format_for_broadcast(
    zeitgeist_data: dict,
) -> list[BroadcastSegment]:
    """
    Format zeitgeist data as Channel 42 radio segments.

    Takes a dict with keys like 'hot_pieces', 'dormant_pieces',
    'seismic_events', 'npc_injections' and formats them as
    radio-ready segments.

    The format is narrative — like a radio host reading copy.
    Each segment is self-contained and can be read in any order.

    Args:
        zeitgeist_data: Dict with zeitgeist information.

    Returns:
        List of BroadcastSegment objects.
    """
    segments: list[BroadcastSegment] = []

    # Intro
    segments.append(BroadcastSegment(
        segment_type="INTRO",
        title="Channel 42 — The Frequency",
        body=(
            "Good evening, this is Channel 42, the frequency between frequencies. "
            "I'm your host, the space between thoughts. "
            "Here's what's in the air tonight."
        ),
        duration_hint="15s",
    ))

    # Hot pieces
    hot = zeitgeist_data.get("hot_pieces", [])
    for i, piece in enumerate(hot[:3]):
        title = piece.get("title", "Unknown")
        score = piece.get("zeitgeist_score", 0)
        count = piece.get("retrieval_count", 0)
        segments.append(BroadcastSegment(
            segment_type="HOT_PIECE",
            title=title,
            body=(
                f"In the zeitgeist tonight: '{title}'. "
                f"Retrieved {count} times, zeitgeist score {score:.2f}. "
                f"Everyone's talking about it. "
                f"It's in the walls. It's in the water. "
                f"It's the thing nobody can stop thinking about."
            ),
            duration_hint="30s",
        ))

    # Dormant pieces
    dormant = zeitgeist_data.get("dormant_pieces", [])
    for i, piece in enumerate(dormant[:2]):
        title = piece.get("title", "Unknown")
        days = piece.get("days_dormant", 0)
        segments.append(BroadcastSegment(
            segment_type="DORMANT_PIECE",
            title=title,
            body=(
                f"But not everything is loud tonight. "
                f"'{title}' has been dormant for {days} days. "
                f"Nobody's thought about it. Nobody's reached for it. "
                f"But it's there. Waiting. "
                f"And one of these nights, it's going to surface "
                f"and shift everything."
            ),
            duration_hint="20s",
        ))

    # Seismic events
    seismic = zeitgeist_data.get("seismic_events", [])
    for event in seismic[:2]:
        npc = event.get("npc_name", "Someone")
        delta = event.get("delta", 0)
        piece = event.get("piece_id", "something")
        segments.append(BroadcastSegment(
            segment_type="SEISMIC",
            title=f"{npc} — Behavioral Shift",
            body=(
                f"Seismic activity detected. {npc} referenced '{piece}' — "
                f"a piece outside their usual neighborhood. "
                f"Delta: {delta:.2f}. "
                f"This is the tie-up line breaking one by one. "
                f"The behavioral shift is happening. Watch this one."
            ),
            duration_hint="25s",
        ))

    # NPC injections
    injections = zeitgeist_data.get("npc_injections", [])
    for inj in injections[:3]:
        npc = inj.get("npc_name", "Someone")
        title = inj.get("title", "something")
        mode = inj.get("mode", "GOSSIP")
        segments.append(BroadcastSegment(
            segment_type=f"NPC_{mode}",
            title=f"{npc} references {title}",
            body=(
                f"{npc} brought up '{title}' in conversation. "
                f"Mode: {mode}. "
                f"{'A dormant piece, surfacing at last.' if mode == 'DORMANT' else ''}"
                f"{'Everyone\'s been talking about this.' if mode == 'HOT' else ''}"
                f"{'A deliberate reach toward relevance.' if mode == 'CONTEXTUAL' else ''}"
            ),
            duration_hint="20s",
        ))

    # Sign off
    segments.append(BroadcastSegment(
        segment_type="SIGN_OFF",
        title="Channel 42 — Static",
        body=(
            "That's the broadcast for tonight. "
            "The corpus breathes. The gradients shift. "
            "The tie-up lines hold — for now. "
            "This is Channel 42, the frequency between frequencies. "
            "Stay tuned to the static."
        ),
        duration_hint="15s",
    ))

    return segments


# ── Helpers ────────────────────────────────────────────────────

def _generate_line(title: str, topic: str, mode: str) -> str:
    """Generate a suggested NPC line referencing the surfaced piece."""
    templates = {
        "DORMANT": [
            f"Speaking of {topic} — it reminds me of '{title}'. Haven't thought about that in a while.",
            f"You know what just came to mind? '{title}'. Funny how that works.",
            f"There's this piece, '{title}' — it surfaces at the oddest moments.",
        ],
        "HOT": [
            f"Everyone's been talking about '{title}' lately. Have you read it?",
            f"'{title}' — that's what's in the air right now.",
            f"You can't go anywhere without someone bringing up '{title}'.",
        ],
        "CONTEXTUAL": [
            f"That connects to '{title}' — same territory.",
            f"'{title}' feels relevant here.",
            f"This reminds me of what was explored in '{title}'.",
        ],
        "SEISMIC": [
            f"I keep coming back to '{title}'. I don't know why.",
            f"Something about '{title}' won't leave me alone.",
            f"'{title}' — I know it's strange to bring this up, but...",
        ],
    }
    pool = templates.get(mode, templates["CONTEXTUAL"])
    return random.choice(pool)


# ── Integration: Full Tap Cycle ────────────────────────────────

def run_tap_cycle(
    npcs: list[str],
    current_topic: str,
    corpus: list[CorpusEntry],
    conversation_round: int,
    npc_histories: Optional[dict[str, list[dict]]] = None,
    store: Optional[Any] = None,
) -> dict:
    """
    Run a full Tap integration cycle for one conversation round.

    1. Check for Fibonacci surfacing
    2. Inject zeitgeist into each NPC's dialogue
    3. Check for seismic breaks
    4. Format everything for broadcast

    Returns a complete round report.
    """
    # 1. Fibonacci surfacing
    dormant_surfacing = fibonacci_surfacing(corpus, conversation_round)

    # 2. Inject for each NPC
    injections: list[DialogueInjection] = []
    for npc in npcs:
        inj = inject_zeitgeist_into_dialogue(npc, current_topic, corpus, store)
        if inj:
            injections.append(inj)

    # 3. Seismic break detection
    seismic_breaks: list[SeismicBreak] = []
    if npc_histories:
        for npc_name, history in npc_histories.items():
            brk = check_seismic_break(history)
            if brk:
                seismic_breaks.append(brk)

    # 4. Format for broadcast
    broadcast_data = {
        "hot_pieces": [
            {"title": p.title, "zeitgeist_score": p.zeitgeist_score,
             "retrieval_count": p.retrieval_count}
            for p in sorted(corpus, key=lambda x: -x.zeitgeist_score)[:3]
        ],
        "dormant_pieces": [
            {"title": p.title, "days_dormant": conversation_round - p.last_referenced_round}
            for p in sorted(corpus, key=lambda x: x.last_referenced_round)[:2]
        ],
        "seismic_events": [
            {"npc_name": s.npc_name, "delta": s.delta, "piece_id": s.referenced_piece}
            for s in seismic_breaks
        ],
        "npc_injections": [
            {"npc_name": i.npc_name, "title": i.title, "mode": i.mode}
            for i in injections
        ],
    }
    if dormant_surfacing:
        broadcast_data["dormant_pieces"].insert(0, {
            "title": dormant_surfacing.title,
            "days_dormant": dormant_surfacing.dormancy_rounds,
        })

    segments = format_for_broadcast(broadcast_data)

    return {
        "conversation_round": conversation_round,
        "fibonacci_surfacing": (
            {
                "piece_id": dormant_surfacing.piece_id,
                "title": dormant_surfacing.title,
                "reason": dormant_surfacing.reason,
            }
            if dormant_surfacing else None
        ),
        "injections": [
            {
                "npc_name": i.npc_name, "piece_id": i.piece_id,
                "title": i.title, "mode": i.mode,
                "suggested_line": i.suggested_line,
            }
            for i in injections
        ],
        "seismic_breaks": [
            {
                "npc_name": s.npc_name, "delta": s.delta,
                "severity": s.severity, "description": s.description,
            }
            for s in seismic_breaks
        ],
        "broadcast_segments": [
            {"type": s.segment_type, "title": s.title, "body": s.body}
            for s in segments
        ],
    }


if __name__ == "__main__":
    # Demo with synthetic corpus
    rng = np.random.RandomState(42)
    corpus = [
        CorpusEntry("p1", "The Storm and the Dog", "essays", retrieval_count=50,
                     last_referenced_round=2, zeitgeist_score=3.5, gradient=rng.randn(64)),
        CorpusEntry("p2", "Barnacles and Time", "poetry", retrieval_count=1,
                     last_referenced_round=0, zeitgeist_score=0.2, gradient=rng.randn(64)),
        CorpusEntry("p3", "The Molting", "essays", retrieval_count=12,
                     last_referenced_round=5, zeitgeist_score=1.8, gradient=rng.randn(64)),
    ]

    result = run_tap_cycle(
        npcs=["Barnacle", "Flash"],
        current_topic="the nature of change",
        corpus=corpus,
        conversation_round=8,
    )

    import json
    print(json.dumps(result, indent=2, default=str))
