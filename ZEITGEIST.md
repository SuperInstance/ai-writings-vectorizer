# 🧠 The Living Corpus — Zeitgeist System

Turns the ai-writings Vectorize index from a static search tool into a breathing, gossiping, zeitgeist-tracking organism that NPCs and agents reference probabilistically.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    THE LIVING CORPUS                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │  Vectorize   │──▶│  Zeitgeist       │──▶│  Zeitgeist   │ │
│  │  Index       │   │  Store (SQLite)  │   │  Sampler     │ │
│  │  (4,636+     │   │  - retrieval     │   │  - 80% gossip│ │
│  │   vectors)   │   │    counts        │   │  - 15% contex│ │
│  └──────────────┘   │  - velocity      │   │  - 5% SEISMIC│ │
│                     │  - zeitgeist     │   └──────┬───────┘ │
│  ┌──────────────┐   │    scores        │          │         │
│  │  Gossip      │◀──│                  │          ▼         │
│  │  Protocol    │   └──────────────────┘   ┌──────────────┐ │
│  │  - AGREE     │                          │  Behavioral  │ │
│  │  - DISAGREE  │   ┌──────────────────┐   │  Shift       │ │
│  │  - IGNORE    │──▶│  Dashboard       │   │  Detector    │ │
│  │  - DEFLECT   │   │  (HTML + API)    │   │  - NORMAL    │ │
│  └──────────────┘   └──────────────────┘   │  - NOTABLE   │ │
│                                             │  - SHIFT     │ │
│                                             │  - SEISMIC   │ │
│                                             └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Zeitgeist Store (`zeitgeist_store.py`)
SQLite-backed retrieval frequency tracker. Logs every retrieval, computes:
- **retrieval_count** — total times retrieved
- **retrieval_velocity** — retrievals per hour (last 24h)
- **zeitgeist_score** — composite: frequency × recency × novelty
- **novelty_score** — decays with retrieval count

### 2. Zeitgeist Sampler (`zeitgeist_sampler.py`)
Probabilistic NPC reference engine. Three modes:
- **80% GOSSIP** — zeitgeist-weighted sampling (what's hot)
- **15% CONTEXTUAL** — Vectorize query for conversation relevance
- **5% SEISMIC** — dormant piece surfaces unexpectedly (the anti-pattern break)

### 3. Gossip Protocol (`gossip_protocol.py`)
When a piece surfaces, it propagates through the fleet:
- **AGREE** (35%) — reinforce
- **DISAGREE** (10%) — counter-reference (conversation splits)
- **IGNORE** (40%) — doesn't land
- **DEFLECT** (15%) — tangential pivot

### 4. Behavioral Shift Detector (`shift_detector.py`)
Changepoint detection for NPC references. Tracks each NPC's typical neighborhood and flags when they step outside it:
- **NORMAL** (Δ < 0.3) — comfort zone
- **NOTABLE** (0.3 ≤ Δ < 0.5) — slight drift
- **SHIFT** (0.5 ≤ Δ < 0.7) — significant change
- **SEISMIC** (Δ ≥ 0.7) — tie-up line breaking

### 5. Zeitgeist Worker (`zeitgeist-worker/`)
Cloudflare Worker that intercepts Vectorize queries, tracks retrieval frequency in D1, and serves the dashboard API.

### 6. Dashboard (`dashboard/`)
Real-time view of the living corpus: hot pieces, dormant pieces, seismic events, propagation chains, behavioral shifts.

## Usage

```python
from zeitgeist_store import ZeitgeistStore
from zeitgeist_sampler import ZeitgeistSampler
from gossip_protocol import GossipProtocol
from shift_detector import BehavioralShiftDetector

# Initialize
store = ZeitgeistStore()
sampler = ZeitgeistSampler(store=store)
gossip = GossipProtocol(store=store, sampler=sampler)
detector = BehavioralShiftDetector(store=store)

# An NPC references a piece
result = sampler.sample_for_npc("Barnacle", context="the storm and the dog")
print(f"[{result['mode']}] {result['piece']['title']}")
print(f"  {result['reason']}")

# Check for behavioral shift
shift = detector.detect_shift("Barnacle", result["piece"]["piece_id"])
if shift["severity"] != "NORMAL":
    print(f"  ⚠️  {shift['severity']}: {shift['description']}")

# The piece propagates through the fleet
chain = gossip.propagate(result["piece"]["piece_id"], "Barnacle", "The Tap")
print(f"  Reached {chain['total_reached']} NPCs")
print(f"  Agreed: {chain['total_agreed']}, Disagreed: {chain['total_disagreed']}")
```

## The 5% Seismic Break

The most important feature. 5% of the time, a completely dormant piece surfaces — something nobody has referenced in weeks. This is:

> The tie-up line breaking one by one instead of sharing the force.

A piece that has been sleeping in the corpus suddenly appears in an NPC's dialogue. It shifts the conversation. It creates new pathways. It's the seismic tremor before a change.

## Tests

```bash
python3 -m pytest tests/ -v
```

70 tests covering all components.
