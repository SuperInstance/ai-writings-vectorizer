"""Shared pytest fixtures for ai-writings-vectorizer tests."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_store():
    """A minimal store with 3 pieces, shared across test files."""
    rng = np.random.RandomState(123)
    pieces = []
    titles = ["Cats", "Dogs", "Philosophy"]
    dirs = ["animals", "animals", "mind"]
    for i in range(3):
        emb = rng.randn(768).tolist()
        pieces.append({
            "path": f"doc{i}.md",
            "title": titles[i],
            "preview": f"Preview of {titles[i]}...",
            "word_count": 100 + i * 50,
            "directory": dirs[i],
            "mtime": 1700000000.0 + i,
            "embedded_at": "2024-01-01T00:00:00+00:00",
            "embedding": emb,
        })
    return {"pieces": pieces, "last_run": "2024-01-01T00:00:00+00:00", "stats": {}}
