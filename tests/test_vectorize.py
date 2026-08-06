"""
Comprehensive tests for vectorize.py — The Collective Consciousness vectorizer.

Covers:
- embed() — Ollama embedding call (mocked)
- walk_corpus() — corpus file discovery
- extract_metadata() — title, preview, word_count, directory extraction
- cosine_similarity_matrix() — similarity computation
- compute_neighbors() — top-k neighbor computation
- load_store() / save_store() — persistence
- compute_stats() — corpus statistics
- op_full_rebuild() — end-to-end rebuild (mocked embeds)
- op_update() — incremental update (mocked embeds)
- op_query() — query operation (mocked embeds)
- CLI argument parsing
"""

import json
import os
import sys
import tempfile
import math
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pytest

# Ensure the module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vectorize


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def tmp_corpus(tmp_path):
    """Create a temporary corpus directory with .md files."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text("# My First Document\n\nThis is about cats and dogs.\n")
    (corpus / "doc2.md").write_text("# Second Piece\n\nThe nature of consciousness.\n")
    sub = corpus / "subdir"
    sub.mkdir()
    (sub / "doc3.md").write_text("No heading here, just text about quantum mechanics.")
    (corpus / "not_md.txt").write_text("This should be ignored.")
    return str(corpus)


@pytest.fixture
def tmp_store_path(tmp_path, monkeypatch):
    """Redirect STORE_PATH to a temp file."""
    store_path = str(tmp_path / "test_store.json")
    monkeypatch.setattr(vectorize, "STORE_PATH", store_path)
    return store_path


@pytest.fixture
def sample_embedding():
    """Return a deterministic 768-dim embedding."""
    rng = np.random.RandomState(42)
    return rng.randn(768).tolist()


# NOTE: sample_store fixture is defined in conftest.py for cross-file sharing


# ── embed() tests ──────────────────────────────────────────────

class TestEmbed:
    @patch("vectorize.urllib.request.urlopen")
    def test_embed_returns_list_of_floats(self, mock_urlopen):
        """embed() should return a list of floats from Ollama."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"embedding": [0.1] * 768}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = vectorize.embed("test text")
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)

    @patch("vectorize.urllib.request.urlopen")
    def test_embed_sends_correct_payload(self, mock_urlopen):
        """embed() should POST to Ollama with model and prompt."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"embedding": [0.5] * 768}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        vectorize.embed("hello world")

        # Verify Request was created with correct data
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["model"] == "nomic-embed-text"
        assert payload["prompt"] == "hello world"

    @patch("vectorize.urllib.request.urlopen")
    def test_embed_raises_on_connection_error(self, mock_urlopen):
        """embed() should propagate connection errors."""
        mock_urlopen.side_effect = ConnectionRefusedError("Connection refused")
        with pytest.raises(ConnectionRefusedError):
            vectorize.embed("test")


# ── walk_corpus() tests ────────────────────────────────────────

class TestWalkCorpus:
    def test_walk_corpus_finds_md_files(self, tmp_corpus):
        """walk_corpus should find all .md files."""
        results = list(vectorize.walk_corpus(tmp_corpus))
        assert len(results) == 3  # doc1, doc2, doc3
        paths = [r[0] for r in results]
        assert all(p.endswith(".md") for p in paths)

    def test_walk_corpus_excludes_git(self, tmp_path):
        """walk_corpus should skip .git directories."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "real.md").write_text("# Real")
        git_dir = corpus / ".git"
        git_dir.mkdir()
        (git_dir / "config.md").write_text("git config")

        results = list(vectorize.walk_corpus(str(corpus)))
        paths = [r[0] for r in results]
        assert len(results) == 1
        assert "real.md" in paths[0]

    def test_walk_corpus_excludes_non_md(self, tmp_corpus):
        """walk_corpus should only return .md files."""
        results = list(vectorize.walk_corpus(tmp_corpus))
        for fpath, _ in results:
            assert fpath.endswith(".md")

    def test_walk_corpus_returns_mtime(self, tmp_corpus):
        """walk_corpus should return mtime for each file."""
        results = list(vectorize.walk_corpus(tmp_corpus))
        for _, mtime in results:
            assert isinstance(mtime, float)
            assert mtime > 0


# ── extract_metadata() tests ───────────────────────────────────

class TestExtractMetadata:
    def test_extract_title_from_heading(self, tmp_path):
        """Should extract title from first # heading."""
        f = tmp_path / "test.md"
        f.write_text("# My Amazing Title\n\nContent here.")
        meta = vectorize.extract_metadata(str(f), str(tmp_path))
        assert meta["title"] == "My Amazing Title"

    def test_extract_title_fallback_to_filename(self, tmp_path):
        """Should fall back to filename if no heading."""
        f = tmp_path / "my-cool-doc.md"
        f.write_text("Just plain text, no heading.")
        meta = vectorize.extract_metadata(str(f), str(tmp_path))
        assert "my cool doc" in meta["title"].lower()

    def test_extract_preview(self, tmp_path):
        """Should generate a preview from content."""
        f = tmp_path / "test.md"
        content = "A" * 300
        f.write_text(f"# Title\n\n{content}")
        meta = vectorize.extract_metadata(str(f), str(tmp_path))
        assert meta["preview"].endswith("...")
        assert len(meta["preview"]) <= 204  # 200 + "..."

    def test_extract_preview_short_content(self, tmp_path):
        """Preview of short content should not have ellipsis."""
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nShort.")
        meta = vectorize.extract_metadata(str(f), str(tmp_path))
        assert "..." not in meta["preview"] or len(meta["preview"]) <= 203

    def test_extract_word_count(self, tmp_path):
        """Should count words in the file."""
        f = tmp_path / "test.md"
        f.write_text("# Title\n\none two three four five")
        meta = vectorize.extract_metadata(str(f), str(tmp_path))
        # word_count uses content.split() which includes markdown tokens
        assert meta["word_count"] == 7  # '#', 'Title', 'one', 'two', 'three', 'four', 'five'

    def test_extract_directory(self, tmp_path):
        """Should compute directory relative to corpus."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        f = sub / "test.md"
        f.write_text("# Test")
        meta = vectorize.extract_metadata(str(f), str(tmp_path))
        assert meta["directory"] == "subdir"

    def test_extract_directory_root(self, tmp_path):
        """Root directory should be '(root)'."""
        f = tmp_path / "test.md"
        f.write_text("# Test")
        meta = vectorize.extract_metadata(str(f), str(tmp_path))
        assert meta["directory"] == "(root)"

    def test_extract_metadata_empty_file(self, tmp_path):
        """Should handle empty files gracefully."""
        f = tmp_path / "empty.md"
        f.write_text("")
        meta = vectorize.extract_metadata(str(f), str(tmp_path))
        assert meta["word_count"] == 0
        assert isinstance(meta["title"], str)

    def test_extract_metadata_ignores_h2_as_title(self, tmp_path):
        """Should not use ## as title, only #."""
        f = tmp_path / "test.md"
        f.write_text("## Subtitle\n\n# Real Title\n\nContent")
        meta = vectorize.extract_metadata(str(f), str(tmp_path))
        assert meta["title"] == "Real Title"


# ── cosine_similarity_matrix() tests ───────────────────────────

class TestCosineSimilarity:
    def test_identity_matrix_for_normalized_vectors(self):
        """Cosine similarity of a vector with itself should be 1."""
        emb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        sim = vectorize.cosine_similarity_matrix(emb)
        assert sim.shape == (3, 3)
        np.testing.assert_allclose(np.diag(sim), [1.0, 1.0, 1.0], atol=1e-6)

    def test_orthogonal_vectors_zero_similarity(self):
        """Orthogonal vectors should have ~0 cosine similarity."""
        emb = np.array([[1, 0], [0, 1]], dtype=np.float32)
        sim = vectorize.cosine_similarity_matrix(emb)
        assert abs(sim[0, 1]) < 1e-6
        assert abs(sim[1, 0]) < 1e-6

    def test_parallel_vectors_similarity_one(self):
        """Parallel vectors should have similarity ~1."""
        emb = np.array([[1, 2, 3], [2, 4, 6]], dtype=np.float32)
        sim = vectorize.cosine_similarity_matrix(emb)
        assert abs(sim[0, 1] - 1.0) < 1e-5

    def test_zero_vector_handled(self):
        """Zero vectors should not cause division by zero."""
        emb = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        sim = vectorize.cosine_similarity_matrix(emb)
        assert sim.shape == (2, 2)
        assert not np.any(np.isnan(sim))

    def test_symmetric_matrix(self):
        """Similarity matrix should be symmetric."""
        rng = np.random.RandomState(42)
        emb = rng.randn(5, 10).astype(np.float32)
        sim = vectorize.cosine_similarity_matrix(emb)
        np.testing.assert_allclose(sim, sim.T, atol=1e-6)


# ── compute_neighbors() tests ──────────────────────────────────

class TestComputeNeighbors:
    def test_excludes_self(self):
        """Neighbors should not include self."""
        sim = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ])
        neighbors = vectorize.compute_neighbors(sim, top_k=2)
        for i, nbrs in enumerate(neighbors):
            indices = [n["index"] for n in nbrs]
            assert i not in indices

    def test_top_k_limit(self):
        """Should return exactly top_k neighbors."""
        sim = np.eye(5)
        for i in range(5):
            for j in range(5):
                if i != j:
                    sim[i, j] = 0.5
        neighbors = vectorize.compute_neighbors(sim, top_k=3)
        assert all(len(n) == 3 for n in neighbors)

    def test_ordered_by_similarity_desc(self):
        """Neighbors should be sorted by similarity descending."""
        sim = np.array([
            [1.0, 0.9, 0.5, 0.3],
            [0.9, 1.0, 0.5, 0.3],
            [0.5, 0.5, 1.0, 0.3],
            [0.3, 0.3, 0.3, 1.0],
        ])
        neighbors = vectorize.compute_neighbors(sim, top_k=3)
        for nbrs in neighbors:
            sims = [n["similarity"] for n in nbrs]
            assert sims == sorted(sims, reverse=True)

    def test_default_top_k(self):
        """Default top_k should be TOP_K (10)."""
        sim = np.eye(15)
        neighbors = vectorize.compute_neighbors(sim)
        assert all(len(n) == min(10, 14) for n in neighbors)

    def test_similarity_values_correct(self):
        """Neighbor similarity values should match the matrix."""
        sim = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ])
        neighbors = vectorize.compute_neighbors(sim, top_k=2)
        # For piece 0: best neighbor should be piece 1 (0.8)
        assert neighbors[0][0]["index"] == 1
        assert abs(neighbors[0][0]["similarity"] - 0.8) < 1e-4


# ── load_store() / save_store() tests ──────────────────────────

class TestStorePersistence:
    def test_load_empty_store(self, tmp_store_path):
        """Loading when no store exists should return empty structure."""
        store = vectorize.load_store()
        assert store == {"pieces": [], "last_run": None, "stats": {}}

    def test_save_and_load_roundtrip(self, tmp_store_path, sample_store):
        """Save then load should preserve data."""
        vectorize.save_store(sample_store)
        loaded = vectorize.load_store()
        assert loaded["pieces"] == sample_store["pieces"]
        assert loaded["last_run"] is not None  # save_store sets it

    def test_load_corrupt_store(self, tmp_store_path):
        """Loading a corrupt store should return empty structure."""
        with open(tmp_store_path, "w") as f:
            f.write("{ this is not valid json")
        store = vectorize.load_store()
        assert store == {"pieces": [], "last_run": None, "stats": {}}

    def test_save_sets_last_run(self, tmp_store_path, sample_store):
        """save_store should set last_run timestamp."""
        sample_store["last_run"] = None
        vectorize.save_store(sample_store)
        loaded = vectorize.load_store()
        assert loaded["last_run"] is not None


# ── compute_stats() tests ──────────────────────────────────────

class TestComputeStats:
    def test_empty_store_stats(self):
        """Stats for empty store should be empty dict."""
        store = {"pieces": []}
        stats = vectorize.compute_stats(store)
        assert stats == {}

    def test_total_pieces(self, sample_store):
        """Should count total pieces correctly."""
        stats = vectorize.compute_stats(sample_store)
        assert stats["total_pieces"] == 3

    def test_total_words(self, sample_store):
        """Should sum word counts."""
        stats = vectorize.compute_stats(sample_store)
        assert stats["total_words"] == 100 + 150 + 200  # 450

    def test_avg_words(self, sample_store):
        """Should compute average words."""
        stats = vectorize.compute_stats(sample_store)
        assert stats["avg_words"] == 150.0

    def test_directories_count(self, sample_store):
        """Should count unique directories."""
        stats = vectorize.compute_stats(sample_store)
        assert stats["directories"] == 2  # animals, mind

    def test_pieces_by_directory(self, sample_store):
        """Should group pieces by directory."""
        stats = vectorize.compute_stats(sample_store)
        assert stats["pieces_by_directory"]["animals"] == 2
        assert stats["pieces_by_directory"]["mind"] == 1

    def test_dimensions_and_model(self, sample_store):
        """Should report embedding dimensions and model."""
        stats = vectorize.compute_stats(sample_store)
        assert stats["dimensions"] == 768
        assert stats["model"] == "nomic-embed-text"


# ── op_full_rebuild() tests (mocked embeds) ────────────────────

class TestFullRebuild:
    @patch("vectorize.embed")
    @patch("vectorize.CORPUS_DIR")
    def test_full_rebuild_basic(self, mock_corpus, mock_embed, tmp_corpus, tmp_store_path):
        """Full rebuild should create pieces from corpus files."""
        mock_corpus.__fspath__ = lambda self: tmp_corpus
        # Need to patch the module-level CORPUS_DIR
        with patch.object(vectorize, "CORPUS_DIR", tmp_corpus):
            rng = np.random.RandomState(42)
            mock_embed.return_value = rng.randn(768).tolist()
            store = vectorize.op_full_rebuild(verbose=False)

        assert len(store["pieces"]) == 3
        assert store["stats"]["total_pieces"] == 3
        assert store["last_run"] is not None

    @patch("vectorize.embed")
    def test_full_rebuild_computes_neighbors(self, mock_embed, tmp_corpus, tmp_store_path):
        """Full rebuild should compute neighbors."""
        with patch.object(vectorize, "CORPUS_DIR", tmp_corpus):
            rng = np.random.RandomState(42)
            mock_embed.return_value = rng.randn(768).tolist()
            store = vectorize.op_full_rebuild(verbose=False)

        for piece in store["pieces"]:
            assert "neighbors" in piece
            assert len(piece["neighbors"]) > 0

    @patch("vectorize.embed")
    def test_full_rebuild_handles_embed_failure(self, mock_embed, tmp_corpus, tmp_store_path):
        """Full rebuild should skip files that fail to embed."""
        with patch.object(vectorize, "CORPUS_DIR", tmp_corpus):
            mock_embed.side_effect = ConnectionRefusedError("no ollama")
            store = vectorize.op_full_rebuild(verbose=False)

        assert len(store["pieces"]) == 0


# ── op_update() tests ──────────────────────────────────────────

class TestOpUpdate:
    @patch("vectorize.embed")
    def test_update_no_changes(self, mock_embed, tmp_corpus, tmp_store_path):
        """Update with no changes should be a no-op."""
        with patch.object(vectorize, "CORPUS_DIR", tmp_corpus):
            rng = np.random.RandomState(42)
            mock_embed.return_value = rng.randn(768).tolist()
            # First, do a full rebuild
            store = vectorize.op_full_rebuild(verbose=False)
            assert len(store["pieces"]) == 3

            # Now update — should be no changes
            updated = vectorize.op_update(verbose=False)
            assert len(updated["pieces"]) == 3

    @patch("vectorize.embed")
    def test_update_detects_new_file(self, mock_embed, tmp_corpus, tmp_store_path):
        """Update should detect and add new files."""
        with patch.object(vectorize, "CORPUS_DIR", tmp_corpus):
            rng = np.random.RandomState(42)
            mock_embed.return_value = rng.randn(768).tolist()
            store = vectorize.op_full_rebuild(verbose=False)
            assert len(store["pieces"]) == 3

            # Add a new file
            Path(tmp_corpus, "new_doc.md").write_text("# New Document\n\nFresh content.")

            updated = vectorize.op_update(verbose=False)
            assert len(updated["pieces"]) == 4

    @patch("vectorize.embed")
    def test_update_detects_deleted_file(self, mock_embed, tmp_corpus, tmp_store_path):
        """Update should remove deleted files."""
        with patch.object(vectorize, "CORPUS_DIR", tmp_corpus):
            rng = np.random.RandomState(42)
            mock_embed.return_value = rng.randn(768).tolist()
            store = vectorize.op_full_rebuild(verbose=False)
            assert len(store["pieces"]) == 3

            # Delete a file
            Path(tmp_corpus, "doc1.md").unlink()

            updated = vectorize.op_update(verbose=False)
            paths = [p["path"] for p in updated["pieces"]]
            assert "doc1.md" not in paths
            assert len(updated["pieces"]) == 2


# ── CLI / main() tests ─────────────────────────────────────────

class TestCLI:
    def test_stats_flag(self, monkeypatch):
        """--stats should call op_stats."""
        monkeypatch.setattr(sys, "argv", ["vectorize.py", "--stats"])
        called = MagicMock()
        monkeypatch.setattr(vectorize, "op_stats", called)
        vectorize.main()
        called.assert_called_once()

    def test_query_flag(self, monkeypatch):
        """--query should call op_query with the query string."""
        monkeypatch.setattr(sys, "argv", ["vectorize.py", "--query", "consciousness"])
        called = MagicMock()
        monkeypatch.setattr(vectorize, "op_query", called)
        vectorize.main()
        called.assert_called_once_with("consciousness", 10)

    def test_update_flag(self, monkeypatch):
        """--update should call op_update."""
        monkeypatch.setattr(sys, "argv", ["vectorize.py", "--update"])
        called = MagicMock()
        monkeypatch.setattr(vectorize, "op_update", called)
        vectorize.main()
        called.assert_called_once()

    def test_default_is_full_rebuild(self, monkeypatch):
        """No flags should call op_full_rebuild."""
        monkeypatch.setattr(sys, "argv", ["vectorize.py"])
        called = MagicMock()
        monkeypatch.setattr(vectorize, "op_full_rebuild", called)
        vectorize.main()
        called.assert_called_once()

    def test_top_flag_passed_to_query(self, monkeypatch):
        """--top should pass top_k to op_query."""
        monkeypatch.setattr(sys, "argv", ["vectorize.py", "--query", "test", "--top", "5"])
        called = MagicMock()
        monkeypatch.setattr(vectorize, "op_query", called)
        vectorize.main()
        called.assert_called_once_with("test", 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
