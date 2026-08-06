"""
Comprehensive tests for query.py — Cloudflare Vectorize semantic search client.

Covers:
- get_token() — credential resolution from env and wrangler config
- embed_query() — local Ollama embedding (mocked)
- query_vectorize() — Cloudflare API call (mocked)
- format_score() — score color formatting
- display_results() — terminal and JSON output
- CLI argument parsing
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import query


# ── get_token() tests ──────────────────────────────────────────

class TestGetToken:
    def test_token_from_env(self, monkeypatch):
        """Should read CLOUDFLARE_API_TOKEN from env."""
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token-123")
        assert query.get_token() == "test-token-123"

    def test_token_from_cf_token_env(self, monkeypatch):
        """Should read CLOUDFLARE_TOKEN as fallback."""
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        monkeypatch.setenv("CLOUDFLARE_TOKEN", "fallback-token")
        assert query.get_token() == "fallback-token"

    def test_token_from_wrangler_config(self, monkeypatch, tmp_path):
        """Should read token from wrangler config file."""
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        monkeypatch.delenv("CLOUDFLARE_TOKEN", raising=False)

        config = tmp_path / "config.toml"
        config.write_text('oauth_token = "wrangler-oauth-abc"\n')

        monkeypatch.setattr(os.path, "expanduser", lambda p: str(config) if "wrangler" in p else p)
        assert query.get_token() == "wrangler-oauth-abc"

    def test_no_token_raises(self, monkeypatch):
        """Should raise RuntimeError if no token found."""
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        monkeypatch.delenv("CLOUDFLARE_TOKEN", raising=False)
        monkeypatch.setattr(os.path, "expanduser", lambda p: "/nonexistent/path/" + p)
        with pytest.raises(RuntimeError, match="No Cloudflare API token"):
            query.get_token()

    def test_env_takes_priority_over_wrangler(self, monkeypatch, tmp_path):
        """Env token should take priority over wrangler config."""
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "env-priority")
        config = tmp_path / "config.toml"
        config.write_text('oauth_token = "wrangler-fallback"\n')
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(config) if "wrangler" in p else p)
        assert query.get_token() == "env-priority"


# ── embed_query() tests ────────────────────────────────────────

class TestEmbedQuery:
    @patch("query.urllib.request.urlopen")
    def test_returns_embedding_list(self, mock_urlopen):
        """Should return embedding from Ollama."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"embedding": [0.1] * 768}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = query.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 768

    @patch("query.urllib.request.urlopen")
    def test_sends_correct_model(self, mock_urlopen):
        """Should send nomic-embed-text model."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"embedding": [0.1] * 768}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        query.embed_query("hello")
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["model"] == "nomic-embed-text"
        assert payload["prompt"] == "hello"


# ── query_vectorize() tests ────────────────────────────────────

class TestQueryVectorize:
    @patch("query.urllib.request.urlopen")
    def test_returns_api_response(self, mock_urlopen):
        """Should return parsed JSON response."""
        api_response = {"success": True, "result": {"matches": [{"id": "abc", "score": 0.9}]}}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(api_response).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = query.query_vectorize("token", [0.1] * 768, top_k=5)
        assert result["success"] is True
        assert len(result["result"]["matches"]) == 1

    @patch("query.urllib.request.urlopen")
    def test_sends_correct_payload(self, mock_urlopen):
        """Should send vector, topK, and metadata flags."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"success": True}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        vec = [0.1] * 768
        query.query_vectorize("tok", vec, top_k=15)
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["vector"] == vec
        assert payload["topK"] == 15
        assert payload["returnMetadata"] == "all"
        assert payload["returnValues"] is False
        assert req.headers["Authorization"] == "Bearer tok"


# ── format_score() tests ───────────────────────────────────────

class TestFormatScore:
    def test_high_score_has_green(self):
        """Scores >= 0.75 should include green color code."""
        formatted = query.format_score(0.85)
        assert "\033[32m" in formatted
        assert "0.8500" in formatted

    def test_medium_score_has_yellow(self):
        """Scores 0.5–0.75 should include yellow color code."""
        formatted = query.format_score(0.6)
        assert "\033[33m" in formatted
        assert "0.6000" in formatted

    def test_low_score_has_dim(self):
        """Scores < 0.5 should include dim color code."""
        formatted = query.format_score(0.3)
        assert "\033[2m" in formatted
        assert "0.3000" in formatted

    def test_boundary_075(self):
        """Exactly 0.75 should be green."""
        formatted = query.format_score(0.75)
        assert "\033[32m" in formatted

    def test_boundary_050(self):
        """Exactly 0.50 should be yellow."""
        formatted = query.format_score(0.50)
        assert "\033[33m" in formatted

    def test_all_formatted_have_reset(self):
        """All formatted scores should contain reset code."""
        for score in [0.1, 0.4, 0.55, 0.7, 0.9]:
            assert "\033[0m" in query.format_score(score)


# ── display_results() tests ────────────────────────────────────

class TestDisplayResults:
    def test_json_output_format(self, capsys):
        """JSON output should be valid JSON with query and results."""
        results = [
            {"score": 0.9, "id": "a", "metadata": {"title": "Test", "path": "test.md"}},
        ]
        query.display_results("my query", results, json_output=True)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["query"] == "my query"
        assert parsed["count"] == 1
        assert parsed["results"][0]["title"] == "Test"

    def test_empty_results_display(self, capsys):
        """Should handle empty results gracefully."""
        query.display_results("nothing", [])
        captured = capsys.readouterr()
        assert "No results" in captured.out or "nothing" in captured.out.lower()

    def test_text_output_shows_title(self, capsys):
        """Text output should show title and score."""
        results = [
            {
                "score": 0.92,
                "id": "vec1",
                "metadata": {
                    "title": "Consciousness Explained",
                    "path": "mind/consciousness.md",
                    "directory": "mind",
                    "word_count": 5000,
                    "preview": "What is it like to be a bat?",
                },
            },
        ]
        query.display_results("consciousness", results)
        captured = capsys.readouterr()
        assert "Consciousness Explained" in captured.out
        assert "consciousness" in captured.out

    def test_json_output_has_timestamp(self, capsys):
        """JSON output should include a timestamp."""
        query.display_results("q", [], json_output=True)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "timestamp" in parsed

    def test_multiple_results_display(self, capsys):
        """Should display all results."""
        results = [
            {"score": 0.9, "id": "a", "metadata": {"title": "A", "path": "a.md", "directory": "x", "word_count": 10, "preview": "prev A"}},
            {"score": 0.8, "id": "b", "metadata": {"title": "B", "path": "b.md", "directory": "y", "word_count": 20, "preview": "prev B"}},
        ]
        query.display_results("test", results)
        captured = capsys.readouterr()
        assert "A" in captured.out
        assert "B" in captured.out


# ── explore.py integration tests ───────────────────────────────

class TestExploreModule:
    """Test the explore.py module functions."""

    def test_build_matrices_shape(self, sample_store):
        """build_matrices should return embeddings and sim matrix of correct shape."""
        import explore
        with patch.object(explore, "STORE_PATH", "dummy"):
            pieces, embeddings, sim_matrix = explore.build_matrices(sample_store)
        assert embeddings.shape == (3, 768)
        assert sim_matrix.shape == (3, 3)

    def test_find_central_piece(self, sample_store):
        """find_central_piece should return ranked results."""
        import explore
        import numpy as np
        pieces, embeddings, sim_matrix = explore.build_matrices(sample_store)
        result = explore.find_central_piece(pieces, sim_matrix.copy(), top_n=3)
        assert len(result) == 3
        assert result[0]["rank"] == 1
        for r in result:
            assert "avg_similarity" in r
            assert "title" in r

    def test_find_loneliest_piece(self, sample_store):
        """find_loneliest_piece should return ranked results."""
        import explore
        pieces, embeddings, sim_matrix = explore.build_matrices(sample_store)
        result = explore.find_loneliest_piece(pieces, sim_matrix.copy(), top_n=3)
        assert len(result) == 3
        assert result[0]["rank"] == 1

    def test_find_surprising_connections(self, sample_store):
        """find_surprising_connections should find cross-directory pairs."""
        import explore
        pieces, embeddings, sim_matrix = explore.build_matrices(sample_store)
        result = explore.find_surprising_connections(pieces, sim_matrix, top_n=5)
        assert len(result) <= 5
        for item in result:
            assert item["piece_a"]["directory"] != item["piece_b"]["directory"]

    def test_find_bridge_pieces(self, sample_store):
        """find_bridge_pieces should identify cross-directory connectors."""
        import explore
        pieces, embeddings, sim_matrix = explore.build_matrices(sample_store)
        result = explore.find_bridge_pieces(pieces, sim_matrix, top_n=3)
        assert len(result) <= 3
        for r in result:
            assert "bridge_dirs" in r
            assert "neighbors" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
