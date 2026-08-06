"""
Comprehensive tests for sync_to_cloudflare.py — Cloudflare Vectorize sync.

Covers:
- get_token() — credential resolution
- load_sync_state() / save_sync_state() — persistence
- make_vector_id() — deterministic ID generation
- build_vector() — vector payload construction
- api_post() — API call with retries (mocked)
- sync_full() — full upload (mocked)
- sync_update() — incremental sync (mocked)
- CLI argument parsing
"""

import json
import os
import sys
import hashlib
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sync_to_cloudflare as sync


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    """Redirect file paths to temp."""
    store = tmp_path / "store.json"
    state = tmp_path / "state.json"
    monkeypatch.setattr(sync, "STORE_PATH", str(store))
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", str(state))
    return store, state


@pytest.fixture
def sample_piece():
    """One sample piece with embedding."""
    return {
        "path": "essays/test.md",
        "title": "Test Essay",
        "preview": "A preview of the test essay...",
        "word_count": 500,
        "directory": "essays",
        "mtime": 1700000000.0,
        "embedding": [0.1] * 768,
    }


@pytest.fixture
def sample_store(tmp_paths, sample_piece):
    """Create a store file with one piece."""
    store_path, _ = tmp_paths
    store_data = {
        "pieces": [sample_piece],
        "last_run": "2024-01-01T00:00:00+00:00",
        "stats": {"total_pieces": 1},
    }
    store_path.write_text(json.dumps(store_data))
    return store_path


# ── get_token() tests ──────────────────────────────────────────

class TestGetToken:
    def test_env_token(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "env-tok")
        assert sync.get_token() == "env-tok"

    def test_env_token_fallback(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        monkeypatch.setenv("CLOUDFLARE_TOKEN", "cf-tok")
        assert sync.get_token() == "cf-tok"

    def test_no_token_raises(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        monkeypatch.delenv("CLOUDFLARE_TOKEN", raising=False)
        monkeypatch.setattr(os.path, "expanduser", lambda p: "/nonexistent/" + p)
        with pytest.raises(RuntimeError):
            sync.get_token()


# ── make_vector_id() tests ─────────────────────────────────────

class TestMakeVectorId:
    def test_id_is_deterministic(self):
        """Same path should produce same ID."""
        assert sync.make_vector_id("foo/bar.md") == sync.make_vector_id("foo/bar.md")

    def test_id_is_hex(self):
        """ID should be a 16-char hex string."""
        vid = sync.make_vector_id("test.md")
        assert len(vid) == 16
        int(vid, 16)  # should not raise

    def test_id_matches_sha256_prefix(self):
        """ID should be first 16 chars of SHA-256."""
        path = "essays/test.md"
        expected = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
        assert sync.make_vector_id(path) == expected

    def test_different_paths_different_ids(self):
        """Different paths should produce different IDs."""
        assert sync.make_vector_id("a.md") != sync.make_vector_id("b.md")

    def test_id_length_under_64_bytes(self):
        """ID must be <= 64 bytes for Vectorize."""
        vid = sync.make_vector_id("very/long/path/" + "x" * 200 + ".md")
        assert len(vid.encode()) <= 64


# ── load/save_sync_state() tests ───────────────────────────────

class TestSyncState:
    def test_load_empty_state(self, tmp_paths):
        """Loading when no state exists returns empty structure."""
        _, _ = tmp_paths
        state = sync.load_sync_state()
        assert state["last_sync"] is None
        assert state["synced_ids"] == {}
        assert state["total_synced"] == 0

    def test_save_load_roundtrip(self, tmp_paths):
        """Save then load should preserve state."""
        state = {
            "last_sync": None,
            "synced_ids": {"doc.md": {"id": "abc123", "mtime": 123.0, "synced_at": "2024-01-01"}},
            "total_synced": 1,
        }
        sync.save_sync_state(state)
        loaded = sync.load_sync_state()
        assert loaded["synced_ids"] == state["synced_ids"]
        assert loaded["total_synced"] == 1
        assert loaded["last_sync"] is not None  # save sets it

    def test_load_corrupt_state(self, tmp_paths):
        """Corrupt state file should return empty structure."""
        _, state_path = tmp_paths
        state_path.write_text("not json {{{")
        loaded = sync.load_sync_state()
        assert loaded["synced_ids"] == {}


# ── build_vector() tests ───────────────────────────────────────

class TestBuildVector:
    def test_returns_dict_with_required_fields(self, sample_piece):
        """build_vector should return dict with id, values, metadata."""
        vec = sync.build_vector(sample_piece)
        assert "id" in vec
        assert "values" in vec
        assert "metadata" in vec

    def test_id_is_hash_based(self, sample_piece):
        """Vector ID should match make_vector_id."""
        vec = sync.build_vector(sample_piece)
        assert vec["id"] == sync.make_vector_id(sample_piece["path"])

    def test_values_are_embedding(self, sample_piece):
        """Values should be the embedding."""
        vec = sync.build_vector(sample_piece)
        assert vec["values"] == sample_piece["embedding"]

    def test_metadata_includes_path_title_directory(self, sample_piece):
        """Metadata should include key fields."""
        vec = sync.build_vector(sample_piece)
        assert vec["metadata"]["path"] == sample_piece["path"]
        assert vec["metadata"]["title"] == sample_piece["title"]
        assert vec["metadata"]["directory"] == sample_piece["directory"]
        assert vec["metadata"]["word_count"] == sample_piece["word_count"]

    def test_metadata_preview_capped(self, sample_piece):
        """Preview in metadata should be capped at 300 chars."""
        sample_piece["preview"] = "x" * 500
        vec = sync.build_vector(sample_piece)
        assert len(vec["metadata"]["preview"]) <= 300

    def test_metadata_includes_mtime(self, sample_piece):
        """Metadata should include mtime."""
        vec = sync.build_vector(sample_piece)
        assert vec["metadata"]["mtime"] == sample_piece["mtime"]


# ── api_post() tests ───────────────────────────────────────────

class TestApiPost:
    @patch("sync_to_cloudflare.urllib.request.urlopen")
    def test_successful_post(self, mock_urlopen):
        """Should return (True, result) on success."""
        api_resp = {"success": True, "result": {}}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(api_resp).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        success, result = sync.api_post("http://example.com", "tok", [{"id": "x"}])
        assert success is True

    @patch("sync_to_cloudflare.urllib.request.urlopen")
    def test_api_error_response(self, mock_urlopen):
        """Should return (False, errors) when API returns success=false."""
        api_resp = {"success": False, "errors": [{"code": 1, "message": "bad"}]}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(api_resp).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        success, result = sync.api_post("http://example.com", "tok", [])
        assert success is False

    @patch("sync_to_cloudflare.time.sleep")
    @patch("sync_to_cloudflare.urllib.request.urlopen")
    def test_rate_limit_retry(self, mock_urlopen, mock_sleep):
        """Should retry on 429."""
        import urllib.error

        # First call raises 429, second succeeds
        error_resp = urllib.error.HTTPError(
            "url", 429, "Rate Limited",
            {},
            MagicMock(read=MagicMock(return_value=b"rate limited")),
        )
        good_resp = MagicMock()
        good_resp.read.return_value = json.dumps({"success": True}).encode()
        good_resp.__enter__ = MagicMock(return_value=good_resp)
        good_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [error_resp, good_resp]

        success, _ = sync.api_post("http://example.com", "tok", [], retries=3)
        assert success is True
        assert mock_sleep.call_count >= 1


# ── sync_full() tests ──────────────────────────────────────────

class TestSyncFull:
    @patch("sync_to_cloudflare.api_post")
    def test_full_sync_uploads_all(self, mock_api, sample_store, tmp_paths):
        """Full sync should upload all pieces."""
        mock_api.return_value = (True, {"success": True})
        sync.sync_full("token", verbose=False)
        assert mock_api.call_count == 1  # 1 piece, batch_size=100
        args = mock_api.call_args[0]
        assert args[0] == sync.INSERT_URL
        assert args[1] == "token"
        assert len(args[2]) == 1  # 1 vector

    @patch("sync_to_cloudflare.api_post")
    def test_full_sync_tracks_state(self, mock_api, sample_store, tmp_paths):
        """Full sync should save sync state."""
        mock_api.return_value = (True, {"success": True})
        sync.sync_full("token", verbose=False)
        _, state_path = tmp_paths
        state = json.loads(state_path.read_text())
        assert state["total_synced"] == 1
        assert "essays/test.md" in state["synced_ids"]

    @patch("sync_to_cloudflare.api_post")
    def test_full_sync_handles_failure(self, mock_api, sample_store, tmp_paths):
        """Full sync should continue on batch failure."""
        mock_api.return_value = (False, [{"error": "fail"}])
        sync.sync_full("token", verbose=False)
        # Should not crash, state should still be saved
        _, state_path = tmp_paths
        state = json.loads(state_path.read_text())
        assert state is not None


# ── sync_update() tests ────────────────────────────────────────

class TestSyncUpdate:
    @patch("sync_to_cloudflare.api_post")
    def test_update_no_changes(self, mock_api, sample_store, tmp_paths):
        """Update with no changes should not call API."""
        # Pre-populate sync state to match store
        _, state_path = tmp_paths
        state = {
            "last_sync": "2024-01-01T00:00:00+00:00",
            "synced_ids": {
                "essays/test.md": {
                    "id": sync.make_vector_id("essays/test.md"),
                    "mtime": 1700000000.0,
                    "synced_at": "2024-01-01",
                }
            },
            "total_synced": 1,
        }
        state_path.write_text(json.dumps(state))

        sync.sync_update("token", verbose=False)
        assert mock_api.call_count == 0

    @patch("sync_to_cloudflare.api_post")
    def test_update_new_piece(self, mock_api, sample_store, tmp_paths):
        """Update should upload pieces not yet synced."""
        mock_api.return_value = (True, {"success": True})
        # No pre-existing sync state -> everything is new
        sync.sync_update("token", verbose=False)
        assert mock_api.call_count >= 1


# ── CLI tests ──────────────────────────────────────────────────

class TestCLI:
    def test_status_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sync_to_cloudflare.py", "--status"])
        called = MagicMock()
        monkeypatch.setattr(sync, "show_status", called)
        sync.main()
        called.assert_called_once()

    def test_update_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sync_to_cloudflare.py", "--update"])
        called = MagicMock()
        monkeypatch.setattr(sync, "sync_update", called)
        sync.main()
        called.assert_called_once()

    def test_default_is_full_sync(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sync_to_cloudflare.py"])
        called = MagicMock()
        monkeypatch.setattr(sync, "sync_full", called)
        sync.main()
        called.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
