"""Vision3D server.py — automated tests.

Phase 8.3: 20 tests covering helpers (unit) and endpoints (integration).
No GPU, CUDA, or ML models required.
"""

import io
import re
import time

import pytest
from fastapi import HTTPException

import server


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS (no FastAPI TestClient needed)
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateOutputSubdir:
    """Tests for _validate_output_subdir() — path traversal protection."""

    def test_valid_values(self):
        """Test 1: Valid subdirs pass through unchanged."""
        assert server._validate_output_subdir("0") == "0"
        assert server._validate_output_subdir("my_job") == "my_job"
        assert server._validate_output_subdir("test123") == "test123"

    def test_path_traversal_rejected(self):
        """Test 2: Path traversal patterns raise HTTP 400."""
        for bad_value in ["..", "../etc", "a/b", "a\\b", "..\\windows"]:
            with pytest.raises(HTTPException) as exc_info:
                server._validate_output_subdir(bad_value)
            assert exc_info.value.status_code == 400


class TestResolvePreset:
    """Tests for _resolve_preset() — quality preset resolution."""

    def test_no_preset_defaults(self):
        """Test 3: Without preset, returns sensible defaults."""
        params = server._resolve_preset(
            target_faces=0, preset="", model="", octree_resolution=0, num_inference_steps=0
        )
        assert params["model"] == "turbo"
        assert params["octree_resolution"] == 384
        assert params["num_inference_steps"] == 30

    def test_with_preset(self):
        """Test 4: Preset 'high' returns preset values."""
        params = server._resolve_preset(
            target_faces=0, preset="high", model="", octree_resolution=0, num_inference_steps=0
        )
        assert params["model"] == "full"
        assert params["target_faces"] == 150000
        assert params["octree_resolution"] == 384
        assert params["num_inference_steps"] == 30

    def test_target_faces_override(self):
        """Test 5: Explicit target_faces overrides preset value."""
        params = server._resolve_preset(
            target_faces=25000, preset="high", model="", octree_resolution=0, num_inference_steps=0
        )
        # target_faces=25000 should override high's 150000
        assert params["target_faces"] == 25000
        # Other preset values should remain
        assert params["model"] == "full"


class TestResolveOutputSubdir:
    """Tests for _resolve_output_subdir() — Bug #6 collision fix."""

    def test_default_replaced_with_uuid(self):
        """Test 6: Default '0' is replaced with 8-char UUID."""
        result = server._resolve_output_subdir("0")
        assert result != "0"
        assert len(result) == 8
        # Should be valid hex (uuid4 first 8 chars)
        assert re.match(r"^[0-9a-f]{8}$", result)

    def test_explicit_preserved(self):
        """Test 7: Explicit values are kept unchanged."""
        assert server._resolve_output_subdir("my_custom_dir") == "my_custom_dir"
        assert server._resolve_output_subdir("web_0") == "web_0"
        assert server._resolve_output_subdir("123") == "123"

    def test_default_uniqueness(self):
        """Test 6b: Multiple calls with '0' produce different UUIDs."""
        results = {server._resolve_output_subdir("0") for _ in range(10)}
        # All 10 should be unique
        assert len(results) == 10


class TestCleanupOldJobs:
    """Tests for _cleanup_old_jobs() — TTL-based job eviction."""

    def test_cleanup_removes_old_jobs(self, mock_job_old_completed, mock_job_old_failed, mock_job_running):
        """Test 8: Old completed/failed jobs are removed; running jobs within
        JOB_RUNNING_TTL are kept."""
        assert len(server._jobs) == 3

        removed, _dirs = server._cleanup_old_jobs()

        assert removed == 2
        assert mock_job_old_completed not in server._jobs
        assert mock_job_old_failed not in server._jobs
        # Running job (still within JOB_RUNNING_TTL) must NOT be removed
        assert mock_job_running in server._jobs

    def test_cleanup_keeps_recent_jobs(self, mock_job_completed):
        """Test 8b: Recent completed jobs are NOT removed."""
        assert len(server._jobs) == 1
        removed, _dirs = server._cleanup_old_jobs()
        assert removed == 0
        assert mock_job_completed in server._jobs

    def test_cleanup_removes_zombie_running_jobs(self, mock_job_zombie_running, mock_job_running):
        """Test 8c: Running jobs older than JOB_RUNNING_TTL are evicted as
        zombies; running jobs within the TTL are preserved."""
        assert len(server._jobs) == 2
        removed, _dirs = server._cleanup_old_jobs()
        assert removed == 1
        assert mock_job_zombie_running not in server._jobs
        assert mock_job_running in server._jobs


class TestValidateUpload:
    """Tests for _validate_upload() — MIME type and size checking."""

    @pytest.mark.asyncio
    async def test_wrong_mime_type(self):
        """Test 9: Incorrect MIME type raises HTTP 400."""
        from unittest.mock import AsyncMock

        fake_file = AsyncMock()
        fake_file.content_type = "text/plain"
        fake_file.filename = "bad.txt"

        with pytest.raises(HTTPException) as exc_info:
            await server._validate_upload(fake_file, server.ALLOWED_IMAGE_TYPES)
        assert exc_info.value.status_code == 400
        assert "text/plain" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_file_too_large(self):
        """Test 10: File exceeding 50MB raises HTTP 400."""
        from unittest.mock import AsyncMock

        fake_file = AsyncMock()
        fake_file.content_type = "image/png"
        fake_file.filename = "huge.png"
        # Return >50MB of data
        fake_file.read.return_value = b"\x00" * (51 * 1024 * 1024)

        with pytest.raises(HTTPException) as exc_info:
            await server._validate_upload(fake_file, server.ALLOWED_IMAGE_TYPES)
        assert exc_info.value.status_code == 400
        assert "too large" in exc_info.value.detail


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT TESTS (using FastAPI TestClient)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client):
        """Test 11: Health endpoint returns 200 with status ok."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "api_key_required" in data


class TestPresetsEndpoint:
    """Tests for GET /api/presets."""

    def test_presets_returns_all(self, client):
        """Test 12: Presets endpoint returns 200 with all 4 presets."""
        response = client.get("/api/presets")
        assert response.status_code == 200
        data = response.json()
        assert "low" in data
        assert "medium" in data
        assert "high" in data
        assert "ultra" in data


class TestModelsEndpoint:
    """Tests for GET /api/models."""

    def test_models_returns_200(self, client):
        """Test 13: Models endpoint returns 200 with expected structure."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "default" in data
        assert "all" in data
        # 'all' should contain the 3 model names
        assert "turbo" in data["all"]
        assert "fast" in data["all"]
        assert "full" in data["all"]


class TestGenerateShapeValidation:
    """Tests for POST /api/generate-shape validation."""

    def test_no_image_returns_422(self, client):
        """Test 14: POST without image returns 422 (missing required field)."""
        response = client.post("/api/generate-shape")
        assert response.status_code == 422

    def test_wrong_mime_returns_400(self, client):
        """Test 15: POST with wrong MIME type returns 400."""
        fake_file = io.BytesIO(b"not an image")
        response = client.post(
            "/api/generate-shape",
            files={"image": ("test.txt", fake_file, "text/plain")},
        )
        assert response.status_code == 400
        assert "text/plain" in response.json()["detail"]


class TestGpuBusy:
    """Tests for GPU semaphore protection."""

    def test_gpu_busy_returns_429(self, client, gpu_semaphore_locked):
        """Test 16: When GPU semaphore is locked, POST returns 429."""
        fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        response = client.post(
            "/api/generate-shape",
            files={"image": ("test.png", fake_image, "image/png")},
        )
        assert response.status_code == 429
        assert "GPU busy" in response.json()["detail"]


class TestAuthentication:
    """Tests for API key authentication."""

    def test_auth_required_no_key(self, client, api_key_configured, mock_job_completed):
        """Test 17: With API key configured, request without key returns 401."""
        job_id = mock_job_completed
        # GET /api/jobs/{id} requires auth — try without key
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]

    def test_auth_query_param_sse(self, client, api_key_configured, mock_job_completed):
        """Test 18: SSE endpoint accepts API key as query param."""
        key = api_key_configured
        job_id = mock_job_completed

        # Request SSE endpoint with key as query param (like EventSource does)
        response = client.get(
            f"/api/jobs/{job_id}/stream?x_api_key={key}",
            headers={"Accept": "text/event-stream"},
        )
        # Should NOT be 401 — the query param auth should work
        assert response.status_code != 401
        # Should be 200 (streaming) since the job exists
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# PAINT UNLOAD TESTS (v1.6.2 — VRAM leak fix)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPaintUnload:
    """Tests for _unload_paint_pipeline() and the idle-timer state machine.

    The real paint pipeline is a ~14 GB CUDA model; these tests substitute a
    lightweight stand-in so the unload logic (not the model) is what gets
    exercised.
    """

    def setup_method(self):
        """Save and clear paint module state before each test."""
        self._saved_pipeline = server._paint_pipeline
        self._saved_last_use = server._paint_pipeline_last_use
        server._paint_pipeline = None
        server._paint_pipeline_last_use = None

    def teardown_method(self):
        """Restore paint module state."""
        server._paint_pipeline = self._saved_pipeline
        server._paint_pipeline_last_use = self._saved_last_use

    def test_unload_is_noop_when_paint_is_none(self):
        """Test 22: unload is a safe no-op when paint pipeline is not loaded."""
        assert server._paint_pipeline is None
        server._unload_paint_pipeline()
        assert server._paint_pipeline is None
        assert server._paint_pipeline_last_use is None

    def test_unload_clears_state_when_loaded(self):
        """Test 23: unload clears both _paint_pipeline and _paint_pipeline_last_use."""
        class _FakePipeline:
            def to(self, device):
                self.moved_to = device
                return self
        server._paint_pipeline = _FakePipeline()
        server._paint_pipeline_last_use = time.monotonic()
        server._unload_paint_pipeline()
        assert server._paint_pipeline is None
        assert server._paint_pipeline_last_use is None

    def test_unload_tolerates_pipelines_without_to_method(self):
        """Test 24: unload does not raise when .to('cpu') fails."""
        class _Bad:
            def to(self, device):
                raise RuntimeError("no .to() support")
        server._paint_pipeline = _Bad()
        server._paint_pipeline_last_use = time.monotonic()
        server._unload_paint_pipeline()
        assert server._paint_pipeline is None

    def test_paint_idle_constants_positive(self):
        """Test 25: idle threshold + check interval are sane (> 0)."""
        assert server.PAINT_IDLE_SECONDS > 0
        assert server.PAINT_IDLE_CHECK_INTERVAL > 0
        assert server.PAINT_IDLE_SECONDS >= server.PAINT_IDLE_CHECK_INTERVAL

    def test_paint_idle_env_override(self, monkeypatch):
        """Test 26: PAINT_IDLE_SECONDS honours VISION3D_PAINT_IDLE_SECONDS env."""
        monkeypatch.setenv("VISION3D_PAINT_IDLE_SECONDS", "42")
        import importlib
        reloaded = importlib.reload(server)
        try:
            assert reloaded.PAINT_IDLE_SECONDS == 42
        finally:
            monkeypatch.delenv("VISION3D_PAINT_IDLE_SECONDS", raising=False)
            importlib.reload(server)
