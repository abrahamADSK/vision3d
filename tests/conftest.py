"""Fixtures for Vision3D server tests.

These tests do NOT require GPU, CUDA, or any ML models.
They test the FastAPI app, helper functions, and endpoint behavior
using mocks where needed.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Ensure torch/cuda imports don't fail in test environment ──────────────

# If torch is not installed, create a minimal mock so server.py can be imported.
if "torch" not in sys.modules:
    _torch_mock = MagicMock()
    _torch_mock.cuda.is_available.return_value = False
    _torch_mock.cuda.empty_cache = MagicMock()
    _torch_mock.cuda.synchronize = MagicMock()
    _torch_mock.backends.mps.is_available.return_value = False
    _torch_mock.mps.empty_cache = MagicMock()
    sys.modules["torch"] = _torch_mock

# Mock ML-heavy optional imports that server.py may try to load
for mod_name in [
    "hy3dgen", "hy3dgen.rembg", "trimesh", "pyfqmr", "numpy",
    "diffusers", "transformers", "accelerate", "PIL", "PIL.Image",
    "psutil",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# ── Import server module AFTER mocks are in place ─────────────────────────

# Add the project root to sys.path so we can import server
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import server  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def app():
    """Return the FastAPI app instance."""
    return server.app


@pytest.fixture
def client(app):
    """Return a Starlette TestClient for the FastAPI app."""
    from starlette.testclient import TestClient
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_jobs():
    """Ensure _jobs is empty before and after each test."""
    server._jobs.clear()
    yield
    server._jobs.clear()


@pytest.fixture
def mock_job_completed():
    """Insert a completed job into _jobs and return its id."""
    job_id = "test1234"
    server._jobs[job_id] = {
        "id": job_id,
        "type": "shape-image",
        "status": "completed",
        "detail": "test",
        "created": time.time(),
        "output_dir": "/tmp/test",
        "files": ["mesh.glb"],
        "error": None,
        "log": ["step 1", "step 2"],
    }
    return job_id


@pytest.fixture
def mock_job_old_completed():
    """Insert a completed job with an old timestamp (2 hours ago)."""
    job_id = "old12345"
    server._jobs[job_id] = {
        "id": job_id,
        "type": "shape-image",
        "status": "completed",
        "detail": "old test",
        "created": time.time() - 7200,  # 2 hours ago
        "output_dir": "/tmp/old",
        "files": ["mesh.glb"],
        "error": None,
        "log": [],
    }
    return job_id


@pytest.fixture
def mock_job_old_failed():
    """Insert a failed job with an old timestamp (2 hours ago)."""
    job_id = "fail1234"
    server._jobs[job_id] = {
        "id": job_id,
        "type": "texture",
        "status": "failed",
        "detail": "old fail",
        "created": time.time() - 7200,  # 2 hours ago
        "output_dir": "/tmp/fail",
        "files": [],
        "error": "test error",
        "log": [],
    }
    return job_id


@pytest.fixture
def mock_job_running():
    """Insert a running job old enough to trigger completed-TTL but still
    exempt from the running-TTL (so cleanup must NOT touch it)."""
    job_id = "run12345"
    server._jobs[job_id] = {
        "id": job_id,
        "type": "full-pipeline",
        "status": "running",
        "detail": "running test",
        # > JOB_TTL_SECONDS (3600) but < JOB_RUNNING_TTL (7200)
        "created": time.time() - 4000,
        "output_dir": None,
        "files": [],
        "error": None,
        "log": [],
    }
    return job_id


@pytest.fixture
def mock_job_zombie_running():
    """Insert a running job older than JOB_RUNNING_TTL — must be cleaned."""
    job_id = "zom12345"
    server._jobs[job_id] = {
        "id": job_id,
        "type": "full-pipeline",
        "status": "running",
        "detail": "zombie",
        "created": time.time() - 10000,  # > JOB_RUNNING_TTL (7200)
        "output_dir": None,
        "files": [],
        "error": None,
        "log": [],
    }
    return job_id


@pytest.fixture
def gpu_semaphore_locked():
    """Acquire the GPU semaphore so it appears busy, release after test."""
    # We need to acquire the asyncio semaphore from sync context.
    # The semaphore starts unlocked (value=1). We decrement it to 0.
    # Since _gpu_semaphore._value is the internal counter, we manipulate it directly.
    original_value = server._gpu_semaphore._value
    server._gpu_semaphore._value = 0
    yield
    server._gpu_semaphore._value = original_value


@pytest.fixture
def api_key_configured(monkeypatch):
    """Temporarily set an API key on the server module."""
    monkeypatch.setattr(server, "API_KEY", "test-secret-key-12345")
    yield "test-secret-key-12345"
