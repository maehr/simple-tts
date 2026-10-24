import sys
from types import SimpleNamespace

import numpy as np
import pytest


class MockKPipeline:
    """Replace the model without importing Kokoro or PyTorch."""

    def __init__(self, lang_code=None, repo_id=None, device=None):
        """Record the requested model settings."""
        self.lang_code = lang_code
        self.repo_id = repo_id
        self.device = device

    def __call__(self, text, voice=None, speed=1.0):
        """Return a short deterministic chunk."""
        yield (text, text, np.zeros(2400, dtype=np.float32))


def pytest_configure(config):
    """Prevent model downloads during tests and doctests."""
    sys.modules["kokoro"] = SimpleNamespace(KPipeline=MockKPipeline)


@pytest.fixture(autouse=True)
def mock_models(monkeypatch):
    """Use CPU-only availability unless a test supplies other devices."""
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=MockKPipeline))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        ),
    )
