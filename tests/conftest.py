import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


class MockKPipeline:
    """Mock KPipeline to prevent model download during tests."""

    def __init__(self, lang_code=None, repo_id=None):
        """Initialize the mock pipeline."""
        pass

    def __call__(self, text, voice=None, speed=1.0):
        """Synthesize speech (mocked)."""
        # Yield dummy data: (graphemes, phonemes, audio)
        yield ("test", "test", np.zeros(int(24000 * 0.1), dtype=np.float32))


def pytest_configure(config):
    """Patch kokoro globally for all tests including doctests."""
    try:
        import kokoro

        # Patch KPipeline directly on the module
        kokoro.KPipeline = MockKPipeline
    except ImportError:
        # If kokoro is not installed, mock the module
        mock_kokoro = MagicMock()
        mock_kokoro.KPipeline = MockKPipeline
        sys.modules["kokoro"] = mock_kokoro


@pytest.fixture(autouse=True)
def mock_kokoro_fixture(monkeypatch):
    """Fixture to ensure kokoro is mocked for regular tests.

    This is redundant if pytest_configure works, but good as a backup
    and for proper cleanup in unit tests if we didn't use global patch.
    """
    try:
        import kokoro

        monkeypatch.setattr(kokoro, "KPipeline", MockKPipeline)
    except ImportError:
        pass
