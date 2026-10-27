import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from simple_tts import tts

DISPLAY_DEVICES = {"cpu": "cpu", "metal": "Metal (Test GPU)", "cuda": "CUDA (Test GPU)"}


class FakeStream:
    """Yield fixed frames like a ptts stream and record its closure."""

    def __init__(self, frames):
        """Use the frames of one text chunk."""
        self.frames = iter(frames)
        self.closed = False

    def __enter__(self):
        """Return the stream itself, as ptts does."""
        return self

    def __exit__(self, *args):
        """Stop the stream."""
        self.closed = True
        return False

    def __iter__(self):
        """Iterate over the frames."""
        return self

    def __next__(self):
        """Return the next frame until the stream closes."""
        if self.closed:
            raise StopIteration
        return next(self.frames)


class FakeTTS:
    """Replace the model without weights, voices, or downloads."""

    def __init__(self, model=None, device=None, quant=None, *, lang, **kwargs):
        """Record the load options and expose the loaded model attributes."""
        self.model = model
        self.lang = lang
        self.conditions = kwargs.get("conditions")
        self.device = DISPLAY_DEVICES[device]
        self.quant = "q8_0" if quant == "q8" else quant
        self.sample_rate = 24000
        self.languages = ["en", "fr", "de", "pt", "es"]
        self.frames = [np.zeros(2400, dtype=np.float32)]
        self.streams = []

    def stream(self, text, *, voice=None):
        """Return one 0.1 s frame per text chunk, or the frames of the voice."""
        if not text.strip():
            raise ValueError("nothing to synthesize: the text is empty")
        frames = self.frames[voice] if isinstance(self.frames, dict) else self.frames
        stream = FakeStream(frames)
        self.streams.append((text, voice, stream))
        return stream


def fake_ptts():
    """Return a ptts module with CPU support only."""
    return SimpleNamespace(
        TTS=Mock(side_effect=FakeTTS),
        available_devices=Mock(return_value=["cpu"]),
        set_num_threads=Mock(),
    )


def pytest_configure(config):
    """Prevent model downloads during tests and doctests."""
    sys.modules["ptts"] = fake_ptts()


@pytest.fixture(scope="session")
def empty_hub(tmp_path_factory):
    """Return a Hub cache without a model snapshot."""
    return tmp_path_factory.mktemp("hub")


@pytest.fixture(autouse=True)
def ptts(monkeypatch, empty_hub):
    """Replace ptts and the model source settings in every test."""
    module = fake_ptts()
    monkeypatch.setitem(sys.modules, "ptts", module)
    monkeypatch.delenv("SIMPLE_TTS_MODEL", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(empty_hub))
    tts._set_threads.cache_clear()
    return module
