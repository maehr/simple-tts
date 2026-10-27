import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from simple_tts import tts

# Real model tests are opt-in. They need the gated model in the Hub cache.
MODEL_TESTS = "SIMPLE_TTS_MODEL_TESTS"
REAL_PTTS = pytest.StashKey[ModuleType | None]()
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


# A small copy of the pinned catalogue metadata, with one unsupported language.
CATALOG = [
    {"name": "Zoey", "voice_id": "NbpkqMVS3CJeq2j8", "language": "en"},
    {"name": "Hadley", "voice_id": "bXwrkHhURL1VFJdG", "language": "en"},
    {"name": "Nele", "voice_id": "2A2fFf8SkDEJ8nzx", "language": "de"},
    {"name": "Anton", "voice_id": "3ZKKapPOvuWFcw9f", "language": "de"},
    {"name": "Noa", "voice_id": "b6FvJAiokjdqIti4", "language": "es"},
    {"name": "Noémie", "voice_id": "FXxJ9mANRq6BCTX5", "language": "fr"},
    {"name": "Manuela-Lu", "voice_id": "KgC2Nqnjj48NUiyV", "language": "pt"},
]


def fake_ptts():
    """Return a ptts module with CPU support only."""
    return SimpleNamespace(
        TTS=Mock(side_effect=FakeTTS),
        available_devices=Mock(return_value=["cpu"]),
        set_num_threads=Mock(),
        catalog_voices=Mock(
            return_value=[
                {**voice, "description": f"{voice['name']} test voice."}
                for voice in CATALOG
            ]
        ),
    )


def model_tests_enabled():
    """Return whether the opt-in real model tests run."""
    return os.environ.get(MODEL_TESTS) == "1"


def pytest_configure(config):
    """Prevent model downloads during tests and doctests.

    Only an opt-in model test run keeps the real ptts module for model tests.
    """
    if model_tests_enabled():
        try:
            import ptts as real
        except ImportError:
            real = None
        config.stash[REAL_PTTS] = real
    sys.modules["ptts"] = fake_ptts()


def pytest_collection_modifyitems(config, items):
    """Skip the model tests unless SIMPLE_TTS_MODEL_TESTS is 1."""
    if model_tests_enabled():
        return
    skip = pytest.mark.skip(reason=f"Set {MODEL_TESTS}=1 to run real model tests.")
    for item in items:
        if item.get_closest_marker("model"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def real_ptts(pytestconfig):
    """Return the real ptts module for the model tests."""
    module = pytestconfig.stash.get(REAL_PTTS, None)
    if module is None:
        pytest.skip("The real ptts module is not installed.")
    return module


@pytest.fixture(scope="session")
def empty_hub(tmp_path_factory):
    """Return a Hub cache without a model snapshot."""
    return tmp_path_factory.mktemp("hub")


@pytest.fixture(autouse=True)
def ptts(request, monkeypatch, empty_hub):
    """Replace ptts and the model source settings in every test.

    Model tests get the real ptts module and keep the real Hub cache.
    """
    if request.node.get_closest_marker("model"):
        module = request.getfixturevalue("real_ptts")
        monkeypatch.setitem(sys.modules, "ptts", module)
        return module
    module = fake_ptts()
    monkeypatch.setitem(sys.modules, "ptts", module)
    monkeypatch.delenv("SIMPLE_TTS_MODEL", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(empty_hub))
    tts._set_threads.cache_clear()
    return module
