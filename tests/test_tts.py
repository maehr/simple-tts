"""Tests for device selection and complete-chunk synthesis."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from simple_tts import tts


def set_devices(monkeypatch, *, cuda=False, mps=False):
    """Set the devices available to the selection code."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        ),
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_selection(monkeypatch, cuda, mps, expected):
    """Auto mode prefers CUDA, then MPS, then CPU."""
    set_devices(monkeypatch, cuda=cuda, mps=mps)
    report = Mock()
    pipeline = tts.load_tts("bf_emma", report=report)
    assert pipeline.device == expected
    assert pipeline.lang_code == "b"
    assert any(f"Device: {expected}" in call.args[0] for call in report.call_args_list)
    if expected in {"cpu", "mps"}:
        assert any("CPU" in call.args[0] for call in report.call_args_list)


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_explicit_device(monkeypatch, device):
    """Explicit device selection reaches Kokoro unchanged."""
    set_devices(monkeypatch, cuda=True, mps=True)
    pipeline = tts.load_tts(device=device)
    assert pipeline.device == device


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_unavailable_explicit_gpu(monkeypatch, device):
    """Unavailable explicit GPUs fail without loading a model."""
    factory = Mock()
    monkeypatch.setattr(tts, "_create_pipeline", factory)
    with pytest.raises(RuntimeError, match="unavailable"):
        tts.load_tts(device=device)
    factory.assert_not_called()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_initialization_falls_back_once(monkeypatch, device):
    """Auto mode retries initialization on CPU exactly once."""
    set_devices(monkeypatch, cuda=device == "cuda", mps=device == "mps")
    cpu = Mock()
    factory = Mock(side_effect=[RuntimeError("GPU init error"), cpu])
    report = Mock()
    monkeypatch.setattr(tts, "_create_pipeline", factory)
    pipeline = tts.load_tts(report=report)
    assert pipeline is cpu
    assert [call.args[1] for call in factory.call_args_list] == [device, "cpu"]
    assert any("Retry on CPU" in call.args[0] for call in report.call_args_list)


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_explicit_initialization_failure(monkeypatch, device):
    """Explicit device failures produce an error without a retry."""
    set_devices(monkeypatch, cuda=True, mps=True)
    factory = Mock(side_effect=RuntimeError("init error"))
    monkeypatch.setattr(tts, "_create_pipeline", factory)
    with pytest.raises(RuntimeError, match=f"initialize TTS on {device}"):
        tts.load_tts(device=device)
    assert factory.call_count == 1


def test_failed_cpu_initialization_retry(monkeypatch):
    """Failed CPU initialization stops auto mode after its single retry."""
    set_devices(monkeypatch, cuda=True)
    factory = Mock(side_effect=RuntimeError("init error"))
    monkeypatch.setattr(tts, "_create_pipeline", factory)
    with pytest.raises(RuntimeError, match="initialize TTS on CPU"):
        tts.load_tts()
    assert factory.call_count == 2


def test_missing_kokoro_is_an_error(monkeypatch):
    """A missing Kokoro package never returns silence."""
    monkeypatch.setitem(sys.modules, "kokoro", None)
    with pytest.raises(RuntimeError, match=r"pip install.*kokoro>=0.9.4"):
        tts.load_tts()
    with pytest.raises(RuntimeError, match="No Kokoro model"):
        tts.synthesize(None, "Hello")


def test_synthesis_failure_raises():
    """Synthesis errors propagate directly."""
    failed = Mock(side_effect=RuntimeError("synthesis error"))
    pipeline = failed
    with pytest.raises(RuntimeError, match="synthesis error"):
        tts.synthesize(pipeline, "Hello")


def test_array_helpers_remain_available():
    """Library callers can still synthesize one chunk or a list."""
    pipeline = tts.load_tts()
    result = tts.synthesize(pipeline, "Hello", speed=0.5)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(tts.synthesize_chunks(pipeline, ["First", "Last"])) == 2
    assert tts.synthesize_chunks(pipeline, []) == []
    assert tts.SAMPLE_RATE == 24000


def test_mps_setting_precedes_heavy_imports():
    """A fresh application process sets fallback before either import."""
    script = """
import builtins
import os
import sys
from types import SimpleNamespace
from simple_tts.tts import load_tts
assert "torch" not in sys.modules
assert "kokoro" not in sys.modules
original = builtins.__import__
events = []
def checked_import(name, *args, **kwargs):
    if name in {"torch", "kokoro"}:
        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        events.append(name)
    if name == "torch":
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
    if name == "kokoro":
        return SimpleNamespace(KPipeline=lambda **kwargs: kwargs)
    return original(name, *args, **kwargs)
builtins.__import__ = checked_import
pipeline = load_tts()
assert pipeline["device"] == "mps"
assert events == ["torch", "kokoro"]
"""
    environment = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_invalid_device_rejected():
    """Unknown devices fail before model initialization."""
    with pytest.raises(ValueError, match="Device must be"):
        tts.load_tts(device="unknown")


def test_missing_torch_reports_install_hint(monkeypatch):
    """Missing PyTorch reports the required package installation."""
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="PyTorch is missing"):
        tts.load_tts()


def test_synthesis_skips_empty_segments_and_moves_tensors_to_cpu():
    """Tensor segments move to CPU before conversion and preserve sample order."""
    tensor = Mock()
    tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(
        [0.25, -0.5]
    )
    pipeline = Mock(
        return_value=iter(
            [
                ("", "", None),
                ("", "", tensor),
                ("", "", np.array([1.0])),
            ]
        )
    )
    audio = tts.synthesize(pipeline, "Hello", voice="am_adam", speed=0.9)
    np.testing.assert_array_equal(audio, [0.25, -0.5, 1.0])
    assert audio.dtype == np.float32
    tensor.detach.return_value.cpu.assert_called_once()
    pipeline.assert_called_once_with("Hello", voice="am_adam", speed=0.9)


def test_synthesis_without_segments_returns_empty_audio():
    """A model with no audio segments returns an empty float32 array."""
    audio = tts.synthesize(lambda *args, **kwargs: iter([]), "Hello")
    assert audio.size == 0
    assert audio.dtype == np.float32
