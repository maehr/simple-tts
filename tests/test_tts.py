"""Tests for Phonon model loading, device mapping, and streaming."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from simple_tts import tts
from simple_tts.processes import ConversionCancelled

SNAPSHOT = Path(
    "models--gradium--phonon-40m-multilingual",
    "snapshots",
    "1537ce46c64f42903e0d608b132ccb073dac46d2",
)


def cache_snapshot(cache, *names):
    """Create a cached snapshot that holds the named files."""
    snapshot = cache / SNAPSHOT
    snapshot.mkdir(parents=True)
    for name in names:
        (snapshot / name).touch()
    return snapshot


@pytest.mark.parametrize(
    ("device", "backend", "quant", "display"),
    [
        ("auto", "cpu", "q8", "Device: cpu, quantization: q8_0"),
        ("cpu", "cpu", "q8", "Device: cpu, quantization: q8_0"),
        ("mps", "metal", "f32", "Device: Metal (Test GPU), quantization: f32"),
        ("cuda", "cuda", "f32", "Device: CUDA (Test GPU), quantization: f32"),
    ],
)
def test_device_selects_backend_and_weights(ptts, device, backend, quant, display):
    """Auto uses CPU/q8 even with a GPU; explicit GPUs use f32 weights."""
    ptts.available_devices.return_value = ["cuda", "metal", "cpu"]
    report = Mock()
    model = tts.load_tts(language="de", device=device, report=report)
    ptts.TTS.assert_called_once_with(
        tts.MODEL_SOURCE,
        device=backend,
        quant=quant,
        lang="de",
        conditions={"duration_delta": 0.0},
    )
    assert model.lang == "de"
    assert [call.args[0] for call in report.call_args_list] == [
        f"Model: {tts.MODEL_SOURCE}",
        display,
    ]


def test_pinned_model_source():
    """The default source is the tested multilingual revision, not the 31M model."""
    assert tts.MODEL_SOURCE == (
        "hf://gradium/phonon-40m-multilingual@1537ce46c64f42903e0d608b132ccb073dac46d2"
    )


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_unavailable_gpu_never_runs_on_cpu(ptts, device):
    """An explicit GPU that the ptts build lacks fails before loading."""
    with pytest.raises(RuntimeError, match=f"'{device}' is unavailable"):
        tts.load_tts(language="en", device=device)
    ptts.TTS.assert_not_called()
    ptts.set_num_threads.assert_not_called()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"language": "it"}, "Language must be one of: en, de, es, fr"),
        ({"language": "pt"}, "Language must be one of"),
        ({"language": "DE"}, "Language must be one of"),
        ({"device": "metal"}, "Device must be"),
        ({"device": "gpu"}, "Device must be"),
        ({"duration_delta": -0.31}, "Duration delta must be from -0.3 through 0.3"),
        ({"duration_delta": 0.31}, "Duration delta"),
        ({"duration_delta": float("nan")}, "Duration delta"),
        ({"duration_delta": float("inf")}, "Duration delta"),
    ],
)
def test_invalid_options_fail_before_loading(ptts, options, message):
    """Invalid options never reach ptts."""
    with pytest.raises(ValueError, match=message):
        tts.load_tts(**{"language": "en", **options})
    ptts.TTS.assert_not_called()
    ptts.available_devices.assert_not_called()


@pytest.mark.parametrize("delta", [-0.3, -0.15, 0.3])
def test_duration_delta_reaches_conditions(ptts, delta):
    """The rate condition is passed through within the validated range."""
    model = tts.load_tts(language="fr", duration_delta=delta, report=Mock())
    assert model.conditions == {"duration_delta": delta}
    assert tts.DURATION_DELTA_RANGE == (-0.3, 0.3)


def test_local_model_variable_takes_precedence(ptts, tmp_path, monkeypatch):
    """SIMPLE_TTS_MODEL selects a local checkpoint before a cached snapshot."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    cache_snapshot(tmp_path, "config.json", "tokenizer.json", "model.q8.gguf")
    monkeypatch.setenv("SIMPLE_TTS_MODEL", "~/checkpoints/phonon")
    model = tts.load_tts(language="en", report=Mock())
    assert model.model == str(Path.home() / "checkpoints" / "phonon")


@pytest.mark.parametrize(
    ("device", "files", "cached"),
    [
        ("auto", ["config.json", "tokenizer.json", "model.q8.gguf"], True),
        ("mps", ["config.json", "tokenizer.json", "model.safetensors"], True),
        ("mps", ["config.json", "tokenizer.json", "model.q8.gguf"], False),
        ("cpu", ["config.json", "model.q8.gguf"], False),
        ("cpu", ["tokenizer.json", "model.q8.gguf"], False),
    ],
)
def test_complete_cached_snapshot_loads_locally(
    ptts, tmp_path, monkeypatch, device, files, cached
):
    """A complete snapshot avoids the Hub request of a pinned hf:// load."""
    ptts.available_devices.return_value = ["metal", "cpu"]
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snapshot = cache_snapshot(tmp_path, *files)
    model = tts.load_tts(language="en", device=device, report=Mock())
    assert model.model == (str(snapshot) if cached else tts.MODEL_SOURCE)


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"HF_HUB_CACHE": "a", "HUGGINGFACE_HUB_CACHE": "b", "HF_HOME": "c"}, "a"),
        ({"HUGGINGFACE_HUB_CACHE": "b", "HF_HOME": "c"}, "b"),
        ({"HF_HOME": "c", "XDG_CACHE_HOME": "d"}, "c/hub"),
        ({"XDG_CACHE_HOME": "d"}, "d/huggingface/hub"),
        ({}, "home/.cache/huggingface/hub"),
    ],
)
def test_hub_cache_follows_ptts_order(tmp_path, monkeypatch, variables, expected):
    """The cache lookup uses the same variables as the ptts runtime."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, str(tmp_path / value))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert tts.hub_cache() == tmp_path / expected


@pytest.mark.parametrize(("cores", "threads"), [(18, 6), (4, 4), (None, 1)])
def test_threads_are_set_once_before_the_first_load(ptts, monkeypatch, cores, threads):
    """The CPU thread pool is sized before ptts builds it."""
    monkeypatch.setattr(tts.os, "cpu_count", lambda: cores)
    ptts.set_num_threads.side_effect = lambda n: ptts.TTS.assert_not_called()
    tts.load_tts(language="en", report=Mock())
    tts.load_tts(language="de", device="cpu", report=Mock())
    ptts.set_num_threads.assert_called_once_with(threads)
    assert ptts.TTS.call_count == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sample_rate": 22050}, "requires 24000 Hz"),
        ({"languages": ["en"]}, "does not support language 'de'"),
    ],
)
def test_loaded_model_is_checked(ptts, change, message):
    """A checkpoint without the language or sample rate is rejected."""
    create = ptts.TTS.side_effect

    def load(*args, **kwargs):
        model = create(*args, **kwargs)
        vars(model).update(change)
        return model

    ptts.TTS.side_effect = load
    with pytest.raises(RuntimeError, match=message):
        tts.load_tts(language="de", report=Mock())


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (LookupError("no checkpoint at /x"), "cannot find a checkpoint.*no checkpoint"),
        (
            OSError("403 (error_code=GatedRepo) token hf_secret"),
            r"Accept its terms at https://huggingface.co/gradium/phonon-40m-multi"
            r"lingual, then run: hf auth login$",
        ),
        (OSError("connection refused"), "cannot download.*connection refused"),
        (
            NotImplementedError("device 'cuda' is not available in this build"),
            "not available in this build. Use --device auto or cpu.",
        ),
    ],
)
def test_load_errors_are_explained(ptts, error, message):
    """Runtime errors become clear messages; gated errors omit the raw text."""
    ptts.TTS.side_effect = error
    with pytest.raises(RuntimeError, match=message) as raised:
        tts.load_tts(language="en", report=Mock())
    assert "hf_secret" not in str(raised.value)
    assert raised.value.__cause__ is error


def test_missing_ptts_reports_install_hint(monkeypatch):
    """A missing runtime package never returns silence."""
    monkeypatch.setitem(sys.modules, "ptts", None)
    with pytest.raises(RuntimeError, match="ptts is missing"):
        tts.load_tts(language="en")


def test_stream_yields_mono_float32_frames_and_closes(ptts):
    """Frames keep their order and leaving the block closes the stream."""
    model = tts.load_tts(language="en", report=Mock())
    model.frames = [np.array([0.25, -0.5], dtype=np.float64), np.ones(1920)]
    with tts.stream_speech(model, "Hello.", voice="Zoey") as frames:
        audio = list(frames)
    text, voice, stream = model.streams[0]
    assert (text, voice, stream.closed) == ("Hello.", "Zoey", True)
    assert [frame.dtype for frame in audio] == [np.float32, np.float32]
    np.testing.assert_array_equal(np.concatenate(audio)[:2], [0.25, -0.5])
    assert sum(frame.size for frame in audio) == 1922


@pytest.mark.parametrize("text", ["", "  \n\t "])
def test_empty_chunks_are_skipped(ptts, text):
    """The helper yields no frames, because ptts rejects empty text."""
    model = tts.load_tts(language="en", report=Mock())
    with tts.stream_speech(model, text, voice="Zoey") as frames:
        assert list(frames) == []
    assert model.streams == []


def test_stream_closes_after_cancellation(ptts):
    """Cancellation between frames stops generation."""
    model = tts.load_tts(language="en", report=Mock())
    model.frames = [np.zeros(1920)] * 3
    with pytest.raises(ConversionCancelled):
        with tts.stream_speech(model, "Hello.", voice="Zoey") as frames:
            next(frames)
            raise ConversionCancelled
    assert model.streams[0][2].closed


def test_stream_rejects_multichannel_audio(ptts):
    """Only mono frames reach the PCM writer."""
    model = tts.load_tts(language="en", report=Mock())
    model.frames = [np.zeros((2, 1920))]
    with pytest.raises(ValueError, match="not mono"):
        with tts.stream_speech(model, "Hello.", voice="Zoey") as frames:
            list(frames)
    assert model.streams[0][2].closed


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (LookupError("no voice 'Hadley' is registered"), "no voice 'Hadley'"),
        (OSError("cannot fetch `voices/x.safetensors`"), "Connect to the internet"),
    ],
)
def test_voice_errors_are_explained(ptts, error, message):
    """Voice errors occur when the stream starts and get clear messages."""
    model = tts.load_tts(language="en", report=Mock())
    model.stream = Mock(side_effect=error)
    with pytest.raises(RuntimeError, match=message):
        with tts.stream_speech(model, "Hello.", voice="Hadley"):
            pass


def test_imports_stay_lazy():
    """Importing the application loads neither ptts nor PyTorch settings."""
    script = """
import os
import sys
import simple_tts.cli
import simple_tts.service
assert "ptts" not in sys.modules
assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
"""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name != "PYTORCH_ENABLE_MPS_FALLBACK"
    }
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
