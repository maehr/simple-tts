"""Verify shared conversion behavior independently of either interface."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from simple_tts import dependencies, service
from simple_tts.processes import ConversionCancelled
from simple_tts.service import ConversionRequest, execute_conversion, prepare_conversion


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"text": None}, "exactly one source"),
        ({"input_file": Path("book.md")}, "exactly one source"),
        ({"voices": ()}, "at least one voice"),
        ({"voices": ("af_bella,am_adam",)}, "Unknown voice"),
        ({"voices": ("af_bella", "bf_emma")}, "share one"),
        ({"speed": float("nan")}, "Speed"),
        ({"speed": float("inf")}, "Speed"),
        ({"speed": 0.4}, "Speed"),
        ({"speed": 2.1}, "Speed"),
        ({"chapter_level": 0}, "Chapter level"),
        ({"chapter_level": True}, "Chapter level"),
        ({"chapter_level": 1.5}, "Chapter level"),
        ({"device": "gpu"}, "Device"),
    ],
)
def test_shared_request_validation(tmp_path, changes, message):
    """Both interfaces use the same source and option constraints."""
    request = ConversionRequest(text="Hello.", output=tmp_path / "audio.m4a")
    with pytest.raises(ValueError, match=message):
        service.validate_request(replace(request, **changes))


def test_preparation_preview_and_execution(tmp_path, monkeypatch):
    """Preparation does not load a model and execution preserves chapter rotation."""
    output = tmp_path / "audio.M4B"
    request = ConversionRequest(
        text="Intro.\n\n## First\n\n" + "A" * 2000 + "\n\n## Last\n\nEnd.",
        voices=("af_bella", "am_adam"),
        speed=0.8,
        output=output,
        chapter_level=2,
        device="mps",
    )
    events = []
    load = Mock(return_value=object())
    monkeypatch.setattr(service, "load_tts", load)
    prepared = prepare_conversion(request, tmp_path, report=events.append)
    load.assert_not_called()
    assert prepared.format_type == "m4b"
    assert prepared.document.matched_headings == 2
    assert len(prepared.chunks) == 5
    spoken = []

    def synthesize(pipeline, text, voice, speed):
        assert pipeline is load.return_value
        assert speed == 0.8
        spoken.append(voice)
        return np.zeros(2400, dtype=np.float32)

    monkeypatch.setattr(service, "synthesize", synthesize)
    staged = tmp_path / "staged.m4b"
    result = execute_conversion(prepared, tmp_path, staged, report=events.append)
    load.assert_called_once()
    assert load.call_args.args == ("af_bella",)
    assert load.call_args.kwargs["device"] == "mps"
    assert spoken == ["af_bella"] + ["am_adam"] * 3 + ["af_bella"]
    assert [chapter.title for chapter in result.chapters] == [
        "Introduction",
        "First",
        "Last",
    ]
    assert result.duration == 0.5
    assert staged.exists()
    assert not output.exists()
    assert {event.phase for event in events} == {
        "extracting",
        "loading",
        "synthesizing",
        "encoding",
    }
    assert events[-1].completed_chunks == events[-1].total_chunks == 5


@pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda"])
def test_shared_device_selection(tmp_path, monkeypatch, device):
    """The service retains automatic selection and explicit GPU behavior."""
    from tests.test_tts import set_devices

    set_devices(monkeypatch, cuda=True, mps=True)
    request = ConversionRequest(
        text="Hello.", output=tmp_path / "audio.wav", device=device
    )
    prepared = prepare_conversion(request, tmp_path)
    events = []
    execute_conversion(
        prepared, tmp_path, tmp_path / "staged.wav", report=events.append
    )
    expected = "cuda" if device == "auto" else device
    assert any(event.message == f"Device: {expected}" for event in events)


@pytest.mark.parametrize("phase", ["extracting", "loading", "synthesizing", "encoding"])
def test_cancellation_between_stages_and_chunks(tmp_path, monkeypatch, phase):
    """A cancelled stage cannot publish audio or continue to the next stage."""
    request = ConversionRequest(text="Hello.", output=tmp_path / "audio.m4a")
    current = "before"

    def report(event):
        nonlocal current
        current = event.phase

    def check():
        if current == phase:
            raise ConversionCancelled

    staged = tmp_path / "staged.m4a"
    with pytest.raises(ConversionCancelled):
        prepared = prepare_conversion(
            request, tmp_path, report=report, check_cancelled=check
        )
        execute_conversion(
            prepared, tmp_path, staged, report=report, check_cancelled=check
        )
    assert not request.output.exists()


def test_missing_tools_do_not_print(tmp_path, monkeypatch, capsys):
    """The shared service reports missing tools through an exception."""
    monkeypatch.setattr(dependencies, "is_command_available", lambda command: False)
    with pytest.raises(RuntimeError, match="pandoc ffmpeg"):
        prepare_conversion(
            ConversionRequest(text="Hello.", output=tmp_path / "audio.m4a"), tmp_path
        )
    assert capsys.readouterr().out == ""
