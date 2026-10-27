"""Verify shared conversion behavior independently of either interface."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from simple_tts import dependencies, service, tts
from simple_tts.audio import Chapter, PCMWriter
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

    @contextmanager
    def stream(model, text, *, voice):
        assert model is load.return_value
        spoken.append(voice)
        yield iter([np.zeros(1200, dtype=np.float32)] * 2)

    monkeypatch.setattr(service, "stream_speech", stream)
    staged = tmp_path / "staged.m4b"
    result = execute_conversion(prepared, tmp_path, staged, report=events.append)
    load.assert_called_once()
    assert load.call_args.args == ()
    assert load.call_args.kwargs.keys() == {"language", "device", "report"}
    assert load.call_args.kwargs["language"] == "en"
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


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("auto", "cpu, quantization: q8_0"),
        ("cpu", "cpu, quantization: q8_0"),
        ("mps", "Metal (Test GPU), quantization: f32"),
        ("cuda", "CUDA (Test GPU), quantization: f32"),
    ],
)
def test_shared_device_selection(tmp_path, ptts, device, expected):
    """Auto stays on CPU/q8 and explicit GPUs report their f32 backend."""
    ptts.available_devices.return_value = ["cuda", "metal", "cpu"]
    request = ConversionRequest(
        text="Hello.", output=tmp_path / "audio.wav", device=device
    )
    prepared = prepare_conversion(request, tmp_path)
    events = []
    execute_conversion(
        prepared, tmp_path, tmp_path / "staged.wav", report=events.append
    )
    assert any(event.message == f"Device: {expected}" for event in events)
    ptts.TTS.assert_called_once()


def test_unavailable_gpu_fails_before_synthesis(tmp_path):
    """An explicit GPU never falls back to the CPU."""
    request = ConversionRequest(text="Hello.", output=tmp_path / "audio.m4a")
    prepared = prepare_conversion(replace(request, device="mps"), tmp_path)
    staged = tmp_path / "staged.m4a"
    with pytest.raises(RuntimeError, match="'mps' is unavailable"):
        execute_conversion(prepared, tmp_path, staged)
    assert not (tmp_path / "narration.f32le").exists()
    assert not staged.exists()


class RecordingWriter(PCMWriter):
    """Record each write so tests can check the order of generation."""

    events: list[str] = []

    def write(self, audio):
        """Write the frame and record the new sample count."""
        super().write(audio)
        self.events.append(f"write {self.samples}")


def frames(events, name, sizes, failure=None):
    """Generate frames lazily and record each generation."""
    for size in sizes:
        events.append(f"{name} {size}")
        yield np.full(size, 0.5, dtype=np.float32)
    if failure is not None:
        raise failure


def test_frames_stream_to_pcm_with_sample_boundaries(tmp_path, monkeypatch):
    """Three frames over two chapters give sample-derived chapter bounds."""
    request = ConversionRequest(
        text="## One\n\nFirst.\n\n## Two\n\nSecond.",
        voices=("af_bella", "am_adam"),
        output=tmp_path / "audio.m4b",
        chapter_level=2,
    )
    prepared = prepare_conversion(request, tmp_path)
    events = []
    model = tts.load_tts(language="en", report=Mock())
    model.frames = {
        "af_bella": frames(events, "generate", [1000, 920]),
        "am_adam": frames(events, "generate", [2400]),
    }
    monkeypatch.setattr(service, "load_tts", lambda **kwargs: model)
    monkeypatch.setattr(RecordingWriter, "events", events)
    monkeypatch.setattr(service, "PCMWriter", RecordingWriter)
    encoded = []

    def encode(pcm, output, format_type, chapters):
        assert pcm.stat().st_size == 4320 * 4
        encoded.extend(chapters)
        output.touch()

    monkeypatch.setattr(service, "encode_pcm", encode)
    progress = []
    result = execute_conversion(
        prepared, tmp_path, tmp_path / "staged.m4b", report=progress.append
    )
    # Each frame reaches the writer before the next frame is generated.
    assert events == [
        "generate 1000",
        "write 1000",
        "generate 920",
        "write 1920",
        "generate 2400",
        "write 4320",
    ]
    assert encoded == [Chapter("One", 0, 1920), Chapter("Two", 1920, 4320)]
    assert [(c.title, c.start, c.end) for c in result.chapters] == [
        ("One", 0.0, 0.08),
        ("Two", 0.08, 0.18),
    ]
    assert result.duration == 0.18
    # Progress counts text chunks, not audio frames.
    assert [(event.phase, event.completed_chunks) for event in progress] == [
        ("loading", 0),
        ("synthesizing", 0),
        ("synthesizing", 1),
        ("synthesizing", 2),
        ("encoding", 2),
    ]
    assert [(text, voice) for text, voice, _ in model.streams] == [
        ("One\n\nFirst.", "af_bella"),
        ("Two\n\nSecond.", "am_adam"),
    ]
    assert all(stream.closed for _, _, stream in model.streams)


@pytest.mark.parametrize("failure", ["cancellation", "error"])
def test_stream_closes_without_publication(tmp_path, monkeypatch, failure):
    """Cancellation between frames and stream errors close the stream."""
    request = ConversionRequest(text="Hello.", output=tmp_path / "audio.m4a")
    prepared = prepare_conversion(request, tmp_path)
    events = []
    model = tts.load_tts(language="en", report=Mock())
    error = RuntimeError("decoder failure") if failure == "error" else None
    model.frames = {"af_bella": frames(events, "generate", [1920, 1920], error)}
    monkeypatch.setattr(service, "load_tts", lambda **kwargs: model)
    monkeypatch.setattr(RecordingWriter, "events", events)
    monkeypatch.setattr(service, "PCMWriter", RecordingWriter)
    encode = Mock()
    monkeypatch.setattr(service, "encode_pcm", encode)

    def check():
        if failure == "cancellation" and "write 1920" in events:
            raise ConversionCancelled

    staged = tmp_path / "staged.m4a"
    expected = ConversionCancelled if failure == "cancellation" else RuntimeError
    with pytest.raises(expected):
        execute_conversion(prepared, tmp_path, staged, check_cancelled=check)
    assert model.streams[0][2].closed
    if failure == "cancellation":
        assert events == ["generate 1920", "write 1920"]
    encode.assert_not_called()
    assert not staged.exists()
    assert not request.output.exists()


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
