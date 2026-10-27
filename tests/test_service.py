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
from simple_tts.voices import Voice

ZOEY = "NbpkqMVS3CJeq2j8"
HADLEY = "bXwrkHhURL1VFJdG"
NELE = "2A2fFf8SkDEJ8nzx"
ANTON = "3ZKKapPOvuWFcw9f"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"text": None}, "exactly one source"),
        ({"input_file": Path("book.md")}, "exactly one source"),
        ({"language": "it"}, "Unsupported language"),
        ({"language": "de-DE"}, "Unsupported language"),
        ({"language": ""}, "Unsupported language"),
        ({"voices": ("Zoey,Hadley",)}, "contain commas"),
        ({"voices": ("Zoey", " ")}, "must not be empty"),
        ({"duration_delta": float("nan")}, "Duration delta"),
        ({"duration_delta": float("inf")}, "Duration delta"),
        ({"duration_delta": -0.31}, "Duration delta"),
        ({"duration_delta": 0.31}, "Duration delta"),
        ({"duration_delta": True}, "Duration delta"),
        ({"duration_delta": "0.1"}, "Duration delta"),
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


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"language": "DE", "voices": ("Unknown",)},
        {"language": "fr", "voices": ("Zoey",)},
        {"duration_delta": -0.3},
        {"duration_delta": 0.3},
        {"duration_delta": 0},
    ],
)
def test_validation_reads_no_catalogue(tmp_path, ptts, changes):
    """Request validation checks syntax only, also for unknown voice names."""
    request = ConversionRequest(text="Hello.", output=tmp_path / "audio.m4a")
    service.validate_request(replace(request, **changes))
    ptts.catalog_voices.assert_not_called()


def test_preparation_preview_and_execution(tmp_path, monkeypatch):
    """Preparation does not load a model and execution preserves chapter rotation."""
    output = tmp_path / "audio.M4B"
    request = ConversionRequest(
        text="Intro.\n\n## First\n\n" + "A" * 2000 + "\n\n## Last\n\nEnd.",
        voices=("zoey", HADLEY),
        duration_delta=-0.2,
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
    assert [voice.name for voice in prepared.voices] == ["Zoey", "Hadley"]
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
    assert load.call_args.kwargs.keys() == {
        "language",
        "device",
        "duration_delta",
        "report",
    }
    assert load.call_args.kwargs["language"] == "en"
    assert load.call_args.kwargs["device"] == "mps"
    assert load.call_args.kwargs["duration_delta"] == -0.2
    # ptts receives the unique catalogue voice ID, not the typed name.
    assert spoken == [ZOEY] + [HADLEY] * 3 + [ZOEY]
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


@pytest.mark.parametrize(
    ("language", "voice", "duration_delta"),
    [
        ("en", "NbpkqMVS3CJeq2j8", 0.0),
        ("de", "2A2fFf8SkDEJ8nzx", -0.3),
        ("es", "b6FvJAiokjdqIti4", 0.15),
        ("FR", "FXxJ9mANRq6BCTX5", 0.3),
    ],
)
def test_language_selects_model_voice_and_rate(
    tmp_path, ptts, monkeypatch, language, voice, duration_delta
):
    """The language picks the native default voice and reaches ptts with the rate."""
    models = []

    def load(**kwargs):
        models.append(tts.load_tts(**kwargs))
        return models[-1]

    monkeypatch.setattr(service, "load_tts", load)
    request = ConversionRequest(
        text="Hello.",
        language=language,
        duration_delta=duration_delta,
        output=tmp_path / "audio.wav",
    )
    prepared = prepare_conversion(request, tmp_path)
    assert prepared.request.language == language.lower()
    execute_conversion(prepared, tmp_path, tmp_path / "staged.wav")
    ptts.TTS.assert_called_once()
    assert ptts.TTS.call_args.kwargs == {
        "device": "cpu",
        "quant": "q8",
        "lang": language.lower(),
        "conditions": {"duration_delta": duration_delta},
    }
    assert [stream_voice for _, stream_voice, _ in models[0].streams] == [voice]


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"voices": ("Unknown",)}, ValueError, "Unknown voice 'Unknown'"),
        ({"language": "de", "voices": ("Zoey",)}, ValueError, "'Zoey' is English"),
        (
            {"language": "de", "voices": ("Nele", "Noémie")},
            ValueError,
            "'Noémie' is French",
        ),
        ({"catalog": OSError("offline")}, RuntimeError, "Connect to the internet"),
    ],
)
def test_voice_errors_precede_extraction(
    tmp_path, ptts, monkeypatch, changes, error, message
):
    """Voice resolution fails before Pandoc, the model, or a voice download."""
    if "catalog" in changes:
        ptts.catalog_voices.side_effect = changes.pop("catalog")
    extract = Mock()
    monkeypatch.setattr(service, "extract_sections", extract)
    request = ConversionRequest(text="Hello.", output=tmp_path / "audio.m4a", **changes)
    with pytest.raises(error, match=message):
        prepare_conversion(request, tmp_path)
    extract.assert_not_called()
    ptts.TTS.assert_not_called()


def test_default_voice_lists_no_other_language(tmp_path, ptts):
    """A missing default fails and never switches to an English voice."""
    ptts.catalog_voices.return_value = [
        {"name": "Zoey", "voice_id": ZOEY, "language": "en"},
        {"name": "Anton", "voice_id": ANTON, "language": "de"},
    ]
    request = ConversionRequest(
        text="Hallo.", language="de", output=tmp_path / "audio.m4a"
    )
    with pytest.raises(ValueError, match="default German \\(de\\) voice 'Nele'"):
        prepare_conversion(request, tmp_path)


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
    """Two German voices rotate per chapter with sample-derived chapter bounds."""
    request = ConversionRequest(
        text="## One\n\nErstens.\n\n## Two\n\nZweitens.",
        language="de",
        voices=("Nele", "Anton"),
        output=tmp_path / "audio.m4b",
        chapter_level=2,
    )
    prepared = prepare_conversion(request, tmp_path)
    assert prepared.voices == (
        Voice("Nele", NELE, "de", "Nele test voice."),
        Voice("Anton", ANTON, "de", "Anton test voice."),
    )
    events = []
    model = tts.load_tts(language="de", report=Mock())
    model.frames = {
        NELE: frames(events, "generate", [1000, 920]),
        ANTON: frames(events, "generate", [2400]),
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
        ("One\n\nErstens.", NELE),
        ("Two\n\nZweitens.", ANTON),
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
    model.frames = {ZOEY: frames(events, "generate", [1920, 1920], error)}
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
