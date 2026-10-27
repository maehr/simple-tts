"""Tests for the CLI module."""

import re
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from simple_tts.cli import app, format_duration
from tests.workers import one_frame

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NELE = "2A2fFf8SkDEJ8nzx"
ANTON = "3ZKKapPOvuWFcw9f"


def plain(output):
    """Remove ANSI codes and borders, and join wrapped table and help lines."""
    return " ".join(ANSI.sub("", output).replace("│", " ").split())


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_formats_seconds_only(self):
        """Test formatting of seconds only."""
        assert format_duration(30) == "30s"

    def test_formats_minutes_and_seconds(self):
        """Test formatting of minutes and seconds."""
        assert format_duration(65) == "1m 5s"

    def test_formats_hours_minutes_seconds(self):
        """Test formatting of hours, minutes, and seconds."""
        assert format_duration(3661) == "1h 1m 1s"

    def test_handles_zero(self):
        """Test handling of zero duration."""
        assert format_duration(0) == "0s"


class TestCLIVersion:
    """Tests for the version command."""

    def test_version_command(self):
        """Test that version command outputs version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "simple-tts" in result.stdout
        from simple_tts import __version__

        assert __version__ in result.stdout


class TestCLIVoices:
    """Tests for the voices command."""

    def test_voices_command(self, ptts):
        """The command lists catalogue voices of the four languages only."""
        result = runner.invoke(app, ["voices"])
        assert result.exit_code == 0, result.output
        clean = plain(result.stdout)
        for name in ["Zoey (default)", "Hadley", "Nele (default)", "Anton"]:
            assert name in clean
        assert "Noa (default)" in clean
        assert "Noémie (default)" in clean
        assert "Hadley (default)" not in clean
        assert NELE in clean
        assert "German (de)" in clean
        assert "Nele test voice." in clean
        assert "Manuela-Lu" not in clean
        assert "af_bella" not in clean
        ptts.catalog_voices.assert_called_once_with()
        ptts.TTS.assert_not_called()

    def test_voices_filter_by_language(self):
        """A language filter shows only the voices of that language."""
        result = runner.invoke(app, ["voices", "--language", "DE"])
        assert result.exit_code == 0, result.output
        clean = plain(result.stdout)
        assert "Nele" in clean
        assert "Anton" in clean
        assert "Zoey" not in clean

    @pytest.mark.parametrize("arguments", [["--language", "it"], ["-l", "pt"]])
    def test_voices_rejects_other_languages(self, arguments):
        """Unsupported languages give an error."""
        result = runner.invoke(app, ["voices", *arguments])
        assert result.exit_code == 1
        assert "Unsupported language" in plain(result.output)

    def test_voices_without_catalogue(self, ptts):
        """An uncached catalogue without network gives a clear error."""
        ptts.catalog_voices.side_effect = OSError("cannot fetch voices.jsonl")
        result = runner.invoke(app, ["voices"])
        assert result.exit_code == 1
        clean = plain(result.output)
        assert "voice catalogue" in clean
        assert "Connect to the internet once" in clean


class TestCLIConvert:
    """Tests for the convert command."""

    def test_convert_missing_file(self):
        """Test convert with non-existent file."""
        result = runner.invoke(app, ["convert", "/nonexistent/file.md"])
        assert result.exit_code != 0

    def test_convert_shows_help(self):
        """Test that convert shows help."""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Prepared input" in result.stdout
        assert "PDF" not in result.stdout
        clean = plain(result.stdout)
        assert "--voice" in clean
        assert "--yes" in clean
        assert "--language" in clean
        assert "en, de, es, or fr" in clean
        assert "--duration-delta" in clean
        assert "-0.3<=x<=0.3" in clean
        assert "--speed" not in clean
        assert "af_bella" not in clean
        assert "Kokoro" not in clean

    def test_app_help_names_phonon(self):
        """The top-level help names the new engine."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        clean = plain(result.stdout)
        assert "Phonon 40M Multilingual" in clean
        assert "Kokoro" not in clean


class TestCLICheckDeps:
    """Tests for the check-deps command."""

    def test_check_deps_command(self):
        """Test that check-deps command works."""
        result = runner.invoke(app, ["check-deps"])
        assert result.exit_code == 0
        assert "pandoc" in result.stdout
        assert "ffmpeg" in result.stdout


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    """Prepare a deterministic conversion without loading a real model."""
    from simple_tts import service

    source = tmp_path / "book.md"
    source.write_text("Preamble.\n\n## First\n\nBody.\n\n## Last\n\nEnd.")
    monkeypatch.setattr(service, "require_dependencies", lambda *args: True)
    monkeypatch.setattr(service, "load_tts", lambda *args, **kwargs: object())
    return service, source


@pytest.mark.parametrize("suffix", [".pdf", ".PDF"])
def test_pdf_rejected_before_external_tools_or_synthesis(tmp_path, monkeypatch, suffix):
    """PDF input points to preparation guidance without loading tools or models."""
    from unittest.mock import Mock

    from simple_tts import converter, service

    source = tmp_path / f"paper{suffix}"
    source.write_bytes(b"%PDF-1.7")
    run = Mock()
    check = Mock()
    load = Mock()
    monkeypatch.setattr(converter.subprocess, "run", run)
    monkeypatch.setattr(service, "require_dependencies", check)
    monkeypatch.setattr(service, "load_tts", load)
    result = runner.invoke(app, ["convert", str(source), "-y"])
    assert result.exit_code == 1
    assert "PDF input is not supported" in result.output
    assert "Markdown or plain text" in result.output
    assert "Check reading order and remove unwanted text" in result.output
    assert "https://github.com/maehr/simple-tts#prepare-input" in result.output
    run.assert_not_called()
    check.assert_not_called()
    load.assert_not_called()
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize(
    ("arguments", "expected_format", "output_name"),
    [
        ([], "m4a", "book.m4a"),
        (["--format", "m4b"], "m4b", "book.m4b"),
        (["--format", "mp4"], "mp4", "book.mp4"),
        (["--output", "named.MP4"], "mp4", "named.MP4"),
        (["--output", "named.M4B"], "m4b", "named.M4B"),
        (["--output", "named.mp3"], "mp3", "named.mp3"),
        (["--output", "named"], "m4a", "named.m4a"),
    ],
)
def test_output_format_resolution(
    conversion, monkeypatch, arguments, expected_format, output_name
):
    """Suffix inference preserves M4A as the default."""
    import numpy as np

    service, source = conversion
    calls = []

    def encode(pcm, output, format_type, chapters):
        calls.append(format_type)
        assert not (source.parent / output_name).exists()
        assert output.parent.parent == source.parent
        output.write_bytes(b"encoded")

    monkeypatch.setattr(
        service, "stream_speech", one_frame(lambda *args: np.zeros(240))
    )
    monkeypatch.setattr(service, "encode_pcm", encode)
    arguments = list(arguments)
    if "--output" in arguments:
        arguments[-1] = str(source.parent / arguments[-1])
    result = runner.invoke(app, ["convert", str(source), "-y", *arguments])
    assert result.exit_code == 0, result.output
    assert calls == [expected_format]
    assert (source.parent / output_name).read_bytes() == b"encoded"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--format", "m4b", "-o", "book.mp3"],
        ["--format", "invalid"],
        ["-o", "book.flac"],
        ["--device", "invalid"],
        ["--chapter-level", "0"],
        ["--chapter-level", "7"],
        ["--voice", ""],
        ["--voice", " "],
        ["--voice", "Zoey,"],
        ["--voice", "Zoey,unknown"],
        ["--voice", "Zoey,Hadley"],
        ["--voice", "af_bella"],
        ["--voice", "Zoey", "--language", "de"],
        ["--voice", "Nele,Anton", "--language", "de"],
        ["--voice", "nbpkqmvs3cjeq2j8"],
        ["--language", "it"],
        ["--language", "pt"],
        ["--language", "de-DE"],
        ["--duration-delta", "0.31"],
        ["--duration-delta", "-0.5"],
        ["--duration-delta", "nan"],
        ["--duration-delta", "fast"],
        ["--speed", "1.2"],
        ["-s", "1.2"],
        ["--no-vtt"],
        ["--vtt"],
        ["--split"],
    ],
)
def test_rejected_options_precede_synthesis(conversion, monkeypatch, arguments):
    """Invalid options never start extraction or synthesis."""
    from unittest.mock import Mock

    service, source = conversion
    extract = Mock(side_effect=AssertionError("Must not extract"))
    load = Mock(side_effect=AssertionError("Must not load"))
    monkeypatch.setattr(service, "extract_sections", extract)
    monkeypatch.setattr(service, "load_tts", load)
    result = runner.invoke(app, ["convert", str(source), "-y", *arguments])
    assert result.exit_code != 0
    extract.assert_not_called()
    load.assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--voice", "Zoey", "--language", "de"], "'Zoey' is English (en)"),
        (["--voice", "Unknown"], "Unknown voice 'Unknown'"),
        (
            ["-l", "de", "-v", "Nele,Anton"],
            "Voice 'Nele,Anton' contains a comma. Each conversion uses one voice. "
            "Voice rotation is removed.",
        ),
        (["--voice", ""], "The voice name or voice ID must not be empty."),
        (["--language", "it"], "Unsupported language 'it'"),
        (["--duration-delta", "nan"], "Duration delta must be from -0.3"),
    ],
)
def test_option_errors_explain_the_fix(conversion, arguments, message):
    """Voice, language, and rate errors name the problem."""
    service, source = conversion
    result = runner.invoke(app, ["convert", str(source), "-y", *arguments])
    assert result.exit_code == 1
    assert message in plain(result.output)


@pytest.mark.parametrize(
    ("language", "name", "voice_id"),
    [
        (None, "Zoey", "NbpkqMVS3CJeq2j8"),
        ("en", "Zoey", "NbpkqMVS3CJeq2j8"),
        ("de", "Nele", NELE),
        ("es", "Noa", "b6FvJAiokjdqIti4"),
        ("FR", "Noémie", "FXxJ9mANRq6BCTX5"),
    ],
)
def test_default_voice_follows_language(
    conversion, monkeypatch, ptts, language, name, voice_id
):
    """An omitted voice selects the native default voice of the language."""
    from simple_tts import tts

    service, source = conversion
    monkeypatch.setattr(service, "load_tts", tts.load_tts)
    monkeypatch.setattr(
        service, "encode_pcm", lambda pcm, output, *args: output.touch()
    )
    arguments = ["--language", language] if language else []
    result = runner.invoke(app, ["convert", str(source), "-y", *arguments])
    assert result.exit_code == 0, result.output
    clean = plain(result.output)
    code = (language or "en").lower()
    assert f"Voice: {name}" in clean
    assert f"({code})" in clean
    assert ptts.TTS.call_args.kwargs["lang"] == code
    assert ptts.TTS.call_args.kwargs["conditions"] == {"duration_delta": 0.0}


@pytest.mark.parametrize("value", ["-0.3", "-0.15", "0.25", "0.3"])
def test_duration_delta_reaches_ptts(conversion, monkeypatch, ptts, value):
    """The rate option becomes the native ptts duration condition."""
    from simple_tts import tts

    service, source = conversion
    monkeypatch.setattr(service, "load_tts", tts.load_tts)
    monkeypatch.setattr(
        service, "encode_pcm", lambda pcm, output, *args: output.touch()
    )
    result = runner.invoke(
        app,
        ["convert", str(source), "-y", "-l", "de", "--duration-delta", value],
    )
    assert result.exit_code == 0, result.output
    ptts.TTS.assert_called_once()
    assert ptts.TTS.call_args.kwargs == {
        "device": "cpu",
        "quant": "q8",
        "lang": "de",
        "conditions": {"duration_delta": float(value)},
    }


@pytest.mark.parametrize("format_type", ["m4a", "m4b", "mp4", "mp3", "wav", "opus"])
def test_sample_boundaries(conversion, monkeypatch, format_type):
    """Each chapter begins before its heading and ends at its final sample."""
    import numpy as np

    from simple_tts.audio import Chapter

    service, source = conversion
    lengths = iter([24000, 36000, 12000])
    spoken = []
    pcm_paths = []
    chapters_seen = []

    def synthesize(model, text, voice):
        spoken.append(text)
        return np.full(next(lengths), len(spoken), dtype=np.float32)

    def encode(pcm, output, format_type, chapters):
        pcm_paths.append(pcm)
        samples = np.fromfile(pcm, dtype="<f4")
        assert samples.tolist() == [1] * 24000 + [2] * 36000 + [3] * 12000
        chapters_seen.extend(chapters)
        assert not source.with_suffix(".vtt").exists()
        output.write_bytes(b"encoded")

    monkeypatch.setattr(service, "stream_speech", one_frame(synthesize))
    monkeypatch.setattr(service, "encode_pcm", encode)
    result = runner.invoke(
        app,
        [
            "convert",
            str(source),
            "--format",
            format_type,
            "--chapter-level",
            "2",
            "-y",
        ],
    )
    assert result.exit_code == 0, result.output
    assert chapters_seen == [
        Chapter("Introduction", 0, 24000),
        Chapter("First", 24000, 60000),
        Chapter("Last", 60000, 72000),
    ]
    assert spoken == ["Preamble.", "First\n\nBody.", "Last\n\nEnd."]
    assert all(not path.parent.exists() for path in pcm_paths)
    assert not list(source.parent.glob(".simple-tts-*"))


@pytest.mark.parametrize("arguments", [[], ["--format", "m4b"], ["--format", "mp4"]])
def test_missing_headings_warning(conversion, monkeypatch, arguments):
    """Every conversion reports absent heading matches."""
    import numpy as np

    service, source = conversion
    source.write_text("## Different level\n\nBody.")
    titles = []

    def encode(pcm, output, format_type, chapters):
        titles.extend(chapter.title for chapter in chapters)
        output.write_bytes(b"encoded")

    monkeypatch.setattr(
        service, "stream_speech", one_frame(lambda *args: np.zeros(240))
    )
    monkeypatch.setattr(service, "encode_pcm", encode)
    result = runner.invoke(app, ["convert", str(source), "-y", *arguments])
    assert result.exit_code == 0, result.output
    assert "No headings match" in result.output
    assert titles == ["book"]


def test_long_conversion_releases_audio_arrays(conversion, monkeypatch):
    """Hundreds of chunks retain at most the last written frame."""
    import weakref

    import numpy as np

    from simple_tts.converter import ExtractedDocument, Section

    service, source = conversion
    references = []
    chunk_count = 300
    monkeypatch.setattr(
        service,
        "extract_sections",
        lambda *args, **kwargs: ExtractedDocument(
            [Section("Long", "A" * 1000 * chunk_count)], 1
        ),
    )

    def live_frames():
        return sum(reference() is not None for reference in references)

    def synthesize(*args):
        assert live_frames() <= 1
        audio = np.zeros(24000, dtype=np.float32)
        references.append(weakref.ref(audio))
        return audio

    def encode(pcm, output, format_type, chapters):
        assert live_frames() <= 1
        assert pcm.stat().st_size == chunk_count * 24000 * 4
        assert chapters[-1].end_sample == chunk_count * 24000
        output.write_bytes(b"encoded")

    monkeypatch.setattr(service, "stream_speech", one_frame(synthesize))
    monkeypatch.setattr(service, "encode_pcm", encode)
    result = runner.invoke(app, ["convert", str(source), "-y"])
    assert result.exit_code == 0, result.output
    assert len(references) == chunk_count


@pytest.mark.parametrize("failure_stage", ["synthesis", "encoding", "cancellation"])
def test_failure_cleanup_and_publication(conversion, monkeypatch, failure_stage):
    """Errors and cancellation remove temporary files and preserve old outputs."""
    import numpy as np

    service, source = conversion
    output = source.with_suffix(".m4b")
    output.write_bytes(b"old audio")
    directories = []
    temporary_directory = tempfile.TemporaryDirectory

    def track_directory(*args, **kwargs):
        directory = temporary_directory(*args, **kwargs)
        directories.append(Path(directory.name))
        return directory

    calls = 0

    def synthesize(*args):
        nonlocal calls
        calls += 1
        if calls == 2 and failure_stage == "cancellation":
            raise KeyboardInterrupt
        if calls == 2 and failure_stage == "synthesis":
            raise RuntimeError("synthesis error")
        return np.zeros(2400, dtype=np.float32)

    def encode(pcm, staged, *args):
        staged.write_bytes(b"partial encode")
        raise RuntimeError("encoding error")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", track_directory)
    monkeypatch.setattr(service, "stream_speech", one_frame(synthesize))
    monkeypatch.setattr(service, "encode_pcm", encode)
    result = runner.invoke(app, ["convert", str(source), "--format", "m4b", "-y"])
    assert result.exit_code != 0
    assert output.read_bytes() == b"old audio"
    assert directories and all(not directory.exists() for directory in directories)


def test_no_audio_is_an_error(conversion, monkeypatch):
    """Empty synthesis does not publish a silent placeholder file."""
    from unittest.mock import Mock

    import numpy as np

    service, source = conversion
    monkeypatch.setattr(service, "stream_speech", one_frame(lambda *args: np.array([])))
    encode = Mock()
    monkeypatch.setattr(service, "encode_pcm", encode)
    result = runner.invoke(app, ["convert", str(source), "-y"])
    assert result.exit_code == 1
    assert "Synthesis produced no audio" in result.output
    assert not source.with_suffix(".m4a").exists()
    assert not source.with_suffix(".vtt").exists()
    encode.assert_not_called()


@pytest.mark.parametrize("format_type", ["m4b", "mp4"])
def test_real_cli_embedded_chapters(tmp_path, format_type):
    """The CLI writes one file with embedded chapter titles and times."""
    import json
    import subprocess

    source = tmp_path / "book.qmd"
    source.write_text(
        "## Über & &lt;雪&gt;\n\nFirst.\n\n## Last\n\nEnd.", encoding="utf-8"
    )
    output = tmp_path / f"book.{format_type}"
    result = runner.invoke(
        app,
        [
            "convert",
            str(source),
            "-o",
            str(output),
            "--chapter-level",
            "2",
            "-y",
        ],
    )
    assert result.exit_code == 0, result.output
    probe = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_chapters",
                "-show_streams",
                "-of",
                "json",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert probe["streams"][0]["codec_name"] == "aac"
    assert [chapter["tags"]["title"] for chapter in probe["chapters"]] == [
        "Über & <雪>",
        "Last",
    ]
    assert [
        (chapter["start_time"], chapter["end_time"]) for chapter in probe["chapters"]
    ] == [("0.000000", "0.100000"), ("0.100000", "0.200000")]
    assert {path.name for path in tmp_path.iterdir()} == {source.name, output.name}


@pytest.mark.parametrize(
    ("arguments", "name", "voice_id"),
    [
        ([], "Nele", NELE),
        (["-v", " anton "], "Anton", ANTON),
        (["--voice", ANTON], "Anton", ANTON),
    ],
)
def test_one_voice_speaks_every_chapter(
    conversion, monkeypatch, arguments, name, voice_id
):
    """One model and one voice serve every chunk of every chapter."""
    import numpy as np

    from simple_tts.converter import ExtractedDocument, Section

    service, source = conversion
    document = ExtractedDocument(
        [Section(title, "A" * 2000) for title in ["Introduction", "First", "Last"]], 2
    )
    monkeypatch.setattr(service, "extract_sections", lambda *args, **kwargs: document)
    loaded = []
    spoken = []
    engine = object()

    def load(*, language, device, duration_delta, report):
        loaded.append(language)
        return engine

    def synthesize(model, text, voice):
        assert model is engine
        spoken.append(voice)
        return np.zeros(240, dtype=np.float32)

    monkeypatch.setattr(service, "load_tts", load)
    monkeypatch.setattr(service, "stream_speech", one_frame(synthesize))
    monkeypatch.setattr(
        service, "encode_pcm", lambda pcm, output, *args: output.touch()
    )
    result = runner.invoke(
        app, ["convert", str(source), "--language", "de", *arguments, "-y"]
    )
    assert result.exit_code == 0, result.output
    assert f"Voice: {name} " in plain(result.output)
    assert loaded == ["de"]
    assert spoken == [voice_id] * 6


def test_output_directory_collision_precedes_synthesis(conversion, monkeypatch):
    """Output paths cannot use an existing directory."""
    from unittest.mock import Mock

    service, source = conversion
    output = source.with_suffix(".m4a")
    output.mkdir()
    load = Mock()
    monkeypatch.setattr(service, "load_tts", load)
    result = runner.invoke(app, ["convert", str(source), "-o", str(output), "-y"])
    assert result.exit_code == 1
    load.assert_not_called()
    assert source.is_file()


def test_confirmation_shows_the_voice(conversion, monkeypatch):
    """The checkpoint identifies the one voice before conversion starts."""
    import numpy as np

    service, source = conversion
    monkeypatch.setattr(
        service, "stream_speech", one_frame(lambda *args: np.zeros(240))
    )
    monkeypatch.setattr(service, "encode_pcm", lambda pcm, path, *args: path.touch())
    result = runner.invoke(
        app,
        [
            "convert",
            str(source),
            "--language",
            "de",
            "--voice",
            "Anton",
            "--duration-delta",
            "-0.1",
        ],
        input="\n",
    )
    assert result.exit_code == 0, result.output
    clean = plain(result.output)
    assert "Language German (de)" in clean
    assert f"Voice Anton ({ANTON}): Anton test voice." in clean
    assert clean.count("Voice Anton") == 1
    assert "Nele" not in clean
    assert "Duration Delta -0.1" in clean
    # Only the summary's speed ratio remains. The speed setting is gone.
    assert clean.count("Speed") == clean.count("Speed Ratio") == 1


@pytest.mark.parametrize(
    "arguments", [["--format", "ogg"], ["--format", "vorbis"], ["-o", "book.ogg"]]
)
def test_vorbis_is_rejected_before_extraction(conversion, monkeypatch, arguments):
    """Removed formats fail before extraction or model loading."""
    from unittest.mock import Mock

    service, source = conversion
    extract = Mock()
    monkeypatch.setattr(service, "extract_sections", extract)
    result = runner.invoke(app, ["convert", str(source), "-y", *arguments])
    assert result.exit_code == 1
    assert "Unsupported" in result.output
    extract.assert_not_called()


@pytest.mark.parametrize("success", [True, False])
def test_install_deps_exit_status(monkeypatch, success):
    """The installer command returns the installation result as its exit status."""
    from unittest.mock import Mock

    install = Mock(return_value=success)
    from simple_tts import cli

    monkeypatch.setattr(cli, "install_dependencies_brew", install)
    result = runner.invoke(app, ["install-deps", "--brew"])
    assert result.exit_code == (0 if success else 1)
    install.assert_called_once_with(cli.console)


def test_install_deps_shows_manual_commands():
    """The installer provides commands when no installer option is selected."""
    result = runner.invoke(app, ["install-deps"])
    assert result.exit_code == 0
    assert "brew install pandoc ffmpeg" in result.output
    assert "sudo apt install pandoc ffmpeg" in result.output


def test_empty_input_stops_before_model_loading(conversion, monkeypatch):
    """An empty document reports a useful error without loading a model."""
    from unittest.mock import Mock

    service, source = conversion
    source.write_text("")
    load = Mock()
    monkeypatch.setattr(service, "load_tts", load)
    result = runner.invoke(app, ["convert", str(source), "-y"])
    assert result.exit_code == 1
    assert "no narration text" in result.output
    load.assert_not_called()


def test_missing_output_directory_stops_before_extraction(conversion, monkeypatch):
    """Invalid output directories fail before document extraction."""
    from unittest.mock import Mock

    service, source = conversion
    extract = Mock()
    monkeypatch.setattr(service, "extract_sections", extract)
    result = runner.invoke(
        app,
        [
            "convert",
            str(source),
            "-o",
            str(source.parent / "absent" / "book.m4b"),
            "-y",
        ],
    )
    assert result.exit_code == 1
    assert "Output directory does not exist" in result.output
    extract.assert_not_called()


def test_cli_without_mcp_extra(tmp_path):
    """The CLI converts files without FastMCP and explains how to install it."""
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
from pathlib import Path
class BlockMCP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'fastmcp', 'anyio'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockMCP())
import numpy as np
from typer.testing import CliRunner
from simple_tts.cli import app
from simple_tts import service
assert 'fastmcp' not in sys.modules
runner = CliRunner()
assert runner.invoke(app, ['version']).exit_code == 0
source = Path(sys.argv[1]) / 'input.md'
source.write_text('Hello.')
from contextlib import contextmanager
from simple_tts import voices
@contextmanager
def stream(*args, **kwargs):
    yield iter([np.zeros(2400, dtype=np.float32)])
# Never read the real voice catalogue in a test process.
voices.load_catalog = lambda: [voices.Voice('Zoey', 'NbpkqMVS3CJeq2j8', 'en')]
service.load_tts = lambda **kwargs: object()
service.stream_speech = stream
converted = runner.invoke(app, ['convert', str(source), '-y'])
assert converted.exit_code == 0, converted.output
assert source.with_suffix('.m4a').exists()
missing = runner.invoke(app, ['mcp', '--workspace', str(source.parent)])
assert missing.exit_code == 1
assert missing.stdout == ''
assert 'pip install "simple-tts[mcp]"' in missing.stderr
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("platform", "message"),
    [
        (None, 'Install MCP support: pip install "simple-tts[mcp]"'),
        ("win32", "requires POSIX support"),
    ],
)
def test_mcp_without_extra_checks_the_platform_first(
    tmp_path, monkeypatch, platform, message
):
    """Without the MCP extra, POSIX gets the install hint, Windows a platform error."""
    import sys

    from simple_tts import jobs

    # None in sys.modules makes the import fail, as in a CLI-only install.
    monkeypatch.setitem(sys.modules, "simple_tts.mcp", None)
    if platform is not None:
        monkeypatch.setattr(jobs.sys, "platform", platform)
    result = runner.invoke(app, ["mcp", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert message in " ".join(result.stderr.split())
    assert "Traceback" not in result.stderr


def test_mcp_windows_startup_error_uses_only_stderr(tmp_path, monkeypatch):
    """Windows gets a useful MCP startup error while ordinary commands still work."""
    from simple_tts import jobs

    monkeypatch.setattr(jobs.sys, "platform", "win32")
    result = runner.invoke(app, ["mcp", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
    assert result.stdout == ""
    stderr = " ".join(result.stderr.split())
    assert "requires POSIX support" in stderr
    assert "Windows is not supported" in stderr
    assert "simple-tts convert" in stderr
    assert "Traceback" not in result.stderr
    assert runner.invoke(app, ["version"]).exit_code == 0
