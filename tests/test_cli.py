"""Tests for the CLI module."""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from simple_tts.cli import app, build_vtt, format_duration, format_timestamp

runner = CliRunner()


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


class TestFormatTimestamp:
    """Tests for WebVTT timestamp formatting."""

    def test_formats_milliseconds(self):
        """Test timestamp formatting to hh:mm:ss.mmm."""
        assert format_timestamp(1.234) == "00:00:01.234"
        assert format_timestamp(3661.789) == "01:01:01.789"


class TestBuildVTT:
    """Tests for building WebVTT captions."""

    def test_builds_captions(self):
        """Test WebVTT construction from chunks and durations."""
        vtt = build_vtt(["Hello world", "Next line"], [1.0, 2.5])
        assert "WEBVTT" in vtt
        assert "00:00:00.000 --> 00:00:01.000" in vtt
        assert "00:00:01.000 --> 00:00:03.500" in vtt
        assert "Hello world" in vtt and "Next line" in vtt


class TestCLIVersion:
    """Tests for the version command."""

    def test_version_command(self):
        """Test that version command outputs version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "simple-tts" in result.stdout
        assert "0.2.0" in result.stdout


class TestCLIVoices:
    """Tests for the voices command."""

    def test_voices_command(self):
        """Test that voices command lists available voices."""
        result = runner.invoke(app, ["voices"])
        assert result.exit_code == 0
        # Strip ANSI codes for checking
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        clean = ansi_escape.sub("", result.stdout)
        assert "af_bella" in clean
        assert "English" in clean


class TestCLIConvert:
    """Tests for the convert command."""

    def test_convert_missing_file(self):
        """Test convert with non-existent file."""
        result = runner.invoke(app, ["convert", "/nonexistent/file.pdf"])
        assert result.exit_code != 0

    def test_convert_shows_help(self):
        """Test that convert shows help."""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Input file" in result.stdout

    def test_convert_has_voice_option(self):
        """Test that convert has --voice option."""
        result = runner.invoke(app, ["convert", "--help"])
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        clean = ansi_escape.sub("", result.stdout)
        assert "--voice" in clean

    def test_convert_has_yes_option(self):
        """Test that convert has -y/--yes option."""
        result = runner.invoke(app, ["convert", "--help"])
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        clean = ansi_escape.sub("", result.stdout)
        assert "--yes" in clean

    def test_invalid_voice_rejected(self):
        """Test that invalid voice is rejected."""
        result = runner.invoke(app, ["convert", "test.txt", "--voice", "invalid_voice"])
        assert result.exit_code != 0


class TestCLICheckDeps:
    """Tests for the check-deps command."""

    def test_check_deps_command(self):
        """Test that check-deps command works."""
        result = runner.invoke(app, ["check-deps"])
        assert result.exit_code == 0
        assert "Dependencies" in result.stdout or "pdftotext" in result.stdout


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    """Prepare a deterministic conversion without loading a real model."""
    from simple_tts import cli

    source = tmp_path / "book.md"
    source.write_text("Preamble.\n\n## First\n\nBody.\n\n## Last\n\nEnd.")
    monkeypatch.setattr(cli, "check_dependencies", lambda *args: True)
    monkeypatch.setattr(cli, "load_tts", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli, "export_chapters", lambda pcm, directory, *args: directory.mkdir()
    )
    return cli, source


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

    cli, source = conversion
    calls = []

    def encode(pcm, output, format_type, chapters):
        calls.append(format_type)
        assert not (source.parent / output_name).exists()
        output.write_bytes(b"encoded")

    monkeypatch.setattr(cli, "synthesize", lambda *args, **kwargs: np.zeros(240))
    monkeypatch.setattr(cli, "encode_pcm", encode)
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
        ["--voice", "af_bella,"],
        ["--voice", "af_bella,unknown"],
        ["--voice", "af_bella,bf_emma"],
        ["--vtt"],
        ["--no-vtt"],
        ["--split"],
    ],
)
def test_rejected_options_precede_synthesis(conversion, monkeypatch, arguments):
    """Invalid options never start extraction or synthesis."""
    from unittest.mock import Mock

    cli, source = conversion
    extract = Mock(side_effect=AssertionError("Must not extract"))
    load = Mock(side_effect=AssertionError("Must not load"))
    monkeypatch.setattr(cli, "extract_sections", extract)
    monkeypatch.setattr(cli, "load_tts", load)
    result = runner.invoke(app, ["convert", str(source), "-y", *arguments])
    assert result.exit_code != 0
    extract.assert_not_called()
    load.assert_not_called()


@pytest.mark.parametrize(
    "format_type", ["m4a", "m4b", "mp4", "mp3", "wav", "ogg", "opus"]
)
def test_sample_boundaries_and_vtt(conversion, monkeypatch, format_type):
    """Each chapter begins before its heading and ends at its final sample."""
    import numpy as np

    from simple_tts.audio import Chapter

    cli, source = conversion
    lengths = iter([24000, 36000, 12000])
    spoken = []
    pcm_paths = []
    chapters_seen = []

    def synthesize(pipeline, text, **kwargs):
        spoken.append(text)
        return np.full(next(lengths), len(spoken), dtype=np.float32)

    def encode(pcm, output, format_type, chapters):
        pcm_paths.append(pcm)
        samples = np.fromfile(pcm, dtype="<f4")
        assert samples.tolist() == [1] * 24000 + [2] * 36000 + [3] * 12000
        chapters_seen.extend(chapters)
        assert not source.with_suffix(".vtt").exists()
        output.write_bytes(b"encoded")

    monkeypatch.setattr(cli, "synthesize", synthesize)
    monkeypatch.setattr(cli, "encode_pcm", encode)
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
    assert source.with_suffix(".vtt").read_text() == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nIntroduction\n\n"
        "00:00:01.000 --> 00:00:02.500\nFirst\n\n"
        "00:00:02.500 --> 00:00:03.000\nLast\n\n"
    )
    assert all(not path.parent.exists() for path in pcm_paths)
    assert not list(source.parent.glob(".simple-tts-*"))


@pytest.mark.parametrize("arguments", [[], ["--format", "m4b"], ["--format", "mp4"]])
def test_missing_headings_warning(conversion, monkeypatch, arguments):
    """Every conversion reports absent heading matches."""
    import numpy as np

    cli, source = conversion
    source.write_text("## Different level\n\nBody.")
    titles = []

    def encode(pcm, output, format_type, chapters):
        titles.extend(chapter.title for chapter in chapters)
        output.write_bytes(b"encoded")

    monkeypatch.setattr(cli, "synthesize", lambda *args, **kwargs: np.zeros(240))
    monkeypatch.setattr(cli, "encode_pcm", encode)
    result = runner.invoke(app, ["convert", str(source), "-y", *arguments])
    assert result.exit_code == 0, result.output
    assert "No headings match" in result.output
    assert titles == ["book"]


def test_long_conversion_releases_audio_arrays(conversion, monkeypatch):
    """Hundreds of chunks never retain earlier audio arrays."""
    import weakref

    import numpy as np

    from simple_tts.converter import ExtractedDocument, Section

    cli, source = conversion
    references = []
    chunk_count = 300
    monkeypatch.setattr(
        cli,
        "extract_sections",
        lambda *args: ExtractedDocument([Section("Long", "A" * 1000 * chunk_count)], 1),
    )

    def synthesize(*args, **kwargs):
        assert all(reference() is None for reference in references)
        audio = np.zeros(24000, dtype=np.float32)
        references.append(weakref.ref(audio))
        return audio

    def encode(pcm, output, format_type, chapters):
        assert all(reference() is None for reference in references)
        assert pcm.stat().st_size == chunk_count * 24000 * 4
        assert chapters[-1].end_sample == chunk_count * 24000
        output.write_bytes(b"encoded")

    monkeypatch.setattr(cli, "synthesize", synthesize)
    monkeypatch.setattr(cli, "encode_pcm", encode)
    result = runner.invoke(app, ["convert", str(source), "-y"])
    assert result.exit_code == 0, result.output
    assert len(references) == chunk_count


@pytest.mark.parametrize("failure_stage", ["synthesis", "encoding", "cancellation"])
def test_failure_cleanup_and_publication(conversion, monkeypatch, failure_stage):
    """Errors and cancellation remove temporary files and preserve old outputs."""
    import numpy as np

    cli, source = conversion
    output = source.with_suffix(".m4b")
    sidecar = source.with_suffix(".vtt")
    output.write_bytes(b"old audio")
    sidecar.write_text("old navigation")
    directories = []
    temporary_directory = cli.tempfile.TemporaryDirectory

    def track_directory(*args, **kwargs):
        directory = temporary_directory(*args, **kwargs)
        directories.append(Path(directory.name))
        return directory

    calls = 0

    def synthesize(*args, **kwargs):
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

    monkeypatch.setattr(cli.tempfile, "TemporaryDirectory", track_directory)
    monkeypatch.setattr(cli, "synthesize", synthesize)
    monkeypatch.setattr(cli, "encode_pcm", encode)
    result = runner.invoke(app, ["convert", str(source), "--format", "m4b", "-y"])
    assert result.exit_code != 0
    assert output.read_bytes() == b"old audio"
    assert sidecar.read_text() == "old navigation"
    assert directories and all(not directory.exists() for directory in directories)


def test_no_audio_is_an_error(conversion, monkeypatch):
    """Empty synthesis does not publish a silent placeholder file."""
    from unittest.mock import Mock

    import numpy as np

    cli, source = conversion
    monkeypatch.setattr(cli, "synthesize", lambda *args, **kwargs: np.array([]))
    encode = Mock()
    monkeypatch.setattr(cli, "encode_pcm", encode)
    result = runner.invoke(app, ["convert", str(source), "-y"])
    assert result.exit_code == 1
    assert "Synthesis produced no audio" in result.output
    assert not source.with_suffix(".m4a").exists()
    assert not source.with_suffix(".vtt").exists()
    encode.assert_not_called()


def test_cli_gpu_retry_has_no_duplicate_samples(tmp_path, monkeypatch):
    """A real CLI spool contains only the completed CPU retry."""
    from unittest.mock import Mock

    import numpy as np

    from simple_tts import cli, tts

    source = tmp_path / "book.md"
    source.write_text("# First\n\nBody.\n\n# Last\n\nEnd.")

    def gpu(text, **kwargs):
        yield text, text, np.ones(77, dtype=np.float32)
        raise RuntimeError("GPU failure")

    retried_voices = []

    def cpu(text, **kwargs):
        retried_voices.append(kwargs["voice"])
        yield text, text, np.full(23, 2, dtype=np.float32)

    engine = tts.TTSPipeline(gpu, "af_bella", "auto", "mps", print)
    factory = Mock(return_value=cpu)
    monkeypatch.setattr(tts, "_create_pipeline", factory)
    monkeypatch.setattr(cli, "load_tts", lambda *args, **kwargs: engine)

    def encode(pcm, output, *args):
        assert np.fromfile(pcm, dtype="<f4").tolist() == [2] * 46
        output.write_bytes(b"encoded")

    monkeypatch.setattr(cli, "encode_pcm", encode)
    result = runner.invoke(
        app, ["convert", str(source), "--voice", "af_bella,am_adam", "-y"]
    )
    assert result.exit_code == 0, result.output
    assert factory.call_count == 1
    assert retried_voices == ["af_bella", "am_adam"]


@pytest.mark.parametrize("format_type", ["m4b", "mp4"])
def test_real_cli_m4b_and_vtt(tmp_path, format_type):
    """The CLI sends matching chapter titles and times to FFmpeg and WebVTT."""
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
    assert output.with_suffix(".vtt").read_text() == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:00.100\nÜber &amp; &lt;雪&gt;\n\n"
        "00:00:00.100 --> 00:00:00.200\nLast\n\n"
    )
    directory = tmp_path / "book_chapters"
    assert len(list(directory.glob(f"*.{format_type}"))) == 2
    assert len(list(directory.glob("*.vtt"))) == 2
    for path in sorted(directory.glob(f"*.{format_type}")):
        chapter_probe = json.loads(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(path)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        assert chapter_probe["chapters"][0]["start_time"] == "0.000000"
        assert chapter_probe["chapters"][0]["end_time"] == "0.100000"
        assert "00:00:00.000 --> 00:00:00.100" in path.with_suffix(".vtt").read_text()


def test_voice_rotation_reuses_one_pipeline(conversion, monkeypatch):
    """Every section retains one voice across chunks and cycles in order."""
    import numpy as np

    from simple_tts.converter import ExtractedDocument, Section

    cli, source = conversion
    document = ExtractedDocument(
        [Section(title, "A" * 2000) for title in ["Introduction", "First", "Last"]], 2
    )
    monkeypatch.setattr(cli, "extract_sections", lambda *args: document)
    loaded = []
    spoken = []
    engine = object()

    def load(voice, **kwargs):
        loaded.append(voice)
        return engine

    def synthesize(pipeline, text, voice, speed):
        assert pipeline is engine
        spoken.append(voice)
        return np.zeros(240, dtype=np.float32)

    monkeypatch.setattr(cli, "load_tts", load)
    monkeypatch.setattr(cli, "synthesize", synthesize)
    monkeypatch.setattr(cli, "encode_pcm", lambda pcm, output, *args: output.touch())
    result = runner.invoke(
        app, ["convert", str(source), "--voice", " af_bella, am_adam ", "-y"]
    )
    assert result.exit_code == 0, result.output
    assert loaded == ["af_bella"]
    assert spoken == ["af_bella"] * 2 + ["am_adam"] * 2 + ["af_bella"] * 2


def test_all_outputs_replace_previous_conversion(conversion, monkeypatch):
    """Default conversion replaces the full generated set without stale tracks."""
    import numpy as np

    from simple_tts import audio

    cli, source = conversion
    source.write_text("# First\n\nBody.\n\n# Last\n\nEnd.")
    spoken = []

    def synthesize(*args, **kwargs):
        spoken.append(kwargs["voice"])
        return np.zeros(2400, dtype=np.float32)

    def encode(pcm, output, *args):
        output.write_bytes(b"encoded")

    monkeypatch.setattr(cli, "synthesize", synthesize)
    monkeypatch.setattr(cli, "encode_pcm", encode)
    monkeypatch.setattr(audio, "encode_pcm", encode)
    monkeypatch.setattr(cli, "export_chapters", audio.export_chapters)
    arguments = ["convert", str(source), "-y"]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    directory = source.parent / "book_chapters"
    assert {path.name for path in directory.iterdir()} == {
        "01_first.m4a",
        "01_first.vtt",
        "02_last.m4a",
        "02_last.vtt",
    }
    assert len(spoken) == 2

    source.write_text("No headings now.")
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert {path.name for path in directory.iterdir()} == {"01_book.m4a", "01_book.vtt"}
    assert source.with_suffix(".m4a").read_bytes() == b"encoded"
    assert "00:00:00.000 --> 00:00:00.100" in source.with_suffix(".vtt").read_text()
    assert len(spoken) == 3
    assert not list(source.parent.glob(".simple-tts-*"))


@pytest.mark.parametrize("cancel", [False, True])
def test_chapter_encoding_failure_preserves_outputs(conversion, monkeypatch, cancel):
    """Late export failures and cancellation preserve every previous output."""
    import numpy as np

    cli, source = conversion
    output = source.with_suffix(".m4a")
    sidecar = source.with_suffix(".vtt")
    directory = source.parent / "book_chapters"
    output.write_bytes(b"old audio")
    sidecar.write_text("old VTT")
    directory.mkdir()
    (directory / "old.m4a").write_bytes(b"old track")

    def fail(pcm, staging, *args):
        staging.mkdir()
        (staging / "partial.m4a").touch()
        if cancel:
            raise KeyboardInterrupt
        raise RuntimeError("chapter encoder failed")

    monkeypatch.setattr(cli, "synthesize", lambda *args, **kwargs: np.zeros(240))
    monkeypatch.setattr(cli, "encode_pcm", lambda pcm, path, *args: path.touch())
    monkeypatch.setattr(cli, "export_chapters", fail)
    result = runner.invoke(app, ["convert", str(source), "-y"])
    assert result.exit_code == (130 if cancel else 1)
    assert output.read_bytes() == b"old audio"
    assert sidecar.read_text() == "old VTT"
    assert [path.name for path in directory.iterdir()] == ["old.m4a"]
    assert (directory / "old.m4a").read_bytes() == b"old track"
    assert not list(source.parent.glob(".simple-tts-*"))


@pytest.mark.parametrize("failure_step", range(6))
def test_publication_failure_restores_files_and_directory(
    tmp_path, monkeypatch, failure_step
):
    """A failure at any backup or publication step restores previous outputs."""
    from simple_tts.cli import publish_outputs

    staging = tmp_path / "staging"
    staging.mkdir()
    outputs = []
    for name in ["book.m4a", "book.vtt", "book_chapters"]:
        source = staging / name
        destination = tmp_path / name
        if name == "book_chapters":
            source.mkdir()
            destination.mkdir()
            (source / "new").write_text("new")
            (destination / "old").write_text("old")
        else:
            source.write_text("new")
            destination.write_text("old")
        outputs.append((source, destination))
    replace = Path.replace
    calls = 0

    def fail_once(source, destination):
        nonlocal calls
        step = calls
        calls += 1
        if step == failure_step:
            raise OSError("publication failed")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(OSError, match="publication failed"):
        publish_outputs(outputs)
    assert (tmp_path / "book.m4a").read_text() == "old"
    assert (tmp_path / "book.vtt").read_text() == "old"
    assert [path.name for path in (tmp_path / "book_chapters").iterdir()] == ["old"]
    assert not list(tmp_path.glob(".simple-tts-backup-*"))


def test_failed_restoration_preserves_recovery_files(tmp_path, monkeypatch):
    """A second filesystem failure leaves old output recoverable outside staging."""
    from simple_tts.cli import publish_outputs

    staging = tmp_path / "staging"
    staging.mkdir()
    source = staging / "book.m4a"
    source.write_text("new")
    destination = tmp_path / "book.m4a"
    destination.write_text("old")
    replace = Path.replace

    def fail_after_backup(path, target):
        if path == destination:
            return replace(path, target)
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(Path, "replace", fail_after_backup)
    with pytest.raises(RuntimeError, match="Recover previous files"):
        publish_outputs([(source, destination)])
    backups = list(tmp_path.glob(".simple-tts-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "0").read_text() == "old"


@pytest.mark.parametrize("collision", ["master", "vtt", "chapter-file", "input"])
def test_output_collisions_precede_synthesis(conversion, monkeypatch, collision):
    """Output paths cannot replace the input or use incompatible file types."""
    from unittest.mock import Mock

    cli, source = conversion
    output = source.with_suffix(".m4a")
    if collision == "master":
        output.mkdir()
    elif collision == "vtt":
        output.with_suffix(".vtt").mkdir()
    elif collision == "chapter-file":
        (source.parent / "book_chapters").touch()
    else:
        directory = source.parent / "book_chapters"
        directory.mkdir()
        source = source.replace(directory / source.name)
    load = Mock()
    monkeypatch.setattr(cli, "load_tts", load)
    result = runner.invoke(app, ["convert", str(source), "-o", str(output), "-y"])
    assert result.exit_code == 1
    load.assert_not_called()
    assert source.is_file()


def test_confirmation_lists_each_narrator(conversion, monkeypatch):
    """The checkpoint identifies every narrator before conversion starts."""
    import numpy as np

    cli, source = conversion
    monkeypatch.setattr(cli, "synthesize", lambda *args, **kwargs: np.zeros(240))
    monkeypatch.setattr(cli, "encode_pcm", lambda pcm, path, *args: path.touch())
    result = runner.invoke(
        app, ["convert", str(source), "--voice", "af_bella,am_adam"], input="\n"
    )
    assert result.exit_code == 0, result.output
    assert "af_bella (American female voice)" in result.output
    assert "am_adam (American male voice)" in result.output
