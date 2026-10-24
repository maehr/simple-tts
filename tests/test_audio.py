"""Tests for streamed PCM encoding and embedded chapters."""

from pathlib import Path

import numpy as np
import pytest


def test_pcm_writer_counts_samples(tmp_path):
    """PCM storage uses mono little-endian float32 samples."""
    from simple_tts.audio import PCMWriter

    pcm = tmp_path / "audio.f32le"
    with pcm.open("wb") as stream:
        writer = PCMWriter(stream)
        writer.write(np.array([0.25, -0.5], dtype=np.float64))
        writer.write(np.array([1.0], dtype=np.float32))
        assert writer.samples == 3
        with pytest.raises(ValueError, match="mono"):
            writer.write(np.zeros((3, 2)))
    assert pcm.stat().st_size == 12
    np.testing.assert_array_equal(np.fromfile(pcm, dtype="<f4"), [0.25, -0.5, 1.0])


def test_ffmetadata_escaping(tmp_path):
    """FFmetadata escapes reserved characters in chapter titles."""
    from simple_tts.audio import Chapter, write_ffmetadata

    path = tmp_path / "chapters.ffmeta"
    write_ffmetadata([Chapter("A=B;C#D\\E\n雪 & <F>", 0, 24000)], path)
    assert path.read_text() == (
        ";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/24000\nSTART=0\nEND=24000\n"
        "title=A\\=B\\;C\\#D\\\\E\\\n雪 & <F>\n"
    )


@pytest.mark.parametrize("format_type", ["m4b", "mp4"])
def test_real_m4b_aac_chapters(tmp_path, format_type):
    """Real FFmpeg embeds AAC chapters with sample-derived times and titles."""
    import json
    import subprocess

    from simple_tts.audio import Chapter, encode_pcm

    pcm = tmp_path / "audio.f32le"
    np.zeros(60000, dtype="<f4").tofile(pcm)
    chapters = [
        Chapter("Über = ; # \\ & <雪>", 0, 24000),
        Chapter("Last chapter", 24000, 60000),
    ]
    output = tmp_path / f"book.{format_type}"
    encode_pcm(pcm, output, format_type, chapters)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_chapters",
            "-show_format",
            "-of",
            "json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    probe = json.loads(result.stdout)
    audio = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "audio"
    )
    assert audio["codec_name"] == "aac"
    assert int(audio["sample_rate"]) == 24000
    assert audio["channels"] == 1
    assert float(probe["format"]["duration"]) == pytest.approx(2.5, abs=0.05)
    for actual, expected in zip(probe["chapters"], chapters, strict=True):
        assert actual["tags"]["title"] == expected.title
        assert float(actual["start_time"]) == expected.start_sample / 24000
        assert float(actual["end_time"]) == expected.end_sample / 24000


@pytest.mark.parametrize("format_type", ["m4a", "mp3", "opus", "wav"])
def test_real_pcm_export_formats(tmp_path, format_type):
    """All output formats accept the raw spool file."""
    from simple_tts.audio import encode_pcm

    pcm = tmp_path / "audio.f32le"
    np.zeros(2400, dtype="<f4").tofile(pcm)
    output = tmp_path / f"audio.{format_type}"
    encode_pcm(pcm, output, format_type)
    assert output.stat().st_size > 0


def test_wav_rf64_auto(tmp_path, monkeypatch):
    """Large WAV output selects RF64 through FFmpeg's automatic size check."""
    from simple_tts import audio

    pcm = tmp_path / "large.f32le"
    # A sparse file crosses the WAV limit without allocating samples in memory.
    with pcm.open("wb") as stream:
        stream.truncate(2**33)
    commands = []
    monkeypatch.setattr(audio, "_run_ffmpeg", commands.append)
    audio.encode_pcm(pcm, tmp_path / "large.wav", "wav")
    command = commands[0]
    assert command[command.index("-rf64") + 1] == "auto"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"


def test_metadata_cleanup_on_encoding_error(tmp_path, monkeypatch):
    """An encoding failure removes its temporary chapter metadata."""
    import subprocess

    from simple_tts import audio

    pcm = tmp_path / "audio.f32le"
    np.zeros(240, dtype="<f4").tofile(pcm)
    metadata = []

    def fail(command, **kwargs):
        path = Path(command[command.index("ffmetadata") + 2])
        metadata.append(path)
        assert path.exists()
        raise subprocess.CalledProcessError(1, command, stderr="encoder error")

    monkeypatch.setattr(audio.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="encoder error"):
        audio.encode_pcm(
            pcm, tmp_path / "book.m4b", "m4b", [audio.Chapter("Title", 0, 240)]
        )
    assert not metadata[0].parent.exists()


@pytest.mark.parametrize("format_type", ["ogg", "vorbis", "flac"])
def test_unsupported_formats_fail_before_encoding(tmp_path, monkeypatch, format_type):
    """Unsupported formats cannot launch FFmpeg or create output files."""
    from unittest.mock import Mock

    from simple_tts import audio

    run = Mock()
    monkeypatch.setattr(audio.subprocess, "run", run)
    output = tmp_path / f"audio.{format_type}"
    with pytest.raises(ValueError, match="Unsupported audio format"):
        audio.encode_pcm(tmp_path / "absent.f32le", output, format_type)
    run.assert_not_called()
    assert not output.exists()


def test_empty_pcm_rejected(tmp_path):
    """An empty spool file cannot produce an output file."""
    from simple_tts.audio import encode_pcm

    pcm = tmp_path / "empty.f32le"
    pcm.touch()
    output = tmp_path / "empty.m4b"
    with pytest.raises(ValueError, match="no audio"):
        encode_pcm(pcm, output, "m4b")
    assert not output.exists()


def test_missing_ffmpeg_reports_install_hint(tmp_path, monkeypatch):
    """Missing FFmpeg reports a useful error instead of a subprocess traceback."""
    from unittest.mock import Mock

    from simple_tts import audio

    pcm = tmp_path / "audio.f32le"
    np.zeros(240, dtype="<f4").tofile(pcm)
    monkeypatch.setattr(audio.subprocess, "run", Mock(side_effect=FileNotFoundError))
    with pytest.raises(RuntimeError, match="ffmpeg not found. Install ffmpeg"):
        audio.encode_pcm(pcm, tmp_path / "audio.mp3", "mp3")
