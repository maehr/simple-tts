"""Tests for the audio module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from simple_tts.audio import (
    concatenate_audio,
    get_audio_duration,
    get_output_format,
    save_audio,
    save_audio_wav,
)


class TestConcatenateAudio:
    """Tests for concatenate_audio function."""

    def test_concatenates_arrays(self):
        """Test that arrays are concatenated correctly."""
        chunks = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        result = concatenate_audio(chunks)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])

    def test_handles_empty_list(self):
        """Test handling of empty list."""
        result = concatenate_audio([])
        assert len(result) == 0

    def test_handles_single_chunk(self):
        """Test handling of single chunk."""
        chunk = np.array([1, 2, 3])
        result = concatenate_audio([chunk])
        np.testing.assert_array_equal(result, chunk)


class TestGetOutputFormat:
    """Tests for get_output_format function."""

    def test_extracts_mp3_format(self):
        """Test extraction of mp3 format."""
        assert get_output_format("output.mp3") == "mp3"

    def test_extracts_m4a_format(self):
        """Test extraction of m4a format."""
        assert get_output_format(Path("audio.m4a")) == "m4a"

    def test_extracts_wav_format(self):
        """Test extraction of wav format."""
        assert get_output_format("test.wav") == "wav"

    def test_handles_uppercase_extension(self):
        """Test handling of uppercase extension."""
        assert get_output_format("file.MP3") == "mp3"


class TestGetAudioDuration:
    """Tests for get_audio_duration function."""

    def test_calculates_duration_correctly(self):
        """Test that duration is calculated correctly."""
        audio = np.zeros(24000)  # 1 second at 24kHz
        assert get_audio_duration(audio, 24000) == 1.0

    def test_handles_empty_audio(self):
        """Test handling of empty audio."""
        assert get_audio_duration(np.array([]), 24000) == 0.0

    def test_different_sample_rate(self):
        """Test with different sample rate."""
        audio = np.zeros(44100)  # 1 second at 44.1kHz
        assert get_audio_duration(audio, 44100) == 1.0


class TestSaveAudioWav:
    """Tests for save_audio_wav function."""

    def test_saves_wav_file(self):
        """Test that WAV file is saved correctly."""
        audio = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            save_audio_wav(audio, path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_raises_for_empty_audio(self):
        """Test that empty audio raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            with pytest.raises(ValueError, match="empty"):
                save_audio_wav(np.array([]), path)


class TestSaveAudio:
    """Tests for save_audio function."""

    def test_saves_wav_format(self):
        """Test saving WAV format."""
        audio = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            save_audio(audio, path)
            assert path.exists()

    def test_saves_mp3_format(self):
        """Test saving MP3 format (requires ffmpeg)."""
        audio = np.sin(np.linspace(0, 100, 24000)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mp3"
            save_audio(audio, path)
            assert path.exists()

    def test_accepts_string_path(self):
        """Test that string paths are accepted."""
        audio = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.wav")
            save_audio(audio, path)
            assert Path(path).exists()


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


def test_vtt_escaping_unicode_and_long_timestamps(tmp_path):
    """VTT cues retain Unicode and cannot inject markup or extra cues."""
    from simple_tts.audio import SAMPLE_RATE, Chapter, write_vtt

    late = (25 * 3600 + 61) * SAMPLE_RATE + 12000
    chapters = [
        Chapter("Über <b> & -->\n\n雪", 0, late),
        Chapter("End", late, late + 36000),
    ]
    path = tmp_path / "book.vtt"
    write_vtt(chapters, path)
    text = path.read_text(encoding="utf-8")
    assert text == (
        "WEBVTT\n\n00:00:00.000 --> 25:01:01.500\n"
        "Über &lt;b&gt; &amp; --&gt; 雪\n\n"
        "25:01:01.500 --> 25:01:03.000\nEnd\n\n"
    )
    cues = text.strip().split("\n\n")[1:]
    first_end = cues[0].splitlines()[0].split(" --> ")[1]
    second_start = cues[1].splitlines()[0].split(" --> ")[0]
    assert first_end == second_start


def test_ffmetadata_escaping(tmp_path):
    """FFmetadata escapes differ from WebVTT escaping."""
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


@pytest.mark.parametrize("format_type", ["m4a", "mp3", "ogg", "opus", "wav"])
def test_real_pcm_export_formats(tmp_path, format_type):
    """All output formats accept the raw spool file."""
    import subprocess

    from simple_tts.audio import encode_pcm

    if format_type == "ogg":
        encoders = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, check=True
        ).stdout
        if "libvorbis" not in encoders:
            pytest.skip("This FFmpeg build has no libvorbis encoder.")
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


def test_chapter_export_exact_ranges_and_names(tmp_path, monkeypatch):
    """Chapter exports copy exact samples and use safe, unique Unicode names."""
    from simple_tts import audio

    pcm = tmp_path / "audio.f32le"
    samples = np.arange(600013, dtype="<f4")
    samples.tofile(pcm)
    chapters = [
        audio.Chapter("../Same / title", 0, 300007),
        audio.Chapter("../Same / title", 300007, 600011),
        audio.Chapter("雪" * 100, 600011, 600012),
        audio.Chapter("???", 600012, 600013),
    ]
    seen = []

    def encode(source, output, format_type, local):
        expected = chapters[len(seen)]
        np.testing.assert_array_equal(
            np.fromfile(source, dtype="<f4"),
            samples[expected.start_sample : expected.end_sample],
        )
        assert local == [
            audio.Chapter(
                expected.title, 0, expected.end_sample - expected.start_sample
            )
        ]
        assert format_type == "mp4"
        seen.append(source)
        output.write_bytes(b"encoded")

    monkeypatch.setattr(audio, "encode_pcm", encode)
    directory = tmp_path / "chapters"
    audio.export_chapters(pcm, directory, "mp4", chapters)
    assert sorted(path.name for path in directory.glob("*.mp4")) == [
        "01_same-title.mp4",
        "02_same-title.mp4",
        f"03_{'雪' * 60}.mp4",
        "04_chapter.mp4",
    ]
    assert (
        "00:00:00.000 --> 00:00:12.500" in (directory / "02_same-title.vtt").read_text()
    )
    assert all(not path.parent.exists() for path in seen)


@pytest.mark.parametrize("bounds", [(-1, 1), (0, 0), (0, 3)])
def test_chapter_export_rejects_invalid_ranges(tmp_path, bounds):
    """Invalid sample bounds cannot silently produce truncated chapters."""
    from simple_tts.audio import Chapter, export_chapters

    pcm = tmp_path / "audio.f32le"
    np.zeros(2, dtype="<f4").tofile(pcm)
    with pytest.raises(ValueError, match="Chapter bounds"):
        export_chapters(pcm, tmp_path / "chapters", "wav", [Chapter("A", *bounds)])


def test_real_chapter_wav_samples(tmp_path):
    """Real chapter WAVs preserve every sample across a non-round boundary."""
    import soundfile as sf

    from simple_tts.audio import Chapter, export_chapters

    pcm = tmp_path / "audio.f32le"
    samples = np.tile(np.array([0.25, -0.5], dtype="<f4"), 24001)
    samples.tofile(pcm)
    directory = tmp_path / "chapters"
    export_chapters(
        pcm,
        directory,
        "wav",
        [Chapter("First", 0, 24001), Chapter("Last", 24001, len(samples))],
    )
    first, sample_rate = sf.read(directory / "01_first.wav")
    last, _ = sf.read(directory / "02_last.wav")
    assert sample_rate == 24000
    assert len(first) == len(last) == 24001
    np.testing.assert_array_equal(np.concatenate([first, last]), samples)
