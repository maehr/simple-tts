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
