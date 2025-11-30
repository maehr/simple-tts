"""Tests for the TTS module."""

import numpy as np
import pytest

from simple_tts.tts import AudioChunk, TTSConfig, TTSEngine


class TestTTSConfig:
    """Tests for TTSConfig dataclass."""

    def test_default_values(self):
        """Test that default values are correct."""
        config = TTSConfig()
        assert config.voice == "af_bella"
        assert config.speed == 1.1
        assert config.device == "cpu"
        assert config.sample_rate == 24000

    def test_custom_values(self):
        """Test that custom values are applied."""
        config = TTSConfig(voice="bf_emma", speed=1.5, device="cuda")
        assert config.voice == "bf_emma"
        assert config.speed == 1.5
        assert config.device == "cuda"


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_duration_calculation(self):
        """Test that duration is calculated correctly."""
        data = np.zeros(24000)  # 1 second at 24kHz
        chunk = AudioChunk(data=data, sample_rate=24000)
        assert chunk.duration == 1.0

    def test_duration_empty(self):
        """Test duration for empty audio."""
        chunk = AudioChunk(data=np.array([]), sample_rate=24000)
        assert chunk.duration == 0.0

    def test_duration_different_sample_rate(self):
        """Test duration with different sample rate."""
        data = np.zeros(44100)  # 1 second at 44.1kHz
        chunk = AudioChunk(data=data, sample_rate=44100)
        assert chunk.duration == 1.0


class TestTTSEngine:
    """Tests for TTSEngine class."""

    def test_initialization_with_default_config(self):
        """Test engine initialization with default config."""
        engine = TTSEngine()
        assert engine.config.voice == "af_bella"
        assert engine.is_loaded is False

    def test_initialization_with_custom_config(self):
        """Test engine initialization with custom config."""
        config = TTSConfig(voice="bf_emma", speed=1.5)
        engine = TTSEngine(config=config)
        assert engine.config.voice == "bf_emma"
        assert engine.config.speed == 1.5

    def test_load_marks_as_loaded(self):
        """Test that load() marks the engine as loaded."""
        engine = TTSEngine()
        assert engine.is_loaded is False
        engine.load()
        assert engine.is_loaded is True

    def test_synthesize_requires_load(self):
        """Test that synthesize() requires load() first."""
        engine = TTSEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.synthesize("Hello")

    def test_synthesize_returns_audio_chunk(self):
        """Test that synthesize() returns an AudioChunk."""
        engine = TTSEngine()
        engine.load()
        chunk = engine.synthesize("Hello world")
        assert isinstance(chunk, AudioChunk)
        assert chunk.sample_rate == 24000
        assert len(chunk.data) > 0

    def test_synthesize_stream(self):
        """Test synthesizing multiple chunks."""
        engine = TTSEngine()
        engine.load()
        chunks = engine.synthesize_stream(["Hello", "World"])
        assert len(chunks) == 2
        assert all(isinstance(c, AudioChunk) for c in chunks)

    def test_set_voice(self):
        """Test setting voice."""
        engine = TTSEngine()
        engine.set_voice("bf_emma")
        assert engine.config.voice == "bf_emma"

    def test_set_speed(self):
        """Test setting speed."""
        engine = TTSEngine()
        engine.set_speed(1.5)
        assert engine.config.speed == 1.5
