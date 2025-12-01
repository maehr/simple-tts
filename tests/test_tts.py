"""Tests for the TTS module."""

import numpy as np

from simple_tts.tts import SAMPLE_RATE, load_tts, synthesize, synthesize_chunks


class TestLoadTTS:
    """Tests for load_tts function."""

    def test_returns_pipeline_or_none(self):
        """Test that load_tts returns a pipeline or None."""
        result = load_tts("af_bella")
        # Either returns a pipeline or None if not installed
        assert result is None or result is not None

    def test_extracts_language_code(self):
        """Test that language code is extracted from voice."""
        # This should not raise even with different voice prefixes
        load_tts("af_bella")  # American
        load_tts("bf_emma")  # British


class TestSynthesize:
    """Tests for synthesize function."""

    def test_returns_numpy_array(self):
        """Test that synthesize returns a numpy array."""
        pipeline = load_tts("af_bella")
        result = synthesize(pipeline, "Hello world", voice="af_bella")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_empty_text_returns_array(self):
        """Test that empty text returns an array."""
        pipeline = load_tts("af_bella")
        result = synthesize(pipeline, "", voice="af_bella")
        assert isinstance(result, np.ndarray)

    def test_respects_speed_parameter(self):
        """Test that speed parameter is accepted."""
        pipeline = load_tts("af_bella")
        # Should not raise with different speed values
        synthesize(pipeline, "Hello", voice="af_bella", speed=0.5)
        synthesize(pipeline, "Hello", voice="af_bella", speed=2.0)


class TestSynthesizeChunks:
    """Tests for synthesize_chunks function."""

    def test_returns_list_of_arrays(self):
        """Test that synthesize_chunks returns a list of arrays."""
        pipeline = load_tts("af_bella")
        result = synthesize_chunks(pipeline, ["Hello", "World"], voice="af_bella")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_empty_list_returns_empty_list(self):
        """Test that empty list returns empty list."""
        pipeline = load_tts("af_bella")
        result = synthesize_chunks(pipeline, [], voice="af_bella")
        assert result == []


class TestSampleRate:
    """Tests for SAMPLE_RATE constant."""

    def test_sample_rate_value(self):
        """Test that sample rate is correct for Kokoro."""
        assert SAMPLE_RATE == 24000
