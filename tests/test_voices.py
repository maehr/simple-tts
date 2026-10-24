"""Tests for the voices module."""

from simple_tts.voices import VOICES, get_voice_info


class TestVoices:
    """Tests for VOICES constant."""

    def test_contains_expected_voices(self):
        """Test that VOICES contains expected voice IDs."""
        assert "af_bella" in VOICES
        assert "bf_emma" in VOICES
        assert "jf_alpha" in VOICES

    def test_voice_info_has_required_keys(self):
        """Test that voice info has required keys."""
        for voice_id, info in VOICES.items():
            assert "language" in info
            assert "description" in info


class TestGetVoiceInfo:
    """Tests for get_voice_info function."""

    def test_returns_info_for_known_voice(self):
        """Test that info is returned for known voice."""
        info = get_voice_info("af_bella")
        assert info["language"] == "English (US)"
        assert "American" in info["description"]

    def test_returns_unknown_for_unknown_voice(self):
        """Test that Unknown is returned for unknown voice."""
        info = get_voice_info("xyz_unknown")
        assert info["language"] == "Unknown"
        assert info["description"] == "Unknown voice"
