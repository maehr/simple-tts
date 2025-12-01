"""Tests for the voices module."""

from simple_tts.voices import VOICES, get_voice_info, list_voices


class TestVoices:
    """Tests for VOICES constant."""

    def test_voices_is_dict(self):
        """Test that VOICES is a dictionary."""
        assert isinstance(VOICES, dict)

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


class TestListVoices:
    """Tests for list_voices function."""

    def test_returns_list(self):
        """Test that list_voices returns a list."""
        voices = list_voices()
        assert isinstance(voices, list)

    def test_list_is_sorted(self):
        """Test that voice list is sorted."""
        voices = list_voices()
        assert voices == sorted(voices)

    def test_contains_expected_voices(self):
        """Test that list contains expected voices."""
        voices = list_voices()
        assert "af_bella" in voices
        assert "bf_emma" in voices
