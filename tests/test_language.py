"""Tests for the language module."""

from simple_tts.language import (
    LANGUAGE_VOICE_MAP,
    detect_language,
    get_sample_text,
    get_voice_for_language,
)


class TestGetSampleText:
    """Tests for get_sample_text function."""

    def test_returns_full_text_when_short(self):
        """Test that short text is returned in full."""
        text = "Hello world"
        assert get_sample_text(text, max_length=100) == text

    def test_extracts_from_middle(self):
        """Test that sample is extracted from the middle."""
        text = "ABCDEFGHIJ"
        sample = get_sample_text(text, max_length=4)
        # (10 - 4) // 2 = 3, so start at index 3: "DEFG"
        assert sample == "DEFG"

    def test_respects_max_length(self):
        """Test that sample respects max_length."""
        text = "A" * 1000
        sample = get_sample_text(text, max_length=100)
        assert len(sample) == 100


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detects_english(self):
        """Test detection of English text."""
        text = "The quick brown fox jumps over the lazy dog."
        assert detect_language(text) == "en"

    def test_detects_french(self):
        """Test detection of French text."""
        text = "Le renard brun rapide saute par-dessus le chien paresseux."
        assert detect_language(text) == "fr"

    def test_detects_spanish(self):
        """Test detection of Spanish text."""
        text = "El zorro marrón rápido salta sobre el perro perezoso."
        assert detect_language(text) == "es"

    def test_defaults_to_english_for_empty(self):
        """Test that empty text defaults to English."""
        assert detect_language("") == "en"

    def test_defaults_to_english_for_unknown(self):
        """Test that unknown text defaults to English."""
        # Numbers only, no language indicators
        assert detect_language("12345 67890") == "en"


class TestGetVoiceForLanguage:
    """Tests for get_voice_for_language function."""

    def test_returns_correct_voice_for_english(self):
        """Test that English returns the correct voice."""
        assert get_voice_for_language("en") == "af_bella"

    def test_returns_correct_voice_for_french(self):
        """Test that French returns the correct voice."""
        assert get_voice_for_language("fr") == "ff_siwis"

    def test_returns_default_for_unknown(self):
        """Test that unknown language returns default voice."""
        assert get_voice_for_language("xyz") == "af_bella"

    def test_is_case_insensitive(self):
        """Test that language code is case insensitive."""
        assert get_voice_for_language("EN") == get_voice_for_language("en")


class TestLanguageVoiceMap:
    """Tests for LANGUAGE_VOICE_MAP constant."""

    def test_contains_expected_languages(self):
        """Test that map contains expected languages."""
        expected_langs = ["en", "fr", "es", "de"]
        for lang in expected_langs:
            assert lang in LANGUAGE_VOICE_MAP

    def test_all_voices_are_strings(self):
        """Test that all voice IDs are strings."""
        for voice in LANGUAGE_VOICE_MAP.values():
            assert isinstance(voice, str)
