"""Tests for the language module."""

import warnings

from simple_tts.language import (
    KOKORO_SUPPORTED_LANGUAGES,
    LANGUAGE_VOICE_MAP,
    check_language_support,
    detect_language,
    get_sample_text,
    get_supported_languages,
    get_voice_for_language,
    is_language_supported,
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
        expected_langs = ["en", "fr", "es", "it"]
        for lang in expected_langs:
            assert lang in LANGUAGE_VOICE_MAP

    def test_all_voices_are_strings(self):
        """Test that all voice IDs are strings."""
        for voice in LANGUAGE_VOICE_MAP.values():
            assert isinstance(voice, str)


class TestIsLanguageSupported:
    """Tests for is_language_supported function."""

    def test_english_is_supported(self):
        """Test that English is supported."""
        assert is_language_supported("en") is True

    def test_french_is_supported(self):
        """Test that French is supported."""
        assert is_language_supported("fr") is True

    def test_german_is_not_supported(self):
        """Test that German is not supported by Kokoro."""
        assert is_language_supported("de") is False

    def test_case_insensitive(self):
        """Test that check is case insensitive."""
        assert is_language_supported("EN") is True


class TestCheckLanguageSupport:
    """Tests for check_language_support function."""

    def test_returns_supported_language(self):
        """Test that supported language is returned as-is."""
        assert check_language_support("en") == "en"
        assert check_language_support("fr") == "fr"

    def test_warns_for_unsupported_language(self):
        """Test that warning is issued for unsupported language."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_language_support("de")
            assert result == "en"  # Falls back to English
            assert len(w) == 1
            assert "not supported" in str(w[0].message)


class TestGetSupportedLanguages:
    """Tests for get_supported_languages function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_supported_languages()
        assert isinstance(result, list)

    def test_contains_expected_languages(self):
        """Test that list contains expected languages."""
        result = get_supported_languages()
        assert "en" in result
        assert "fr" in result
        assert "ja" in result

    def test_is_sorted(self):
        """Test that list is sorted."""
        result = get_supported_languages()
        assert result == sorted(result)


class TestKokoroSupportedLanguages:
    """Tests for KOKORO_SUPPORTED_LANGUAGES constant."""

    def test_is_set(self):
        """Test that constant is a set."""
        assert isinstance(KOKORO_SUPPORTED_LANGUAGES, set)

    def test_contains_kokoro_languages(self):
        """Test that set contains Kokoro TTS supported languages."""
        # Based on https://github.com/hexgrad/kokoro
        assert "en" in KOKORO_SUPPORTED_LANGUAGES
        assert "ja" in KOKORO_SUPPORTED_LANGUAGES
        assert "zh" in KOKORO_SUPPORTED_LANGUAGES
        assert "ko" in KOKORO_SUPPORTED_LANGUAGES
