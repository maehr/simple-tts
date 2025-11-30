"""Language detection utilities for identifying text language.

This module provides functions for detecting the language of text,
which is used to select the appropriate TTS voice.

Examples:
    >>> from simple_tts.language import get_sample_text
    >>> text = "Hello world, this is a test."
    >>> sample = get_sample_text(text, max_length=10)
    >>> len(sample) <= 10
    True
"""

# Language to voice mapping for Kokoro TTS
LANGUAGE_VOICE_MAP: dict[str, str] = {
    "en": "af_bella",  # American English
    "en-us": "af_bella",  # American English
    "en-gb": "bf_emma",  # British English
    "es": "ef_dora",  # Spanish
    "fr": "ff_siwis",  # French
    "de": "gf_eva",  # German (fallback)
    "it": "if_lisa",  # Italian (fallback)
    "pt": "pf_maria",  # Portuguese (fallback)
    "ja": "jf_nanami",  # Japanese (fallback)
    "zh": "zf_ming",  # Chinese (fallback)
    "ko": "kf_jihye",  # Korean (fallback)
}

DEFAULT_VOICE = "af_bella"


def get_sample_text(text: str, max_length: int = 500) -> str:
    """Extract a sample from the middle of text for language detection.

    Takes a sample from the middle of the text to avoid headers/footers
    that might not be representative of the main content.

    Args:
        text: The full text to sample from.
        max_length: Maximum length of the sample.

    Returns:
        A sample of text from the middle of the input.

    Examples:
        >>> get_sample_text("Hello world", max_length=5)
        'lo wo'
        >>> get_sample_text("Short", max_length=100)
        'Short'
        >>> get_sample_text("ABCDEFGHIJ", max_length=4)
        'DEFG'
    """
    if len(text) <= max_length:
        return text

    # Get sample from the middle
    start = (len(text) - max_length) // 2
    return text[start : start + max_length]


def detect_language(text: str) -> str:
    """Detect the language of the given text.

    Uses simple heuristics for language detection. For production use,
    consider using fasttext or langdetect.

    Args:
        text: The text to analyze.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'fr', 'es').

    Examples:
        >>> detect_language("Hello world, this is English text.")
        'en'
        >>> detect_language("Bonjour le monde, ceci est du texte français.")
        'fr'
        >>> detect_language("")
        'en'
    """
    if not text.strip():
        return "en"

    sample = get_sample_text(text.lower())

    # Simple language detection based on common words
    # This is a basic implementation; for production, use fasttext
    lang_indicators = {
        "en": ["the", "is", "are", "and", "of", "to", "in", "it", "you", "that"],
        "fr": ["le", "la", "les", "est", "sont", "et", "de", "un", "une", "que"],
        "es": ["el", "la", "los", "es", "son", "y", "de", "un", "una", "que"],
        "de": ["der", "die", "das", "ist", "sind", "und", "ein", "eine", "zu", "den"],
        "it": ["il", "la", "gli", "è", "sono", "e", "di", "un", "una", "che"],
        "pt": ["o", "a", "os", "é", "são", "e", "de", "um", "uma", "que"],
    }

    scores: dict[str, int] = {}
    words = sample.split()

    for lang, indicators in lang_indicators.items():
        score = sum(1 for word in words if word in indicators)
        scores[lang] = score

    # Return the language with the highest score, default to 'en'
    if scores:
        best_lang = max(scores, key=lambda k: scores[k])
        if scores[best_lang] > 0:
            return best_lang

    return "en"


def get_voice_for_language(language: str) -> str:
    """Get the recommended voice for a language.

    Maps ISO 639-1 language codes to Kokoro TTS voice IDs.

    Args:
        language: ISO 639-1 language code (e.g., 'en', 'fr').

    Returns:
        Voice ID for the given language.

    Examples:
        >>> get_voice_for_language("en")
        'af_bella'
        >>> get_voice_for_language("fr")
        'ff_siwis'
        >>> get_voice_for_language("unknown")
        'af_bella'
    """
    return LANGUAGE_VOICE_MAP.get(language.lower(), DEFAULT_VOICE)
