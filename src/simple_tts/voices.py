"""Voice definitions for Kokoro TTS.

This module provides voice information and utilities for the available
Kokoro TTS voices.

Example:
    >>> from simple_tts.voices import VOICES, get_voice_info
    >>> get_voice_info("af_bella")
    {'language': 'English (US)', 'description': 'American female voice'}
"""

# Narration languages of Phonon 40M Multilingual, one per conversion.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
}
DEFAULT_LANGUAGE = "en"
# Catalog voices per language. Provisional until a listening review confirms them.
DEFAULT_VOICES: dict[str, str] = {
    "en": "Zoey",
    "de": "Nele",
    "es": "Noa",
    "fr": "Noémie",
}

# Available voices from Kokoro TTS
# Voice ID format: {lang_code}{gender}_{name}
# Lang codes: a=American, b=British, e=Spanish, f=French, h=Hindi,
#             i=Italian, j=Japanese, k=Korean, p=Portuguese, z=Chinese
VOICES: dict[str, dict[str, str]] = {
    # American English
    "af_bella": {"language": "English (US)", "description": "American female voice"},
    "af_nicole": {"language": "English (US)", "description": "American female voice"},
    "af_sarah": {"language": "English (US)", "description": "American female voice"},
    "af_sky": {"language": "English (US)", "description": "American female voice"},
    "am_adam": {"language": "English (US)", "description": "American male voice"},
    "am_michael": {"language": "English (US)", "description": "American male voice"},
    # British English
    "bf_emma": {"language": "English (UK)", "description": "British female voice"},
    "bf_isabella": {"language": "English (UK)", "description": "British female voice"},
    "bm_george": {"language": "English (UK)", "description": "British male voice"},
    "bm_lewis": {"language": "English (UK)", "description": "British male voice"},
    # Spanish
    "ef_dora": {"language": "Spanish", "description": "Spanish female voice"},
    "em_alex": {"language": "Spanish", "description": "Spanish male voice"},
    "em_santa": {"language": "Spanish", "description": "Spanish male voice"},
    # French
    "ff_siwis": {"language": "French", "description": "French female voice"},
    # Hindi
    "hf_alpha": {"language": "Hindi", "description": "Hindi female voice"},
    "hf_beta": {"language": "Hindi", "description": "Hindi female voice"},
    "hm_omega": {"language": "Hindi", "description": "Hindi male voice"},
    "hm_psi": {"language": "Hindi", "description": "Hindi male voice"},
    # Italian
    "if_sara": {"language": "Italian", "description": "Italian female voice"},
    "im_nicola": {"language": "Italian", "description": "Italian male voice"},
    # Japanese
    "jf_alpha": {"language": "Japanese", "description": "Japanese female voice"},
    "jf_gongitsune": {"language": "Japanese", "description": "Japanese female voice"},
    "jf_nezumi": {"language": "Japanese", "description": "Japanese female voice"},
    "jf_tebukuro": {"language": "Japanese", "description": "Japanese female voice"},
    "jm_kumo": {"language": "Japanese", "description": "Japanese male voice"},
    # Korean
    "kf_alpha": {"language": "Korean", "description": "Korean female voice"},
    "kf_beta": {"language": "Korean", "description": "Korean female voice"},
    "km_alpha": {"language": "Korean", "description": "Korean male voice"},
    "km_beta": {"language": "Korean", "description": "Korean male voice"},
    # Portuguese (Brazilian)
    "pf_dora": {"language": "Portuguese (BR)", "description": "Brazilian female voice"},
    "pm_alex": {"language": "Portuguese (BR)", "description": "Brazilian male voice"},
    "pm_santa": {"language": "Portuguese (BR)", "description": "Brazilian male voice"},
    # Chinese (Mandarin)
    "zf_xiaobei": {"language": "Chinese", "description": "Chinese female voice"},
    "zf_xiaoni": {"language": "Chinese", "description": "Chinese female voice"},
    "zf_xiaoxiao": {"language": "Chinese", "description": "Chinese female voice"},
    "zf_xiaoyi": {"language": "Chinese", "description": "Chinese female voice"},
    "zm_yunjian": {"language": "Chinese", "description": "Chinese male voice"},
    "zm_yunxi": {"language": "Chinese", "description": "Chinese male voice"},
    "zm_yunxia": {"language": "Chinese", "description": "Chinese male voice"},
    "zm_yunyang": {"language": "Chinese", "description": "Chinese male voice"},
}


def get_voice_info(voice_id: str) -> dict[str, str]:
    """Get information about a voice.

    Args:
        voice_id: The voice ID to look up.

    Returns:
        Dictionary with 'language' and 'description' keys.

    Example:
        >>> get_voice_info("af_bella")
        {'language': 'English (US)', 'description': 'American female voice'}
        >>> get_voice_info("unknown")
        {'language': 'Unknown', 'description': 'Unknown voice'}
    """
    return VOICES.get(voice_id, {"language": "Unknown", "description": "Unknown voice"})
