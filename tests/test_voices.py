"""Tests for narration languages and catalogue voice resolution."""

import sys

import pytest

from simple_tts import voices
from simple_tts.voices import (
    DEFAULT_LANGUAGE,
    DEFAULT_VOICES,
    LANGUAGES,
    Voice,
    available_voices,
    check_language,
    check_voice,
    language_code,
    load_catalog,
    normalize_name,
    resolve_voice,
)


def test_languages_and_confirmed_defaults():
    """Each narration language has one confirmed native default voice."""
    assert LANGUAGES == {
        "en": "English",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
    }
    assert DEFAULT_LANGUAGE == "en"
    assert DEFAULT_VOICES == {"en": "Zoey", "de": "Nele", "es": "Noa", "fr": "Noémie"}


@pytest.mark.parametrize(
    ("value", "expected"), [("de", "de"), (" DE ", "de"), ("Fr", "fr")]
)
def test_check_language_normalizes_case(value, expected):
    """Language codes ignore case and surrounding space."""
    assert check_language(value) == expected


@pytest.mark.parametrize("value", ["", "it", "pt", "de-DE", "english"])
def test_check_language_rejects_unsupported_codes(value):
    """Only the four narration languages are accepted."""
    with pytest.raises(ValueError, match="Use one of: en, de, es, fr"):
        check_language(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("de", "de"), ("de-DE", "de"), ("es_MX", "es"), ("EN-gb", "en"), ("", "")],
)
def test_language_code_reads_locale_variants(value, expected):
    """Catalogue locales reduce to their primary language subtag."""
    assert language_code(value) == expected


@pytest.mark.parametrize("value", ["Noémie", "noemie", "NOÉMIE", " Noémie "])
def test_normalize_name_ignores_case_and_accents(value):
    """Composed and decomposed accents give the same key."""
    assert normalize_name(value) == "noemie"


@pytest.mark.parametrize("value", ["Nele", " Anton ", "2A2fFf8SkDEJ8nzx"])
def test_check_voice_accepts_one_value(value):
    """One name or voice ID passes the syntax check."""
    check_voice(value)


@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_check_voice_rejects_an_empty_value(value):
    """An empty voice is a syntax error."""
    with pytest.raises(ValueError, match="must not be empty"):
        check_voice(value)


@pytest.mark.parametrize("value", ["Nele,Anton", " Nele, Anton ", "Nele,", ","])
def test_check_voice_rejects_a_rotation(value):
    """A comma-separated list fails, because each conversion uses one voice."""
    with pytest.raises(ValueError, match="one voice. Voice rotation is removed"):
        check_voice(value)


def test_load_catalog_reads_default_catalogue_metadata(ptts):
    """Catalogue entries become voices without a model or voice download."""
    ptts.catalog_voices.return_value = [
        {"name": "Nele", "voice_id": "2A2fFf8SkDEJ8nzx", "language": "de"},
        {"name": "Lea", "voice_id": "id-lea", "language": "de-DE", "description": None},
        {"name": "Ann", "voice_id": "id-ann", "language": None, "locale": "en_GB"},
        {"name": "Nobody", "voice_id": "id-nobody"},
        {"name": None, "voice_id": "id-none", "language": "de"},
        {"name": "No ID", "voice_id": "", "language": "de"},
    ]
    assert load_catalog() == [
        Voice("Nele", "2A2fFf8SkDEJ8nzx", "de"),
        Voice("Lea", "id-lea", "de"),
        Voice("Ann", "id-ann", "en"),
        Voice("Nobody", "id-nobody", ""),
    ]
    # No argument keeps the pinned default catalogue, which ptts caches.
    ptts.catalog_voices.assert_called_once_with()
    ptts.TTS.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [OSError("cannot fetch voices.jsonl"), LookupError("voices.jsonl is not in hf")],
)
def test_load_catalog_explains_a_missing_catalogue(ptts, error):
    """An uncached catalogue without network gives a useful error."""
    ptts.catalog_voices.side_effect = error
    with pytest.raises(RuntimeError, match="Connect to the internet once"):
        load_catalog()


def test_load_catalog_rejects_an_empty_catalogue(ptts):
    """An empty catalogue cannot supply a default voice."""
    ptts.catalog_voices.return_value = []
    with pytest.raises(RuntimeError, match="contains no voices"):
        load_catalog()


def test_load_catalog_without_ptts(monkeypatch):
    """A missing runtime gives an installation hint."""
    monkeypatch.setitem(sys.modules, "ptts", None)
    with pytest.raises(RuntimeError, match=r"^ptts is missing. Reinstall simple-tts.$"):
        load_catalog()


def test_voice_default_flag():
    """Only the default name of the same language is a default."""
    assert Voice("Nele", "id", "de").is_default
    assert not Voice("Anton", "id", "de").is_default
    assert not Voice("Nele", "id", "pt").is_default


def test_available_voices_lists_narration_languages_in_order():
    """Unsupported catalogue languages are left out."""
    listed = available_voices()
    assert [(voice.language, voice.name) for voice in listed] == [
        ("en", "Hadley"),
        ("en", "Zoey"),
        ("de", "Anton"),
        ("de", "Nele"),
        ("es", "Noa"),
        ("fr", "Noémie"),
    ]
    assert all(voice.description for voice in listed)


def test_available_voices_filters_one_language():
    """A language filter returns only its voices."""
    assert [voice.name for voice in available_voices("DE")] == ["Anton", "Nele"]
    with pytest.raises(ValueError, match="Unsupported language"):
        available_voices("pt")


@pytest.mark.parametrize(
    ("language", "name", "voice_id"),
    [
        ("en", "Zoey", "NbpkqMVS3CJeq2j8"),
        ("de", "Nele", "2A2fFf8SkDEJ8nzx"),
        ("es", "Noa", "b6FvJAiokjdqIti4"),
        ("fr", "Noémie", "FXxJ9mANRq6BCTX5"),
    ],
)
def test_default_voice_follows_the_language(language, name, voice_id):
    """No voice selects the native default voice of the language."""
    voice = resolve_voice(None, language)
    assert (voice.name, voice.voice_id, voice.language) == (name, voice_id, language)


@pytest.mark.parametrize("value", ["Anton", "anton", " ANTON ", "3ZKKapPOvuWFcw9f"])
def test_explicit_voice_replaces_the_default(value):
    """An explicit German voice replaces the default voice Nele."""
    voice = resolve_voice(value, "de")
    assert voice == Voice("Anton", "3ZKKapPOvuWFcw9f", "de", "Anton test voice.")
    assert not voice.is_default


@pytest.mark.parametrize(
    "value", ["Noémie", "noemie", "NOEMIE", " noémie ", "FXxJ9mANRq6BCTX5"]
)
def test_names_and_voice_ids_resolve(value):
    """Names ignore case and accents. Voice IDs match exactly."""
    assert resolve_voice(value, "fr").name == "Noémie"


FRENCH_HINT = (
    "List French voices with 'simple-tts voices --language fr' "
    "or the MCP tool list_voices."
)
GERMAN_HINT = (
    "List German voices with 'simple-tts voices --language de' "
    "or the MCP tool list_voices."
)


@pytest.mark.parametrize("value", ["fxxj9manrq6bctx5", "Unknown", "af_bella"])
def test_unknown_voices_and_lowercase_ids_are_rejected(value):
    """Lowercase voice IDs fail here first, because ptts rejects them too."""
    with pytest.raises(ValueError) as raised:
        resolve_voice(value, "fr")
    assert str(raised.value) == f"Unknown voice '{value}'. {FRENCH_HINT}"


@pytest.mark.parametrize(
    ("value", "language", "message"),
    [
        ("Zoey", "de", r"'Zoey' is English \(en\), .* German \(de\)"),
        ("Nele", "en", r"'Nele' is German \(de\), .* English \(en\)"),
        ("Manuela-Lu", "es", r"'Manuela-Lu' is an unsupported language \(pt\)"),
    ],
)
def test_language_mismatch_is_rejected(value, language, message):
    """A voice never narrates another language without notice."""
    with pytest.raises(ValueError, match=message) as raised:
        resolve_voice(value, language)
    # The CLI and MCP clients both get a command that lists the right voices.
    assert f"'simple-tts voices --language {language}'" in str(raised.value)
    assert str(raised.value).endswith(" or the MCP tool list_voices.")


def test_language_mismatch_message_is_complete():
    """The mismatch error names both languages and both voice lists."""
    with pytest.raises(ValueError) as raised:
        resolve_voice("Zoey", "de")
    assert str(raised.value) == (
        "Voice 'Zoey' is English (en), but the narration language is "
        f"German (de). {GERMAN_HINT}"
    )


def test_unknown_language_and_syntax_fail_before_the_catalogue(ptts):
    """Invalid requests do not read the catalogue."""
    with pytest.raises(ValueError, match="Unsupported language"):
        resolve_voice(None, "it")
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_voice(" ", "de")
    with pytest.raises(ValueError, match="Voice rotation is removed"):
        resolve_voice("Nele,Anton", "de")
    ptts.catalog_voices.assert_not_called()


def test_missing_default_voice_is_an_error():
    """An absent default never falls back to another voice or language."""
    catalog = [Voice("Zoey", "z", "en"), Voice("Nele", "n", "pt")]
    with pytest.raises(ValueError) as raised:
        resolve_voice(None, "de", catalog)
    assert str(raised.value) == (
        "The default German (de) voice 'Nele' is not in the voice catalogue. "
        f"Select another voice. {GERMAN_HINT}"
    )
    assert "--voice" not in str(raised.value)


def test_missing_catalogue_is_an_error(ptts):
    """Resolution reports a missing catalogue, also for default voices."""
    ptts.catalog_voices.side_effect = OSError("offline")
    with pytest.raises(RuntimeError, match="voice catalogue"):
        resolve_voice(None, "de")


def test_shared_name_prefers_the_narration_language():
    """A name in two languages selects the voice of the narration language."""
    catalog = [Voice("Alex", "alex-en", "en"), Voice("Alex", "alex-es", "es")]
    assert resolve_voice("alex", "es", catalog).voice_id == "alex-es"
    assert resolve_voice("Alex", "en", catalog).voice_id == "alex-en"
    with pytest.raises(ValueError, match="'Alex' is English"):
        resolve_voice("Alex", "de", catalog)


def test_locale_language_metadata_resolves(ptts):
    """A catalogue with locale codes still matches the narration language."""
    ptts.catalog_voices.return_value = [
        {"name": "Lea", "voice_id": "id-lea", "language": "de-DE"}
    ]
    assert resolve_voice("Lea", "de") == Voice("Lea", "id-lea", "de")


def test_module_has_no_kokoro_voices_or_rotation():
    """The Kokoro voice table and the rotation helpers are gone."""
    removed = ["VOICES", "get_voice_info", "parse_voices", "check_voice_names"]
    for name in [*removed, "resolve_voices"]:
        assert not hasattr(voices, name)
