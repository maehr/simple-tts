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
    language_code,
    load_catalog,
    normalize_name,
    parse_voices,
    resolve_voices,
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


def test_parse_voices_trims_names():
    """Rotation entries lose surrounding space."""
    assert parse_voices(" Nele , Anton ") == ("Nele", "Anton")


@pytest.mark.parametrize("value", ["", " ", "Nele,", ",Nele", "Nele,,Anton"])
def test_parse_voices_rejects_empty_entries(value):
    """Empty rotation entries are syntax errors."""
    with pytest.raises(ValueError, match="must not be empty"):
        parse_voices(value)


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
    with pytest.raises(RuntimeError, match="ptts is missing"):
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
    (voice,) = resolve_voices((), language)
    assert (voice.name, voice.voice_id, voice.language) == (name, voice_id, language)


def test_rotation_keeps_order_and_repeats():
    """Two German voices rotate in the given order."""
    resolved = resolve_voices(("Nele", "anton", "NELE"), "de")
    assert [voice.name for voice in resolved] == ["Nele", "Anton", "Nele"]


@pytest.mark.parametrize(
    "value", ["Noémie", "noemie", "NOEMIE", " noémie ", "FXxJ9mANRq6BCTX5"]
)
def test_names_and_voice_ids_resolve(value):
    """Names ignore case and accents. Voice IDs match exactly."""
    (voice,) = resolve_voices((value,), "fr")
    assert voice.name == "Noémie"


@pytest.mark.parametrize("value", ["fxxj9manrq6bctx5", "Unknown", "af_bella"])
def test_unknown_voices_and_lowercase_ids_are_rejected(value):
    """Lowercase voice IDs fail here first, because ptts rejects them too."""
    with pytest.raises(ValueError, match=f"Unknown voice '{value}'.*--language fr"):
        resolve_voices((value,), "fr")


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
    with pytest.raises(ValueError, match=message):
        resolve_voices((value,), language)


def test_mismatch_in_a_rotation_is_rejected():
    """Every voice of a rotation must match the narration language."""
    with pytest.raises(ValueError, match="'Zoey' is English"):
        resolve_voices(("Nele", "Zoey"), "de")


def test_unknown_language_and_syntax_fail_before_the_catalogue(ptts):
    """Invalid requests do not read the catalogue."""
    with pytest.raises(ValueError, match="Unsupported language"):
        resolve_voices((), "it")
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_voices(("Nele", " "), "de")
    ptts.catalog_voices.assert_not_called()


def test_missing_default_voice_is_an_error():
    """An absent default never falls back to another voice or language."""
    catalog = [Voice("Zoey", "z", "en"), Voice("Nele", "n", "pt")]
    with pytest.raises(ValueError, match="default German \\(de\\) voice 'Nele'"):
        resolve_voices((), "de", catalog)


def test_missing_catalogue_is_an_error(ptts):
    """Resolution reports a missing catalogue, also for default voices."""
    ptts.catalog_voices.side_effect = OSError("offline")
    with pytest.raises(RuntimeError, match="voice catalogue"):
        resolve_voices((), "de")


def test_shared_name_prefers_the_narration_language():
    """A name in two languages selects the voice of the narration language."""
    catalog = [Voice("Alex", "alex-en", "en"), Voice("Alex", "alex-es", "es")]
    assert resolve_voices(("alex",), "es", catalog)[0].voice_id == "alex-es"
    assert resolve_voices(("Alex",), "en", catalog)[0].voice_id == "alex-en"
    with pytest.raises(ValueError, match="'Alex' is English"):
        resolve_voices(("Alex",), "de", catalog)


def test_locale_language_metadata_resolves(ptts):
    """A catalogue with locale codes still matches the narration language."""
    ptts.catalog_voices.return_value = [
        {"name": "Lea", "voice_id": "id-lea", "language": "de-DE"}
    ]
    assert resolve_voices(("Lea",), "de") == (Voice("Lea", "id-lea", "de"),)


def test_module_has_no_kokoro_voices():
    """The Kokoro voice table is gone."""
    assert not hasattr(voices, "VOICES")
    assert not hasattr(voices, "get_voice_info")
