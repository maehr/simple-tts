"""Narration languages and Phonon catalogue voices.

Each conversion uses one narration language. Every voice comes from the
Phonon voice catalogue and must have the same language as the narration.
Names match without case or accents. Voice IDs match exactly.

Example:
    >>> catalog = [
    ...     Voice("Nele", "2A2fFf8SkDEJ8nzx", "de"),
    ...     Voice("Anton", "3ZKKapPOvuWFcw9f", "de"),
    ...     Voice("Noémie", "FXxJ9mANRq6BCTX5", "fr"),
    ... ]
    >>> [voice.name for voice in resolve_voices((), "de", catalog)]
    ['Nele']
    >>> [voice.name for voice in resolve_voices(parse_voices("noemie"), "fr", catalog)]
    ['Noémie']
    >>> try:
    ...     resolve_voices(("Noémie",), "de", catalog)
    ... except ValueError as error:
    ...     print(str(error).split(", but")[0])
    Voice 'Noémie' is French (fr)
"""

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

# Narration languages of Phonon 40M Multilingual, one per conversion.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
}
DEFAULT_LANGUAGE = "en"
# The user confirmed these defaults on 2026-10-27. Change a default only here.
DEFAULT_VOICES: dict[str, str] = {
    "en": "Zoey",
    "de": "Nele",
    "es": "Noa",
    "fr": "Noémie",
}


@dataclass(frozen=True)
class Voice:
    """One catalogue voice and its narration language."""

    name: str
    voice_id: str
    language: str
    description: str = ""

    @property
    def is_default(self) -> bool:
        """Return whether the voice is the default of its language."""
        return DEFAULT_VOICES.get(self.language) == self.name


def check_language(value: str) -> str:
    """Return a supported narration language code in lower case.

    Example:
        >>> check_language(" DE ")
        'de'
    """
    code = value.strip().lower()
    if code not in LANGUAGES:
        raise ValueError(
            f"Unsupported language '{value}'. Use one of: {', '.join(LANGUAGES)}."
        )
    return code


def language_code(value: str) -> str:
    """Return the primary language subtag of a catalogue language or locale.

    Example:
        >>> language_code("en-US"), language_code("de_DE"), language_code(" FR ")
        ('en', 'de', 'fr')
    """
    return value.strip().replace("_", "-").split("-")[0].lower()


def normalize_name(value: str) -> str:
    """Remove surrounding space, case, and accents from a voice name.

    Example:
        >>> normalize_name(" NOÉMIE ")
        'noemie'
    """
    decomposed = unicodedata.normalize("NFKD", value.strip())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def check_voice_names(names: Sequence[str]) -> None:
    """Reject empty entries and entries with commas without the catalogue."""
    for name in names:
        if not name.strip() or "," in name:
            raise ValueError("Voice names must not be empty or contain commas.")


def parse_voices(value: str) -> tuple[str, ...]:
    """Split a comma-separated voice rotation and check only its syntax.

    Example:
        >>> parse_voices(" Nele, Anton ")
        ('Nele', 'Anton')
    """
    names = tuple(name.strip() for name in value.split(","))
    check_voice_names(names)
    return names


def load_catalog() -> list[Voice]:
    """Read the pinned Phonon voice catalogue without a model or voice weights.

    ptts reads a cached catalogue without a network request. Only the first
    use downloads the catalogue file (about 33 KB).

    Raises:
        RuntimeError: ptts is missing, or the catalogue is not available.
    """
    try:
        import ptts
    except ImportError as e:
        raise RuntimeError("ptts is missing. Run: pip install -U simple-tts") from e
    try:
        # The default catalogue is pinned. An unpinned repo sends a request per call.
        entries = ptts.catalog_voices()
    except (OSError, LookupError) as e:
        raise RuntimeError(
            f"Phonon cannot read the voice catalogue: {e} "
            "Connect to the internet once to cache the catalogue."
        ) from e
    catalog = []
    for entry in entries:
        name, voice_id = entry.get("name"), entry.get("voice_id")
        if not name or not voice_id:
            continue
        language = entry.get("language") or entry.get("locale") or ""
        description = entry.get("description") or ""
        catalog.append(Voice(name, voice_id, language_code(language), description))
    if not catalog:
        raise RuntimeError("The Phonon voice catalogue contains no voices.")
    return catalog


def available_voices(
    language: str | None = None, catalog: Sequence[Voice] | None = None
) -> list[Voice]:
    """Return the catalogue voices of the narration languages in a fixed order.

    Voices of other catalogue languages, such as Portuguese, are left out.
    """
    voices = load_catalog() if catalog is None else catalog
    codes = list(LANGUAGES) if language is None else [check_language(language)]
    return sorted(
        (voice for voice in voices if voice.language in codes),
        key=lambda voice: (codes.index(voice.language), normalize_name(voice.name)),
    )


def find_voice(value: str, language: str, catalog: Sequence[Voice]) -> Voice:
    """Find a voice by its exact voice ID, or by a name without case or accents.

    A name that occurs in more than one language selects the voice of the
    narration language.
    """
    value = value.strip()
    for voice in catalog:
        if voice.voice_id == value:
            return voice
    key = normalize_name(value)
    matches = [voice for voice in catalog if normalize_name(voice.name) == key]
    if not matches:
        raise ValueError(
            f"Unknown voice '{value}'. "
            f"List voices with: simple-tts voices --language {language}"
        )
    native = [voice for voice in matches if voice.language == language]
    return (native or matches)[0]


def resolve_voices(
    names: Sequence[str], language: str, catalog: Sequence[Voice] | None = None
) -> tuple[Voice, ...]:
    """Resolve a chapter rotation for one narration language.

    Args:
        names: Voice names or voice IDs. No names select the language default.
        language: Narration language code from ``LANGUAGES``.
        catalog: Catalogue voices. Default: the pinned Phonon catalogue.

    Returns:
        The voices in rotation order.

    Raises:
        ValueError: A voice is unknown or does not speak the narration language.
        RuntimeError: The catalogue is not available.
    """
    code = check_language(language)
    check_voice_names(names)
    voices = load_catalog() if catalog is None else catalog
    label = f"{LANGUAGES[code]} ({code})"
    if not names:
        default = DEFAULT_VOICES[code]
        if not any(
            voice.name == default and voice.language == code for voice in voices
        ):
            raise ValueError(
                f"The default {label} voice '{default}' is not in the voice "
                "catalogue. Select a voice with --voice."
            )
        names = (default,)
    resolved = []
    for name in names:
        voice = find_voice(name, code, voices)
        if voice.language != code:
            spoken = LANGUAGES.get(voice.language, "an unsupported language")
            raise ValueError(
                f"Voice '{voice.name}' is {spoken} ({voice.language or '?'}), "
                f"but the narration language is {label}. List {LANGUAGES[code]} "
                f"voices with: simple-tts voices --language {code}"
            )
        resolved.append(voice)
    return tuple(resolved)
