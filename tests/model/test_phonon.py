"""Opt-in tests with the real Phonon 40M Multilingual model and voices.

Run: ``SIMPLE_TTS_MODEL_TESTS=1 uv run --extra mcp pytest -m model``.
The gated model and the catalogue voices must be in the Hugging Face cache.
``SIMPLE_TTS_TEST_DEVICE`` selects cpu (default, CPU/q8), mps (Metal/f32) or
cuda (CUDA/f32). The tests write audio only to pytest temporary folders.
"""

import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from simple_tts import service, tts
from simple_tts.cli import app
from simple_tts.service import (
    ConversionRequest,
    ConversionResult,
    PreparedConversion,
    execute_conversion,
    prepare_conversion,
)
from simple_tts.voices import (
    DEFAULT_VOICES,
    LANGUAGES,
    available_voices,
    load_catalog,
    resolve_voice,
)

pytestmark = pytest.mark.model

DEVICE = os.environ.get("SIMPLE_TTS_TEST_DEVICE", "cpu")
# Display name prefix and weight format that ptts reports for each test device.
BACKENDS = {"cpu": ("cpu", "q8_0"), "mps": ("Metal", "f32"), "cuda": ("CUDA", "f32")}
SAMPLE_RATE = tts.SAMPLE_RATE
FIXTURES = Path(__file__).parents[1] / "fixtures" / "narration"
DEFAULTS = {
    "en": ("Zoey", "NbpkqMVS3CJeq2j8"),
    "de": ("Nele", "2A2fFf8SkDEJ8nzx"),
    "es": ("Noa", "b6FvJAiokjdqIti4"),
    "fr": ("Noémie", "FXxJ9mANRq6BCTX5"),
}
ANTON = "3ZKKapPOvuWFcw9f"
# Voices per language in the pinned catalogue (findings, section 1).
CATALOG_COUNTS = {"en": 37, "de": 13, "es": 25, "fr": 15, "pt": 12}
# Step 01: fixture audio with 1000-character chunks on CPU/q8 (findings, section 6).
# Metal/f32 differed by less than 1 %. The smallest chunk is 4 % of a fixture.
FIXTURE_SECONDS = {"en": 186.0, "de": 172.0, "es": 178.9, "fr": 157.8}
FIXTURE_TOLERANCE = 0.03
# Step 01 measured 16.6 to 20.9 characters per second. Allow a wide margin.
CHARACTERS_PER_SECOND = (8.0, 35.0)
# Step 01: -0.3 gave 0.82 to 0.87 of the base duration and +0.3 gave 1.10 to 1.17.
FASTER_RATIO = (0.75, 0.95)
SLOWER_RATIO = (1.03, 1.30)
SENTENCES = {
    "en": "The keeper lit the lamp and watched the fog drift over the harbour.",
    "de": "Die Wärterin entzündete die Lampe und sah den Nebel über den Hafen ziehen.",
    "es": "La guardiana encendió la lámpara y miró la niebla sobre el puerto.",
    "fr": "La gardienne alluma la lampe et regarda la brume couvrir le port.",
}


@dataclass
class Call:
    """One synthesized chunk with its voice and its real output samples."""

    voice: str
    text: str
    samples: int = 0


@dataclass(frozen=True)
class Loaded:
    """A loaded model and the load messages of ``load_tts``."""

    model: object
    messages: list[str]


def load_model(language: str, duration_delta: float = 0.0) -> Loaded:
    """Load the pinned model on the test device, or skip with a clear reason."""
    if DEVICE not in BACKENDS:
        pytest.fail(f"SIMPLE_TTS_TEST_DEVICE must be cpu, mps or cuda, not {DEVICE}.")
    messages: list[str] = []
    # ptts sizes its thread pool at the first real load. Other tests used a fake.
    tts._set_threads.cache_clear()
    try:
        model = tts.load_tts(
            language=language,
            device=DEVICE,
            duration_delta=duration_delta,
            report=messages.append,
        )
    except RuntimeError as error:
        reason = str(error)
        if "unavailable in this ptts build" in reason:
            pytest.skip(f"Device '{DEVICE}' is not available: {reason}")
        if "gated model" in reason or "cannot download" in reason:
            pytest.skip(f"No access to the Phonon model: {reason}")
        raise
    return Loaded(model, messages)


@pytest.fixture(autouse=True)
def pinned_model(monkeypatch):
    """Load the pinned model, never a local SIMPLE_TTS_MODEL override."""
    monkeypatch.delenv("SIMPLE_TTS_MODEL", raising=False)


@pytest.fixture(scope="session")
def phonon() -> Callable[..., Loaded]:
    """Load the model once per language and keep it for the session.

    Other duration deltas load a new model, because the condition is a load
    option. The test releases that model.
    """
    models: dict[str, Loaded] = {}

    def load(language: str, duration_delta: float = 0.0) -> Loaded:
        if duration_delta:
            return load_model(language, duration_delta)
        if language not in models:
            models[language] = load_model(language)
        return models[language]

    return load


@pytest.fixture
def reuse_models(monkeypatch, phonon):
    """Let each conversion use the session model of its language."""

    def load(*, language, device, duration_delta, report):
        assert device == DEVICE
        return phonon(language, duration_delta).model

    monkeypatch.setattr(service, "load_tts", load)


@pytest.fixture
def calls(monkeypatch) -> list[Call]:
    """Record the voice and the real output samples of each chunk."""
    recorded: list[Call] = []

    @contextmanager
    def record(model, text, *, voice):
        call = Call(voice, text)
        recorded.append(call)
        with tts.stream_speech(model, text, voice=voice) as frames:

            def counted():
                for frame in frames:
                    call.samples += frame.size
                    yield frame

            yield counted()

    monkeypatch.setattr(service, "stream_speech", record)
    return recorded


@pytest.fixture
def tools():
    """Skip conversions when Pandoc or FFmpeg is missing."""
    missing = [
        name for name in ("pandoc", "ffmpeg", "ffprobe") if not shutil.which(name)
    ]
    if missing:
        pytest.skip(f"Missing system tools: {', '.join(missing)}")


@pytest.fixture(scope="session")
def catalog_entries(real_ptts) -> list[dict]:
    """Read the pinned voice catalogue, or skip when it is not available."""
    try:
        return real_ptts.catalog_voices()
    except (OSError, LookupError) as error:
        pytest.skip(f"The Phonon voice catalogue is not available: {error}")


@pytest.fixture
def catalog(catalog_entries):
    """Return the catalogue voices as simple-tts reads them."""
    return load_catalog()


def two_chapters(language: str, folder: Path) -> Path:
    """Write the first paragraph of the first two fixture chapters."""
    text = (FIXTURES / f"{language}.md").read_text(encoding="utf-8")
    chapters = []
    for chapter in re.split(r"^# ", text, flags=re.MULTILINE)[1:3]:
        title, body = chapter.split("\n", 1)
        paragraph = body.strip().split("\n\n")[0]
        chapters.append(f"# {title}\n\n{paragraph}\n")
    path = folder / f"{language}-two-chapters.md"
    path.write_text("\n".join(chapters), encoding="utf-8")
    return path


def convert(
    folder: Path,
    source: Path,
    *,
    language: str,
    voice: str | None = None,
    duration_delta: float = 0.0,
    format_type: str = "wav",
    name: str = "book",
) -> tuple[PreparedConversion, ConversionResult]:
    """Run a conversion and publish the file in the test folder."""
    scratch = folder / f"{name}-scratch"
    scratch.mkdir()
    request = ConversionRequest(
        input_file=source,
        language=language,
        voice=voice,
        duration_delta=duration_delta,
        output=folder / f"{name}.{format_type}",
        device=DEVICE,
    )
    prepared = prepare_conversion(request, scratch)
    staged = scratch / f"staged.{format_type}"
    result = execute_conversion(prepared, scratch, staged)
    staged.replace(prepared.output)
    return prepared, result


def chapter_samples(prepared: PreparedConversion, calls: list[Call]) -> list[int]:
    """Add the recorded samples of the chunks of each chapter."""
    assert [call.text for call in calls] == prepared.chunks
    totals, position = [], 0
    for _, chunks in prepared.sections:
        end = position + len(chunks)
        totals.append(sum(call.samples for call in calls[position:end]))
        position = end
    return totals


def check_samples(
    prepared: PreparedConversion, result: ConversionResult, calls: list[Call]
) -> None:
    """Check that the chapter bounds count the real output samples."""
    start = 0
    totals = chapter_samples(prepared, calls)
    assert [chapter.title for chapter in result.chapters] == [
        title for title, _ in prepared.sections
    ]
    for chapter, samples in zip(result.chapters, totals, strict=True):
        assert samples > 0
        assert round(chapter.start * SAMPLE_RATE) == start
        assert round(chapter.end * SAMPLE_RATE) == start + samples
        start += samples
    assert round(result.duration * SAMPLE_RATE) == start


def probe(path: Path) -> dict:
    """Read streams, chapters and duration with ffprobe."""
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-show_chapters"]
        + ["-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)


def check_markers(path: Path, result: ConversionResult) -> None:
    """Check the embedded chapter titles, bounds and file duration."""
    info = probe(path)
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert [(s["codec_name"], s["sample_rate"], s["channels"]) for s in audio] == [
        ("aac", str(SAMPLE_RATE), 1)
    ]
    markers = info["chapters"]
    assert [marker["tags"]["title"] for marker in markers] == [
        chapter.title for chapter in result.chapters
    ]
    for marker, chapter in zip(markers, result.chapters, strict=True):
        assert float(marker["start_time"]) == pytest.approx(chapter.start, abs=1e-3)
        assert float(marker["end_time"]) == pytest.approx(chapter.end, abs=1e-3)
    # AAC frames hold 1024 samples, so the container can round the duration.
    assert float(info["format"]["duration"]) == pytest.approx(result.duration, abs=0.05)


@pytest.mark.parametrize("language", list(LANGUAGES))
def test_pinned_multilingual_model_loads(phonon, language):
    """The pinned multilingual checkpoint loads on the test device."""
    loaded = phonon(language)
    model = loaded.model
    source = loaded.messages[0].removeprefix("Model: ")
    # Never the English-only Phonon 31M default model.
    assert "phonon-40m-multilingual" in source
    assert tts.MODEL_REVISION in source
    assert set(LANGUAGES) <= set(model.languages)
    assert model.sample_rate == SAMPLE_RATE
    display, quant = BACKENDS[DEVICE]
    assert model.device.startswith(display)
    assert model.quant == quant
    assert loaded.messages[1] == f"Device: {model.device}, quantization: {quant}"


def test_catalogue_lists_native_default_voices(catalog_entries):
    """The pinned catalogue holds each default voice with its language."""
    assert Counter(entry["language"] for entry in catalog_entries) == CATALOG_COUNTS
    by_id = {entry["voice_id"]: entry for entry in catalog_entries}
    assert len(by_id) == len(catalog_entries)
    for language, (name, voice_id) in DEFAULTS.items():
        assert DEFAULT_VOICES[language] == name
        entry = by_id[voice_id]
        assert (entry["name"], entry["language"]) == (name, language)
        assert entry["description"]
        voice = resolve_voice(None, language)
        assert (voice.name, voice.voice_id, voice.language) == (
            name,
            voice_id,
            language,
        )
        assert voice.is_default


@pytest.mark.parametrize("language", list(LANGUAGES))
def test_default_voice_speaks_its_language(phonon, language):
    """Each default voice synthesizes a sentence in its own language."""
    model = phonon(language).model
    name, voice_id = DEFAULTS[language]
    with tts.stream_speech(model, SENTENCES[language], voice=voice_id) as stream:
        frames = list(stream)
    assert frames
    assert all(frame.ndim == 1 and frame.dtype == np.float32 for frame in frames)
    audio = np.concatenate(frames)
    assert 1.0 < audio.size / SAMPLE_RATE < 10.0
    assert 0.05 < np.abs(audio).max() < 0.99
    assert name in model.voices


@pytest.mark.parametrize("language", list(LANGUAGES))
def test_narration_fixture_converts(reuse_models, calls, tools, tmp_path, language):
    """A three-chapter fixture gives complete 24 kHz mono audio."""
    prepared, result = convert(tmp_path, FIXTURES / f"{language}.md", language=language)
    name, voice_id = DEFAULTS[language]
    assert prepared.voice.name == name
    assert {call.voice for call in calls} == {voice_id}
    # Step 01 used 4 or 5 chunks of at most 1000 characters.
    assert len(prepared.sections) == 3
    assert len(prepared.sections) <= len(prepared.chunks) <= 6
    assert all(len(chunk) <= 1000 for chunk in prepared.chunks)
    check_samples(prepared, result, calls)

    info = sf.info(prepared.output)
    assert (info.samplerate, info.channels) == (SAMPLE_RATE, 1)
    assert info.frames == round(result.duration * SAMPLE_RATE)
    expected = FIXTURE_SECONDS[language]
    assert abs(result.duration / expected - 1) <= FIXTURE_TOLERANCE, result.duration

    # Each chunk gives speech at a plausible rate, so no chunk is cut short.
    low, high = CHARACTERS_PER_SECOND
    for call in calls:
        seconds = call.samples / SAMPLE_RATE
        assert low <= len(call.text) / seconds <= high, call.text[:60]
    audio, _ = sf.read(prepared.output, dtype="float32")
    assert np.abs(audio).max() < 0.99
    for chapter in result.chapters:
        start = round(chapter.start * SAMPLE_RATE)
        end = round(chapter.end * SAMPLE_RATE)
        assert np.sqrt(np.mean(audio[start:end] ** 2)) > 0.01, chapter


@pytest.mark.parametrize("format_type", ["m4b", "mp4"])
def test_one_voice_marks_chapters(reuse_models, calls, tools, tmp_path, format_type):
    """Anton speaks both chapters, and the markers match the samples."""
    source = two_chapters("de", tmp_path)
    prepared, result = convert(
        tmp_path, source, language="de", voice="anton", format_type=format_type
    )
    assert (prepared.voice.name, prepared.voice.voice_id) == ("Anton", ANTON)
    assert len(prepared.sections) == 2
    assert [call.voice for call in calls] == [ANTON] * len(prepared.chunks)
    check_samples(prepared, result, calls)
    check_markers(prepared.output, result)
    assert [chapter.title for chapter in result.chapters] == [
        "Der Wächter des Lichts",
        "Eine Karte der kleinen Dinge",
    ]


@pytest.mark.parametrize("language", list(LANGUAGES))
def test_duration_delta_changes_samples_and_markers(
    reuse_models, calls, tools, tmp_path, language
):
    """Two rate values change the real samples and the chapter markers.

    The test checks only the direction and a wide range, not a linear scale.
    """
    source = two_chapters(language, tmp_path)
    seconds = {}
    for name, delta in [("base", 0.0), ("faster", -0.3), ("slower", 0.3)]:
        calls.clear()
        prepared, result = convert(
            tmp_path,
            source,
            language=language,
            duration_delta=delta,
            format_type="m4b",
            name=name,
        )
        check_samples(prepared, result, calls)
        check_markers(prepared.output, result)
        seconds[name] = np.array([c.end - c.start for c in result.chapters])
    base, faster, slower = seconds["base"], seconds["faster"], seconds["slower"]
    assert np.all(faster < base)
    assert np.all(base < slower)
    assert FASTER_RATIO[0] <= faster.sum() / base.sum() <= FASTER_RATIO[1]
    assert SLOWER_RATIO[0] <= slower.sum() / base.sum() <= SLOWER_RATIO[1]


@pytest.mark.parametrize(
    ("language", "voice", "spoken"),
    [
        ("de", "Zoey", "English (en)"),
        ("de", DEFAULTS["fr"][1], "French (fr)"),
        ("en", "Nele", "German (de)"),
        ("es", "Noémie", "French (fr)"),
        ("fr", "Noa", "Spanish (es)"),
    ],
)
def test_mismatched_voice_is_rejected(catalog, language, voice, spoken):
    """The real catalogue language rejects a voice of another language."""
    label = f"{LANGUAGES[language]} ({language})"
    message = f"is {spoken}, but the narration language is {label}"
    with pytest.raises(ValueError, match=re.escape(message)):
        resolve_voice(voice, language)


def test_unsupported_catalogue_language_is_rejected(catalog_entries, catalog):
    """A Portuguese catalogue voice is never a narration voice."""
    name = next(e["name"] for e in catalog_entries if e["language"] == "pt")
    with pytest.raises(ValueError, match=r"an unsupported language \(pt\)"):
        resolve_voice(name, "es")


def test_cli_rejects_mismatch_before_synthesis(catalog, monkeypatch, tmp_path):
    """The CLI names the mismatch and loads no model."""
    load = Mock()
    monkeypatch.setattr(service, "load_tts", load)
    output = tmp_path / "book.m4a"
    result = CliRunner().invoke(
        app,
        ["convert", str(FIXTURES / "de.md"), "-l", "de", "-v", "Zoey"]
        + ["-o", str(output), "-y"],
    )
    assert result.exit_code == 1
    text = " ".join(result.output.split())
    assert "Voice 'Zoey' is English (en), but the narration language is German" in text
    assert "simple-tts voices --language de" in text
    load.assert_not_called()
    assert not output.exists()


def test_list_voices_reads_the_real_catalogue(catalog, tmp_path):
    """MCP list_voices returns the four languages and their catalogue voices."""
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from simple_tts.mcp import create_server

    async def scenario():
        async with Client(create_server(tmp_path)) as client:
            everything = await client.call_tool("list_voices")
            german = await client.call_tool("list_voices", {"language": "de"})
        return everything.structured_content, german.structured_content

    everything, german = asyncio.run(scenario())
    languages = everything["languages"]
    assert [language["code"] for language in languages] == list(LANGUAGES)
    assert {item["code"]: item["default_voice"] for item in languages} == (
        DEFAULT_VOICES
    )
    voices = everything["voices"]
    expected = {code: CATALOG_COUNTS[code] for code in LANGUAGES}
    assert Counter(voice["language"] for voice in voices) == expected
    assert [voice["name"] for voice in voices] == [
        voice.name for voice in available_voices(catalog=catalog)
    ]
    assert sorted(
        (voice["language"], voice["name"], voice["id"])
        for voice in voices
        if voice["default"]
    ) == sorted((code, name, voice_id) for code, (name, voice_id) in DEFAULTS.items())
    assert len(german["voices"]) == CATALOG_COUNTS["de"]
    assert {voice["language"] for voice in german["voices"]} == {"de"}
    assert [voice["name"] for voice in german["voices"] if voice["default"]] == ["Nele"]


def closed_port() -> int:
    """Return a local port without a listener."""
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def test_uncached_voice_fails_at_the_first_chunk(
    real_ptts, phonon, calls, tools, tmp_path, monkeypatch
):
    """An uncached voice without a network fails at the first chunk.

    ptts downloads a catalogue voice (256 KB) before its first frame. Offline
    and without that voice in the cache, the first chunk fails before any
    audio exists. The conversion publishes no file.
    """
    # Skip early when the device or the model is not available.
    phonon("de")
    snapshot = Path(tts.model_source(tts.BACKENDS[DEVICE][1]))
    revision = real_ptts.DEFAULT_VOICE_CATALOG.rsplit("@", 1)[1]
    catalogue = (
        tts.hub_cache() / "models--gradium--phonon-voices" / "snapshots" / revision
    )
    if not snapshot.is_dir() or not (catalogue / "voices.jsonl").is_file():
        pytest.skip("The model snapshot or the voice catalogue is not in the cache.")
    # The copy holds the catalogue, but no voice file.
    home = tmp_path / "hf"
    copy = home / "hub" / "models--gradium--phonon-voices" / "snapshots" / revision
    copy.mkdir(parents=True)
    shutil.copyfile(catalogue / "voices.jsonl", copy / "voices.jsonl")
    # A new HF_HOME holds no token. A closed local port stands in for the Hub.
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"] + [
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "XDG_CACHE_HOME",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(home))
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{closed_port()}")
    monkeypatch.setenv("SIMPLE_TTS_MODEL", str(snapshot))
    tts._set_threads.cache_clear()

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    request = ConversionRequest(
        input_file=two_chapters("de", tmp_path),
        language="de",
        voice="Anton",
        output=tmp_path / "book.m4b",
        device=DEVICE,
    )
    # The cached catalogue resolves the voice without a network request.
    prepared = prepare_conversion(request, scratch)
    assert prepared.voice.voice_id == ANTON
    staged = scratch / "staged.m4b"
    with pytest.raises(RuntimeError, match=rf"(?s)cannot download.*{ANTON}"):
        execute_conversion(prepared, scratch, staged)

    assert [(call.voice, call.samples) for call in calls] == [(ANTON, 0)]
    assert not (scratch / "narration.f32le").stat().st_size
    assert not staged.exists()
    assert not prepared.output.exists()
    assert not (copy / "voices" / f"{ANTON}.safetensors").exists()
