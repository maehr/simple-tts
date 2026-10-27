"""Prepare and execute conversions without a user interface."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from simple_tts.audio import SUPPORTED_FORMATS, Chapter, PCMWriter, encode_pcm
from simple_tts.converter import (
    ExtractedDocument,
    extract_sections,
    split_into_chunks,
    validate_input,
)
from simple_tts.dependencies import get_missing_dependencies
from simple_tts.tts import DEVICES, SAMPLE_RATE, load_tts, stream_speech
from simple_tts.voices import DEFAULT_LANGUAGE, VOICES

Phase = Literal[
    "extracting", "loading", "synthesizing", "encoding", "publishing", "done"
]


@dataclass(frozen=True)
class ConversionRequest:
    """Source and options shared by the CLI and MCP server."""

    input_file: Path | None = None
    text: str | None = None
    voices: tuple[str, ...] = ("af_bella",)
    speed: float = 1.1
    output: Path | None = None
    format_type: str | None = None
    chapter_level: int = 1
    device: str = "auto"


@dataclass(frozen=True)
class ProgressEvent:
    """Current phase and the number of completed text chunks."""

    phase: Phase
    completed_chunks: int = 0
    total_chunks: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class ChapterResult:
    """Chapter boundaries in seconds."""

    title: str
    start: float
    end: float


@dataclass(frozen=True)
class ConversionResult:
    """Metadata for a completed conversion."""

    path: str
    uri: str
    format: str
    duration: float
    processing_time: float
    chapters: list[ChapterResult]


@dataclass(frozen=True)
class PreparedConversion:
    """Extracted text for a preview and later synthesis."""

    request: ConversionRequest
    output: Path
    format_type: str
    document: ExtractedDocument
    sections: list[tuple[str, list[str]]]
    started: float

    @property
    def chunks(self) -> list[str]:
        """Return the narration chunks in reading order."""
        return [chunk for _, chunks in self.sections for chunk in chunks]


def resolve_output(
    input_file: Path, output: Path | None, format_type: str | None
) -> tuple[Path, str]:
    """Resolve the format before extraction or synthesis starts."""
    requested = format_type.lower() if format_type is not None else None
    if requested is not None and requested not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {requested}")
    suffix = output.suffix.lstrip(".").lower() if output is not None else ""
    if suffix and suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported output suffix: .{suffix}")
    if requested and suffix and requested != suffix:
        raise ValueError(
            f"Format '{requested}' conflicts with output suffix '.{suffix}'."
        )
    selected = requested or suffix or "m4a"
    if output is None:
        output = input_file.with_suffix(f".{selected}")
    elif not suffix:
        output = output.with_suffix(f".{selected}")
    return output.absolute(), selected


def parse_voices(value: str) -> list[str]:
    """Validate a rotation that shares one Kokoro language code."""
    voices = [voice.strip() for voice in value.split(",")]
    for voice in voices:
        if voice not in VOICES:
            raise ValueError(f"Unknown voice '{voice}'. Run 'simple-tts voices'.")
    if len({voice[0] for voice in voices}) != 1:
        raise ValueError("All voices must share one Kokoro language code.")
    return voices


def validate_request(request: ConversionRequest) -> tuple[Path, str]:
    """Reject invalid options before extraction or model loading."""
    if (request.input_file is None) == (request.text is None):
        raise ValueError("Supply exactly one source: input_file or text.")
    if request.text is not None and not request.text.strip():
        raise ValueError("The document contains no narration text.")
    if not request.voices:
        raise ValueError("Select at least one voice.")
    if any(voice not in VOICES for voice in request.voices):
        raise ValueError("Unknown voice. Run 'simple-tts voices'.")
    parse_voices(",".join(request.voices))
    if not math.isfinite(request.speed) or not 0.5 <= request.speed <= 2.0:
        raise ValueError("Speed must be from 0.5 through 2.0.")
    if type(request.chapter_level) is not int or not 1 <= request.chapter_level <= 6:
        raise ValueError("Chapter level must be from 1 through 6.")
    if request.device not in DEVICES:
        raise ValueError("Device must be auto, cuda, mps, or cpu.")
    source = request.input_file or Path("text.md")
    output, format_type = resolve_output(source, request.output, request.format_type)
    if not output.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output.parent}")
    if request.input_file is not None:
        if not source.is_file():
            raise ValueError(f"Input file does not exist: {source}")
        if source.resolve() == output.resolve():
            raise ValueError("The output path must differ from the input path.")
        validate_input(source)
    if output.is_dir():
        raise ValueError(f"An output file path is a directory: {output}")
    return output, format_type


def require_dependencies() -> None:
    """Report missing system tools without printing to stdout."""
    missing = get_missing_dependencies()
    if missing:
        names = " ".join(dep.name for dep in missing)
        raise RuntimeError(f"Missing system dependencies: {names}. Install {names}.")


def _ignore_event(event: ProgressEvent) -> None:
    pass


def _no_cancellation() -> None:
    pass


def prepare_conversion(
    request: ConversionRequest,
    scratch: Path,
    *,
    sandbox: bool = False,
    report: Callable[[ProgressEvent], None] = _ignore_event,
    check_cancelled: Callable[[], None] = _no_cancellation,
) -> PreparedConversion:
    """Extract narration before the CLI requests confirmation."""
    started = time.monotonic()
    output, format_type = validate_request(request)
    check_cancelled()
    report(ProgressEvent("extracting"))
    require_dependencies()
    source = request.input_file
    if source is None:
        source = scratch / "text.md"
        source.write_text(request.text or "", encoding="utf-8")
    document = extract_sections(source, request.chapter_level, sandbox=sandbox)
    check_cancelled()
    sections = [
        (section.title, split_into_chunks(section.text))
        for section in document.sections
        if section.text.strip()
    ]
    total = sum(len(chunks) for _, chunks in sections)
    if not total:
        raise ValueError("The document contains no narration text.")
    report(ProgressEvent("extracting", total_chunks=total))
    return PreparedConversion(request, output, format_type, document, sections, started)


def execute_conversion(
    prepared: PreparedConversion,
    scratch: Path,
    staged_output: Path,
    *,
    report: Callable[[ProgressEvent], None] = _ignore_event,
    check_cancelled: Callable[[], None] = _no_cancellation,
) -> ConversionResult:
    """Write streamed PCM frames and encode once without publishing a file."""
    request = prepared.request
    total = sum(len(chunks) for _, chunks in prepared.sections)
    completed = 0
    check_cancelled()
    report(ProgressEvent("loading", total_chunks=total))
    # One model serves every chapter voice of the conversion.
    model = load_tts(
        language=DEFAULT_LANGUAGE,
        device=request.device,
        report=lambda message: report(ProgressEvent("loading", 0, total, message)),
    )
    check_cancelled()
    report(ProgressEvent("synthesizing", 0, total))
    pcm_path = scratch / "narration.f32le"
    chapters = []
    with pcm_path.open("wb") as stream:
        writer = PCMWriter(stream)
        for index, (title, chunks) in enumerate(prepared.sections):
            voice = request.voices[index % len(request.voices)]
            start = writer.samples
            for chunk in chunks:
                check_cancelled()
                with stream_speech(model, chunk, voice=voice) as frames:
                    for frame in frames:
                        writer.write(frame)
                        check_cancelled()
                completed += 1
                report(ProgressEvent("synthesizing", completed, total))
            if writer.samples > start:
                chapters.append(Chapter(title, start, writer.samples))
        total_samples = writer.samples
    if not total_samples:
        raise ValueError("Synthesis produced no audio.")
    check_cancelled()
    report(ProgressEvent("encoding", completed, total))
    encode_pcm(pcm_path, staged_output, prepared.format_type, chapters)
    check_cancelled()
    return ConversionResult(
        str(prepared.output),
        prepared.output.as_uri(),
        prepared.format_type,
        total_samples / SAMPLE_RATE,
        time.monotonic() - prepared.started,
        [
            ChapterResult(
                c.title, c.start_sample / SAMPLE_RATE, c.end_sample / SAMPLE_RATE
            )
            for c in chapters
        ],
    )
