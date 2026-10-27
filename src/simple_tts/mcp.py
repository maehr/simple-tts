"""Optional local MCP tools for asynchronous conversion jobs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from simple_tts import __version__
from simple_tts.dependencies import DEPENDENCIES, is_command_available
from simple_tts.jobs import JobManager, JobStatus, require_mcp_platform
from simple_tts.service import ConversionRequest
from simple_tts.tts import DURATION_DELTA_RANGE
from simple_tts.voices import DEFAULT_VOICES, LANGUAGES, available_voices

if TYPE_CHECKING:
    from fastmcp import FastMCP

Language = Literal["en", "de", "es", "fr"]


@dataclass(frozen=True)
class LanguageDescription:
    """A narration language and its default voice."""

    code: str
    name: str
    default_voice: str


@dataclass(frozen=True)
class VoiceDescription:
    """A Phonon catalogue voice."""

    name: str
    id: str
    language: str
    description: str
    default: bool


@dataclass(frozen=True)
class VoiceCatalog:
    """Structured voice discovery output."""

    languages: list[LanguageDescription]
    voices: list[VoiceDescription]


def create_server(workspace: Path, *, manager: JobManager | None = None) -> FastMCP:
    """Build a stdio server without loading a speech model."""
    require_mcp_platform()

    import anyio
    import anyio.to_thread
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError

    jobs = manager or JobManager(workspace)

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[None]:
        try:
            yield
        finally:
            with anyio.CancelScope(shield=True):
                await jobs.close()

    server = FastMCP(
        "simple-tts",
        version=__version__,
        instructions=(
            "Convert prepared documents or Markdown text to local audio. "
            "Start one job, poll get_conversion, and cancel it when needed. "
            "Select one narration language per job. Voices must match it. "
            "Paths are relative to the configured workspace. "
            "File URIs require access to the local filesystem."
        ),
        lifespan=lifespan,
        strict_input_validation=True,
        tasks=False,
    )

    @server.tool(annotations={"readOnlyHint": True})
    async def list_voices(language: Language | None = None) -> VoiceCatalog:
        """Return narration languages and catalogue voices with descriptions.

        Read only the voice catalogue metadata. Load no model or voice weights.
        """
        try:
            # The first use can download the catalogue, so keep the loop free.
            voices = await anyio.to_thread.run_sync(available_voices, language)
        except (ValueError, RuntimeError) as error:
            raise ToolError(str(error)) from error
        return VoiceCatalog(
            [
                LanguageDescription(code, name, DEFAULT_VOICES[code])
                for code, name in LANGUAGES.items()
            ],
            [
                VoiceDescription(
                    voice.name,
                    voice.voice_id,
                    voice.language,
                    voice.description,
                    voice.is_default,
                )
                for voice in voices
            ],
        )

    @server.tool(annotations={"readOnlyHint": True})
    async def check_dependencies() -> dict[str, bool]:
        """Return Pandoc and FFmpeg availability without loading a model."""
        return {dep.name: is_command_available(dep.name) for dep in DEPENDENCIES}

    @server.tool
    async def start_conversion(
        input_file: str | None = None,
        text: str | None = None,
        language: Language = "en",
        voices: list[str] = [],
        duration_delta: Annotated[
            float, Field(ge=DURATION_DELTA_RANGE[0], le=DURATION_DELTA_RANGE[1])
        ] = 0.0,
        format: str | None = None,
        output: str | None = None,
        chapter_level: Annotated[int, Field(ge=1, le=6)] = 1,
        device: Literal["auto", "cuda", "mps", "cpu"] = "auto",
    ) -> JobStatus:
        """Start one conversion from exactly one document path or Markdown text.

        Poll get_conversion for progress and the final local file URI.
        Voices are catalogue names or voice IDs and rotate once per chapter.
        Omit voices to use the default voice of the language.
        The job fails when a voice is unknown or has another language.
        Negative duration_delta values speak faster, positive values slower.
        Existing outputs are rejected.
        """
        try:
            return jobs.start(
                ConversionRequest(
                    input_file=Path(input_file) if input_file is not None else None,
                    text=text,
                    language=language,
                    voices=tuple(voices),
                    duration_delta=duration_delta,
                    format_type=format,
                    output=Path(output) if output is not None else None,
                    chapter_level=chapter_level,
                    device=device,
                )
            )
        except (ValueError, OSError, RuntimeError) as error:
            raise ToolError(str(error)) from error

    @server.tool(annotations={"readOnlyHint": True})
    async def get_conversion(job_id: str) -> JobStatus:
        """Return progress, a worker error, or completed audio metadata."""
        try:
            return jobs.get(job_id)
        except ValueError as error:
            raise ToolError(str(error)) from error

    @server.tool
    async def cancel_conversion(job_id: str) -> JobStatus:
        """Request cancellation and return the current job state."""
        try:
            return jobs.cancel(job_id)
        except ValueError as error:
            raise ToolError(str(error)) from error

    return server
