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
from simple_tts.jobs import JobManager, JobStatus
from simple_tts.service import ConversionRequest
from simple_tts.voices import VOICES

if TYPE_CHECKING:
    from fastmcp import FastMCP


@dataclass(frozen=True)
class VoiceDescription:
    """A supported Kokoro voice."""

    id: str
    language: str
    description: str


@dataclass(frozen=True)
class VoiceCatalog:
    """Structured voice discovery output."""

    voices: list[VoiceDescription]


def create_server(workspace: Path, *, manager: JobManager | None = None) -> FastMCP:
    """Build a stdio server without loading a speech model."""
    import anyio
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
            "Paths are relative to the configured workspace. "
            "File URIs require access to the local filesystem."
        ),
        lifespan=lifespan,
        strict_input_validation=True,
        tasks=False,
    )

    @server.tool(annotations={"readOnlyHint": True})
    async def list_voices() -> VoiceCatalog:
        """Return voice IDs, languages, and descriptions."""
        return VoiceCatalog(
            [VoiceDescription(voice, **info) for voice, info in sorted(VOICES.items())]
        )

    @server.tool(annotations={"readOnlyHint": True})
    async def check_dependencies() -> dict[str, bool]:
        """Return Pandoc and FFmpeg availability without loading a model."""
        return {dep.name: is_command_available(dep.name) for dep in DEPENDENCIES}

    @server.tool
    async def start_conversion(
        input_file: str | None = None,
        text: str | None = None,
        voices: Annotated[list[str], Field(min_length=1)] = ["af_bella"],
        speed: Annotated[float, Field(ge=0.5, le=2.0)] = 1.1,
        format: str | None = None,
        output: str | None = None,
        chapter_level: Annotated[int, Field(ge=1, le=6)] = 1,
        device: Literal["auto", "cuda", "mps", "cpu"] = "auto",
    ) -> JobStatus:
        """Start one conversion from exactly one document path or Markdown text.

        Poll get_conversion for progress and the final local file URI.
        All voices must share one language code. Existing outputs are rejected.
        """
        try:
            return jobs.start(
                ConversionRequest(
                    input_file=Path(input_file) if input_file is not None else None,
                    text=text,
                    voices=tuple(voices),
                    speed=speed,
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
