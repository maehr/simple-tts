"""Command-line interface for simple-tts.

A simple CLI for converting documents to speech with Phonon 40M Multilingual.

Example:
    $ simple-tts convert document.md --language de
    $ simple-tts convert document.md --language de --voice Nele,Anton
    $ simple-tts voices --language de
    $ simple-tts check-deps
"""

import tempfile
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from simple_tts.dependencies import (
    install_dependencies_brew,
    print_dependency_status,
)
from simple_tts.service import (
    ConversionRequest,
    ConversionResult,
    PreparedConversion,
    ProgressEvent,
    execute_conversion,
    prepare_conversion,
    validate_request,
)
from simple_tts.tts import DURATION_DELTA_RANGE
from simple_tts.voices import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    available_voices,
    check_language,
    parse_voices,
)

app = typer.Typer(
    name="simple-tts",
    help=(
        "Convert prepared text and structured documents to speech "
        "with Phonon 40M Multilingual."
    ),
    add_completion=False,
)
console = Console()


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration.

    Examples:
        >>> format_duration(65)
        '1m 5s'
        >>> format_duration(30)
        '30s'
        >>> format_duration(3661)
        '1h 1m 1s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def language_label(code: str) -> str:
    """Return a language name with its code.

    Example:
        >>> language_label("de")
        'German (de)'
    """
    return f"{LANGUAGES[code]} ({code})"


def display_checkpoint(prepared: PreparedConversion) -> None:
    """Display checkpoint before TTS inference."""
    request = prepared.request
    chunks = prepared.chunks
    text_length = sum(len(section.text) for section in prepared.document.sections)
    config_table = Table(title="TTS Configuration", show_header=False)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Input File", escape(str(request.input_file)))
    config_table.add_row("Output File", escape(str(prepared.output)))
    config_table.add_row("Language", language_label(request.language))
    for voice in prepared.voices:
        config_table.add_row(
            "Voice", escape(f"{voice.name} ({voice.voice_id}): {voice.description}")
        )
    config_table.add_row("Duration Delta", f"{request.duration_delta:g}")
    config_table.add_row("Format", prepared.format_type)
    config_table.add_row("Text Length", f"{text_length:,} characters")
    config_table.add_row("Chunks", str(len(chunks)))

    console.print()
    console.print(config_table)

    # Show first 10 chunks
    console.print()
    chunk_table = Table(title="Text Chunks Preview")
    chunk_table.add_column("#", style="dim", width=4)
    chunk_table.add_column("Length", style="cyan", width=8)
    chunk_table.add_column("Preview", style="white", overflow="ellipsis")

    for i, chunk in enumerate(chunks[:10]):
        preview = chunk[:80].replace("\n", " ")
        if len(chunk) > 80:
            preview += "..."
        chunk_table.add_row(str(i + 1), str(len(chunk)), preview)

    if len(chunks) > 10:
        chunk_table.add_row("...", "...", f"({len(chunks) - 10} more chunks)")

    console.print(chunk_table)
    console.print()


@app.command()
def convert(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Prepared input: MD, QMD, RMD, TXT, DOCX, EPUB, RST, or HTML.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Narration language: en, de, es, or fr."),
    ] = DEFAULT_LANGUAGE,
    voice: Annotated[
        Optional[str],
        typer.Option(
            "--voice",
            "-v",
            help=(
                "Catalogue voice names or IDs, separated by commas. Rotate once "
                "per chapter. Default: the default voice of the language."
            ),
            show_default=False,
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o", help="Output path. The suffix selects the format."
        ),
    ] = None,
    duration_delta: Annotated[
        float,
        typer.Option(
            "--duration-delta",
            help="Speaking rate. Negative values speak faster, positive slower.",
            min=DURATION_DELTA_RANGE[0],
            max=DURATION_DELTA_RANGE[1],
        ),
    ] = 0.0,
    format_type: Annotated[
        Optional[str],
        typer.Option(
            "--format",
            "-f",
            help="Audio format: m4a, m4b, mp4, mp3, wav, or opus. Default: m4a.",
            show_default=False,
        ),
    ] = None,
    chapter_level: Annotated[
        int,
        typer.Option(
            "--chapter-level",
            help="Heading level that starts each chapter.",
            min=1,
            max=6,
        ),
    ] = 1,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help=(
                "Device: auto or cpu (CPU, 8-bit weights), "
                "mps (Metal, f32), or cuda (f32)."
            ),
        ),
    ] = "auto",
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Convert a document to speech with optional chapters and voice rotation."""
    total_start = time.time()
    try:
        request = ConversionRequest(
            input_file=input_file,
            language=check_language(language),
            voices=parse_voices(voice) if voice is not None else (),
            duration_delta=duration_delta,
            output=output,
            format_type=format_type,
            chapter_level=chapter_level,
            device=device,
        )
        output, format_type = validate_request(request)
    except ValueError as e:
        console.print("Error:", str(e), style="bold red", markup=False)
        raise typer.Exit(1) from e

    console.print(f"[bold blue]Input:[/] {escape(str(input_file))}")
    console.print(f"[bold blue]Output:[/] {escape(str(output))}")
    console.print(f"[bold blue]Language:[/] {language_label(request.language)}")

    try:
        with tempfile.TemporaryDirectory(prefix="simple-tts-") as scratch:
            prepared = prepare_conversion(request, Path(scratch))
            result = _convert_prepared(prepared, Path(scratch), yes)
    except KeyboardInterrupt:
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(130) from None
    except Exception as e:
        console.print("Error:", str(e), style="bold red", markup=False)
        raise typer.Exit(1) from e

    total_time = time.time() - total_start
    _display_summary(result, total_time)
    console.print("[bold green]Done.[/]")


def _convert_prepared(
    prepared: PreparedConversion, scratch: Path, yes: bool
) -> ConversionResult:
    request = prepared.request
    input_file = request.input_file
    assert input_file is not None
    output = prepared.output
    names = ", ".join(voice.name for voice in prepared.voices)
    console.print(f"[bold blue]Voice:[/] {escape(names)}")
    if not prepared.document.matched_headings:
        console.print(
            f"[yellow]Warning: No headings match level {request.chapter_level}. "
            f"Use one chapter named '{input_file.stem}'.[/yellow]"
        )
    if not yes:
        display_checkpoint(prepared)
        console.print(
            Panel(
                "Press Enter to start synthesis. Press Ctrl+C to cancel.",
                title="Checkpoint",
                border_style="blue",
            )
        )
        input()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Load the TTS model...", total=None)

        def report(event: ProgressEvent) -> None:
            if event.message:
                console.print(event.message, markup=False)
            progress.update(
                task,
                description=f"{event.phase.capitalize()}...",
                completed=event.completed_chunks,
                total=event.total_chunks if event.phase == "synthesizing" else None,
            )

        with tempfile.TemporaryDirectory(
            prefix=".simple-tts-", dir=output.parent
        ) as encoded:
            temp_output = Path(encoded) / output.name
            result = execute_conversion(prepared, scratch, temp_output, report=report)
            temp_output.replace(output)

        progress.update(task, completed=1, total=1)
    return result


def _display_summary(result: ConversionResult, total_time: float) -> None:
    summary = Table(title="Summary", show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Output File", result.path)
    summary.add_row("Chapters", str(len(result.chapters)))
    summary.add_row("Audio Duration", format_duration(result.duration))
    summary.add_row("Processing Time", format_duration(total_time))
    if total_time > 0:
        summary.add_row("Speed Ratio", f"{result.duration / total_time:.1f}x realtime")
    console.print(summary)


@app.command()
def mcp(
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            help="Existing directory for MCP input and output files.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ],
) -> None:
    """Run the local MCP server on macOS or Linux over stdio."""
    try:
        from simple_tts.mcp import create_server

        server = create_server(workspace)
    except ImportError as error:
        Console(stderr=True).print(
            'Install MCP support: pip install "simple-tts[mcp]"', markup=False
        )
        raise typer.Exit(1) from error
    except RuntimeError as error:
        Console(stderr=True).print("Error:", str(error), style="bold red", markup=False)
        raise typer.Exit(1) from error
    server.run(transport="stdio", show_banner=False)


@app.command()
def voices(
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            "-l",
            help="Show only the voices of one language: en, de, es, or fr.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """List catalogue voices for each narration language."""
    try:
        catalog = available_voices(language)
    except (ValueError, RuntimeError) as e:
        console.print("Error:", str(e), style="bold red", markup=False)
        raise typer.Exit(1) from e

    table = Table(title="Phonon Voices")
    table.add_column("Name", style="cyan")
    table.add_column("Voice ID", style="magenta")
    table.add_column("Language", style="green")
    table.add_column("Description", style="white")

    for voice in catalog:
        name = f"{voice.name} (default)" if voice.is_default else voice.name
        table.add_row(
            escape(name),
            escape(voice.voice_id),
            language_label(voice.language),
            escape(voice.description),
        )

    console.print(table)


@app.command()
def version() -> None:
    """Show version."""
    from simple_tts import __version__

    console.print(f"simple-tts version {__version__}")


@app.command()
def check_deps() -> None:
    """Check system dependencies."""
    print_dependency_status(console)


@app.command()
def install_deps(
    brew: Annotated[
        bool,
        typer.Option("--brew", help="Install using Homebrew (macOS)."),
    ] = False,
) -> None:
    """Install missing system dependencies."""
    if brew:
        if not install_dependencies_brew(console):
            raise typer.Exit(1)
    else:
        console.print("[bold]To install dependencies:[/bold]")
        console.print()
        console.print("[bold]macOS:[/bold]")
        console.print("  [cyan]brew install pandoc ffmpeg[/cyan]")
        console.print()
        console.print("[bold]Ubuntu/Debian:[/bold]")
        console.print("  [cyan]sudo apt install pandoc ffmpeg[/cyan]")


if __name__ == "__main__":
    app()
