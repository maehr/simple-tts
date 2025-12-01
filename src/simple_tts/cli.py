"""Command-line interface for simple-tts.

This module provides the CLI entry point for the simple-tts application,
built using Typer for argument parsing and Rich for visual output.

Examples:
    >>> from simple_tts.cli import format_duration
    >>> format_duration(125.5)
    '2m 5s'
"""

import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
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

from simple_tts.audio import concatenate_audio, get_audio_duration, save_audio
from simple_tts.converter import extract_text, normalize_text, split_into_chunks
from simple_tts.dependencies import (
    check_dependencies,
    install_dependencies_brew,
    print_dependency_status,
)
from simple_tts.language import (
    check_language_support,
    detect_language,
    get_supported_languages,
    get_voice_for_language,
)
from simple_tts.tts import AudioChunk, TTSConfig, TTSEngine

app = typer.Typer(
    name="simple-tts",
    help="Convert documents to speech using Kokoro TTS.",
    add_completion=False,
)
console = Console()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.

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


def format_eta(seconds: float) -> str:
    """Format estimated time remaining.

    Args:
        seconds: Estimated seconds remaining.

    Returns:
        Formatted ETA string.

    Examples:
        >>> format_eta(90)
        '~1m 30s remaining'
        >>> format_eta(30)
        '~30s remaining'
    """
    if seconds <= 0:
        return "almost done"
    return f"~{format_duration(seconds)} remaining"


def validate_input_file(file_path: Path) -> None:
    """Validate that the input file exists and is readable.

    Args:
        file_path: Path to the input file.

    Raises:
        typer.BadParameter: If the file doesn't exist or is not readable.
    """
    if not file_path.exists():
        raise typer.BadParameter(f"File not found: {file_path}")
    if not file_path.is_file():
        raise typer.BadParameter(f"Not a file: {file_path}")


def validate_speed(speed: float) -> float:
    """Validate the speed parameter.

    Note: Typer's min/max constraints handle the actual validation.
    This function is kept for API consistency and testing.

    Args:
        speed: Speed multiplier to validate.

    Returns:
        Validated speed value.

    Examples:
        >>> validate_speed(1.0)
        1.0
        >>> validate_speed(1.5)
        1.5
    """
    return speed


def display_checkpoint(
    input_file: Path,
    output: Path,
    voice: str,
    lang: str,
    speed: float,
    format_type: str,
    chunks: list[str],
    text_length: int,
) -> None:
    """Display checkpoint information before TTS inference.

    Args:
        input_file: Path to input file.
        output: Path to output file.
        voice: Voice ID to use.
        lang: Language code.
        speed: Speed multiplier.
        format_type: Output format.
        chunks: List of text chunks.
        text_length: Total text length in characters.
    """
    # Create configuration table
    config_table = Table(title="TTS Configuration", show_header=False)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Input File", str(input_file))
    config_table.add_row("Output File", str(output))
    config_table.add_row("Language", lang)
    config_table.add_row("Voice", voice)
    config_table.add_row("Speed", f"{speed}x")
    config_table.add_row("Format", format_type)
    config_table.add_row("Text Length", f"{text_length:,} characters")
    config_table.add_row("Chunks", str(len(chunks)))

    console.print()
    console.print(config_table)

    # Display chunk details
    console.print()
    chunk_table = Table(title="Text Chunks Preview")
    chunk_table.add_column("#", style="dim", width=4)
    chunk_table.add_column("Length", style="cyan", width=8)
    chunk_table.add_column("Preview", style="white", overflow="ellipsis")

    for i, chunk in enumerate(chunks[:10]):  # Show first 10 chunks
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
            help="Input file to convert (PDF, DOCX, MD, etc.)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path. Defaults to input with .m4a extension.",
        ),
    ] = None,
    voice: Annotated[
        Optional[str],
        typer.Option(
            "--voice",
            "-v",
            help="Voice to use. Auto-detected from language if not specified.",
        ),
    ] = None,
    lang: Annotated[
        Optional[str],
        typer.Option(
            "--lang",
            "-l",
            help="Language code (e.g., 'en', 'fr'). Auto-detected if not specified.",
        ),
    ] = None,
    speed: Annotated[
        float,
        typer.Option(
            "--speed",
            "-s",
            help="Playback speed multiplier (0.5-2.0).",
            min=0.5,
            max=2.0,
        ),
    ] = 1.1,
    format_type: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (m4a, mp3, wav, ogg).",
        ),
    ] = "m4a",
    no_checkpoint: Annotated[
        bool,
        typer.Option(
            "--no-checkpoint",
            help="Skip the checkpoint display before TTS inference.",
        ),
    ] = False,
) -> None:
    """Convert a document to speech.

    Takes an input document (PDF, DOCX, MD, etc.) and converts it to
    an audio file using Kokoro TTS.
    """
    total_start_time = time.time()
    step_times: dict[str, float] = {}

    # Check system dependencies first
    if not check_dependencies(console):
        console.print(
            "\n[bold yellow]⚠️  Some features may not work without "
            "all dependencies.[/bold yellow]"
        )
        console.print()

    # Determine output path
    if output is None:
        output = input_file.with_suffix(f".{format_type}")

    console.print(f"[bold blue]📄 Input:[/] {input_file}")
    console.print(f"[bold blue]🔊 Output:[/] {output}")

    # Handle language selection
    supported_lang = None
    if lang:
        supported_lang = check_language_support(lang)
        if lang != supported_lang:
            console.print(
                f"[bold yellow]⚠️  Language '{lang}' not supported, "
                f"using '{supported_lang}'[/bold yellow]"
            )
        console.print(f"[bold blue]🌍 Language:[/] {supported_lang}")
    else:
        console.print("[bold blue]🌍 Language:[/] not specified; will detect from text")

    # Create progress bar with time tracking
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
        # Extract text
        step_start = time.time()
        task = progress.add_task("Extracting text...", total=100)
        try:
            text = extract_text(input_file)
            text = normalize_text(text)
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            raise typer.Exit(1) from e
        progress.update(task, completed=100)
        step_times["extract"] = time.time() - step_start
        console.print(
            f"[dim]  └─ Extracted {len(text):,} characters "
            f"in {format_duration(step_times['extract'])}[/dim]"
        )

        # Detect language if not provided
        if supported_lang is None:
            step_start = time.time()
            task = progress.add_task("Detecting language...", total=100)
            detected_lang = detect_language(text)
            supported_lang = check_language_support(detected_lang)
            if detected_lang != supported_lang:
                console.print(
                    f"[bold yellow]⚠️  Language '{detected_lang}' not supported, "
                    f"using '{supported_lang}'[/bold yellow]"
                )
            progress.update(task, completed=100)
            step_times["detect"] = time.time() - step_start
            console.print(
                f"[bold green]🌍 Language:[/] {supported_lang} "
                f"[dim](detected in {format_duration(step_times['detect'])})[/dim]"
            )

        # Get voice if not specified
        if voice is None:
            voice = get_voice_for_language(supported_lang)

        console.print(f"[bold green]🗣️ Voice:[/] {voice}")

        # Split text into chunks
        step_start = time.time()
        task = progress.add_task("Preparing text...", total=100)
        chunks = split_into_chunks(text)
        progress.update(task, completed=100)
        step_times["split"] = time.time() - step_start
        console.print(
            f"[bold green]📝 Chunks:[/] {len(chunks)} "
            f"[dim](prepared in {format_duration(step_times['split'])})[/dim]"
        )

    # Display checkpoint if not skipped
    if not no_checkpoint:
        display_checkpoint(
            input_file=input_file,
            output=output,
            voice=voice,
            lang=supported_lang,
            speed=speed,
            format_type=format_type,
            chunks=chunks,
            text_length=len(text),
        )

        # Ask for confirmation
        console.print(
            Panel(
                "[bold]Ready to start TTS inference.[/bold]\n"
                "Press [cyan]Enter[/cyan] to continue or [red]Ctrl+C[/red] to cancel.",
                title="Checkpoint",
                border_style="blue",
            )
        )
        try:
            input()
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled by user.[/yellow]")
            raise typer.Exit(0) from None

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
        # Initialize TTS engine
        step_start = time.time()
        task = progress.add_task("Loading TTS model...", total=100)
        config = TTSConfig(voice=voice, speed=speed)
        engine = TTSEngine(config)
        engine.load()
        progress.update(task, completed=100)
        step_times["load"] = time.time() - step_start
        console.print(
            f"[dim]  └─ Model loaded in {format_duration(step_times['load'])}[/dim]"
        )

        # Synthesize audio with progress tracking
        step_start = time.time()
        task = progress.add_task(
            f"Synthesizing speech (0/{len(chunks)} chunks)...",
            total=len(chunks),
        )
        audio_chunks: list[AudioChunk] = []

        for i, chunk in enumerate(chunks):
            audio_chunk = engine.synthesize(chunk)
            audio_chunks.append(audio_chunk)
            progress.update(task, advance=1)
            progress.update(
                task,
                description=f"Synthesizing speech ({i + 1}/{len(chunks)} chunks)...",
            )

        step_times["synthesize"] = time.time() - step_start
        console.print(
            f"[dim]  └─ Synthesized {len(chunks)} chunks "
            f"in {format_duration(step_times['synthesize'])}[/dim]"
        )

        # Concatenate and save
        step_start = time.time()
        task = progress.add_task("Saving audio...", total=100)
        audio_data = concatenate_audio([chunk.data for chunk in audio_chunks])
        save_audio(audio_data, output, sample_rate=config.sample_rate)
        progress.update(task, completed=100)
        step_times["save"] = time.time() - step_start
        console.print(f"[dim]  └─ Saved in {format_duration(step_times['save'])}[/dim]")

    # Final summary
    total_time = time.time() - total_start_time
    duration = get_audio_duration(audio_data, config.sample_rate)

    console.print()
    summary_table = Table(title="Summary", show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Output File", str(output))
    summary_table.add_row("Audio Duration", format_duration(duration))
    summary_table.add_row("Processing Time", format_duration(total_time))
    if total_time > 0:
        speed_ratio = f"{duration / total_time:.1f}x realtime"
    else:
        speed_ratio = "N/A"
    summary_table.add_row("Speed Ratio", speed_ratio)

    console.print(summary_table)
    console.print(f"\n[bold green]✅ Done![/] Duration: {format_duration(duration)}")


@app.command()
def version() -> None:
    """Show the version of simple-tts."""
    from simple_tts import __version__

    console.print(f"simple-tts version {__version__}")


@app.command()
def check_deps() -> None:
    """Check if all system dependencies are installed."""
    print_dependency_status(console)


@app.command()
def install_deps(
    brew: Annotated[
        bool,
        typer.Option(
            "--brew",
            help="Install dependencies using Homebrew (macOS).",
        ),
    ] = False,
) -> None:
    """Install missing system dependencies.

    Use --brew to install via Homebrew on macOS.
    """
    if brew:
        success = install_dependencies_brew(console)
        if not success:
            raise typer.Exit(1)
    else:
        console.print("[bold]To install dependencies:[/bold]")
        console.print()
        console.print("[bold]macOS (Homebrew):[/bold]")
        console.print("  [cyan]simple-tts install-deps --brew[/cyan]")
        console.print("  or: [cyan]brew install poppler pandoc ffmpeg[/cyan]")
        console.print()
        console.print("[bold]Ubuntu/Debian:[/bold]")
        console.print("  [cyan]sudo apt-get install poppler-utils pandoc ffmpeg[/cyan]")


@app.command()
def languages() -> None:
    """List supported languages for TTS."""
    langs = get_supported_languages()
    console.print("[bold]Supported languages:[/bold]")
    console.print()
    for lang in langs:
        console.print(f"  • {lang}")


if __name__ == "__main__":
    app()
