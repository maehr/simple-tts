"""Command-line interface for simple-tts.

This module provides the CLI entry point for the simple-tts application,
built using Typer for argument parsing and Rich for visual output.

Examples:
    >>> from simple_tts.cli import format_duration
    >>> format_duration(125.5)
    '2m 5s'
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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
) -> None:
    """Convert a document to speech.

    Takes an input document (PDF, DOCX, MD, etc.) and converts it to
    an audio file using Kokoro TTS.
    """
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Extract text
        task = progress.add_task("Extracting text...", total=None)
        try:
            text = extract_text(input_file)
            text = normalize_text(text)
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            raise typer.Exit(1) from e
        progress.update(task, completed=True)

        # Determine language - only detect if not provided
        if lang is not None:
            # Use provided language, check if supported
            task = progress.add_task("Checking language...", total=None)
            supported_lang = check_language_support(lang)
            if lang != supported_lang:
                console.print(
                    f"[bold yellow]⚠️  Language '{lang}' not supported, "
                    f"using '{supported_lang}'[/bold yellow]"
                )
            progress.update(task, completed=True)
        else:
            # Detect language from text
            task = progress.add_task("Detecting language...", total=None)
            detected_lang = detect_language(text)
            supported_lang = check_language_support(detected_lang)
            if detected_lang != supported_lang:
                console.print(
                    f"[bold yellow]⚠️  Language '{detected_lang}' not supported, "
                    f"using '{supported_lang}'[/bold yellow]"
                )
            progress.update(task, completed=True)

        # Get voice if not specified
        if voice is None:
            voice = get_voice_for_language(supported_lang)
        console.print(f"[bold green]🌍 Language:[/] {supported_lang} → Voice: {voice}")

        # Split text into chunks
        task = progress.add_task("Preparing text...", total=None)
        chunks = split_into_chunks(text)
        progress.update(task, completed=True)
        console.print(f"[bold green]📝 Chunks:[/] {len(chunks)}")

        # Initialize TTS engine
        task = progress.add_task("Loading TTS model...", total=None)
        config = TTSConfig(voice=voice, speed=speed)
        engine = TTSEngine(config)
        engine.load()
        progress.update(task, completed=True)

        # Synthesize audio
        task = progress.add_task("Synthesizing speech...", total=None)
        audio_chunks: list[AudioChunk] = engine.synthesize_stream(chunks)
        progress.update(task, completed=True)

        # Concatenate and save
        task = progress.add_task("Saving audio...", total=None)
        audio_data = concatenate_audio([chunk.data for chunk in audio_chunks])
        save_audio(audio_data, output, sample_rate=config.sample_rate)
        progress.update(task, completed=True)

    duration = get_audio_duration(audio_data, config.sample_rate)
    console.print(f"[bold green]✅ Done![/] Duration: {format_duration(duration)}")


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
