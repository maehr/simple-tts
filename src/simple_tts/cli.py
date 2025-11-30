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
from simple_tts.language import detect_language, get_voice_for_language
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

    Args:
        speed: Speed multiplier to validate.

    Returns:
        Validated speed value.

    Raises:
        typer.BadParameter: If speed is out of valid range.

    Examples:
        >>> validate_speed(1.0)
        1.0
        >>> validate_speed(1.5)
        1.5
    """
    if speed < 0.5 or speed > 2.0:
        raise typer.BadParameter("Speed must be between 0.5 and 2.0")
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

        # Detect language
        task = progress.add_task("Detecting language...", total=None)
        detected_lang = detect_language(text)
        if voice is None:
            voice = get_voice_for_language(detected_lang)
        progress.update(task, completed=True)
        console.print(f"[bold green]🌍 Language:[/] {detected_lang} → Voice: {voice}")

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


if __name__ == "__main__":
    app()
