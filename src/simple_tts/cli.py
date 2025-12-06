"""Command-line interface for simple-tts.

A simple CLI for converting documents to speech using Kokoro TTS.

Example:
    $ simple-tts convert document.pdf --voice af_bella
    $ simple-tts voices
    $ simple-tts check-deps
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
from simple_tts.tts import SAMPLE_RATE, load_tts, synthesize
from simple_tts.voices import VOICES, get_voice_info

app = typer.Typer(
    name="simple-tts",
    help="Convert documents to speech using Kokoro TTS.",
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


def format_timestamp(seconds: float) -> str:
    """Format seconds to WebVTT timestamp (hh:mm:ss.mmm)."""
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def build_vtt(chunks: list[str], durations: list[float]) -> str:
    """Build a WebVTT string from text chunks and durations."""
    if len(chunks) != len(durations):
        raise ValueError("Chunks and durations must have the same length.")

    lines = ["WEBVTT", ""]
    current_time = 0.0

    for i, (chunk, duration) in enumerate(zip(chunks, durations)):
        start = current_time
        end = current_time + max(duration, 0.0)
        text = " ".join(chunk.split()) or "(silence)"

        lines.append(str(i + 1))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(text)
        lines.append("")

        current_time = end

    return "\n".join(lines).strip() + "\n"


def display_checkpoint(
    input_file: Path,
    output: Path,
    voice: str,
    speed: float,
    format_type: str,
    chunks: list[str],
    text_length: int,
) -> None:
    """Display checkpoint before TTS inference."""
    voice_info = get_voice_info(voice)

    config_table = Table(title="TTS Configuration", show_header=False)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Input File", str(input_file))
    config_table.add_row("Output File", str(output))
    config_table.add_row("Voice", f"{voice} ({voice_info['description']})")
    config_table.add_row("Speed", f"{speed}x")
    config_table.add_row("Format", format_type)
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
            help="Input file to convert (PDF, DOCX, EPUB, MD, TXT, etc.)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    voice: Annotated[
        str,
        typer.Option(
            "--voice",
            "-v",
            help="Voice ID (run 'simple-tts voices' to list)",
        ),
    ] = "af_bella",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path",
        ),
    ] = None,
    speed: Annotated[
        float,
        typer.Option(
            "--speed",
            "-s",
            help="Playback speed (0.5-2.0)",
            min=0.5,
            max=2.0,
        ),
    ] = 1.1,
    format_type: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output audio format",
        ),
    ] = "m4a",
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt",
        ),
    ] = False,
    vtt: Annotated[
        bool,
        typer.Option(
            "--vtt/--no-vtt",
            help="Write WebVTT sidecar captions next to the audio file",
        ),
    ] = True,
) -> None:
    """Convert a document to speech using Kokoro TTS.

    Example:
        $ simple-tts convert document.pdf --voice af_bella
        $ simple-tts convert essay.md -v bf_emma -s 1.2 -o output.mp3
    """
    total_start = time.time()
    step_times: dict[str, float] = {}

    # Validate voice
    if voice not in VOICES:
        console.print(f"[bold red]Error:[/] Unknown voice '{voice}'")
        console.print("Run 'simple-tts voices' to see available voices.")
        raise typer.Exit(1)

    # Check dependencies
    if not check_dependencies(console):
        console.print(
            "\n[bold yellow]⚠️  Missing dependencies may cause errors.[/bold yellow]\n"
        )

    # Set output path
    if output is None:
        output = input_file.with_suffix(f".{format_type}")

    console.print(f"[bold blue]📄 Input:[/] {input_file}")
    console.print(f"[bold blue]🔊 Output:[/] {output}")
    console.print(f"[bold blue]🗣️ Voice:[/] {voice}")

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
        extract_time = format_duration(step_times["extract"])
        console.print(f"[dim]  └─ {len(text):,} chars in {extract_time}[/dim]")

        # Split into chunks
        step_start = time.time()
        task = progress.add_task("Preparing text...", total=100)
        chunks = split_into_chunks(text)
        progress.update(task, completed=100)
        step_times["split"] = time.time() - step_start
        split_time = format_duration(step_times["split"])
        console.print(f"[dim]  └─ {len(chunks)} chunks in {split_time}[/dim]")

    # Show checkpoint and ask for confirmation
    if not yes:
        display_checkpoint(
            input_file=input_file,
            output=output,
            voice=voice,
            speed=speed,
            format_type=format_type,
            chunks=chunks,
            text_length=len(text),
        )
        console.print(
            Panel(
                "[bold]Ready to synthesize.[/bold]\n"
                "Press [cyan]Enter[/cyan] to continue or [red]Ctrl+C[/red] to cancel.",
                title="Checkpoint",
                border_style="blue",
            )
        )
        try:
            input()
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
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
        # Load TTS model
        step_start = time.time()
        task = progress.add_task("Loading TTS model...", total=100)
        pipeline = load_tts(voice)
        progress.update(task, completed=100)
        step_times["load"] = time.time() - step_start
        console.print(
            f"[dim]  └─ Model loaded in {format_duration(step_times['load'])}[/dim]"
        )

        # Synthesize chunks
        step_start = time.time()
        task = progress.add_task(
            f"Synthesizing (0/{len(chunks)})...", total=len(chunks)
        )
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            audio = synthesize(pipeline, chunk, voice=voice, speed=speed)
            audio_chunks.append(audio)
            progress.update(task, advance=1)
            desc = f"Synthesizing ({i + 1}/{len(chunks)})..."
            progress.update(task, description=desc)

        step_times["synthesize"] = time.time() - step_start
        synth_time = format_duration(step_times["synthesize"])
        console.print(f"[dim]  └─ {len(chunks)} chunks in {synth_time}[/dim]")

        # Save audio
        step_start = time.time()
        task = progress.add_task("Saving audio...", total=100)
        audio_data = concatenate_audio(audio_chunks)
        save_audio(audio_data, output, sample_rate=SAMPLE_RATE)
        progress.update(task, completed=100)
        step_times["save"] = time.time() - step_start
        console.print(f"[dim]  └─ Saved in {format_duration(step_times['save'])}[/dim]")

    vtt_path: Path | None = None
    if vtt:
        step_start = time.time()
        durations = [
            len(audio) / SAMPLE_RATE if len(audio) else 0.0 for audio in audio_chunks
        ]
        vtt_text = build_vtt(chunks, durations)
        vtt_path = output.with_suffix(".vtt")
        vtt_path.write_text(vtt_text, encoding="utf-8")
        console.print(
            f"[dim]  └─ Captions saved to {vtt_path} "
            f"({format_duration(time.time() - step_start)})[/dim]"
        )

    # Summary
    total_time = time.time() - total_start
    duration = get_audio_duration(audio_data, SAMPLE_RATE)

    console.print()
    summary = Table(title="Summary", show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Output File", str(output))
    summary.add_row("Audio Duration", format_duration(duration))
    summary.add_row("Processing Time", format_duration(total_time))
    if total_time > 0:
        summary.add_row("Speed Ratio", f"{duration / total_time:.1f}x realtime")
    if vtt_path:
        summary.add_row("Captions", str(vtt_path))
    console.print(summary)
    console.print("\n[bold green]✅ Done![/]")


@app.command()
def voices() -> None:
    """List available voices."""
    table = Table(title="Available Voices")
    table.add_column("Voice ID", style="cyan")
    table.add_column("Language", style="green")
    table.add_column("Description", style="white")

    for voice_id, info in sorted(VOICES.items()):
        table.add_row(voice_id, info["language"], info["description"])

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
        console.print("  [cyan]brew install poppler pandoc ffmpeg[/cyan]")
        console.print()
        console.print("[bold]Ubuntu/Debian:[/bold]")
        console.print("  [cyan]sudo apt install poppler-utils pandoc ffmpeg[/cyan]")


if __name__ == "__main__":
    app()
