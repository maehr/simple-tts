"""Command-line interface for simple-tts.

A simple CLI for converting documents to speech using Kokoro TTS.

Example:
    $ simple-tts convert document.md --voice af_bella
    $ simple-tts voices
    $ simple-tts check-deps
"""

import tempfile
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

from simple_tts.audio import (
    SUPPORTED_FORMATS,
    Chapter,
    PCMWriter,
    encode_pcm,
)
from simple_tts.converter import extract_sections, split_into_chunks
from simple_tts.dependencies import (
    check_dependencies,
    install_dependencies_brew,
    print_dependency_status,
)
from simple_tts.tts import DEVICES, SAMPLE_RATE, load_tts, synthesize
from simple_tts.voices import VOICES, get_voice_info

app = typer.Typer(
    name="simple-tts",
    help="Convert prepared text and structured documents to speech using Kokoro TTS.",
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
    config_table = Table(title="TTS Configuration", show_header=False)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Input File", str(input_file))
    config_table.add_row("Output File", str(output))
    for voice_id in voice.split(","):
        description = get_voice_info(voice_id)["description"]
        config_table.add_row("Voice", f"{voice_id} ({description})")
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
    return output, selected


def parse_voices(value: str) -> list[str]:
    """Validate a rotation that shares one Kokoro language code."""
    voices = [voice.strip() for voice in value.split(",")]
    for voice in voices:
        if voice not in VOICES:
            raise ValueError(f"Unknown voice '{voice}'. Run 'simple-tts voices'.")
    if len({voice[0] for voice in voices}) != 1:
        raise ValueError("All voices must share one Kokoro language code.")
    return voices


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
    voice: Annotated[
        str,
        typer.Option(
            "--voice",
            "-v",
            help="Voice IDs separated by commas. Rotate once per chapter.",
        ),
    ] = "af_bella",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o", help="Output path. The suffix selects the format."
        ),
    ] = None,
    speed: Annotated[
        float,
        typer.Option("--speed", "-s", help="Speech speed.", min=0.5, max=2.0),
    ] = 1.1,
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
        typer.Option("--device", help="Device: auto, cuda, mps, or cpu."),
    ] = "auto",
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Convert a document to speech with optional chapters and voice rotation."""
    total_start = time.time()
    try:
        output, format_type = resolve_output(input_file, output, format_type)
        if device not in DEVICES:
            raise ValueError("Device must be auto, cuda, mps, or cpu.")
        selected_voices = parse_voices(voice)
        voice = ",".join(selected_voices)
        if not output.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {output.parent}")
        if input_file.resolve() == output.resolve():
            raise ValueError("The output path must differ from the input path.")
        if output.is_dir():
            raise ValueError(f"An output file path is a directory: {output}")
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1) from e

    console.print(f"[bold blue]Input:[/] {input_file}")
    console.print(f"[bold blue]Output:[/] {output}")
    console.print(f"[bold blue]Voice:[/] {voice}")

    try:
        document = extract_sections(input_file, chapter_level)
        if not document.matched_headings:
            console.print(
                f"[yellow]Warning: No headings match level {chapter_level}. "
                f"Use one chapter named '{input_file.stem}'.[/yellow]"
            )
        sections = [
            (section.title, split_into_chunks(section.text))
            for section in document.sections
            if section.text.strip()
        ]
        chunks = [chunk for _, section_chunks in sections for chunk in section_chunks]
        if not chunks:
            raise ValueError("The document contains no narration text.")
        check_dependencies(console)
        if not yes:
            display_checkpoint(
                input_file,
                output,
                voice,
                speed,
                format_type,
                chunks,
                sum(len(section.text) for section in document.sections),
            )
            console.print(
                Panel(
                    "Press Enter to start synthesis. Press Ctrl+C to cancel.",
                    title="Checkpoint",
                    border_style="blue",
                )
            )
            input()

        with (
            tempfile.TemporaryDirectory(prefix="simple-tts-") as scratch,
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
            ) as progress,
        ):
            task = progress.add_task("Load the TTS model...", total=None)
            pipeline = load_tts(
                selected_voices[0],
                device=device,
                report=lambda message: console.print(message, markup=False),
            )
            progress.update(task, completed=1, total=1)
            task = progress.add_task("Synthesize speech...", total=len(chunks))
            pcm_path = Path(scratch) / "narration.f32le"
            chapters = []
            with pcm_path.open("wb") as stream:
                writer = PCMWriter(stream)
                for index, (title, section_chunks) in enumerate(sections):
                    section_voice = selected_voices[index % len(selected_voices)]
                    start_sample = writer.samples
                    for chunk in section_chunks:
                        audio = synthesize(
                            pipeline, chunk, voice=section_voice, speed=speed
                        )
                        writer.write(audio)
                        del audio
                        progress.update(task, advance=1)
                    if writer.samples > start_sample:
                        chapters.append(Chapter(title, start_sample, writer.samples))
                total_samples = writer.samples
            if not total_samples:
                raise ValueError("Synthesis produced no audio.")

            task = progress.add_task("Encode audio...", total=None)
            with tempfile.TemporaryDirectory(
                prefix=".simple-tts-", dir=output.parent
            ) as encoded:
                temp_output = Path(encoded) / output.name
                encode_pcm(pcm_path, temp_output, format_type, chapters)
                temp_output.replace(output)

            progress.update(task, completed=1, total=1)
    except KeyboardInterrupt:
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(130) from None
    except Exception as e:
        console.print("Error:", str(e), style="bold red", markup=False)
        raise typer.Exit(1) from e

    total_time = time.time() - total_start
    duration = total_samples / SAMPLE_RATE
    summary = Table(title="Summary", show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Output File", str(output))
    summary.add_row("Chapters", str(len(chapters)))
    summary.add_row("Audio Duration", format_duration(duration))
    summary.add_row("Processing Time", format_duration(total_time))
    if total_time > 0:
        summary.add_row("Speed Ratio", f"{duration / total_time:.1f}x realtime")
    console.print(summary)
    console.print("[bold green]Done.[/]")


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
        console.print("  [cyan]brew install pandoc ffmpeg[/cyan]")
        console.print()
        console.print("[bold]Ubuntu/Debian:[/bold]")
        console.print("  [cyan]sudo apt install pandoc ffmpeg[/cyan]")


if __name__ == "__main__":
    app()
