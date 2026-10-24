"""Command-line interface for simple-tts.

A simple CLI for converting documents to speech using Kokoro TTS.

Example:
    $ simple-tts convert document.pdf --voice af_bella
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
    write_vtt,
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


@app.command()
def convert(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input file: PDF, DOCX, EPUB, MD, QMD, RMD, TXT, RST, or HTML.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    voice: Annotated[
        str,
        typer.Option(
            "--voice", "-v", help="Voice ID. Run 'simple-tts voices' for the list."
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
            help="Audio format: m4a, m4b, mp3, wav, ogg, or opus. Default: m4a.",
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
    vtt: Annotated[
        bool,
        typer.Option("--vtt/--no-vtt", help="Write chapter navigation to a .vtt file."),
    ] = False,
    device: Annotated[
        str,
        typer.Option("--device", help="Device: auto, cuda, mps, or cpu."),
    ] = "auto",
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Convert a document to speech with optional chapter navigation."""
    total_start = time.time()
    try:
        output, format_type = resolve_output(input_file, output, format_type)
        if device not in DEVICES:
            raise ValueError("Device must be auto, cuda, mps, or cpu.")
        if voice not in VOICES:
            raise ValueError(f"Unknown voice '{voice}'. Run 'simple-tts voices'.")
        if not output.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {output.parent}")
        if output.resolve() == input_file.resolve():
            raise ValueError("The output path must differ from the input path.")
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1) from e

    check_dependencies(console)
    console.print(f"[bold blue]Input:[/] {input_file}")
    console.print(f"[bold blue]Output:[/] {output}")
    console.print(f"[bold blue]Voice:[/] {voice}")

    try:
        document = extract_sections(input_file, chapter_level)
        if (format_type == "m4b" or vtt) and not document.matched_headings:
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
            tempfile.TemporaryDirectory(
                prefix=".simple-tts-", dir=output.parent
            ) as staging,
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
                voice,
                device=device,
                report=lambda message: console.print(message, markup=False),
            )
            progress.update(task, completed=1, total=1)
            task = progress.add_task("Synthesize speech...", total=len(chunks))
            pcm_path = Path(scratch) / "narration.f32le"
            chapters = []
            with pcm_path.open("wb") as stream:
                writer = PCMWriter(stream)
                for title, section_chunks in sections:
                    start_sample = writer.samples
                    for chunk in section_chunks:
                        audio = synthesize(pipeline, chunk, voice=voice, speed=speed)
                        writer.write(audio)
                        del audio
                        progress.update(task, advance=1)
                    if writer.samples > start_sample:
                        chapters.append(Chapter(title, start_sample, writer.samples))
                total_samples = writer.samples
            if not total_samples:
                raise ValueError("Synthesis produced no audio.")

            task = progress.add_task("Encode audio...", total=None)
            staged_audio = Path(staging) / output.name
            encode_pcm(pcm_path, staged_audio, format_type, chapters)
            staged_vtt = Path(staging) / "chapters.vtt"
            if vtt:
                write_vtt(chapters, staged_vtt)
            progress.update(task, completed=1, total=1)
            staged_audio.replace(output)
            if vtt:
                staged_vtt.replace(output.with_suffix(".vtt"))
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
    if vtt:
        summary.add_row("Chapter VTT", str(output.with_suffix(".vtt")))
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
        console.print("  [cyan]brew install poppler pandoc ffmpeg[/cyan]")
        console.print()
        console.print("[bold]Ubuntu/Debian:[/bold]")
        console.print("  [cyan]sudo apt install poppler-utils pandoc ffmpeg[/cyan]")


if __name__ == "__main__":
    app()
