"""Audio processing utilities for output handling.

This module provides functions for audio processing, including
stitching audio chunks and encoding to various formats using ffmpeg.

Examples:
    >>> "mp4" in SUPPORTED_FORMATS
    True
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

SAMPLE_RATE = 24000
SUPPORTED_FORMATS = {"m4a", "m4b", "mp4", "mp3", "wav", "opus"}
CODECS = {
    "m4a": "aac",
    "m4b": "aac",
    "mp4": "aac",
    "mp3": "libmp3lame",
    "opus": "libopus",
    "wav": "pcm_s16le",
}


@dataclass(frozen=True)
class Chapter:
    """Chapter bounds in samples at the Kokoro sample rate."""

    title: str
    start_sample: int
    end_sample: int


class PCMWriter:
    """Write completed chunks as mono float32 PCM with bounded memory."""

    def __init__(self, stream: BinaryIO) -> None:
        """Use an open binary stream and count the written samples."""
        self.stream = stream
        self.samples = 0

    def write(self, audio: np.ndarray) -> None:
        """Write one completed chunk in little-endian sample order."""
        chunk = np.asarray(audio, dtype="<f4")
        if chunk.ndim != 1:
            raise ValueError("Audio must contain mono samples.")
        chunk.tofile(self.stream)
        self.samples += chunk.size


def escape_ffmetadata(title: str) -> str:
    """Escape FFmetadata separators and line breaks."""
    title = title.replace("\r\n", "\n").replace("\r", "\n")
    return "".join("\\" + c if c in "\\=;#\n" else c for c in title)


def write_ffmetadata(chapters: list[Chapter], path: Path) -> None:
    """Write chapter times in the original sample time base."""
    lines = [";FFMETADATA1\n"]
    for chapter in chapters:
        lines.append(
            f"[CHAPTER]\nTIMEBASE=1/{SAMPLE_RATE}\n"
            f"START={chapter.start_sample}\nEND={chapter.end_sample}\n"
            f"title={escape_ffmetadata(chapter.title)}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to encode audio: {e.stderr}") from e
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found. Install ffmpeg.") from e


def encode_pcm(
    input_path: Path,
    output_path: Path,
    format_type: str = "m4a",
    chapters: list[Chapter] | None = None,
    bitrate: str = "128k",
) -> None:
    """Encode raw mono float32 PCM, with optional M4B or MP4 chapters."""
    if format_type not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {format_type}")
    if input_path.stat().st_size == 0:
        raise ValueError("Synthesis produced no audio.")
    with tempfile.TemporaryDirectory(prefix="simple-tts-metadata-") as directory:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            str(input_path),
        ]
        if format_type in {"m4b", "mp4"} and chapters:
            metadata = Path(directory) / "chapters.ffmeta"
            write_ffmetadata(chapters, metadata)
            cmd.extend(
                [
                    "-f",
                    "ffmetadata",
                    "-i",
                    str(metadata),
                    "-map_metadata",
                    "1",
                    "-map_chapters",
                    "1",
                ]
            )
        cmd.extend(["-map", "0:a:0", "-c:a", CODECS[format_type]])
        if format_type == "wav":
            cmd.extend(["-rf64", "auto"])
        else:
            cmd.extend(["-b:a", bitrate])
        if format_type in {"m4a", "m4b", "mp4"}:
            cmd.extend(["-f", "mp4"])
        cmd.append(str(output_path))
        _run_ffmpeg(cmd)
