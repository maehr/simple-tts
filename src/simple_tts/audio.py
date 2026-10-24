"""Audio processing utilities for output handling.

This module provides functions for audio processing, including
stitching audio chunks and encoding to various formats using ffmpeg.

Examples:
    >>> from simple_tts.audio import get_output_format
    >>> get_output_format("output.mp3")
    'mp3'
"""

import html
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
SUPPORTED_FORMATS = {"m4a", "m4b", "mp3", "wav", "ogg", "opus"}
CODECS = {
    "m4a": "aac",
    "m4b": "aac",
    "mp3": "libmp3lame",
    "ogg": "libvorbis",
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


def vtt_timestamp(samples: int) -> str:
    """Convert samples to milliseconds without a 24-hour wrap."""
    milliseconds = samples * 1000 // SAMPLE_RATE
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def write_vtt(chapters: list[Chapter], path: Path) -> None:
    """Write chapter navigation with escaped WebVTT cue text."""
    cues = ["WEBVTT\n\n"]
    for chapter in chapters:
        title = html.escape(" ".join(chapter.title.split()), quote=False)
        cues.append(
            f"{vtt_timestamp(chapter.start_sample)} --> "
            f"{vtt_timestamp(chapter.end_sample)}\n{title}\n\n"
        )
    path.write_text("".join(cues), encoding="utf-8")


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
    """Encode raw mono float32 PCM once, with optional M4B chapters."""
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
        if format_type == "m4b" and chapters:
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
        if format_type in {"m4a", "m4b"}:
            cmd.extend(["-f", "mp4"])
        cmd.append(str(output_path))
        _run_ffmpeg(cmd)


def concatenate_audio(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate multiple audio arrays into one.

    Args:
        chunks: List of audio arrays to concatenate.

    Returns:
        Single concatenated audio array.

    Examples:
        >>> import numpy as np
        >>> chunks = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> result = concatenate_audio(chunks)
        >>> result.tolist()
        [1, 2, 3, 4, 5, 6]
        >>> concatenate_audio([]).tolist()
        []
    """
    if not chunks:
        return np.array([])

    return np.concatenate(chunks)


def get_output_format(output_path: Path | str) -> str:
    """Determine the output format from file extension.

    Args:
        output_path: Path to the output file.

    Returns:
        Audio format string.

    Examples:
        >>> get_output_format("output.mp3")
        'mp3'
        >>> get_output_format(Path("audio.m4a"))
        'm4a'
        >>> get_output_format("test.wav")
        'wav'
    """
    path = Path(output_path) if isinstance(output_path, str) else output_path
    return path.suffix.lstrip(".").lower()


def save_audio_wav(
    audio: np.ndarray, output_path: Path, sample_rate: int = 24000
) -> None:
    """Save audio data to a WAV file.

    Args:
        audio: Audio data as numpy array.
        output_path: Path to save the WAV file.
        sample_rate: Sample rate of the audio.

    Raises:
        ValueError: If audio data is empty.

    Examples:
        >>> import numpy as np
        >>> from pathlib import Path
        >>> import tempfile
        >>> audio = np.sin(np.linspace(0, 100, 1000))
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir) / "test.wav"
        ...     save_audio_wav(audio, path)
        ...     path.exists()
        True
    """
    if len(audio) == 0:
        raise ValueError("Cannot save empty audio data")

    sf.write(str(output_path), audio, sample_rate)


def encode_audio(
    input_path: Path,
    output_path: Path,
    format_type: str = "m4a",
    bitrate: str = "128k",
) -> None:
    """Encode audio file to a compressed format using ffmpeg.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the encoded file.
        format_type: Output format ('m4a', 'mp3', 'ogg').
        bitrate: Audio bitrate.

    Raises:
        RuntimeError: If encoding fails.

    Examples:
        >>> # encode_audio(Path("input.wav"), Path("output.m4a"))
    """
    codec = CODECS.get(format_type, "aac")

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i",
        str(input_path),
        "-c:a",
        codec,
        "-b:a",
        bitrate,
        str(output_path),
    ]

    _run_ffmpeg(cmd)


def save_audio(
    audio: np.ndarray,
    output_path: Path | str,
    sample_rate: int = 24000,
    bitrate: str = "128k",
) -> None:
    """Save audio to a file with automatic format detection.

    Supports WAV (direct save) and compressed formats (via ffmpeg).

    Args:
        audio: Audio data as numpy array.
        output_path: Path to save the audio file.
        sample_rate: Sample rate of the audio.
        bitrate: Bitrate for compressed formats.

    Raises:
        ValueError: If audio data is empty.
        RuntimeError: If encoding fails.

    Examples:
        >>> import numpy as np
        >>> from pathlib import Path
        >>> import tempfile
        >>> audio = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir) / "test.wav"
        ...     save_audio(audio, path)
        ...     path.exists()
        True
    """
    if len(audio) == 0:
        raise ValueError("Cannot save empty audio data")

    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    format_type = get_output_format(output_path)

    if format_type == "wav":
        save_audio_wav(audio, output_path, sample_rate)
    else:
        # Save as WAV first, then encode
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_audio_wav(audio, tmp_path, sample_rate)
            encode_audio(tmp_path, output_path, format_type, bitrate)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def get_audio_duration(audio: np.ndarray, sample_rate: int = 24000) -> float:
    """Calculate the duration of audio data in seconds.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate of the audio.

    Returns:
        Duration in seconds.

    Examples:
        >>> import numpy as np
        >>> audio = np.zeros(24000)  # 1 second at 24kHz
        >>> get_audio_duration(audio, 24000)
        1.0
        >>> get_audio_duration(np.zeros(48000), 24000)
        2.0
    """
    if len(audio) == 0:
        return 0.0
    return len(audio) / sample_rate
