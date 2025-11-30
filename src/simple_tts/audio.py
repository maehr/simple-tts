"""Audio processing utilities for output handling.

This module provides functions for audio processing, including
stitching audio chunks and encoding to various formats using ffmpeg.

Examples:
    >>> from simple_tts.audio import get_output_format
    >>> get_output_format("output.mp3")
    'mp3'
"""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


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
    codec_map = {
        "m4a": "aac",
        "mp3": "libmp3lame",
        "ogg": "libvorbis",
        "opus": "libopus",
    }

    codec = codec_map.get(format_type, "aac")

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

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to encode audio: {e.stderr}") from e
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.") from e


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
        import tempfile

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
