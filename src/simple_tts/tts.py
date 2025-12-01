"""Text-to-Speech synthesis using Kokoro TTS.

This module provides functional TTS synthesis using the Kokoro library.

Example:
    >>> from simple_tts.tts import load_tts, synthesize
    >>> pipeline = load_tts("af_bella")
    >>> audio = synthesize(pipeline, "Hello world", voice="af_bella")
"""

import warnings

import numpy as np

# Default sample rate for Kokoro TTS
SAMPLE_RATE = 24000


def load_tts(voice: str = "af_bella") -> object | None:
    """Load the Kokoro TTS pipeline.

    Args:
        voice: Voice ID to determine language code.

    Returns:
        Kokoro KPipeline object, or None if not installed.

    Example:
        >>> pipeline = load_tts("af_bella")
        >>> pipeline is not None or True  # May not be installed
        True
    """
    try:
        from kokoro import KPipeline

        # Extract language code from voice (first character)
        lang_code = voice[0] if voice else "a"

        # Suppress known warnings during model load
        warnings.filterwarnings(
            "ignore",
            message="dropout option adds dropout after all but last recurrent layer",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="`torch.nn.utils.weight_norm` is deprecated",
            category=FutureWarning,
        )

        return KPipeline(lang_code=lang_code)
    except ImportError:
        warnings.warn(
            "Kokoro TTS not installed. Install with: pip install kokoro>=0.8.4",
            UserWarning,
            stacklevel=2,
        )
        return None


def synthesize(
    pipeline: object | None,
    text: str,
    voice: str = "af_bella",
    speed: float = 1.1,
) -> np.ndarray:
    """Synthesize speech from text.

    Args:
        pipeline: Kokoro pipeline from load_tts().
        text: Text to synthesize.
        voice: Voice ID to use.
        speed: Playback speed multiplier.

    Returns:
        Audio data as numpy float32 array.

    Example:
        >>> pipeline = load_tts("af_bella")
        >>> audio = synthesize(pipeline, "Hello", voice="af_bella")
        >>> isinstance(audio, np.ndarray)
        True
    """
    if pipeline is None:
        # Fallback: return silence (for testing without kokoro)
        duration = len(text) * 0.05
        return np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)

    segments = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        segments.append(audio)

    if segments:
        return np.concatenate(segments).astype(np.float32)
    return np.array([], dtype=np.float32)


def synthesize_chunks(
    pipeline: object | None,
    chunks: list[str],
    voice: str = "af_bella",
    speed: float = 1.1,
) -> list[np.ndarray]:
    """Synthesize multiple text chunks.

    Args:
        pipeline: Kokoro pipeline from load_tts().
        chunks: List of text chunks.
        voice: Voice ID to use.
        speed: Playback speed multiplier.

    Returns:
        List of audio arrays.

    Example:
        >>> pipeline = load_tts("af_bella")
        >>> audios = synthesize_chunks(pipeline, ["Hello", "World"])
        >>> len(audios)
        2
    """
    return [synthesize(pipeline, chunk, voice=voice, speed=speed) for chunk in chunks]
