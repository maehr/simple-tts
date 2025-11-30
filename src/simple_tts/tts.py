"""Text-to-Speech engine interface.

This module provides the TTS synthesis functionality. It abstracts the
TTS backend to allow for different implementations (e.g., Kokoro, local models).

Examples:
    >>> from simple_tts.tts import TTSConfig
    >>> config = TTSConfig()
    >>> config.speed
    1.1
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TTSConfig:
    """Configuration for TTS synthesis.

    Attributes:
        voice: Voice ID to use for synthesis.
        speed: Playback speed multiplier.
        device: Device to use ('cpu', 'cuda', 'mps').
        sample_rate: Output audio sample rate in Hz.

    Examples:
        >>> config = TTSConfig()
        >>> config.voice
        'af_bella'
        >>> config.speed
        1.1
        >>> config = TTSConfig(speed=1.5, voice="bf_emma")
        >>> config.speed
        1.5
    """

    voice: str = "af_bella"
    speed: float = 1.1
    device: str = "cpu"
    sample_rate: int = 24000


@dataclass
class AudioChunk:
    """Represents a chunk of synthesized audio.

    Attributes:
        data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        duration: Duration in seconds.

    Examples:
        >>> import numpy as np
        >>> chunk = AudioChunk(data=np.zeros(24000), sample_rate=24000)
        >>> chunk.duration
        1.0
    """

    data: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_rate: int = 24000

    @property
    def duration(self) -> float:
        """Calculate the duration of the audio chunk in seconds.

        Returns:
            Duration in seconds.

        Examples:
            >>> import numpy as np
            >>> chunk = AudioChunk(data=np.zeros(48000), sample_rate=24000)
            >>> chunk.duration
            2.0
        """
        if len(self.data) == 0:
            return 0.0
        return len(self.data) / self.sample_rate


class TTSEngine:
    """Text-to-Speech synthesis engine.

    This class provides an interface for text-to-speech synthesis.
    The actual TTS backend (Kokoro) would be loaded here in production.

    Attributes:
        config: TTS configuration settings.
        is_loaded: Whether the model is loaded.

    Examples:
        >>> engine = TTSEngine()
        >>> engine.is_loaded
        False
        >>> engine.config.voice
        'af_bella'
    """

    def __init__(self, config: TTSConfig | None = None):
        """Initialize the TTS engine.

        Args:
            config: Optional TTS configuration. Uses defaults if not provided.

        Examples:
            >>> engine = TTSEngine()
            >>> engine.config.speed
            1.1
            >>> custom_config = TTSConfig(speed=1.5)
            >>> engine2 = TTSEngine(config=custom_config)
            >>> engine2.config.speed
            1.5
        """
        self.config = config or TTSConfig()
        self.is_loaded = False
        self._model = None

    def load(self) -> None:
        """Load the TTS model.

        Loads the Kokoro TTS model into memory. This should be called
        before synthesizing audio.

        Examples:
            >>> engine = TTSEngine()
            >>> engine.is_loaded
            False
            >>> engine.load()
            >>> engine.is_loaded
            True
        """
        # In production, this would load the Kokoro model
        # For now, we just mark as loaded for the interface
        self.is_loaded = True

    def synthesize(self, text: str) -> AudioChunk:
        """Synthesize speech from text.

        Converts the input text to audio using the configured voice.

        Args:
            text: The text to synthesize.

        Returns:
            AudioChunk containing the synthesized audio.

        Raises:
            RuntimeError: If the model is not loaded.

        Examples:
            >>> engine = TTSEngine()
            >>> engine.load()
            >>> chunk = engine.synthesize("Hello world")
            >>> chunk.sample_rate
            24000
        """
        if not self.is_loaded:
            raise RuntimeError("TTS model not loaded. Call load() first.")

        # In production, this would use Kokoro for synthesis
        # For now, return a placeholder audio chunk
        # Generate a simple sine wave for testing
        duration = len(text) * 0.05  # ~50ms per character
        t = np.linspace(0, duration, int(self.config.sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone

        return AudioChunk(
            data=audio.astype(np.float32), sample_rate=self.config.sample_rate
        )

    def synthesize_stream(self, chunks: list[str]) -> list[AudioChunk]:
        """Synthesize speech from multiple text chunks.

        Processes multiple text chunks and returns audio for each.

        Args:
            chunks: List of text chunks to synthesize.

        Returns:
            List of AudioChunks, one for each input chunk.

        Raises:
            RuntimeError: If the model is not loaded.

        Examples:
            >>> engine = TTSEngine()
            >>> engine.load()
            >>> chunks = engine.synthesize_stream(["Hello", "World"])
            >>> len(chunks)
            2
        """
        if not self.is_loaded:
            raise RuntimeError("TTS model not loaded. Call load() first.")

        return [self.synthesize(chunk) for chunk in chunks]

    def set_voice(self, voice: str) -> None:
        """Set the voice for synthesis.

        Args:
            voice: Voice ID to use.

        Examples:
            >>> engine = TTSEngine()
            >>> engine.config.voice
            'af_bella'
            >>> engine.set_voice("bf_emma")
            >>> engine.config.voice
            'bf_emma'
        """
        self.config.voice = voice

    def set_speed(self, speed: float) -> None:
        """Set the playback speed.

        Args:
            speed: Speed multiplier (1.0 = normal, 1.5 = 50% faster).

        Examples:
            >>> engine = TTSEngine()
            >>> engine.config.speed
            1.1
            >>> engine.set_speed(1.5)
            >>> engine.config.speed
            1.5
        """
        self.config.speed = speed
