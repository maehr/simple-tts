"""Text-to-Speech engine interface.

This module provides the TTS synthesis functionality using Kokoro TTS.

Examples:
    >>> from simple_tts.tts import TTSConfig
    >>> config = TTSConfig()
    >>> config.speed
    1.1
"""

from dataclasses import dataclass, field
import warnings

import numpy as np


@dataclass
class TTSConfig:
    """Configuration for TTS synthesis.

    Attributes:
        voice: Voice ID to use for synthesis.
        speed: Playback speed multiplier.
        device: Device to use ('cpu', 'cuda', 'mps').
        sample_rate: Output audio sample rate in Hz.
        repo_id: Kokoro model repository ID (e.g., 'hexgrad/Kokoro-82M').

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
    repo_id: str = "hexgrad/Kokoro-82M"


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
    """Text-to-Speech synthesis engine using Kokoro TTS.

    This class provides an interface for text-to-speech synthesis
    using the Kokoro TTS library.

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
        self._pipeline = None
        self._voice_tensor = None

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
        try:
            from kokoro import KPipeline

            # Voice ID format: {lang_prefix}{gender}_{name}
            # e.g., "af_bella" -> lang_code "a" (American English)
            # Extract language code from voice ID prefix (before underscore)
            voice_prefix = self.config.voice.split("_")[0]
            lang_code = voice_prefix[0] if voice_prefix else "a"

            # Silence known upstream warnings from PyTorch/Kokoro during model init
            warnings.filterwarnings(
                "ignore",
                message=(
                    "dropout option adds dropout after all but last recurrent layer"
                ),
                category=UserWarning,
                module=r"torch\.nn\.modules\.rnn",
            )
            warnings.filterwarnings(
                "ignore",
                message=("`torch.nn.utils.weight_norm` is deprecated in favor of"),
                category=FutureWarning,
                module=r"torch\.nn\.utils\.weight_norm",
            )

            # Prefer explicit repo_id; fall back if not supported by installed kokoro
            try:
                self._pipeline = KPipeline(
                    lang_code=lang_code,
                    repo_id=self.config.repo_id,
                )
            except TypeError:
                self._pipeline = KPipeline(lang_code=lang_code)

            self.is_loaded = True
        except ImportError:
            # Kokoro not installed - use fallback mode for testing
            warnings.warn(
                "Kokoro TTS library not installed. "
                "Install with: pip install kokoro>=0.8.4. "
                "Using silent audio fallback.",
                UserWarning,
                stacklevel=2,
            )
            self.is_loaded = True
            self._pipeline = None

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

        if self._pipeline is not None:
            # Use Kokoro TTS
            audio_segments = []
            generator = self._pipeline(
                text,
                voice=self.config.voice,
                speed=self.config.speed,
            )
            for _, _, audio in generator:
                audio_segments.append(audio)

            if audio_segments:
                audio = np.concatenate(audio_segments)
            else:
                audio = np.array([], dtype=np.float32)
        else:
            # Fallback: generate silence with correct duration estimate
            # This is only used when kokoro is not installed (testing)
            duration = len(text) * 0.05  # ~50ms per character
            num_samples = int(self.config.sample_rate * duration)
            audio = np.zeros(num_samples, dtype=np.float32)

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
        # For Kokoro, passing the voice ID string is sufficient

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
