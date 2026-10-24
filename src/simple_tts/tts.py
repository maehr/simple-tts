"""Text-to-speech synthesis with explicit device selection and CPU retry."""

import os
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

# PyTorch reads this setting at import time. Keep both heavy imports lazy.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

SAMPLE_RATE = 24000
DEVICES = {"auto", "cuda", "mps", "cpu"}


class SpeechPipeline(Protocol):
    """The callable interface of a Kokoro pipeline."""

    def __call__(self, text: str, *, voice: str, speed: float) -> Iterable[Any]:
        """Yield graphemes, phonemes, and audio segments."""
        ...


def _create_pipeline(voice: str, device: str) -> SpeechPipeline:
    from kokoro import KPipeline

    with warnings.catch_warnings():
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
        return KPipeline(
            lang_code=voice[0] if voice else "a",
            repo_id="hexgrad/Kokoro-82M",
            device=device,
        )


@dataclass
class TTSPipeline:
    """A Kokoro pipeline with device state for one conversion."""

    pipeline: SpeechPipeline | None
    voice: str
    requested_device: str
    device: str
    report: Callable[[str], None]

    def __call__(self, text: str, *, voice: str, speed: float):
        """Keep the Kokoro callable interface available to callers."""
        if self.pipeline is None:
            raise RuntimeError("The TTS model is not loaded.")
        return self.pipeline(text, voice=voice, speed=speed)


def load_tts(
    voice: str = "af_bella",
    device: str = "auto",
    *,
    report: Callable[[str], None] = print,
) -> TTSPipeline:
    """Load Kokoro on CUDA, MPS, or CPU and report the selected device.

    Auto mode retries GPU initialization on CPU once. Explicit GPU requests
    fail if the device is unavailable or initialization fails.
    """
    if device not in DEVICES:
        raise ValueError("Device must be auto, cuda, mps, or cpu.")
    # Also set it here for callers that change their environment after import.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch is missing. Run: pip install -U simple-tts") from e

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    selected = device
    if device == "auto":
        selected = next(name for name, ready in available.items() if ready)
    elif not available[device]:
        raise RuntimeError(
            f"Requested device '{device}' is unavailable. Use --device auto or cpu."
        )
    try:
        pipeline = _create_pipeline(voice, selected)
    except ImportError as e:
        raise RuntimeError(
            "Kokoro or a required package is missing. Run: pip install -U simple-tts "
            "(requires kokoro>=0.9.4)."
        ) from e
    except Exception as e:
        if device != "auto" or selected == "cpu":
            raise RuntimeError(f"Failed to initialize TTS on {selected}: {e}") from e
        report(f"Device {selected} initialization failed: {e}. Retry on CPU.")
        selected = "cpu"
        try:
            pipeline = _create_pipeline(voice, selected)
        except Exception as cpu_error:
            raise RuntimeError(
                f"Failed to initialize TTS on CPU: {cpu_error}"
            ) from cpu_error

    report(f"Device: {selected}")
    if selected == "mps":
        report("MPS operations can fall back to CPU when unsupported.")
    elif device == "auto" and not available["cuda"] and not available["mps"]:
        report("No GPU is available. Use CPU.")
    return TTSPipeline(pipeline, voice, device, selected, report)


def _synthesize(
    pipeline: SpeechPipeline, text: str, voice: str, speed: float
) -> np.ndarray:
    segments = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is None:
            continue
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        segments.append(np.asarray(audio, dtype=np.float32))
    if segments:
        return np.concatenate(segments).astype(np.float32, copy=False)
    return np.array([], dtype=np.float32)


def synthesize(
    pipeline: SpeechPipeline | None,
    text: str,
    voice: str = "af_bella",
    speed: float = 1.1,
) -> np.ndarray:
    """Complete a chunk before returning audio, with one auto-mode CPU retry.

    A failed GPU chunk contributes no samples. After a CPU retry, subsequent
    chunks use CPU. Array storage stays bounded by the input chunk size.
    """
    if pipeline is None:
        raise RuntimeError("No Kokoro model is loaded. Run: pip install -U simple-tts")
    try:
        return _synthesize(pipeline, text, voice, speed)
    except Exception as e:
        if not isinstance(pipeline, TTSPipeline):
            raise
        if pipeline.requested_device != "auto" or pipeline.device == "cpu":
            raise RuntimeError(f"Synthesis failed on {pipeline.device}: {e}") from e
        pipeline.report(
            f"Device {pipeline.device} synthesis failed: {e}. Retry this chunk on CPU."
        )
    # Leave the exception scope first so its traceback releases partial audio.
    pipeline.pipeline = None
    pipeline.device = "cpu"
    try:
        pipeline.pipeline = _create_pipeline(pipeline.voice, "cpu")
        pipeline.report("Device: cpu")
        return _synthesize(pipeline, text, voice, speed)
    except Exception as e:
        raise RuntimeError(f"CPU retry failed: {e}") from e


def synthesize_chunks(
    pipeline: SpeechPipeline | None,
    chunks: list[str],
    voice: str = "af_bella",
    speed: float = 1.1,
) -> list[np.ndarray]:
    """Return audio arrays for library callers that need all chunks in memory."""
    return [synthesize(pipeline, chunk, voice=voice, speed=speed) for chunk in chunks]
