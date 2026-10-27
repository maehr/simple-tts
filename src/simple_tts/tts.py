"""Phonon 40M Multilingual synthesis with explicit device and weight selection.

Each simple-tts device maps to a ptts device and a weight format. 8-bit
weights run only on the CPU, so GPU devices use f32 weights.

Example:
    >>> BACKENDS["auto"], BACKENDS["mps"]
    (('cpu', 'q8'), ('metal', 'f32'))
"""

import functools
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from simple_tts.voices import LANGUAGES

if TYPE_CHECKING:
    from ptts import TTS, AudioStream

SAMPLE_RATE = 24000
DEVICES = {"auto", "cuda", "mps", "cpu"}
# Auto selects the CPU deliberately: it was faster than Metal in benchmarks.
BACKENDS = {
    "auto": ("cpu", "q8"),
    "cpu": ("cpu", "q8"),
    "mps": ("metal", "f32"),
    "cuda": ("cuda", "f32"),
}
MODEL_REPO = "gradium/phonon-40m-multilingual"
MODEL_REVISION = "1537ce46c64f42903e0d608b132ccb073dac46d2"
MODEL_SOURCE = f"hf://{MODEL_REPO}@{MODEL_REVISION}"
WEIGHT_FILES = {"q8": "model.q8.gguf", "f32": "model.safetensors"}
DURATION_DELTA_RANGE = (-0.3, 0.3)
# Six threads were fastest on the tested Apple M5 Pro. More threads were slower.
MAX_THREADS = 6


def hub_cache() -> Path:
    """Return the Hugging Face Hub cache directory that ptts uses."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if value := os.environ.get(name):
            return Path(value).expanduser()
    if value := os.environ.get("HF_HOME"):
        return Path(value).expanduser() / "hub"
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache).expanduser() / "huggingface" / "hub"


def model_source(quant: str) -> str:
    """Select a local checkpoint first, then the pinned Hub revision.

    A pinned Hub load sends one request on every load, also when the files
    are cached. A complete cached snapshot therefore loads as a local path.
    Set ``SIMPLE_TTS_MODEL`` to use another local checkpoint.
    """
    if local := os.environ.get("SIMPLE_TTS_MODEL"):
        return os.path.expanduser(local)
    snapshot = (
        hub_cache()
        / f"models--{MODEL_REPO.replace('/', '--')}"
        / "snapshots"
        / MODEL_REVISION
    )
    required = ("config.json", "tokenizer.json", WEIGHT_FILES[quant])
    if all((snapshot / name).is_file() for name in required):
        return str(snapshot)
    return MODEL_SOURCE


@functools.cache
def _set_threads() -> None:
    # ptts sizes its thread pool at the first model load and ignores later calls.
    import ptts

    ptts.set_num_threads(min(MAX_THREADS, os.cpu_count() or 1))


def _phonon_error(error: Exception) -> RuntimeError:
    """Explain a ptts error without credentials."""
    if isinstance(error, NotImplementedError):
        return RuntimeError(f"{error}. Use --device auto or cpu.")
    if isinstance(error, LookupError):
        return RuntimeError(f"Phonon cannot find a checkpoint or voice: {error}")
    if "GatedRepo" in str(error):
        return RuntimeError(
            f"No access to the gated model {MODEL_REPO}. Accept its terms at "
            f"https://huggingface.co/{MODEL_REPO}, then run: hf auth login"
        )
    return RuntimeError(
        f"Phonon cannot download a required file: {error} "
        "Connect to the internet once to cache the model and voices."
    )


def load_tts(
    *,
    language: str,
    device: str = "auto",
    duration_delta: float = 0.0,
    report: Callable[[str], None] = print,
) -> "TTS":
    """Load Phonon 40M Multilingual once for a conversion.

    Auto and CPU use 8-bit weights on the CPU, also on Macs with Metal.
    MPS uses f32 weights on Metal, and CUDA uses f32 weights on CUDA.
    Explicit GPU requests fail when the ptts build lacks the device.

    Args:
        language: Narration language code from ``LANGUAGES``.
        device: One of auto, cpu, mps, or cuda.
        duration_delta: Rate condition. Negative values speak faster.
        report: Receives the model source, device, and weight format.

    Returns:
        The loaded ptts model.

    Raises:
        ValueError: An option is invalid.
        RuntimeError: ptts, the device, the model, or model access is missing.
    """
    if language not in LANGUAGES:
        raise ValueError(f"Language must be one of: {', '.join(LANGUAGES)}.")
    if device not in DEVICES:
        raise ValueError("Device must be auto, cuda, mps, or cpu.")
    low, high = DURATION_DELTA_RANGE
    if not math.isfinite(duration_delta) or not low <= duration_delta <= high:
        raise ValueError(f"Duration delta must be from {low} through {high}.")
    try:
        import ptts
    except ImportError as e:
        raise RuntimeError("ptts is missing. Run: pip install -U simple-tts") from e

    backend, quant = BACKENDS[device]
    if backend not in ptts.available_devices():
        raise RuntimeError(
            f"Requested device '{device}' is unavailable in this ptts build. "
            "Use --device auto or cpu."
        )
    _set_threads()
    source = model_source(quant)
    report(f"Model: {source}")
    try:
        model = ptts.TTS(
            source,
            device=backend,
            quant=quant,
            lang=language,
            conditions={"duration_delta": duration_delta},
        )
    except (LookupError, OSError, NotImplementedError) as e:
        raise _phonon_error(e) from e
    if language not in model.languages:
        raise RuntimeError(f"The model does not support language '{language}'.")
    if model.sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"The model produces {model.sample_rate} Hz audio, "
            f"but simple-tts requires {SAMPLE_RATE} Hz."
        )
    report(f"Device: {model.device}, quantization: {model.quant}")
    return model


def _frames(stream: "AudioStream") -> Iterator[np.ndarray]:
    for frame in stream:
        audio = np.asarray(frame, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Phonon returned audio that is not mono.")
        yield audio


@contextmanager
def stream_speech(
    model: "TTS", text: str, *, voice: str
) -> Iterator[Iterator[np.ndarray]]:
    """Stream one text chunk as mono float32 frames.

    Leaving the block closes the stream and stops generation, also after an
    error or cancellation. Empty text yields no frames, because ptts rejects it.

    Example:
        with stream_speech(model, "Hello.", voice="Zoey") as frames:
            for frame in frames:
                writer.write(frame)
    """
    if not text.strip():
        yield iter(())
        return
    # ptts resolves and downloads the voice here, before the first frame.
    try:
        stream = model.stream(text, voice=voice)
    except (LookupError, OSError) as e:
        raise _phonon_error(e) from e
    with stream:
        yield _frames(stream)
