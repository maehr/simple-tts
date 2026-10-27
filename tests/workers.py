"""Importable worker targets for spawn-based integration tests."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np

from simple_tts import service
from simple_tts.jobs import conversion_worker
from simple_tts.processes import run_command


def use_fake_ptts():
    """Install the test ptts module, so no worker reads the real catalogue."""
    from tests.conftest import fake_ptts

    module = fake_ptts()
    sys.modules["ptts"] = module
    return module


def one_frame(synthesize):
    """Stream the array that ``synthesize(model, text, voice)`` returns."""

    @contextmanager
    def stream(model, text, *, voice):
        yield iter([synthesize(model, text, voice)])

    return stream


def mock_worker(request, workspace, scratch, staged, cancelled, sender):
    """Use real extraction and encoding with deterministic synthesis."""
    use_fake_ptts()

    def load(*, language, device, duration_delta, report):
        print("Worker diagnostic", flush=True)
        report(f"Device: {device}")
        return object()

    def synthesize(model, text, voice):
        return np.full(2400, 0.1, dtype=np.float32)

    service.load_tts = load
    service.stream_speech = one_frame(synthesize)
    conversion_worker(request, workspace, scratch, staged, cancelled, sender)


def _engine_worker(devices, request, workspace, scratch, staged, cancelled, sender):
    """Run the real engine with a fake ptts and record model loads and voices."""
    from tests.conftest import FakeTTS

    module = use_fake_ptts()
    module.available_devices.return_value = devices
    models = []

    def load(*args, **kwargs):
        models.append(FakeTTS(*args, **kwargs))
        return models[-1]

    module.TTS.side_effect = load
    try:
        conversion_worker(request, workspace, scratch, staged, cancelled, sender)
    finally:
        loads = [call.kwargs for call in module.TTS.call_args_list]
        voices = [voice for model in models for _, voice, _ in model.streams]
        (workspace / "ptts.json").write_text(
            json.dumps({"loads": loads, "voices": voices})
        )


def engine_worker(request, workspace, scratch, staged, cancelled, sender):
    """Use a ptts build that offers only the CPU."""
    _engine_worker(["cpu"], request, workspace, scratch, staged, cancelled, sender)


def metal_engine_worker(request, workspace, scratch, staged, cancelled, sender):
    """Use a ptts build that offers Metal and the CPU."""
    _engine_worker(
        ["metal", "cpu"], request, workspace, scratch, staged, cancelled, sender
    )


def failing_worker(request, workspace, scratch, staged, cancelled, sender):
    """Fail after writing partial audio so cleanup can be checked."""
    use_fake_ptts()

    def fail(*args, **kwargs):
        (scratch / "partial.f32le").write_bytes(b"partial")
        raise RuntimeError("Mock model failure")

    service.load_tts = fail
    conversion_worker(request, workspace, scratch, staged, cancelled, sender)


def crashing_worker(request, workspace, scratch, staged, cancelled, sender):
    """Exit without a result to simulate an abrupt worker failure."""
    os.setsid()
    (scratch / "partial").touch()
    os._exit(7)


def slow_worker(request, workspace, scratch, staged, cancelled, sender):
    """Stream frames without end, so only cancellation can stop synthesis."""
    use_fake_ptts()
    service.load_tts = lambda **kwargs: object()

    def frames():
        while True:
            time.sleep(0.01)
            yield np.zeros(240, dtype=np.float32)

    @contextmanager
    def stream(*args, **kwargs):
        yield frames()

    service.stream_speech = stream
    conversion_worker(request, workspace, scratch, staged, cancelled, sender)


def external_worker(request, workspace, scratch, staged, cancelled, sender):
    """Block extraction in an external process that ignores SIGTERM."""
    use_fake_ptts()

    def extract(*args, **kwargs):
        run_command(
            [
                sys.executable,
                "-c",
                "import os, signal, time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "Path('child.pid').write_text(str(os.getpid())); time.sleep(60)",
            ]
        )

    os.chdir(scratch)
    service.extract_sections = extract
    conversion_worker(request, workspace, scratch, staged, cancelled, sender)


def stubborn_worker(request, workspace, scratch, staged, cancelled, sender):
    """Ignore cancellation with a child in the worker process group."""
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (scratch / "child.pid").write_text(str(child.pid))
    sender.send(service.ProgressEvent("loading"))
    time.sleep(60)


def publication_worker(request, workspace, scratch, staged, cancelled, sender):
    """Prepare a completed result and wait for a publication race trigger."""
    os.setsid()
    descriptor = os.open(scratch, os.O_RDONLY | os.O_DIRECTORY)
    staged.write_bytes(b"complete audio")
    sender.send(service.ProgressEvent("encoding", 1, 1))
    (scratch / "ready").touch()
    while True:
        try:
            os.stat("release", dir_fd=descriptor)
            break
        except FileNotFoundError:
            time.sleep(0.01)
    os.close(descriptor)
    sender.send(
        service.ConversionResult(
            str(request.output), request.output.as_uri(), "m4a", 1.0, 0.1, []
        )
    )


def stdio_worker(request, workspace, scratch, staged, cancelled, sender):
    """Select a mock conversion or a long job for stdio lifecycle tests."""
    (scratch / "worker.pid").write_text(str(os.getpid()))
    worker = slow_worker if request.text == "Wait for shutdown." else mock_worker
    worker(request, workspace, scratch, staged, cancelled, sender)


def confined_worker(request, workspace, scratch, staged, cancelled, sender):
    """Fail if any secondary file contents reach narration in a real worker."""
    original_extract = service.extract_sections

    def extract(*args, **kwargs):
        document = original_extract(*args, **kwargs)
        assert all(
            "Confidential outside narration." not in section.text
            for section in document.sections
        ), "Pandoc disclosed a secondary file."
        return document

    service.extract_sections = extract
    mock_worker(request, workspace, scratch, staged, cancelled, sender)
