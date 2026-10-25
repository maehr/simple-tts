"""Importable worker targets for spawn-based integration tests."""

import os
import signal
import subprocess
import sys
import time

import numpy as np

from simple_tts import service
from simple_tts.jobs import conversion_worker
from simple_tts.processes import run_command


def mock_worker(request, workspace, scratch, staged, cancelled, sender):
    """Use real extraction and encoding with deterministic synthesis."""

    def load(voice, device, report):
        print("Worker diagnostic", flush=True)
        report(f"Device: {device}")
        return object()

    def synthesize(pipeline, text, voice, speed):
        return np.full(2400, 0.1 if voice == "af_bella" else 0.2, dtype=np.float32)

    service.load_tts = load
    service.synthesize = synthesize
    conversion_worker(request, workspace, scratch, staged, cancelled, sender)


def failing_worker(request, workspace, scratch, staged, cancelled, sender):
    """Fail after writing partial audio so cleanup can be checked."""

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
    """Wait inside synthesis until cancellation is requested."""
    service.load_tts = lambda *args, **kwargs: object()

    def synthesize(*args, **kwargs):
        while not cancelled.is_set():
            time.sleep(0.01)
        return np.zeros(2400, dtype=np.float32)

    service.synthesize = synthesize
    conversion_worker(request, workspace, scratch, staged, cancelled, sender)


def external_worker(request, workspace, scratch, staged, cancelled, sender):
    """Block extraction in an external process that ignores SIGTERM."""

    def extract(*args):
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
