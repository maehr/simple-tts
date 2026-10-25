"""Run external tools with cooperative cancellation in conversion workers."""

import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class ConversionCancelled(Exception):
    """The caller cancelled a conversion before publication."""


_cancel_check: ContextVar[Callable[[], None] | None] = ContextVar(
    "cancel_check", default=None
)


@contextmanager
def cancellable_commands(check: Callable[[], None]) -> Iterator[None]:
    """Check cancellation while Pandoc or FFmpeg runs."""
    token = _cancel_check.set(check)
    try:
        yield
    finally:
        _cancel_check.reset(token)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Capture output and stop external work when cancellation arrives."""
    check = _cancel_check.get()
    if check is None:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    check()
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        try:
            while True:
                check()
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            process.terminate()
            try:
                process.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=stdout, stderr=stderr
            )
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
