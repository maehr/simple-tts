"""Verify cancellation and captured output from external tools."""

import sys

import pytest

from simple_tts.processes import (
    ConversionCancelled,
    cancellable_commands,
    run_command,
)


@pytest.mark.parametrize("exit_code", [0, 3])
def test_cancellable_tool_output_and_errors(exit_code):
    """External tools retain their output and exit status."""
    import subprocess

    command = [sys.executable, "-c", f"print('diagnostic'); exit({exit_code})"]
    with cancellable_commands(lambda: None):
        if exit_code:
            with pytest.raises(subprocess.CalledProcessError) as error:
                run_command(command)
            assert error.value.returncode == 3
            assert error.value.output == "diagnostic\n"
        else:
            assert run_command(command).stdout == "diagnostic\n"


def test_cancelled_command_stops_and_context_resets():
    """Polling stops slow external work and restores the default runner."""
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise ConversionCancelled

    with cancellable_commands(check), pytest.raises(ConversionCancelled):
        run_command([sys.executable, "-c", "import time; time.sleep(60)"])
    assert run_command([sys.executable, "-c", "print('ready')"]).stdout == "ready\n"
