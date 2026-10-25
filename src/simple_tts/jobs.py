"""Manage one spawned conversion worker and retain job records in memory."""

import asyncio
import multiprocessing
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from simple_tts.processes import ConversionCancelled, cancellable_commands
from simple_tts.service import (
    ConversionRequest,
    ConversionResult,
    Phase,
    ProgressEvent,
    execute_conversion,
    prepare_conversion,
    require_dependencies,
    resolve_output,
    validate_request,
)
from simple_tts.workspace import ScratchDirectory, Workspace

JobState = Literal["running", "cancelling", "succeeded", "failed", "cancelled"]
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
CANCEL_TIMEOUT = 5.0


def require_mcp_platform() -> None:
    """Reject platforms without the file and process controls used by MCP."""
    required = ("O_DIRECTORY", "O_NOFOLLOW", "setsid", "killpg")
    if (
        sys.platform == "win32"
        or os.name != "posix"
        or any(not hasattr(os, name) for name in required)
    ):
        raise RuntimeError(
            "The MCP server requires POSIX support (macOS or Linux). "
            "Windows is not supported. Use 'simple-tts convert' on Windows."
        )


class CancellationEvent(Protocol):
    """The process-safe event interface used by workers."""

    def is_set(self) -> bool:
        """Return whether cancellation was requested."""
        ...

    def set(self) -> None:
        """Request cancellation."""
        ...


@dataclass(frozen=True)
class JobStatus:
    """A snapshot returned by each conversion job tool."""

    job_id: str
    state: JobState = "running"
    phase: Phase = "extracting"
    completed_chunks: int = 0
    total_chunks: int | None = None
    message: str | None = None
    error: str | None = None
    result: ConversionResult | None = None


def conversion_worker(
    request: ConversionRequest,
    workspace: Path,
    scratch: Path,
    staged: Path,
    cancelled: CancellationEvent,
    sender: Connection,
) -> None:
    """Execute a conversion in its own process group with stderr diagnostics."""
    os.setsid()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def check() -> None:
        if cancelled.is_set():
            raise ConversionCancelled

    try:
        with cancellable_commands(check):
            check()
            if request.input_file is not None:
                source_dir = scratch / "input"
                source_dir.mkdir()
                source = source_dir / request.input_file.name
                Workspace(workspace).copy_input(request.input_file, source)
                request = replace(request, input_file=source)
            prepared = prepare_conversion(
                request,
                scratch,
                sandbox=True,
                report=sender.send,
                check_cancelled=check,
            )
            result = execute_conversion(
                prepared, scratch, staged, report=sender.send, check_cancelled=check
            )
            check()
            sender.send(result)
    except ConversionCancelled:
        sender.send("cancelled")
    except Exception as error:
        sender.send(f"{type(error).__name__}: {error}")
    finally:
        sender.close()


@dataclass
class _ActiveJob:
    job_id: str
    process: BaseProcess
    cancelled: CancellationEvent
    receiver: Connection
    directory: ScratchDirectory
    staged: Path
    output: Path
    task: asyncio.Task[None] | None = None
    cancel_time: float | None = None


class JobManager:
    """Own the worker, cancellation deadline, and final publication."""

    def __init__(
        self,
        workspace: Path,
        *,
        worker: Callable = conversion_worker,
    ) -> None:
        """Use spawn so model runtimes never inherit a server process."""
        require_mcp_platform()
        self.workspace = Workspace(workspace)
        self.records: dict[str, JobStatus] = {}
        self._context = multiprocessing.get_context("spawn")
        self._worker = worker
        self._active: _ActiveJob | None = None
        self._closed = False

    def start(self, request: ConversionRequest) -> JobStatus:
        """Validate a request and return before extraction starts."""
        if self._closed:
            raise ValueError("The conversion server is closed.")
        if self._active is not None:
            raise ValueError("The server is busy. Wait for the active conversion.")
        job_id = uuid4().hex
        source = (
            self.workspace.resolve(request.input_file)
            if request.input_file is not None
            else None
        )
        # Resolve the suffix before creating the default directory.
        output = (
            self.workspace.resolve(request.output)
            if request.output is not None
            else None
        )
        resolved_output, format_type = resolve_output(
            Path("text.md"), output, request.format_type
        )
        # Validate options and dependencies without running external tools.
        provisional = replace(
            request,
            input_file=source,
            output=(
                resolved_output
                if output is not None
                else self.workspace.path / f"{job_id}.{format_type}"
            ),
            format_type=format_type,
        )
        validate_request(provisional)
        require_dependencies()
        if output is None:
            output = self.workspace.default_output(job_id, format_type)
        else:
            # Inspect the original output as well as the resolved path.
            self.workspace.check_output(request.output or output)
            output = resolved_output
        output = self.workspace.check_output(output)
        request = replace(provisional, output=output)
        directory = ScratchDirectory(self.workspace, output.parent)
        scratch = Path(directory.name)
        staged = scratch / output.name
        receiver, sender = self._context.Pipe(duplex=False)
        cancelled = self._context.Event()
        process = self._context.Process(
            target=self._worker,
            args=(request, self.workspace.path, scratch, staged, cancelled, sender),
            name=f"simple-tts-{job_id}",
        )
        try:
            process.start()
        except BaseException:
            receiver.close()
            directory.cleanup()
            raise
        finally:
            sender.close()
        active = _ActiveJob(
            job_id, process, cancelled, receiver, directory, staged, output
        )
        self.records[job_id] = JobStatus(job_id)
        self._active = active
        active.task = asyncio.create_task(self._monitor(active))
        return self.records[job_id]

    def get(self, job_id: str) -> JobStatus:
        """Return a retained job snapshot or a request error."""
        if job_id not in self.records:
            raise ValueError(f"Unknown conversion job: {job_id}")
        return self.records[job_id]

    def cancel(self, job_id: str) -> JobStatus:
        """Request cancellation without blocking a tool response."""
        status = self.get(job_id)
        if status.state == "running":
            active = self._active
            assert active is not None and active.job_id == job_id
            active.cancel_time = time.monotonic()
            active.cancelled.set()
            self.records[job_id] = replace(status, state="cancelling")
        return self.records[job_id]

    @staticmethod
    def _stop_group(active: _ActiveJob) -> None:
        # The worker calls setsid before it can launch external processes.
        pid = active.process.pid
        assert pid is not None
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            if active.process.is_alive():
                active.process.kill()

    async def _monitor(self, active: _ActiveJob) -> None:
        outcome: ConversionResult | str | None = None
        try:
            eof = False
            while True:
                while not eof and active.receiver.poll():
                    try:
                        event = active.receiver.recv()
                    except EOFError:
                        eof = True
                        break
                    if isinstance(event, ProgressEvent):
                        self.records[active.job_id] = replace(
                            self.get(active.job_id),
                            phase=event.phase,
                            completed_chunks=event.completed_chunks,
                            total_chunks=event.total_chunks,
                            message=event.message,
                        )
                    else:
                        outcome = event
                if not active.process.is_alive():
                    # Drain any final message after the process exits.
                    if not eof and active.receiver.poll():
                        continue
                    break
                if (
                    active.cancel_time is not None
                    and time.monotonic() - active.cancel_time >= CANCEL_TIMEOUT
                ):
                    self._stop_group(active)
                await asyncio.sleep(0.05)
            active.process.join()
            self._stop_group(active)
            status = self.get(active.job_id)
            if active.cancelled.is_set() or outcome == "cancelled":
                self.records[active.job_id] = replace(status, state="cancelled")
            elif isinstance(outcome, ConversionResult) and active.process.exitcode == 0:
                self.records[active.job_id] = replace(status, phase="publishing")
                # No await separates this cancellation check and atomic publication.
                if active.cancelled.is_set():
                    self.records[active.job_id] = replace(status, state="cancelled")
                else:
                    self.workspace.publish(active.staged, active.output)
                    self.records[active.job_id] = replace(
                        status, state="succeeded", phase="done", result=outcome
                    )
            else:
                error = (
                    outcome
                    if isinstance(outcome, str)
                    else f"Worker exited with code {active.process.exitcode}."
                )
                self.records[active.job_id] = replace(
                    status, state="failed", error=error
                )
        except asyncio.CancelledError:
            active.cancelled.set()
            self._stop_group(active)
            active.process.join()
            self.records[active.job_id] = replace(
                self.get(active.job_id), state="cancelled"
            )
            raise
        except Exception as error:
            self._stop_group(active)
            active.process.join()
            self.records[active.job_id] = replace(
                self.get(active.job_id), state="failed", error=str(error)
            )
        finally:
            active.receiver.close()
            active.process.close()
            try:
                active.directory.cleanup()
            finally:
                self._active = None

    async def close(self) -> None:
        """Cancel active work and remove its temporary files on shutdown."""
        self._closed = True
        active = self._active
        if active is not None:
            self.cancel(active.job_id)
            if active.task is not None:
                await asyncio.shield(active.task)
