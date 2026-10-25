"""Exercise spawned workers, cancellation, workspace limits, and publication."""

import asyncio
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from simple_tts import jobs
from simple_tts.jobs import JobManager
from simple_tts.service import ConversionRequest
from simple_tts.workspace import Workspace
from tests.workers import (
    confined_worker,
    crashing_worker,
    external_worker,
    failing_worker,
    mock_worker,
    publication_worker,
    slow_worker,
    stubborn_worker,
)


async def wait_for(manager, job_id, predicate, timeout=15):
    """Poll without blocking the server event loop."""
    async with asyncio.timeout(timeout):
        while not predicate(manager.get(job_id)):
            await asyncio.sleep(0.02)
    return manager.get(job_id)


async def finish(manager, job_id):
    """Wait for a terminal job state."""
    return await wait_for(
        manager, job_id, lambda job: job.state in jobs.TERMINAL_STATES
    )


def assert_clean(workspace):
    """No conversion scratch directories remain after completion."""
    assert not list(workspace.rglob(".simple-tts-*"))


@pytest.mark.parametrize("source_kind", ["document", "text"])
def test_conversion_results_and_cleanup(tmp_path, source_kind):
    """Documents and Markdown text produce local audio and chapter metadata."""
    text = "# First\n\nHello.\n\n# Second\n\nGoodbye."
    if source_kind == "document":
        (tmp_path / "book.md").write_text(text)
        source = {"input_file": Path("book.md")}
    else:
        source = {"text": text}

    async def scenario():
        manager = JobManager(tmp_path, worker=mock_worker)
        try:
            start = manager.start(
                ConversionRequest(
                    **source, voices=("af_bella", "am_adam"), format_type="m4b"
                )
            )
            assert start.state == "running"
            assert start.total_chunks is None
            done = await finish(manager, start.job_id)
            assert done.state == "succeeded", done.error
            assert done.completed_chunks == done.total_chunks == 2
            result = done.result
            output = tmp_path / ".simple-tts" / "audio" / f"{start.job_id}.m4b"
            assert result.path == str(output)
            assert result.uri == output.as_uri()
            assert result.format == "m4b"
            assert result.duration == 0.2
            assert result.processing_time > 0
            assert [(c.title, c.start, c.end) for c in result.chapters] == [
                ("First", 0.0, 0.1),
                ("Second", 0.1, 0.2),
            ]
            assert output.stat().st_size > 0
            assert manager.cancel(start.job_id) == done
            assert_clean(tmp_path)
        finally:
            await manager.close()
        assert output.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("output_name", "format_type", "expected"),
    [
        ("named", None, "named.m4a"),
        ("named", "wav", "named.wav"),
        ("named.MP4", None, "named.MP4"),
    ],
)
def test_custom_output_resolution(tmp_path, output_name, format_type, expected):
    """Custom outputs infer formats or gain the selected format suffix."""

    async def scenario():
        manager = JobManager(tmp_path, worker=mock_worker)
        try:
            start = manager.start(
                ConversionRequest(
                    text="Hello.", output=Path(output_name), format_type=format_type
                )
            )
            done = await finish(manager, start.job_id)
            assert done.state == "succeeded", done.error
            assert done.result.path == str(tmp_path / expected)
            assert (tmp_path / expected).exists()
            assert_clean(tmp_path)
        finally:
            await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("worker", [failing_worker, crashing_worker])
def test_worker_failures_keep_records_and_clean_files(tmp_path, worker):
    """Worker exceptions and abrupt exits appear in retained job status."""

    async def scenario():
        manager = JobManager(tmp_path, worker=worker)
        try:
            start = manager.start(ConversionRequest(text="Hello."))
            done = await finish(manager, start.job_id)
            assert done.state == "failed"
            assert done.error
            assert done.result is None
            assert not list(tmp_path.rglob("*.m4a"))
            assert_clean(tmp_path)
        finally:
            await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_source_is_a_request_error(tmp_path, text):
    """Empty text never starts a worker."""
    manager = JobManager(tmp_path)
    with pytest.raises(ValueError, match="no narration text"):
        manager.start(ConversionRequest(text=text))
    assert not manager.records


def test_empty_narration_is_a_worker_error(tmp_path):
    """Documents with no narration fail after asynchronous extraction."""
    (tmp_path / "empty.md").touch()

    async def scenario():
        manager = JobManager(tmp_path, worker=mock_worker)
        try:
            start = manager.start(ConversionRequest(input_file=Path("empty.md")))
            done = await finish(manager, start.job_id)
            assert done.state == "failed"
            assert "no narration text" in done.error
            assert_clean(tmp_path)
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_busy_cancel_and_shutdown(tmp_path):
    """Cancellation stays responsive and shutdown cancels active work."""

    async def scenario():
        manager = JobManager(tmp_path, worker=slow_worker)
        first = manager.start(ConversionRequest(text="Hello."))
        await wait_for(manager, first.job_id, lambda job: job.phase == "synthesizing")
        with pytest.raises(ValueError, match="busy"):
            manager.start(ConversionRequest(text="Another."))
        assert manager.cancel(first.job_id).state == "cancelling"
        assert manager.cancel(first.job_id).state == "cancelling"
        assert (await finish(manager, first.job_id)).state == "cancelled"
        assert_clean(tmp_path)
        second = manager.start(ConversionRequest(text="Goodbye."))
        await wait_for(manager, second.job_id, lambda job: job.phase == "synthesizing")
        await manager.close()
        assert manager.get(second.job_id).state == "cancelled"
        assert_clean(tmp_path)
        assert len(manager.records) == 2
        with pytest.raises(ValueError, match="closed"):
            manager.start(ConversionRequest(text="Again."))

    asyncio.run(scenario())


@pytest.mark.parametrize("worker", [external_worker, stubborn_worker])
def test_cancellation_stops_external_processes(tmp_path, monkeypatch, worker):
    """Cooperative and forced cancellation stop external work and clean scratch."""
    monkeypatch.setattr(jobs, "CANCEL_TIMEOUT", 0.8)

    async def scenario():
        manager = JobManager(tmp_path, worker=worker)
        try:
            start = manager.start(ConversionRequest(text="Hello."))
            active = manager._active
            child_path = Path(active.directory.name) / "child.pid"
            async with asyncio.timeout(10):
                while not child_path.exists():
                    await asyncio.sleep(0.02)
            child_pid = int(child_path.read_text())
            cancelled_at = time.monotonic()
            manager.cancel(start.job_id)
            done = await finish(manager, start.job_id)
            assert done.state == "cancelled"
            assert time.monotonic() - cancelled_at < 3
            # A killed orphan can briefly remain as a zombie until reaped.
            import subprocess

            async with asyncio.timeout(3):
                while True:
                    state = subprocess.run(
                        ["ps", "-o", "stat=", "-p", str(child_pid)],
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    if not state or state.startswith("Z"):
                        break
                    await asyncio.sleep(0.02)
            assert_clean(tmp_path)
        finally:
            await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("race", ["collision", "symlink", "cancel"])
def test_publication_races(tmp_path, race):
    """New outputs, escaped parents, and cancellation prevent publication."""
    output_dir = tmp_path / "audio"
    output_dir.mkdir()
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir()

    async def scenario():
        manager = JobManager(tmp_path, worker=publication_worker)
        try:
            start = manager.start(
                ConversionRequest(text="Hello.", output=Path("audio/result.m4a"))
            )
            scratch = Path(manager._active.directory.name)
            async with asyncio.timeout(10):
                while not (scratch / "ready").exists():
                    await asyncio.sleep(0.02)
            if race == "collision":
                (output_dir / "result.m4a").write_bytes(b"existing")
            elif race == "symlink":
                output_dir.rename(tmp_path / "original-audio")
                output_dir.symlink_to(outside, target_is_directory=True)
                scratch = tmp_path / "original-audio" / scratch.name
            else:
                manager.cancel(start.job_id)
            (scratch / "release").touch()
            done = await finish(manager, start.job_id)
            assert done.state == ("cancelled" if race == "cancel" else "failed")
            if race == "collision":
                assert (output_dir / "result.m4a").read_bytes() == b"existing"
            assert not list(outside.iterdir())
            assert_clean(tmp_path)
        finally:
            await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kind", ["relative", "absolute", "input_symlink", "output_symlink"]
)
def test_workspace_escapes(tmp_path, kind):
    """Absolute paths, traversal, and symlinks cannot leave the workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("Hello.")
    options = {"text": "Hello."}
    if kind == "relative":
        options = {"input_file": Path("../outside.md")}
    elif kind == "absolute":
        options = {"input_file": outside}
    elif kind == "input_symlink":
        (workspace / "source.md").symlink_to(outside)
        options = {"input_file": Path("source.md")}
    else:
        (workspace / "audio").symlink_to(tmp_path, target_is_directory=True)
        options["output"] = Path("audio/result.m4a")
    manager = JobManager(workspace)
    with pytest.raises(ValueError, match="escapes the workspace"):
        manager.start(ConversionRequest(**options))


def test_output_and_workspace_errors(tmp_path):
    """Output collisions, absent parents, and invalid workspaces fail early."""
    with pytest.raises(FileNotFoundError):
        JobManager(tmp_path / "missing")
    (tmp_path / "file").touch()
    with pytest.raises(ValueError, match="directory"):
        JobManager(tmp_path / "file")
    manager = JobManager(tmp_path)
    (tmp_path / "used.m4a").touch()
    (tmp_path / "dangling.m4a").symlink_to(tmp_path / "absent.m4a")
    for output in ["used.m4a", "dangling.m4a"]:
        with pytest.raises(ValueError, match="already exists"):
            manager.start(ConversionRequest(text="Hello.", output=Path(output)))
    with pytest.raises(ValueError, match="Output directory does not exist"):
        manager.start(
            ConversionRequest(text="Hello.", output=Path("missing/audio.m4a"))
        )
    with pytest.raises(ValueError, match="Unknown conversion job"):
        manager.get("unknown")
    with pytest.raises(ValueError, match="Unknown conversion job"):
        manager.cancel("unknown")


def test_source_replacement_is_rejected(tmp_path):
    """A symlink inserted after validation cannot become a worker input."""
    source = tmp_path / "source.md"
    workspace = Workspace(tmp_path)
    source.symlink_to(tmp_path / "other.md")
    with pytest.raises(ValueError, match="input path changed"):
        workspace.copy_input(source, tmp_path / "copy.md")


def test_process_start_failure_cleanup(tmp_path, monkeypatch):
    """A failed spawn leaves no record or scratch directory."""
    manager = JobManager(tmp_path)
    process = Mock()
    process.start.side_effect = OSError("spawn failed")
    monkeypatch.setattr(manager._context, "Process", Mock(return_value=process))
    with pytest.raises(OSError, match="spawn failed"):
        manager.start(ConversionRequest(text="Hello."))
    assert manager.records == {}
    assert_clean(tmp_path)


def test_atomic_publication_preserves_a_racing_output(tmp_path, monkeypatch):
    """A collision after final validation cannot replace an existing file."""
    workspace = Workspace(tmp_path)
    staged = tmp_path / "staged.m4a"
    staged.write_bytes(b"converted")
    output = tmp_path / "result.m4a"
    check_output = workspace.check_output

    def race(path):
        resolved = check_output(path)
        path.write_bytes(b"another file")
        return resolved

    monkeypatch.setattr(workspace, "check_output", race)
    with pytest.raises(FileExistsError):
        workspace.publish(staged, output)
    assert output.read_bytes() == b"another file"


def test_monitor_cancellation_stops_worker_and_cleans_files(tmp_path):
    """Event loop shutdown cannot leave an unmonitored worker behind."""

    async def scenario():
        manager = JobManager(tmp_path, worker=slow_worker)
        try:
            start = manager.start(ConversionRequest(text="Hello."))
            await wait_for(
                manager, start.job_id, lambda job: job.phase == "synthesizing"
            )
            task = manager._active.task
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert manager.get(start.job_id).state == "cancelled"
            assert_clean(tmp_path)
        finally:
            await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("source_kind", ["rst", "html", "text"])
def test_worker_confines_pandoc_secondary_reads(tmp_path, source_kind):
    """MCP extraction cannot turn referenced files into narration."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "outside.txt"
    secret.write_text("Confidential outside narration.\n")
    if source_kind == "rst":
        source = workspace / "book.rst"
        source.write_text(f"Public narration.\n\n.. include:: {secret}\n")
        options = {"input_file": Path(source.name)}
    else:
        text = f'<p>Public narration.</p><iframe src="{secret.as_uri()}"></iframe>'
        if source_kind == "html":
            source = workspace / "book.html"
            source.write_text(text)
            options = {"input_file": Path(source.name)}
        else:
            options = {"text": text}

    async def scenario():
        manager = JobManager(workspace, worker=confined_worker)
        try:
            start = manager.start(ConversionRequest(**options))
            done = await finish(manager, start.job_id)
            assert done.state == "succeeded", done.error
            assert done.result.duration > 0
            assert_clean(workspace)
        finally:
            await manager.close()

    asyncio.run(scenario())
