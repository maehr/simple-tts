"""Verify MCP discovery, structured tool responses, and stdio startup."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client
from fastmcp.exceptions import ToolError

from simple_tts import dependencies
from simple_tts.jobs import JobManager
from simple_tts.mcp import create_server
from tests.test_jobs import assert_clean, wait_for
from tests.workers import failing_worker, mock_worker, slow_worker


def test_discovery_schemas_and_defaults(tmp_path):
    """FastMCP exposes five ordinary tools with typed inputs and outputs."""

    async def scenario():
        async with Client(create_server(tmp_path)) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert set(tools) == {
                "list_voices",
                "check_dependencies",
                "start_conversion",
                "get_conversion",
                "cancel_conversion",
            }
            schema = tools["start_conversion"].input_schema
            props = schema["properties"]
            assert props["voices"]["default"] == ["af_bella"]
            assert props["voices"]["minItems"] == 1
            assert props["speed"]["default"] == 1.1
            assert props["speed"]["minimum"] == 0.5
            assert props["speed"]["maximum"] == 2.0
            assert props["chapter_level"]["default"] == 1
            assert props["chapter_level"]["minimum"] == 1
            assert props["chapter_level"]["maximum"] == 6
            assert props["device"]["default"] == "auto"
            assert props["device"]["enum"] == ["auto", "cuda", "mps", "cpu"]
            assert props["format"]["default"] is None
            assert props["output"]["default"] is None
            for tool in tools.values():
                assert tool.output_schema is not None
            voices = (await client.call_tool("list_voices")).structured_content[
                "voices"
            ]
            assert {
                "id": "af_bella",
                "language": "English (US)",
                "description": "American female voice",
            } in voices
            deps = (await client.call_tool("check_dependencies")).structured_content
            assert deps == {"pandoc": True, "ffmpeg": True}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "source", [{"text": "# First\n\nHello."}, {"input_file": "book.md"}]
)
def test_tool_conversion_and_polling(tmp_path, source):
    """Clients can convert documents and text, then read structured results."""
    (tmp_path / "book.md").write_text("# First\n\nHello.")

    async def scenario():
        manager = JobManager(tmp_path, worker=mock_worker)
        async with Client(create_server(tmp_path, manager=manager)) as client:
            start = (
                await client.call_tool("start_conversion", source)
            ).structured_content
            assert start["state"] == "running"
            job_id = start["job_id"]
            async with asyncio.timeout(15):
                while True:
                    done = (
                        await client.call_tool("get_conversion", {"job_id": job_id})
                    ).structured_content
                    if done["state"] in {"succeeded", "failed"}:
                        break
                    await asyncio.sleep(0.02)
            assert done["state"] == "succeeded", done
            assert done["completed_chunks"] == done["total_chunks"] == 1
            result = done["result"]
            assert result["format"] == "m4a"
            assert result["chapters"] == [{"title": "First", "start": 0, "end": 0.1}]
            assert Path(result["path"]).is_file()
            assert result["uri"] == Path(result["path"]).as_uri()
            assert result["duration"] == 0.1
            assert result["processing_time"] > 0
        assert Path(result["path"]).exists()
        assert_clean(tmp_path)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"input_file": "book.md", "text": "Hello."},
        {"text": ""},
        {"text": "Hello.", "voices": []},
        {"text": "Hello.", "voices": ["unknown"]},
        {"text": "Hello.", "voices": ["af_bella", "bf_emma"]},
        {"text": "Hello.", "speed": 0.4},
        {"text": "Hello.", "speed": 2.1},
        {"text": "Hello.", "chapter_level": 7},
        {"text": "Hello.", "chapter_level": 1.5},
        {"text": "Hello.", "device": "gpu"},
        {"text": "Hello.", "format": "ogg"},
        {"text": "Hello.", "format": "m4b", "output": "audio.mp3"},
        {"input_file": "missing.md"},
        {"input_file": "book.pdf"},
        {"input_file": "../book.md"},
    ],
)
def test_request_failures_are_tool_errors(tmp_path, arguments):
    """Invalid requests return MCP errors without creating a job."""
    (tmp_path / "book.pdf").touch()

    async def scenario():
        manager = JobManager(tmp_path)
        async with Client(create_server(tmp_path, manager=manager)) as client:
            with pytest.raises(ToolError):
                await client.call_tool("start_conversion", arguments)
            for tool in ["get_conversion", "cancel_conversion"]:
                with pytest.raises(ToolError, match="Unknown conversion job"):
                    await client.call_tool(tool, {"job_id": "missing"})
            assert manager.records == {}

    asyncio.run(scenario())


def test_missing_dependencies_are_tool_errors(tmp_path, monkeypatch):
    """Missing tools fail before spawn and report their names."""
    monkeypatch.setattr(dependencies, "is_command_available", lambda command: False)

    async def scenario():
        async with Client(create_server(tmp_path)) as client:
            deps = (await client.call_tool("check_dependencies")).structured_content
            assert deps == {"pandoc": False, "ffmpeg": False}
            with pytest.raises(ToolError, match="Missing system dependencies"):
                await client.call_tool("start_conversion", {"text": "Hello."})

    # The tool imports this helper directly, so patch that reference too.
    from simple_tts import mcp

    monkeypatch.setattr(mcp, "is_command_available", lambda command: False)
    asyncio.run(scenario())


def test_busy_cancellation_and_lifespan_shutdown(tmp_path):
    """Tool calls remain responsive while a worker runs or cancels."""

    async def scenario():
        manager = JobManager(tmp_path, worker=slow_worker)
        async with Client(create_server(tmp_path, manager=manager)) as client:
            start = (
                await client.call_tool("start_conversion", {"text": "Hello."})
            ).structured_content
            job_id = start["job_id"]
            await wait_for(manager, job_id, lambda job: job.phase == "synthesizing")
            with pytest.raises(ToolError, match="busy"):
                await client.call_tool("start_conversion", {"text": "Another."})
            cancelled = (
                await client.call_tool("cancel_conversion", {"job_id": job_id})
            ).structured_content
            assert cancelled["state"] == "cancelling"
            await wait_for(manager, job_id, lambda job: job.state == "cancelled")
            start = (
                await client.call_tool("start_conversion", {"text": "Goodbye."})
            ).structured_content
            job_id = start["job_id"]
            await wait_for(manager, job_id, lambda job: job.phase == "synthesizing")
        assert manager.get(job_id).state == "cancelled"
        assert_clean(tmp_path)

    asyncio.run(scenario())


def test_worker_failure_is_status_data(tmp_path):
    """An execution error appears in job status instead of a polling error."""

    async def scenario():
        manager = JobManager(tmp_path, worker=failing_worker)
        async with Client(create_server(tmp_path, manager=manager)) as client:
            start = (
                await client.call_tool("start_conversion", {"text": "Hello."})
            ).structured_content
            job_id = start["job_id"]
            await wait_for(manager, job_id, lambda job: job.state == "failed")
            status = (
                await client.call_tool("get_conversion", {"job_id": job_id})
            ).structured_content
            assert status["state"] == "failed"
            assert "Mock model failure" in status["error"]

    asyncio.run(scenario())


def test_stdio_startup_has_only_protocol_stdout(tmp_path):
    """A real stdio process starts without importing Torch or printing banners."""

    async def scenario():
        from fastmcp.client.transports import StdioTransport

        # Fail immediately if startup tries to import a model runtime.
        script = """
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'kokoro'}:
            raise RuntimeError('Model import during startup')
sys.meta_path.insert(0, BlockModels())
from simple_tts.cli import app
app()
"""
        transport = StdioTransport(
            command=sys.executable,
            args=["-c", script, "mcp", "--workspace", str(tmp_path)],
            keep_alive=False,
        )
        async with Client(transport) as client:
            assert len(await client.list_tools()) == 5
            assert (await client.call_tool("check_dependencies")).structured_content

    asyncio.run(scenario())
    # Independently inspect raw stdout, including startup and EOF shutdown.
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    result = subprocess.run(
        [sys.executable, "-m", "simple_tts.cli", "mcp", "--workspace", str(tmp_path)],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        assert json.loads(line)["jsonrpc"] == "2.0"


def test_stdio_conversion_and_shutdown_cleanup(tmp_path):
    """Worker diagnostics preserve stdio and EOF cancels an active conversion."""

    async def scenario():
        from fastmcp.client.transports import StdioTransport

        script = """
import sys
from pathlib import Path
from simple_tts.mcp import create_server
from simple_tts.jobs import JobManager
from tests.workers import stdio_worker
workspace = Path(sys.argv[1])
manager = JobManager(workspace, worker=stdio_worker)
create_server(workspace, manager=manager).run(transport='stdio', show_banner=False)
"""
        transport = StdioTransport(
            command=sys.executable, args=["-c", script, str(tmp_path)], keep_alive=False
        )
        worker_pid = None
        try:
            async with Client(transport) as client:
                start = (
                    await client.call_tool("start_conversion", {"text": "Hello."})
                ).structured_content
                async with asyncio.timeout(15):
                    while True:
                        status = (
                            await client.call_tool(
                                "get_conversion", {"job_id": start["job_id"]}
                            )
                        ).structured_content
                        if status["state"] in {"succeeded", "failed"}:
                            break
                        await asyncio.sleep(0.02)
                assert status["state"] == "succeeded", status
                completed = Path(status["result"]["path"])
                start = (
                    await client.call_tool(
                        "start_conversion", {"text": "Wait for shutdown."}
                    )
                ).structured_content
                async with asyncio.timeout(15):
                    while True:
                        status = (
                            await client.call_tool(
                                "get_conversion", {"job_id": start["job_id"]}
                            )
                        ).structured_content
                        if status["phase"] == "synthesizing":
                            break
                        await asyncio.sleep(0.02)
                worker_pid = int(next(tmp_path.rglob("worker.pid")).read_text())
            assert_clean(tmp_path)
            assert completed.exists()
        finally:
            # Ensure a failed lifecycle test cannot leave model work behind.
            if worker_pid is not None:
                import os
                import signal

                try:
                    os.killpg(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    asyncio.run(scenario())
