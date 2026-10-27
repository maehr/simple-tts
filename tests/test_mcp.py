"""Verify MCP discovery, structured tool responses, and stdio startup."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import call

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client
from fastmcp.exceptions import ToolError

from simple_tts import dependencies
from simple_tts.jobs import TERMINAL_STATES, JobManager
from simple_tts.mcp import create_server
from simple_tts.voices import DEFAULT_LANGUAGE, LANGUAGES
from tests.test_jobs import assert_clean, wait_for
from tests.workers import (
    engine_worker,
    failing_worker,
    metal_engine_worker,
    mock_worker,
    slow_worker,
)


def test_discovery_schemas_and_defaults(tmp_path, ptts):
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
            assert props["language"]["default"] == DEFAULT_LANGUAGE
            assert props["language"]["enum"] == list(LANGUAGES)
            assert props["voice"]["default"] is None
            assert props["voice"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
            assert "voices" not in props
            assert props["duration_delta"]["default"] == 0.0
            assert props["duration_delta"]["minimum"] == -0.3
            assert props["duration_delta"]["maximum"] == 0.3
            assert "speed" not in props
            assert props["chapter_level"]["default"] == 1
            assert props["chapter_level"]["minimum"] == 1
            assert props["chapter_level"]["maximum"] == 6
            assert props["device"]["default"] == "cpu"
            assert props["device"]["enum"] == ["cpu", "mps", "cuda"]
            assert props["format"]["default"] is None
            assert props["output"]["default"] is None
            for tool in tools.values():
                assert tool.output_schema is not None
            # Startup and discovery read no catalogue and load no model.
            ptts.catalog_voices.assert_not_called()
            ptts.TTS.assert_not_called()
            deps = (await client.call_tool("check_dependencies")).structured_content
            assert deps == {"pandoc": True, "ffmpeg": True}

    asyncio.run(scenario())


def test_list_voices_reads_only_catalogue_metadata(tmp_path, ptts):
    """Voice discovery reports languages and voices without model weights."""

    async def scenario():
        async with Client(create_server(tmp_path)) as client:
            catalog = (await client.call_tool("list_voices")).structured_content
            assert catalog["languages"] == [
                {"code": "en", "name": "English", "default_voice": "Zoey"},
                {"code": "de", "name": "German", "default_voice": "Nele"},
                {"code": "es", "name": "Spanish", "default_voice": "Noa"},
                {"code": "fr", "name": "French", "default_voice": "Noémie"},
            ]
            assert {
                "name": "Nele",
                "id": "2A2fFf8SkDEJ8nzx",
                "language": "de",
                "description": "Nele test voice.",
                "default": True,
            } in catalog["voices"]
            assert [voice["name"] for voice in catalog["voices"]] == [
                "Hadley",
                "Zoey",
                "Anton",
                "Nele",
                "Noa",
                "Noémie",
            ]
            german = (
                await client.call_tool("list_voices", {"language": "de"})
            ).structured_content["voices"]
            assert [(voice["name"], voice["default"]) for voice in german] == [
                ("Anton", False),
                ("Nele", True),
            ]
            with pytest.raises(ToolError):
                await client.call_tool("list_voices", {"language": "pt"})

    asyncio.run(scenario())
    # The pinned default catalogue needs no argument. No weights load.
    assert ptts.catalog_voices.call_args_list == [call(), call()]
    ptts.TTS.assert_not_called()


def test_list_voices_without_catalogue_is_a_tool_error(tmp_path, ptts):
    """An uncached catalogue offline gives a clear tool error."""
    ptts.catalog_voices.side_effect = OSError("cannot fetch voices.jsonl")

    async def scenario():
        async with Client(create_server(tmp_path)) as client:
            with pytest.raises(ToolError, match="Connect to the internet once"):
                await client.call_tool("list_voices")

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
        {"text": "Hello.", "voice": ""},
        {"text": "Hello.", "voice": " "},
        {"text": "Hello.", "voice": "Nele,Anton"},
        {"text": "Hello.", "voice": ["Nele", "Anton"]},
        {"text": "Hello.", "voices": ["Zoey"]},
        {"text": "Hello.", "language": "it"},
        {"text": "Hello.", "language": "pt"},
        {"text": "Hello.", "language": "DE"},
        {"text": "Hello.", "duration_delta": -0.31},
        {"text": "Hello.", "duration_delta": 0.5},
        {"text": "Hello.", "duration_delta": "fast"},
        {"text": "Hello.", "speed": 1.1},
        {"text": "Hello.", "chapter_level": 7},
        {"text": "Hello.", "chapter_level": 1.5},
        {"text": "Hello.", "device": "gpu"},
        {"text": "Hello.", "device": "auto"},
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


def test_voice_rotation_is_a_tool_error(tmp_path, ptts):
    """A comma-separated voice fails before a job starts or a catalogue read."""

    async def scenario():
        manager = JobManager(tmp_path)
        async with Client(create_server(tmp_path, manager=manager)) as client:
            arguments = {"text": "Hallo.", "language": "de", "voice": "Nele,Anton"}
            with pytest.raises(ToolError, match="Voice rotation is removed"):
                await client.call_tool("start_conversion", arguments)
            assert manager.records == {}

    asyncio.run(scenario())
    ptts.catalog_voices.assert_not_called()


async def run_job(client, manager, arguments):
    """Start one job and return its terminal status."""
    start = (await client.call_tool("start_conversion", arguments)).structured_content
    assert start["state"] == "running"
    await wait_for(manager, start["job_id"], lambda job: job.state in TERMINAL_STATES)
    return (
        await client.call_tool("get_conversion", {"job_id": start["job_id"]})
    ).structured_content


@pytest.mark.parametrize(
    ("worker", "device", "backend"),
    [
        (engine_worker, None, {"device": "cpu", "quant": "q8"}),
        (engine_worker, "cpu", {"device": "cpu", "quant": "q8"}),
        (metal_engine_worker, None, {"device": "cpu", "quant": "q8"}),
        (metal_engine_worker, "mps", {"device": "metal", "quant": "f32"}),
        (engine_worker, "mps", None),
        (metal_engine_worker, "cuda", None),
    ],
)
def test_job_language_rate_and_device_reach_ptts(
    tmp_path, ptts, worker, device, backend
):
    """The default uses CPU/q8. MPS uses Metal/f32 or fails without a fallback."""
    arguments = {
        "text": "# Eins\n\nHallo.\n\n# Zwei\n\nTschüss.",
        "language": "de",
        "voice": "anton",
        "duration_delta": -0.2,
    }
    if device is not None:
        arguments["device"] = device

    async def scenario():
        manager = JobManager(tmp_path, worker=worker)
        async with Client(create_server(tmp_path, manager=manager)) as client:
            status = await run_job(client, manager, arguments)
        record = json.loads((tmp_path / "ptts.json").read_text())
        if backend is None:
            assert status["state"] == "failed"
            assert f"Requested device '{device}' is unavailable" in status["error"]
            assert record == {"loads": [], "voices": []}
        else:
            assert status["state"] == "succeeded", status
            assert record["loads"] == [
                {
                    **backend,
                    "lang": "de",
                    "conditions": {"duration_delta": -0.2},
                }
            ]
            # One explicit voice speaks both chapters.
            assert record["voices"] == ["3ZKKapPOvuWFcw9f"] * 2
            assert [chapter["title"] for chapter in status["result"]["chapters"]] == [
                "Eins",
                "Zwei",
            ]
        assert_clean(tmp_path)

    asyncio.run(scenario())
    # Submission validates syntax only. The worker reads the catalogue.
    ptts.catalog_voices.assert_not_called()
    ptts.TTS.assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"voice": "Unknown"}, "Unknown voice 'Unknown'"),
        ({"language": "de", "voice": "Zoey"}, "'Zoey' is English (en)"),
        ({"language": "fr", "voice": "Noa"}, "'Noa' is Spanish (es)"),
    ],
)
def test_voice_errors_are_job_status_data(tmp_path, ptts, arguments, message):
    """The worker rejects unknown and mismatched voices after submission."""

    async def scenario():
        manager = JobManager(tmp_path, worker=mock_worker)
        async with Client(create_server(tmp_path, manager=manager)) as client:
            status = await run_job(client, manager, {"text": "Hello.", **arguments})
            assert status["state"] == "failed"
            assert message in status["error"]
            # An MCP client learns the tool that lists the voices.
            assert "or the MCP tool list_voices." in status["error"]
            assert status["result"] is None
        assert_clean(tmp_path)

    asyncio.run(scenario())
    ptts.catalog_voices.assert_not_called()


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
    """A real stdio process starts without importing ptts or printing banners."""

    async def scenario():
        from fastmcp.client.transports import StdioTransport

        # Fail immediately if startup tries to import a model runtime.
        script = """
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] == 'ptts':
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


def test_server_rejects_windows_with_an_injected_manager(tmp_path, monkeypatch):
    """An injected manager cannot bypass the server's platform guard."""
    from unittest.mock import Mock

    from simple_tts import jobs

    monkeypatch.setattr(jobs.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Windows is not supported"):
        create_server(tmp_path, manager=Mock())
