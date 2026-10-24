"""Tests for the dependencies module."""

import pytest

from simple_tts import dependencies
from simple_tts.dependencies import DEPENDENCIES


@pytest.mark.parametrize("missing", [[], ["pandoc"], ["ffmpeg"], ["pandoc", "ffmpeg"]])
def test_dependency_checks_only_require_pandoc_and_ffmpeg(monkeypatch, capsys, missing):
    """Filter missing tools without checking Poppler."""
    from unittest.mock import Mock

    which = Mock(side_effect=lambda name: None if name in missing else f"/bin/{name}")
    monkeypatch.setattr(dependencies.shutil, "which", which)
    assert dependencies.check_dependencies(warn=False) is (not missing)
    assert [call.args[0] for call in which.call_args_list] == ["pandoc", "ffmpeg"]
    assert capsys.readouterr().out == ""
    assert [dep.name for dep in dependencies.get_missing_dependencies()] == missing


@pytest.mark.parametrize("warn", [True, False])
def test_missing_dependencies_report_install_commands(monkeypatch, capsys, warn):
    """Missing tools report package names for both supported installers."""
    monkeypatch.setattr(dependencies, "is_command_available", lambda name: False)
    assert dependencies.check_dependencies(warn=warn) is False
    output = capsys.readouterr().out
    if warn:
        assert "brew install pandoc ffmpeg" in output
        assert "sudo apt-get install pandoc ffmpeg" in output
    else:
        assert output == ""


@pytest.mark.parametrize("state", ["no-brew", "installed", "missing", "failed"])
def test_homebrew_installation(monkeypatch, capsys, state):
    """Installation handles missing Homebrew, existing tools, and command failures."""
    import subprocess
    from unittest.mock import Mock

    monkeypatch.setattr(
        dependencies, "is_command_available", lambda name: state != "no-brew"
    )
    monkeypatch.setattr(
        dependencies,
        "get_missing_dependencies",
        lambda: [] if state == "installed" else DEPENDENCIES,
    )
    run = Mock()
    if state == "failed":
        run.side_effect = subprocess.CalledProcessError(
            1, "brew", stderr="install error"
        )
    monkeypatch.setattr(dependencies.subprocess, "run", run)
    assert dependencies.install_dependencies_brew() is (
        state in {"installed", "missing"}
    )
    output = capsys.readouterr().out
    if state == "missing":
        assert [call.args[0] for call in run.call_args_list] == [
            ["brew", "install", "pandoc"],
            ["brew", "install", "ffmpeg"],
        ]
        assert "All dependencies installed" in output
    elif state == "failed":
        assert run.call_count == 1
        assert "install error" in output
    else:
        run.assert_not_called()
        assert (
            "Homebrew is not installed"
            if state == "no-brew"
            else "All dependencies are installed"
        ) in output


def test_dependency_status_includes_missing_tools(monkeypatch, capsys):
    """Status output includes available and missing dependencies."""
    monkeypatch.setattr(
        dependencies, "is_command_available", lambda name: name == "pandoc"
    )
    dependencies.print_dependency_status()
    output = capsys.readouterr().out
    assert "✓ pandoc" in output
    assert "✗ ffmpeg" in output
