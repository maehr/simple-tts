"""Tests for the dependencies module."""

import pytest

from simple_tts.dependencies import (
    DEPENDENCIES,
    Dependency,
    get_dependency_status,
    get_missing_dependencies,
    is_command_available,
)


class TestIsCommandAvailable:
    """Tests for is_command_available function."""

    def test_python_is_available(self):
        """Test that Python is detected as available."""
        assert is_command_available("python") or is_command_available("python3")

    def test_nonexistent_command(self):
        """Test that non-existent command returns False."""
        assert is_command_available("nonexistent_command_12345") is False


class TestDependencies:
    """Tests for DEPENDENCIES constant."""

    def test_dependencies_is_list(self):
        """Test that DEPENDENCIES is a list."""
        assert isinstance(DEPENDENCIES, list)

    def test_all_dependencies_are_dependency_objects(self):
        """Test that all items are Dependency namedtuples."""
        for dep in DEPENDENCIES:
            assert isinstance(dep, Dependency)

    def test_expected_dependencies_present(self):
        """Test that expected dependencies are present."""
        names = [dep.name for dep in DEPENDENCIES]
        assert "pdftotext" in names
        assert "pandoc" in names
        assert "ffmpeg" in names


class TestGetMissingDependencies:
    """Tests for get_missing_dependencies function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_missing_dependencies()
        assert isinstance(result, list)

    def test_all_items_are_dependency_objects(self):
        """Test that returned items are Dependency namedtuples."""
        result = get_missing_dependencies()
        for dep in result:
            assert isinstance(dep, Dependency)


class TestGetDependencyStatus:
    """Tests for get_dependency_status function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = get_dependency_status()
        assert isinstance(result, dict)

    def test_all_values_are_bool(self):
        """Test that all values are booleans."""
        result = get_dependency_status()
        for value in result.values():
            assert isinstance(value, bool)

    def test_contains_all_dependencies(self):
        """Test that status contains all dependencies."""
        result = get_dependency_status()
        for dep in DEPENDENCIES:
            assert dep.name in result


@pytest.mark.parametrize("warn", [True, False])
def test_missing_dependencies_report_install_commands(monkeypatch, capsys, warn):
    """Missing tools report package names for both supported installers."""
    from simple_tts import dependencies

    monkeypatch.setattr(dependencies, "is_command_available", lambda name: False)
    assert dependencies.check_dependencies(warn=warn) is False
    output = capsys.readouterr().out
    if warn:
        assert "brew install poppler pandoc ffmpeg" in output
        assert "sudo apt-get install poppler-utils pandoc ffmpeg" in output
    else:
        assert output == ""


@pytest.mark.parametrize("state", ["no-brew", "installed", "missing", "failed"])
def test_homebrew_installation(monkeypatch, capsys, state):
    """Installation handles missing Homebrew, existing tools, and command failures."""
    import subprocess
    from unittest.mock import Mock

    from simple_tts import dependencies

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
            ["brew", "install", dep.brew_package] for dep in DEPENDENCIES
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
    from simple_tts import dependencies

    monkeypatch.setattr(
        dependencies, "is_command_available", lambda name: name == "pandoc"
    )
    dependencies.print_dependency_status()
    output = capsys.readouterr().out
    assert "✓ pandoc" in output
    assert "✗ pdftotext" in output
    assert "✗ ffmpeg" in output
