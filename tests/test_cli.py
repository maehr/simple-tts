"""Tests for the CLI module."""

import re

from typer.testing import CliRunner

from simple_tts.cli import app, build_vtt, format_duration, format_timestamp

runner = CliRunner()


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_formats_seconds_only(self):
        """Test formatting of seconds only."""
        assert format_duration(30) == "30s"

    def test_formats_minutes_and_seconds(self):
        """Test formatting of minutes and seconds."""
        assert format_duration(65) == "1m 5s"

    def test_formats_hours_minutes_seconds(self):
        """Test formatting of hours, minutes, and seconds."""
        assert format_duration(3661) == "1h 1m 1s"

    def test_handles_zero(self):
        """Test handling of zero duration."""
        assert format_duration(0) == "0s"


class TestFormatTimestamp:
    """Tests for WebVTT timestamp formatting."""

    def test_formats_milliseconds(self):
        """Test timestamp formatting to hh:mm:ss.mmm."""
        assert format_timestamp(1.234) == "00:00:01.234"
        assert format_timestamp(3661.789) == "01:01:01.789"


class TestBuildVTT:
    """Tests for building WebVTT captions."""

    def test_builds_captions(self):
        """Test WebVTT construction from chunks and durations."""
        vtt = build_vtt(["Hello world", "Next line"], [1.0, 2.5])
        assert "WEBVTT" in vtt
        assert "00:00:00.000 --> 00:00:01.000" in vtt
        assert "00:00:01.000 --> 00:00:03.500" in vtt
        assert "Hello world" in vtt and "Next line" in vtt


class TestCLIVersion:
    """Tests for the version command."""

    def test_version_command(self):
        """Test that version command outputs version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "simple-tts" in result.stdout
        assert "0.1.0" in result.stdout


class TestCLIVoices:
    """Tests for the voices command."""

    def test_voices_command(self):
        """Test that voices command lists available voices."""
        result = runner.invoke(app, ["voices"])
        assert result.exit_code == 0
        # Strip ANSI codes for checking
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        clean = ansi_escape.sub("", result.stdout)
        assert "af_bella" in clean
        assert "English" in clean


class TestCLIConvert:
    """Tests for the convert command."""

    def test_convert_missing_file(self):
        """Test convert with non-existent file."""
        result = runner.invoke(app, ["convert", "/nonexistent/file.pdf"])
        assert result.exit_code != 0

    def test_convert_shows_help(self):
        """Test that convert shows help."""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Input file" in result.stdout

    def test_convert_has_voice_option(self):
        """Test that convert has --voice option."""
        result = runner.invoke(app, ["convert", "--help"])
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        clean = ansi_escape.sub("", result.stdout)
        assert "--voice" in clean

    def test_convert_has_yes_option(self):
        """Test that convert has -y/--yes option."""
        result = runner.invoke(app, ["convert", "--help"])
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        clean = ansi_escape.sub("", result.stdout)
        assert "--yes" in clean

    def test_invalid_voice_rejected(self):
        """Test that invalid voice is rejected."""
        result = runner.invoke(app, ["convert", "test.txt", "--voice", "invalid_voice"])
        assert result.exit_code != 0


class TestCLICheckDeps:
    """Tests for the check-deps command."""

    def test_check_deps_command(self):
        """Test that check-deps command works."""
        result = runner.invoke(app, ["check-deps"])
        assert result.exit_code == 0
        assert "Dependencies" in result.stdout or "pdftotext" in result.stdout
