"""Tests for the CLI module."""

from typer.testing import CliRunner

from simple_tts.cli import app, format_duration, format_eta, validate_speed

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


class TestFormatEta:
    """Tests for format_eta function."""

    def test_formats_seconds(self):
        """Test formatting of ETA in seconds."""
        assert format_eta(30) == "~30s remaining"

    def test_formats_minutes_and_seconds(self):
        """Test formatting of ETA with minutes."""
        assert format_eta(90) == "~1m 30s remaining"

    def test_handles_zero(self):
        """Test handling of zero ETA."""
        assert format_eta(0) == "almost done"

    def test_handles_negative(self):
        """Test handling of negative ETA."""
        assert format_eta(-10) == "almost done"


class TestValidateSpeed:
    """Tests for validate_speed function."""

    def test_valid_speed(self):
        """Test that valid speed is returned unchanged."""
        assert validate_speed(1.0) == 1.0
        assert validate_speed(1.5) == 1.5

    def test_boundary_values(self):
        """Test boundary speed values."""
        assert validate_speed(0.5) == 0.5
        assert validate_speed(2.0) == 2.0


class TestCLIVersion:
    """Tests for the version command."""

    def test_version_command(self):
        """Test that version command outputs version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "simple-tts" in result.stdout
        assert "0.1.0" in result.stdout


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

    def test_convert_has_no_checkpoint_option(self):
        """Test that convert has --no-checkpoint option."""
        import re

        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        # Strip ANSI escape codes and check for the option text
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        clean_output = ansi_escape.sub("", result.stdout)
        assert "--no-checkpoint" in clean_output
