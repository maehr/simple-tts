"""Tests for the dependencies module."""

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
