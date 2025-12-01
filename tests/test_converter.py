"""Tests for the converter module."""

import tempfile
from pathlib import Path

import pytest

from simple_tts.converter import (
    extract_text,
    normalize_text,
    split_into_chunks,
)


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_removes_leading_trailing_whitespace(self):
        """Test that leading and trailing whitespace is removed."""
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        """Test that multiple spaces are collapsed to single space."""
        assert normalize_text("hello   world") == "hello world"

    def test_preserves_paragraph_structure(self):
        """Test that double newlines (paragraphs) are preserved."""
        assert normalize_text("para1\n\npara2") == "para1\n\npara2"

    def test_handles_empty_string(self):
        """Test handling of empty strings."""
        assert normalize_text("") == ""

    def test_handles_whitespace_only(self):
        """Test handling of whitespace-only strings."""
        assert normalize_text("   ") == ""


class TestSplitIntoChunks:
    """Tests for split_into_chunks function."""

    def test_returns_single_chunk_for_short_text(self):
        """Test that short text is returned as a single chunk."""
        text = "Hello world."
        chunks = split_into_chunks(text, max_chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_long_text(self):
        """Test that long text is split into chunks."""
        text = "A" * 100
        chunks = split_into_chunks(text, max_chunk_size=25)
        assert len(chunks) == 4
        assert all(len(chunk) <= 25 for chunk in chunks)

    def test_splits_at_sentence_boundaries(self):
        """Test that splitting prefers sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_into_chunks(text, max_chunk_size=35)
        assert len(chunks) >= 1

    def test_handles_empty_text(self):
        """Test handling of empty text."""
        chunks = split_into_chunks("")
        assert chunks == [""]


class TestExtractText:
    """Tests for extract_text function."""

    def test_raises_for_nonexistent_file(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            extract_text(Path("/nonexistent/file.pdf"))

    def test_raises_for_unsupported_format(self):
        """Test that ValueError is raised for unsupported formats."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"test content")

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                extract_text(temp_path)
        finally:
            temp_path.unlink()

    def test_extracts_text_from_txt(self):
        """Test extraction from plain text files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("Hello world!")
            temp_path = Path(f.name)

        try:
            text = extract_text(temp_path)
            assert "Hello world!" in text
        finally:
            temp_path.unlink()

    def test_extracts_text_from_md(self):
        """Test extraction from Markdown files."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("# Heading\n\nParagraph text here.")
            temp_path = Path(f.name)

        try:
            text = extract_text(temp_path)
            assert "Heading" in text
            assert "Paragraph" in text
        finally:
            temp_path.unlink()
