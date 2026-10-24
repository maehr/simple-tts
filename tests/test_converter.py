"""Tests for the converter module."""

import subprocess
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

    def test_extracts_text_from_epub(self, monkeypatch):
        """Test extraction from EPUB files via pandoc using bundled sample."""
        epub_path = Path(__file__).parent / "minimal.epub"
        assert epub_path.exists()

        def fake_run(cmd, capture_output, text, check):
            assert cmd == ["pandoc", str(epub_path), "-t", "plain"]
            return subprocess.CompletedProcess(cmd, 0, stdout="EPUB text", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        text = extract_text(epub_path)
        assert "EPUB text" in text


@pytest.mark.parametrize("level", range(1, 7))
def test_exact_chapter_level(tmp_path, level):
    """Only the chosen level starts chapters; all headings remain spoken."""
    from simple_tts.converter import extract_sections

    source = tmp_path / "book.md"
    lines = ["Preamble."]
    for heading_level in range(1, 7):
        lines.extend(["#" * heading_level + f" Heading{heading_level}", "Body."])
    source.write_text("\n\n".join(lines))
    result = extract_sections(source, level)
    assert result.matched_headings == 1
    assert [section.title for section in result.sections] == [
        "Introduction",
        f"Heading{level}",
    ]
    narration = " ".join(section.text for section in result.sections)
    for heading_level in range(1, 7):
        assert narration.count(f"Heading{heading_level}") == 1
    assert result.sections[1].text.startswith(f"Heading{level}")


@pytest.mark.parametrize("suffix", ["md", "qmd", "rmd"])
def test_setext_and_formatted_titles(tmp_path, suffix):
    """Pandoc retains Setext headings and removes title formatting."""
    from simple_tts.converter import extract_sections

    source = tmp_path / f"book.{suffix}"
    source.write_text(
        "Preface\n=======\n\nIntro.\n\n"
        "A *formatted* [title](https://example.com) `code`\n"
        "------------------------------------------------------\n\nBody.\n\n"
        "## Last **chapter**\n\nEnd.\n"
    )
    result = extract_sections(source, 2)
    assert [section.title for section in result.sections] == [
        "Introduction",
        "A formatted title code",
        "Last chapter",
    ]
    assert result.sections[0].text == "Preface\n\nIntro."
    assert result.sections[1].text.count("A formatted title code") == 1


def test_code_does_not_start_chapters(tmp_path):
    """Heading-like code has no Header nodes."""
    from simple_tts.converter import extract_sections

    source = tmp_path / "no-headings.md"
    source.write_text("Text.\n\n```markdown\n# False title\n```\n\n    ## Also code\n")
    result = extract_sections(source)
    assert result.matched_headings == 0
    assert len(result.sections) == 1
    assert result.sections[0].title == "no-headings"
    assert "False title" in result.sections[0].text


def test_absent_selected_level(tmp_path):
    """Other heading levels do not create an introduction chapter."""
    from simple_tts.converter import extract_sections

    source = tmp_path / "book.md"
    source.write_text("# Title\n\n### Details\n\nBody.")
    result = extract_sections(source, 2)
    assert result.matched_headings == 0
    assert [section.title for section in result.sections] == ["book"]
    assert "Title" in result.sections[0].text
    assert "Details" in result.sections[0].text


def test_nested_html_headings(tmp_path):
    """Headings inside HTML divisions retain their level."""
    from simple_tts.converter import extract_sections

    source = tmp_path / "book.html"
    source.write_text("<div><h1>Title</h1><h2>First</h2><p>Body</p></div>")
    result = extract_sections(source, 2)
    assert [section.title for section in result.sections] == ["Introduction", "First"]


def test_docx_headings(tmp_path):
    """DOCX styles retain the same chapter boundaries as Markdown."""
    import subprocess

    from simple_tts.converter import extract_sections

    source = tmp_path / "book.docx"
    subprocess.run(
        ["pandoc", "-f", "markdown", "-o", str(source)],
        input="# Book\n\n## First\n\nBody.\n\n## Last\n\nEnd.",
        text=True,
        check=True,
    )
    result = extract_sections(source, 2)
    assert [section.title for section in result.sections] == [
        "Introduction",
        "First",
        "Last",
    ]


def test_pdf_is_one_section(tmp_path, monkeypatch):
    """PDF extraction keeps the existing text extraction helper."""
    from simple_tts import converter

    source = tmp_path / "book.pdf"
    source.touch()
    monkeypatch.setattr(converter.shutil, "which", lambda command: None)
    monkeypatch.setattr(converter, "extract_text_from_pdf", lambda path: "PDF text")
    result = converter.extract_sections(source, 2)
    assert result.sections == [converter.Section("book", "PDF text")]
    assert result.matched_headings == 0


def test_rst_headings(tmp_path):
    """RST heading levels pass through the Pandoc structure."""
    from simple_tts.converter import extract_sections

    source = tmp_path / "book.rst"
    source.write_text("Book\n====\n\nFirst\n-----\n\nBody.\n\nLast\n----\n\nEnd.\n")
    result = extract_sections(source, 2)
    assert [section.title for section in result.sections] == [
        "Introduction",
        "First",
        "Last",
    ]


def test_no_preamble_does_not_add_introduction(tmp_path):
    """A document that starts with a selected heading has no preamble chapter."""
    from simple_tts.converter import extract_sections

    source = tmp_path / "book.md"
    source.write_text("## First\n\n## Last\n")
    result = extract_sections(source, 2)
    assert [section.title for section in result.sections] == ["First", "Last"]
    assert [section.text for section in result.sections] == ["First", "Last"]


def test_epub_chapter_extraction():
    """The bundled EPUB keeps its headings in structured extraction."""
    from simple_tts.converter import extract_sections

    result = extract_sections(Path(__file__).parent / "minimal.epub")
    assert result.matched_headings == 3
    assert [section.title for section in result.sections] == [
        "Chapter 1",
        "Chapter 2",
        "Copyright",
    ]
    for section in result.sections:
        assert section.text.startswith(section.title)
        assert section.text.count(section.title) == 1


@pytest.mark.parametrize("markdown", ["## First\n\nBody.\n\n## Last\n\nEnd.", "Body."])
def test_pdf_markdown_headings(tmp_path, monkeypatch, markdown):
    """Optional pdf2md output uses the real Pandoc chapter parser."""
    import subprocess

    from simple_tts import converter

    source = tmp_path / "book with spaces.pdf"
    source.touch()
    run = subprocess.run
    calls = []

    def convert(command, **kwargs):
        if command[0] == "/local/pdf2md":
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, markdown, "")
        return run(command, **kwargs)

    monkeypatch.setattr(converter.shutil, "which", lambda command: "/local/pdf2md")
    monkeypatch.setattr(converter.subprocess, "run", convert)
    result = converter.extract_sections(source, 2)
    assert calls == [["/local/pdf2md", str(source), "--raw"]]
    if markdown.startswith("##"):
        assert result.matched_headings == 2
        assert result.sections == [
            converter.Section("First", "First\n\nBody."),
            converter.Section("Last", "Last\n\nEnd."),
        ]
    else:
        assert result.matched_headings == 0
        assert result.sections == [converter.Section(source.stem, "Body.")]


@pytest.mark.parametrize("failure", ["empty", "exit", "missing"])
def test_pdf_markdown_failure_falls_back(tmp_path, monkeypatch, failure):
    """Broken optional extraction warns before the pdftotext fallback."""
    import subprocess

    from simple_tts import converter

    source = tmp_path / "book.pdf"
    source.touch()

    def run(command, **kwargs):
        if failure == "exit":
            raise subprocess.CalledProcessError(1, command, stderr="bad PDF")
        if failure == "missing":
            raise FileNotFoundError("pdf2md disappeared")
        return subprocess.CompletedProcess(command, 0, " \n", "")

    monkeypatch.setattr(converter.shutil, "which", lambda command: "/local/pdf2md")
    monkeypatch.setattr(converter.subprocess, "run", run)
    monkeypatch.setattr(
        converter, "extract_text_from_pdf", lambda path: "Fallback text"
    )
    with pytest.warns(RuntimeWarning, match="Use pdftotext"):
        result = converter.extract_sections(source)
    assert result.sections == [converter.Section("book", "Fallback text")]
