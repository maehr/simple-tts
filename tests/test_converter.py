"""Tests for the converter module."""

import shutil
import subprocess
from pathlib import Path

import pytest

from simple_tts.converter import (
    extract_sections,
    normalize_text,
    split_into_chunks,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  hello   world  ", "hello world"),
        ("para1\n\npara2", "para1\n\npara2"),
        ("", ""),
        ("   ", ""),
        ("exam\u00adple", "example"),
    ],
)
def test_normalize_text(text, expected):
    """Normalize whitespace while retaining paragraph boundaries."""
    assert normalize_text(text) == expected


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
        assert chunks == ["First sentence. Second sentence.", "Third sentence."]

    def test_handles_empty_text(self):
        """Test handling of empty text."""
        chunks = split_into_chunks("")
        assert chunks == [""]


def test_plain_text_input(tmp_path):
    """Prepared text without headings forms one chapter."""
    source = tmp_path / "paper.txt"
    source.write_text("First paragraph.\n\nSecond paragraph.")
    result = extract_sections(source)
    assert result.matched_headings == 0
    assert len(result.sections) == 1
    assert result.sections[0].title == "paper"
    assert result.sections[0].text == "First paragraph.\n\nSecond paragraph."


@pytest.mark.parametrize("level", range(1, 7))
def test_exact_chapter_level(tmp_path, level):
    """Only the chosen level starts chapters; all headings remain spoken."""
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
    source = tmp_path / "no-headings.md"
    source.write_text("Text.\n\n```markdown\n# False title\n```\n\n    ## Also code\n")
    result = extract_sections(source)
    assert result.matched_headings == 0
    assert len(result.sections) == 1
    assert result.sections[0].title == "no-headings"
    assert "False title" in result.sections[0].text


def test_absent_selected_level(tmp_path):
    """Other heading levels do not create an introduction chapter."""
    source = tmp_path / "book.md"
    source.write_text("# Title\n\n### Details\n\nBody.")
    result = extract_sections(source, 2)
    assert result.matched_headings == 0
    assert [section.title for section in result.sections] == ["book"]
    assert "Title" in result.sections[0].text
    assert "Details" in result.sections[0].text


def test_nested_html_headings(tmp_path):
    """Headings inside HTML divisions retain their level."""
    source = tmp_path / "book.html"
    source.write_text("<div><h1>Title</h1><h2>First</h2><p>Body</p></div>")
    result = extract_sections(source, 2)
    assert [section.title for section in result.sections] == ["Introduction", "First"]


@pytest.mark.parametrize("sandbox", [False, True])
def test_docx_headings(tmp_path, sandbox):
    """DOCX styles retain the same chapter boundaries as Markdown."""
    source = tmp_path / "book.docx"
    subprocess.run(
        ["pandoc", "-f", "markdown", "-o", str(source)],
        input="# Book\n\n## First\n\nBody.\n\n## Last\n\nEnd.",
        text=True,
        check=True,
    )
    result = extract_sections(source, 2, sandbox=sandbox)
    assert [section.title for section in result.sections] == [
        "Introduction",
        "First",
        "Last",
    ]


def test_rst_headings(tmp_path):
    """RST heading levels pass through the Pandoc structure."""
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
    source = tmp_path / "book.md"
    source.write_text("## First\n\n## Last\n")
    result = extract_sections(source, 2)
    assert [section.title for section in result.sections] == ["First", "Last"]
    assert [section.text for section in result.sections] == ["First", "Last"]


@pytest.mark.parametrize("sandbox", [False, True])
def test_epub_chapter_extraction(sandbox):
    """The bundled EPUB keeps its headings in structured extraction."""
    result = extract_sections(Path(__file__).parent / "minimal.epub", sandbox=sandbox)
    assert result.matched_headings == 3
    assert [section.title for section in result.sections] == [
        "Chapter 1",
        "Chapter 2",
        "Copyright",
    ]
    for section in result.sections:
        assert section.text.startswith(section.title)
        assert section.text.count(section.title) == 1


@pytest.mark.parametrize("reference", ["absolute", "relative", "symlink"])
def test_sandbox_blocks_rst_includes(tmp_path, reference):
    """Pandoc's sandbox blocks secondary files, including symlink targets."""
    secret = tmp_path / "outside.txt"
    secret.write_text("Confidential outside narration.\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if reference == "absolute":
        target = str(secret)
    elif reference == "relative":
        target = "../outside.txt"
    else:
        target = str(workspace / "linked.txt")
        Path(target).symlink_to(secret)
    source = workspace / "book.rst"
    source.write_text(f"Public narration.\n\n.. include:: {target}\n")
    unrestricted = extract_sections(source)
    assert any(
        "Confidential outside narration." in section.text
        for section in unrestricted.sections
    )
    confined = extract_sections(source, sandbox=True)
    assert [section.text for section in confined.sections] == ["Public narration."]


def test_section_extraction_calls_pandoc_once(tmp_path, monkeypatch):
    """Chapter count does not increase the number of Pandoc processes."""
    from simple_tts import converter

    source = tmp_path / "book.epub"
    source.touch()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        markdown = "\n\n".join(f"# Chapter {i}\n\nBody." for i in range(40))
        return subprocess.CompletedProcess(command, 0, stdout=markdown)

    monkeypatch.setattr(converter.subprocess, "run", run)
    result = converter.extract_sections(source)
    assert len(calls) == 1
    assert len(result.sections) == 40
    assert result.matched_headings == 40


def test_inline_formatting_preserves_literal_punctuation(tmp_path):
    """Markdown cleanup retains escaped punctuation and word underscores."""
    source = tmp_path / "book.md"
    source.write_text(
        "# A **bold** title\n\nUse `snake_case` and snake_case. A \\*literal\\* star."
    )
    result = extract_sections(source)
    assert result.sections[0].title == "A bold title"
    assert "snake_case and snake_case" in result.sections[0].text
    assert "*literal*" in result.sections[0].text


@pytest.mark.parametrize("failure", ["missing-tool", "bad-document"])
def test_extractor_errors_include_actionable_details(tmp_path, monkeypatch, failure):
    """External tool failures identify the missing tool or retain its error text."""
    from unittest.mock import Mock

    from simple_tts import converter

    source = tmp_path / "book.md"
    source.touch()
    error = (
        FileNotFoundError()
        if failure == "missing-tool"
        else subprocess.CalledProcessError(1, "extract", stderr="invalid document")
    )
    monkeypatch.setattr(converter.subprocess, "run", Mock(side_effect=error))
    with pytest.raises(
        RuntimeError,
        match="not found" if failure == "missing-tool" else "invalid document",
    ):
        extract_sections(source)


def test_extraction_rejects_missing_files(tmp_path):
    """Reject missing input files before extraction."""
    with pytest.raises(FileNotFoundError, match="not found"):
        extract_sections(tmp_path / "missing.md")


@pytest.mark.parametrize("level", [0, 7])
def test_invalid_chapter_levels_fail_before_extraction(tmp_path, level):
    """Library callers receive the same chapter level limits as CLI callers."""
    with pytest.raises(ValueError, match="Chapter level"):
        extract_sections(tmp_path / "missing.md", level)


def test_sections_reject_unsupported_input(tmp_path):
    """Unsupported documents fail without invoking Pandoc."""
    source = tmp_path / "book.xyz"
    source.touch()
    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_sections(source)


def test_empty_document_has_no_sections(tmp_path):
    """Whitespace without headings produces no narration sections."""
    source = tmp_path / "empty.md"
    source.write_text("\n\n ")
    assert extract_sections(source).sections == []


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("Read https://example.com/paper now.", "Read now."),
        ("Read the notes at https://example.com/notes.", "Read the notes at."),
        ("Read https://example.com/a, then stop.", "Read, then stop."),
        ("He wrote “see https://example.com/q”.", "He wrote “see”."),
        ("See <https://example.com/autolink> for details.", "See for details."),
        ("Read [the label](https://example.com/label) now.", "Read the label now."),
        ("The exam\u00adple text.", "The example text."),
    ],
)
def test_narration_skips_urls_and_soft_hyphens(tmp_path, markdown, expected):
    """Speech omits URLs and soft hyphens but keeps link labels and punctuation."""
    source = tmp_path / "notes.md"
    source.write_text(markdown, encoding="utf-8")
    result = extract_sections(source)
    assert [section.text for section in result.sections] == [expected]


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="Pandoc is not installed.")
def test_page_markers_are_not_spoken(tmp_path):
    """Pandoc drops the HTML comments that pdf2md writes as page markers."""
    source = tmp_path / "paper.md"
    source.write_text(
        "<!-- pdf-page: 1 -->\n\nBody text.\n\n"
        "<!-- pdf-page: 2; half: left -->\n\nMore text.\n",
        encoding="utf-8",
    )
    result = extract_sections(source)
    texts = [section.text for section in result.sections]
    assert texts == ["Body text.\n\nMore text."]
