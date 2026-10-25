"""Extract narration and chapters from prepared documents with Pandoc."""

import html
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from simple_tts.processes import run_command

PANDOC_EXTENSIONS = {".epub", ".docx", ".md", ".qmd", ".rmd", ".txt", ".rst", ".html"}


@dataclass(frozen=True)
class Section:
    """A chapter title and its narration, including the heading once."""

    title: str
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    """Narration sections and the number of matching headings."""

    sections: list[Section]
    matched_headings: int


def _pandoc(args: list[str]) -> str:
    try:
        return run_command(["pandoc", *args]).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert document: {e.stderr}") from e
    except FileNotFoundError as e:
        raise RuntimeError("pandoc not found. Install pandoc.") from e


def validate_input(file_path: Path) -> None:
    """Reject unsupported sources before running external tools."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        raise ValueError(
            "PDF input is not supported. Convert the PDF to Markdown or plain text "
            "first. Check reading order and remove unwanted text. "
            "See https://github.com/maehr/simple-tts#prepare-input"
        )
    if extension not in PANDOC_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {extension}")


def extract_sections(file_path: Path, chapter_level: int = 1) -> ExtractedDocument:
    """Convert once to Markdown, then split at the selected heading level.

    Pandoc normalizes input to ATX headings and indented code blocks.
    """
    if not 1 <= chapter_level <= 6:
        raise ValueError("Chapter level must be from 1 through 6.")
    validate_input(file_path)
    extension = file_path.suffix.lower()
    args = [
        str(file_path),
        "-t",
        "markdown_strict-raw_html",
        "--wrap=none",
        "--markdown-headings=atx",
    ]
    if extension in {".md", ".qmd", ".rmd", ".txt"}:
        args.extend(["-f", "markdown"])
    markdown = _pandoc(args)

    def narration(text: str) -> str:
        # Keep link labels and inline text without Markdown delimiters.
        text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"`+([^`]+)`+", r"\1", text)
        text = re.sub(r"(?<![\w\\])([*_]{1,3})(\S(?:.*?\S)?)\1(?!\w)", r"\2", text)
        text = re.sub(r"\\([!\"#$%&'()*+,./:;<=>?@\[\\\]^{|}~_-])", r"\1", text)
        return normalize_text(html.unescape(text))

    sections = []
    pending = []
    title = "Introduction"
    matches = 0
    for line in markdown.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            line = heading[2]
            if len(heading[1]) == chapter_level:
                text = narration("\n".join(pending))
                if text:
                    sections.append(Section(title, text))
                pending = []
                title = narration(line)
                matches += 1
        pending.append(line)
    text = narration("\n".join(pending))
    if text:
        sections.append(Section(title if matches else file_path.stem, text))
    return ExtractedDocument(sections, matches)


def normalize_text(text: str) -> str:
    r"""Normalize text by removing excess whitespace.

    Removes leading/trailing whitespace and collapses multiple spaces
    into single spaces. Preserves paragraph structure (double newlines).

    Args:
        text: The text to normalize.

    Returns:
        Normalized text with cleaned up whitespace.

    Examples:
        >>> normalize_text("  Hello   World  ")
        'Hello World'
        >>> normalize_text("Line1\\n\\nLine2")
        'Line1\\n\\nLine2'
        >>> normalize_text("  Multiple   spaces   here  ")
        'Multiple spaces here'
    """
    # Split into paragraphs (preserve double newlines)
    paragraphs = text.split("\n\n")

    # Clean each paragraph
    cleaned_paragraphs = []
    for para in paragraphs:
        # Replace newlines with spaces and collapse multiple spaces
        cleaned = " ".join(para.split())
        if cleaned:
            cleaned_paragraphs.append(cleaned)

    return "\n\n".join(cleaned_paragraphs)


def split_into_chunks(text: str, max_chunk_size: int = 1000) -> list[str]:
    """Split text into chunks for processing.

    Splits text into manageable chunks at sentence boundaries when possible,
    to keep memory usage low during TTS processing.

    Args:
        text: The text to split.
        max_chunk_size: Maximum size of each chunk in characters.

    Returns:
        List of text chunks.

    Examples:
        >>> split_into_chunks("Hello. World.", max_chunk_size=100)
        ['Hello. World.']
        >>> chunks = split_into_chunks("A" * 50, max_chunk_size=25)
        >>> len(chunks)
        2
        >>> len(chunks[0])
        25
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    current_chunk = ""

    # Try to split at sentence boundaries
    sentences = text.replace(". ", ".\n").split("\n")

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If a single sentence is too long, split it
            if len(sentence) > max_chunk_size:
                for i in range(0, len(sentence), max_chunk_size):
                    chunks.append(sentence[i : i + max_chunk_size])
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
