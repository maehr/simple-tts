"""Document conversion utilities for extracting text from various formats.

This module provides functions to extract plain text from PDF, DOCX, EPUB, Markdown,
and other document formats using system tools (pdftotext, pandoc).

Examples:
    >>> from simple_tts.converter import normalize_text
    >>> normalize_text("  Hello   World  ")
    'Hello World'
"""

import html
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
        return subprocess.run(
            ["pandoc", *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert document: {e.stderr}") from e
    except FileNotFoundError as e:
        raise RuntimeError("pandoc not found. Install pandoc.") from e


def extract_sections(file_path: Path, chapter_level: int = 1) -> ExtractedDocument:
    """Convert once to Markdown, then split at the selected heading level.

    PDFs form a single section. Pandoc normalizes other input formats to
    ATX headings and indented code blocks before section extraction.
    """
    if not 1 <= chapter_level <= 6:
        raise ValueError("Chapter level must be from 1 through 6.")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        text = normalize_text(extract_text_from_pdf(file_path))
        return ExtractedDocument([Section(file_path.stem, text)], 0)
    if extension not in PANDOC_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {extension}")

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


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using pdftotext.

    Uses the poppler pdftotext utility with layout preservation for
    better reading order.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text content from the PDF.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If pdftotext fails to extract text.

    Examples:
        >>> from pathlib import Path
        >>> import tempfile
        >>> # This would require a real PDF file
        >>> # text = extract_text_from_pdf(Path("document.pdf"))
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(file_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e.stderr}") from e
    except FileNotFoundError as e:
        raise RuntimeError("pdftotext not found. Please install poppler-utils.") from e


def extract_text_from_document(file_path: Path) -> str:
    """Extract text from a document using pandoc.

    Supports DOCX, EPUB, MD, QMD, RMD, and other formats that pandoc can handle.
    Converts the document to plain text.

    Args:
        file_path: Path to the document file.

    Returns:
        Extracted text content from the document.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If pandoc fails to convert the document.

    Examples:
        >>> from pathlib import Path
        >>> # This would require a real document file
        >>> # text = extract_text_from_document(Path("document.docx"))
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    try:
        result = subprocess.run(
            ["pandoc", str(file_path), "-t", "plain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert document: {e.stderr}") from e
    except FileNotFoundError as e:
        raise RuntimeError("pandoc not found. Please install pandoc.") from e


def extract_text(file_path: Path) -> str:
    """Extract text from a file based on its extension.

    Routes to the appropriate extraction function based on file extension:
    - .pdf: Uses pdftotext
    - .docx, .epub, .md, .qmd, .rmd, .txt, .rst, .html: Uses pandoc

    Args:
        file_path: Path to the file to extract text from.

    Returns:
        Extracted text content from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        RuntimeError: If text extraction fails.

    Examples:
        >>> from pathlib import Path
        >>> # extract_text(Path("document.pdf"))  # Uses pdftotext
        >>> # extract_text(Path("document.docx"))  # Uses pandoc
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension in PANDOC_EXTENSIONS:
        return extract_text_from_document(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")


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
