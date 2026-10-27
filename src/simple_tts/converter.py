"""Extract narration and chapters from prepared documents with Pandoc."""

import html
import re
import subprocess
from collections.abc import Callable
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


def extract_sections(
    file_path: Path, chapter_level: int = 1, *, sandbox: bool = False
) -> ExtractedDocument:
    """Convert once to Markdown, then split at the selected heading level.

    Pandoc normalizes input to ATX headings and indented code blocks.
    Sandbox mode limits reader access to the selected input file.
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
    if sandbox:
        args.append("--sandbox")
    markdown = _pandoc(args)

    def narration(text: str) -> str:
        # Keep link labels and inline text without Markdown delimiters.
        text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
        # Remove autolinks and bare URLs with the space before them.
        # Keep punctuation that follows a bare URL.
        text = re.sub(r"[ \t]*<https?://[^>\s]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r'[ \t]*https?://\S*?(?=[.,;:!?)\]”’»"]*(?:\s|$))',
            "",
            text,
            flags=re.IGNORECASE,
        )
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
    r"""Normalize text by removing soft hyphens and excess whitespace.

    Removes soft hyphens, leading/trailing whitespace, and collapses multiple
    spaces into single spaces. Preserves paragraph structure (double newlines).

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
    # Soft hyphens only mark optional line breaks. Speech must not read them.
    text = text.replace("\u00ad", "")

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


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"[.!?…](?:\s*[»”’]|[\"'”’»)\]]*)\s+")
_CLAUSE_BREAK = re.compile(r",[\"'”’»)\]]*\s+")


def _split_after(pattern: re.Pattern[str], text: str) -> list[str]:
    """Split text after each match and keep the matched punctuation."""
    parts: list[str] = []
    start = 0
    for match in pattern.finditer(text):
        parts.append(text[start : match.end()].strip())
        start = match.end()
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _cut_token(token: str, max_chunk_size: int) -> list[str]:
    """Cut a token without whitespace into pieces of the given size."""
    return [
        token[start : start + max_chunk_size]
        for start in range(0, len(token), max_chunk_size)
    ]


def _split_clause(clause: str, max_chunk_size: int) -> list[str]:
    """Split one oversized clause at whitespace, then cut long tokens."""
    return _pack_parts(
        [(" ", token) for token in clause.split()],
        max_chunk_size,
        lambda token: _cut_token(token, max_chunk_size),
    )


def _split_sentence(sentence: str, max_chunk_size: int) -> list[str]:
    """Split an oversized sentence at commas, then at whitespace."""
    return _pack_parts(
        [(" ", clause) for clause in _split_after(_CLAUSE_BREAK, sentence)],
        max_chunk_size,
        lambda clause: _split_clause(clause, max_chunk_size),
    )


def _pack_parts(
    parts: list[tuple[str, str]],
    max_chunk_size: int,
    split_part: Callable[[str], list[str]],
) -> list[str]:
    """Join parts with their separators and split any part above the limit."""
    chunks: list[str] = []
    current = ""
    for separator, part in parts:
        if not part:
            continue
        if len(part) > max_chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_part(part))
        elif not current:
            current = part
        elif len(current) + len(separator) + len(part) <= max_chunk_size:
            current = f"{current}{separator}{part}"
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def split_into_chunks(text: str, max_chunk_size: int = 1000) -> list[str]:
    """Split text into chunks at paragraph and sentence boundaries.

    Paragraphs end at blank lines. Sentences end after a full stop, an
    exclamation mark, a question mark, or an ellipsis. Closing quotes and
    brackets stay with their sentence. Whole sentences are packed up to the
    limit and joined with one space; a chunk that spans paragraphs keeps
    their blank line. An oversized sentence splits at commas first and
    whitespace second. A token without whitespace is cut only when it is
    longer than the limit.

    Args:
        text: The text to split.
        max_chunk_size: Maximum size of each chunk in characters.

    Returns:
        List of text chunks. Empty text returns an empty list.

    Examples:
        >>> split_into_chunks("Hello. World.", max_chunk_size=100)
        ['Hello. World.']
        >>> split_into_chunks("Wait! Now go.", max_chunk_size=10)
        ['Wait!', 'Now go.']
        >>> chunks = split_into_chunks("A" * 50, max_chunk_size=25)
        >>> len(chunks)
        2
        >>> len(chunks[0])
        25
        >>> split_into_chunks("")
        []
    """
    sentences: list[tuple[str, str]] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for index, sentence in enumerate(_split_after(_SENTENCE_BREAK, paragraph)):
            separator = " " if index else ("\n\n" if sentences else "")
            sentences.append((separator, sentence))
    return _pack_parts(
        sentences,
        max_chunk_size,
        lambda sentence: _split_sentence(sentence, max_chunk_size),
    )
