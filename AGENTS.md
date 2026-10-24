# Repository Guidelines

## Project Structure & Module Organization

- `src/simple_tts/cli.py`: Typer entrypoint that wires the CLI commands.
- `src/simple_tts/converter.py`: Pandoc extracts narration and chapters from prepared documents.
- `src/simple_tts/tts.py` and `audio.py`: Text chunking, Kokoro synthesis, and audio muxing/output.
- `src/simple_tts/voices.py` and `dependencies.py`: Voice catalog and external binary checks.
- Tests live in `tests/` (e.g., `tests/test_cli.py`, `tests/test_converter.py`); fixtures in `tests/conftest.py`.
- `tests/minimal.epub`: Offline EPUB fixture for converter tests and usage examples.

## Build, Test, and Development Commands

- Preferred env: `uv sync --dev` to honor `uv.lock` (Python 3.12+).
- Lint & format: `uv run ruff check src tests` and `uv run ruff format src tests`.
- Run tests and source doctests with coverage: `uv run pytest tests src/simple_tts`.
- Type checking: `uv run ty check src`.
- Smoke test: `uv run simple-tts convert README.md -y` writes one audio file.

## Coding Style & Naming Conventions

- Use Python with four-space indentation and an 88-character line limit. See `pyproject.toml`.
- Follow Ruff’s enabled rules (`E/W/F/I/D` with Google docstrings; D100/D104 ignored).
- Use `snake_case` for functions/variables, `PascalCase` for classes, hyphen-less Typer command names.
- Keep CLI output user-friendly; prefer Rich panels/spinners as in existing commands.

## Testing Guidelines

- Add or extend tests in `tests/` using Pytest; mirror target module names (`test_audio.py`, etc.).
- Include regression cases for new voices/options and for error paths (missing deps, invalid inputs).
- Keep doctests in sync with CLI help snippets.
- Aim to maintain coverage reported by `pytest`; add fixtures instead of network/file I/O where possible.

## Commit & Pull Request Guidelines

- Use Conventional Commits: `type(scope): description`. Scope is optional.
- Sign each commit.
- Mark breaking changes with `!` and a `BREAKING CHANGE:` footer.
- Each PR should describe behavior changes, new flags, and dependency impacts; link issues when relevant.
- Provide repro steps and sample CLI output for user-facing changes; attach before/after audio path when useful.
- Include tests or rationale when skipping; keep diffs small and focused.

## Security & Configuration Tips

- Required system tools are `pandoc` and `ffmpeg`. Check them with `simple-tts check-deps`.
- Expect prepared prose in the correct reading order. Markdown headings define chapters.
- Keep PDF extraction in external tools. See the [input guide](README.md#prepare-input).
- Avoid embedding credentials; no external network calls should be added to the CLI path.
- Large docs are chunked before synthesis—respect existing batching logic rather than increasing buffer sizes.
