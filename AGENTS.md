# Repository Guidelines

## Project Structure & Module Organization

- `src/simple_tts/cli.py`: Typer entrypoint that wires the CLI commands.
- `src/simple_tts/converter.py`: Document ingestion and text extraction (pandoc/pdftotext helpers).
- `src/simple_tts/tts.py` and `audio.py`: Text chunking, Kokoro synthesis, and audio muxing/output.
- `src/simple_tts/voices.py` and `dependencies.py`: Voice catalog and external binary checks.
- Tests live in `tests/` (e.g., `tests/test_cli.py`, `tests/test_converter.py`); fixtures in `tests/conftest.py`.
- `tests/minimal.epub`: Offline EPUB fixture for converter tests and usage examples.

## Build, Test, and Development Commands

- Preferred env: `uv sync --dev` to honor `uv.lock` (Python 3.12+).
- Lint & format: `uv run ruff check src tests` and `uv run ruff format src tests`.
- Run tests with coverage & doctests: `uv run pytest` (pyproject adds `--cov` and `--doctest-modules`).
- Type checking: `uv run ty check src`.
- Manual CLI smoke test: `uv run simple-tts convert README.md -y` (writes `README.m4a` and `README.vtt` in-place).

## Coding Style & Naming Conventions

- Python, 4-space indent, line length 88 (see `ruff.toml`/`pyproject.toml`).
- Follow Ruff’s enabled rules (`E/W/F/I/D` with Google docstrings; D100/D104 ignored).
- Use `snake_case` for functions/variables, `PascalCase` for classes, hyphen-less Typer command names.
- Keep CLI output user-friendly; prefer Rich panels/spinners as in existing commands.

## Testing Guidelines

- Add or extend tests in `tests/` using Pytest; mirror target module names (`test_audio.py`, etc.).
- Include regression cases for new voices/options and for error paths (missing deps, invalid inputs).
- Keep doctests in sync with CLI help snippets.
- Aim to maintain coverage reported by `pytest`; add fixtures instead of network/file I/O where possible.

## Commit & Pull Request Guidelines

- Match existing log style: short, imperative summaries (e.g., “Add docstrings to MockKPipeline…”).
- Each PR should describe behavior changes, new flags, and dependency impacts; link issues when relevant.
- Provide repro steps and sample CLI output for user-facing changes; attach before/after audio path when useful.
- Include tests or rationale when skipping; keep diffs small and focused.

## Security & Configuration Tips

- System deps required: `pdftotext` (poppler), `pandoc`, and `ffmpeg`; verify via `simple-tts check-deps`.
- Avoid embedding credentials; no external network calls should be added to the CLI path.
- Large docs are chunked before synthesis—respect existing batching logic rather than increasing buffer sizes.
