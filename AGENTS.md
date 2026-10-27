# Repository Guidelines

## Project Structure and Module Organization

- `src/simple_tts/cli.py`: The Typer entry point. It connects the CLI commands.
- `src/simple_tts/converter.py`: Pandoc extracts narration and chapters from prepared documents. The module splits text into chunks at sentence boundaries.
- `src/simple_tts/tts.py`: Loads Phonon 40M Multilingual through `ptts` and streams speech.
- `src/simple_tts/audio.py`: Writes PCM audio and encodes the output file with FFmpeg.
- `src/simple_tts/voices.py`: Narration languages, default voices, and the Phonon voice catalogue.
- `src/simple_tts/dependencies.py`: Checks the external tools.
- `src/simple_tts/service.py`: The conversion service for the CLI and the MCP server.
- `src/simple_tts/mcp.py` and `jobs.py`: The optional MCP server and its conversion jobs.
- Tests are in `tests/`, for example `tests/test_cli.py` and `tests/test_converter.py`. Fixtures are in `tests/conftest.py`.
- `tests/minimal.epub`: An offline EPUB fixture for converter tests and usage examples.
- `tests/fixtures/narration/`: Narration fixtures in English, German, Spanish, and French.

## Build, Test, and Development Commands

- Set up the environment with `uv sync --dev --extra mcp`. This command uses `uv.lock` and Python 3.12 or later.
- If an old environment reports `No module named 'typer'`, run `uv sync --extra dev --extra mcp --reinstall-package typer`.
- Lint and format: `uv run --extra mcp ruff check src tests` and `uv run --extra mcp ruff format src tests`.
- Run the tests and source doctests with coverage: `uv run --extra mcp pytest tests src/simple_tts`.
- Check the types: `uv run --extra mcp ty check src`.
- Run the tests with the real model: `SIMPLE_TTS_MODEL_TESTS=1 uv run --extra mcp pytest -m model`. These tests need access to the gated model. Set `SIMPLE_TTS_TEST_DEVICE` to `cpu` (default), `mps`, or `cuda` to select the device.
- Smoke test: `uv run simple-tts convert README.md -y` writes one audio file. This test needs access to the gated model. The first run downloads the model.

## Coding Style and Naming Conventions

- Use Python with four-space indentation and an 88-character line limit. See `pyproject.toml`.
- Follow the enabled Ruff rules: `E`, `W`, `F`, `I`, and `D` with Google docstrings. Ruff ignores `D100` and `D104`.
- Use `snake_case` for functions and variables. Use `PascalCase` for classes. Use Typer command names without hyphens.
- Keep the CLI output easy to read. Use Rich panels and spinners, as in the existing commands.

## Testing Guidelines

- Add or extend tests in `tests/` with Pytest. Name each test file after its target module, for example `test_audio.py`.
- Add regression tests for new voices and options. Add tests for error paths, for example missing dependencies and invalid input.
- Keep doctests in sync with the CLI help text.
- Keep the coverage that `pytest` reports. Use fixtures instead of network access or file I/O where possible.
- Unit tests must not download model weights, voices, or the voice catalogue. The autouse `ptts` fixture in `tests/conftest.py` supplies a fake module.

## Commit and Pull Request Guidelines

- Use Conventional Commits: `type(scope): description`. The scope is optional.
- Sign each commit.
- Mark a breaking change with `!` and a `BREAKING CHANGE:` footer.
- Describe behavior changes, new options, and dependency changes in each PR. Link related issues.
- Give steps to reproduce and sample CLI output for user-facing changes. Give the paths of before and after audio files when useful.
- Include tests, or give the reason to skip them. Keep each diff small and focused.

## Security and Configuration Tips

- The required system tools are `pandoc` and `ffmpeg`. Check them with `simple-tts check-deps`.
- Expect prepared prose in the correct reading order. Markdown headings define chapters.
- Keep PDF extraction in external tools. See the [input guide](README.md#prepare-input).
- Do not put credentials in the code. Do not print or log the Hugging Face token.
- Do not add network calls to the CLI path. Only `ptts` downloads the model, the voice catalogue, and the voices.
- The Phonon model weights use the CC BY-NC 4.0 license. Do not commit or distribute model weights, voice files, cache files, or generated audio.
- Keep `cpu` as the default device, with 8-bit weights. A change needs a reproducible benchmark and a separate review.
- The chunker splits large documents before synthesis. Keep the existing batching logic. Do not increase the buffer sizes.
