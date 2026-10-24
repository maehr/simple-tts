# Changelog

Version numbers follow the [versioning policy](README.md#versioning).

## Unreleased

Issue: [#7](https://github.com/maehr/simple-tts/issues/7).

### Breaking changes

- BREAKING CHANGE: PDF input is no longer supported. See the [input guide](README.md#prepare-input) for preparation steps.
- BREAKING CHANGE: Remove `extract_text_from_pdf()`, `extract_text_from_document()`, and `extract_text()`. Use `extract_sections()` for prepared documents.
- BREAKING CHANGE: Remove unused `list_voices()` and `get_dependency_status()` helpers.
- BREAKING CHANGE: `Dependency` removes `package`, `brew_package`, and `apt_package`. Use `name` for command and package names.

### Changed

- Pandoc and FFmpeg are the only required system tools. Dependency checks, installers, and CI no longer require Poppler.
- CLI help and examples focus on prepared text and structured Markdown.
- The input guide includes a `paper.md` example and links to external PDF tools.
- Remove tests for obsolete helpers and redundant checks of return types.
- Keep the Ruff configuration in `pyproject.toml` only.

## 0.2.0 - 2026-10-24

### Breaking changes

- BREAKING CHANGE: OGG output with Vorbis is no longer supported.

- BREAKING CHANGE: Missing Kokoro raises an error. Passing `None` to `synthesize()` also raises an error.
- BREAKING CHANGE: The CLI no longer creates VTT files or accepts `--vtt` and `--no-vtt`.
- BREAKING CHANGE: The audio module removes the unused array helpers.

### Added

- M4B and MP4 output include AAC audio at 128 kbps and embedded chapters.
- Comma-separated `--voice` IDs rotate once per section, including the introduction.
- Voice rotation requires one shared Kokoro language code and reuses one pipeline.
- `--chapter-level` selects one heading level from 1 through 6.
- `--device` selects `auto`, `cuda`, `mps`, or `cpu`.
- One Pandoc call converts each document to Markdown before chapter extraction.
- Tests cover chapter extraction, audio memory use, cleanup, and real FFmpeg output.

### Changed

- M4A remains the default audio format.
- An explicit output suffix selects the audio format.
- The CLI writes each completed chunk to a temporary mono float32 PCM file at 24,000 Hz.
- The CLI encodes the audio once after synthesis completes with bounded RAM usage.
- Auto mode prefers CUDA, then MPS, then CPU.
- Auto mode retries GPU initialization on CPU once.
- The application enables CPU fallback for unsupported MPS operations before importing PyTorch or Kokoro.
- WAV output uses RF64 automatically when the file exceeds the standard WAV size limit.
- PyTorch is a direct dependency.
- Kokoro requires version 0.9.4 or later.
- Development uses Python 3.12.
- A Cython build constraint fixes the dependency build on Python 3.13.

### Fixed

- Conflicting format and output suffix options fail before synthesis.
- Missing Kokoro and empty synthesis results produce clear errors.
- The CLI publishes outputs only after synthesis and encoding succeed.
- The CLI removes temporary files after success, failure, or cancellation.

## 0.1.0

This historical baseline identifies commit `32015172166985b5c39fe21434fff18bbe58c809`.
The tag records the existing code before version 0.2.0.

### Added

- The CLI converts PDF, DOCX, EPUB, Markdown, and other Pandoc inputs to speech.
- Kokoro provides voice selection and speech speed control.
- Audio output supports M4A, MP3, WAV, OGG, and Opus.
- The CLI writes narration captions in a VTT sidecar by default.
