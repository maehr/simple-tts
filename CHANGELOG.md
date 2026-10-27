# Changelog

Version numbers follow the [versioning policy](README.md#versioning).

## 0.4.0 - Unreleased

### Breaking changes

- BREAKING CHANGE: Phonon 40M Multilingual replaces Kokoro. simple-tts uses the `ptts` runtime 0.4.2 and no longer installs Kokoro or PyTorch.
- BREAKING CHANGE: The model weights are gated and use the CC BY-NC 4.0 license. This license does not permit commercial use. Accept the model terms on Hugging Face before the first conversion. See the [license section](README.md#license).
- BREAKING CHANGE: Remove `--speed` and `-s`. Use `--duration-delta` from -0.3 to 0.3. This value is a model condition, not a speed multiplier.
- BREAKING CHANGE: Remove voice rotation. `--voice` and MCP `voice` accept one voice. A value with a comma fails before extraction. Each conversion uses one voice for all chapters.
- BREAKING CHANGE: Kokoro voice IDs, such as `af_bella`, no longer work. `--voice` accepts one name or voice ID from the Phonon voice catalogue.
- BREAKING CHANGE: The default voice depends on `--language`: Zoey (`en`), Nele (`de`), Noa (`es`), and Noémie (`fr`). The previous default was `af_bella`.
- BREAKING CHANGE: Remove `--device auto`. The default device is `cpu`, with 8-bit weights. `--device mps` and `--device cuda` use f32 weights.
- BREAKING CHANGE: MCP `start_conversion` removes `speed` and rejects a request that contains it. It adds `language` and `duration_delta`.
- BREAKING CHANGE: MCP `start_conversion` replaces the `voices` list with one optional `voice` string. Omit `voice` to select the default voice of the language. The server rejects a request that contains `voices`.
- BREAKING CHANGE: MCP voice errors are now job failures, not tool errors. An unknown voice or a voice of a different language sets the job state to `failed`. The `error` field gives the reason. Syntax errors stay tool errors.
- BREAKING CHANGE: MCP `list_voices` returns `languages` and `voices`. Each voice has `name`, `id`, `language`, `description`, and `default`. The tool accepts an optional `language`.
- BREAKING CHANGE: Remove `VOICES` and `get_voice_info()` from `voices.py`. Use `available_voices()` and `resolve_voice()`.
- BREAKING CHANGE: Remove `synthesize()` and `synthesize_chunks()` from `tts.py`. Use `load_tts()` and `stream_speech()`.
- BREAKING CHANGE: `load_tts()` requires `language` and accepts `duration_delta`. It no longer accepts `voice`.
- BREAKING CHANGE: `ConversionRequest` removes `speed` and `voices`. It adds `language`, `voice`, and `duration_delta`. `voice` is one optional voice name or voice ID.
- BREAKING CHANGE: Remove `parse_voices()` from `service.py`. Use `resolve_voice()` to select one voice.

### Added

- `--language` and `-l` select the narration language: `en`, `de`, `es`, or `fr`. The default is `en`.
- `simple-tts voices --language` lists the voices of one language. The list shows the name, voice ID, language, and description of each voice. It marks the default voices.
- Voice names match without case or accents. Voice IDs match exactly.
- The CLI and the MCP worker reject unknown voices and voices of a different language before extraction.
- `--duration-delta` and MCP `duration_delta` control the speaking rate. Negative values speak faster.
- `SIMPLE_TTS_MODEL` selects a local checkpoint folder or an `hf://` source.
- The confirmation table shows the language, the voice with its ID and description, and the duration delta.
- Unit tests use a fake `ptts` module. They download no model weights, voices, or voice catalogue.
- Narration fixtures in English, German, Spanish, and French test the text chunker.

### Changed

- simple-tts pins the model revision `1537ce46c64f42903e0d608b132ccb073dac46d2`.
- simple-tts uses the default voice catalogue of `ptts`, revision `da6c9ec0d47be8df5c87c730a3aa97187f3229c3`.
- A complete cached snapshot of the pinned model loads as a local folder. This load sends no request to the Hub.
- The model loads once per conversion.
- An uncached voice without a network fails at the first text chunk, before any audio exists.
- The service streams 24 kHz mono float32 frames into the PCM file. It checks cancellation after each frame.
- simple-tts uses at most six CPU threads.
- The text chunker splits narration at paragraph and sentence boundaries. A sentence can end with `.`, `!`, `?`, or `…`.
- `list_voices` reads the catalogue in a worker thread and loads no model weights.
- The lock file shrinks from 164 to 87 packages. It no longer contains PyTorch, Kokoro, or the Cython build constraint.
- The application no longer sets `PYTORCH_ENABLE_MPS_FALLBACK`.
- The README documents model access, licenses, the cache, use without a network, devices, and quality evidence.
- The package description names Phonon 40M Multilingual.

### Fixed

- A sentence longer than the chunk limit no longer splits inside a word. The chunker splits it at commas first, then at spaces.
- CLI validation errors and paths with square brackets print as written. Rich no longer reads them as markup.
- Voice errors name the CLI command `simple-tts voices --language <code>` and the MCP tool `list_voices`. MCP jobs return these errors too.
- The error for a missing default voice no longer names `--voice`. It tells the user to select another voice.

## 0.3.0 - Unreleased

Issues: [#7](https://github.com/maehr/simple-tts/issues/7) and [#9](https://github.com/maehr/simple-tts/issues/9).

### Added

- The optional `mcp` extra installs `fastmcp>=4.1.3,<5`.
- `simple-tts mcp --workspace /absolute/path` serves five local tools over stdio.
- Conversion jobs accept a document path or Markdown text with the existing conversion options.
- Spawned workers keep the server responsive during extraction, model loading, synthesis, and encoding.
- Job tools report progress, request cancellation, and return local audio metadata.
- Workspace checks reject escaped paths, symlinks, and output collisions.
- The server publishes completed files atomically and removes temporary files after each job.
- Shutdown cancels active work and keeps completed audio files.
- MCP tests cover client schemas, structured results, cancellation, shutdown, and publication races.

### Breaking changes

- BREAKING CHANGE: PDF input is no longer supported. See the [input guide](README.md#prepare-input) for preparation steps.
- BREAKING CHANGE: Remove `extract_text_from_pdf()`, `extract_text_from_document()`, and `extract_text()`. Use `extract_sections()` for prepared documents.
- BREAKING CHANGE: Remove unused `list_voices()` and `get_dependency_status()` helpers.
- BREAKING CHANGE: `Dependency` removes `package`, `brew_package`, and `apt_package`. Use `name` for command and package names.

### Changed

- The CLI and MCP server use one typed conversion service.
- CI checks the MCP extra on Ubuntu and macOS with Python 3.12 and 3.13.
- CI also checks the CLI without the MCP extra.
- Pandoc and FFmpeg are the only required system tools. Dependency checks, installers, and CI no longer require Poppler.
- CLI help and examples focus on prepared text and structured Markdown.
- The input guide includes a `paper.md` example and links to external PDF tools.
- Remove tests for obsolete helpers and redundant checks of return types.
- Keep the Ruff configuration in `pyproject.toml` only.

### Fixed

- MCP extraction uses Pandoc's sandbox to block secondary file reads from includes and embedded frames.
- MCP startup rejects Windows with a clear error before it accesses workspace files or starts workers.
- The MCP guide states the platform requirements and Pandoc sandbox requirements.
- Narration skips URLs, autolinks, and soft hyphens. Link labels remain spoken.

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
