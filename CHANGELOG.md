# Changelog

Version numbers follow the [versioning policy](README.md#versioning).

## 0.2.0 - 2026-10-24

### Breaking changes

- BREAKING CHANGE: `load_tts()` returns a pipeline wrapper with device state.
- BREAKING CHANGE: Missing Kokoro raises an error. Passing `None` to `synthesize()` also raises an error.
- BREAKING CHANGE: Every conversion writes master audio, chapter audio, and VTT navigation files.
- BREAKING CHANGE: The CLI removes `--vtt` and `--no-vtt`. VTT files contain chapter titles instead of narration captions.

### Added

- M4B and MP4 output include AAC audio at 128 kbps and embedded chapters.
- The generated chapter folder contains numbered audio files and matching VTT files with local timestamps.
- The optional local `pdf2md` command provides PDF headings through raw Markdown and Pandoc.
- PDF extraction uses `pdftotext` when `pdf2md` is missing, fails, or returns empty output.
- Comma-separated `--voice` IDs rotate once per section, including the introduction.
- Voice rotation requires one shared Kokoro language code and reuses one pipeline.
- `--chapter-level` selects one heading level from 1 through 6.
- `--device` selects `auto`, `cuda`, `mps`, or `cpu`.
- Structured extraction preserves Pandoc headings, including EPUB headings, before it renders narration text.
- Tests cover chapter extraction, audio memory use, device retries, cleanup, and real FFmpeg output.

### Changed

- M4A remains the default audio format.
- An explicit output suffix selects the audio format.
- The CLI writes each completed chunk to a temporary mono float32 PCM file at 24,000 Hz.
- The CLI encodes the master audio once after synthesis completes.
- Chapter exports copy exact PCM sample ranges with bounded memory and encode each chapter once.
- Every conversion replaces the complete generated chapter folder after all exports succeed.
- Audio memory use stays bounded by the text chunk size.
- Sample counts determine chapter boundaries, VTT times, and total duration.
- Auto mode prefers CUDA, then MPS, then CPU.
- Auto mode retries GPU failures on CPU once.
- After a synthesis retry, subsequent chunks use CPU.
- The application enables CPU fallback for unsupported MPS operations before it imports PyTorch or Kokoro.
- WAV output uses RF64 automatically when the file exceeds the standard WAV size limit.
- PyTorch is a direct dependency.
- Kokoro requires version 0.9.4 or later.
- Development uses Python 3.12.
- A Cython build constraint fixes the dependency build on Python 3.13.
- The README documents the new options and temporary disk use of approximately 3.5 GB per ten hours, plus output storage.

### Fixed

- OGG output uses Vorbis quality 4 to support mono audio at 24,000 Hz.
- Conflicting format and output suffix options fail before synthesis.
- Failed GPU chunks contribute no samples before a CPU retry.
- Missing Kokoro and empty synthesis results produce clear errors.
- The CLI publishes outputs only after synthesis and encoding succeed.
- The CLI removes temporary files after success, failure, or cancellation.
- Publication failures restore previous outputs. Failed restoration preserves backups for recovery.

## 0.1.0

This historical baseline identifies commit `32015172166985b5c39fe21434fff18bbe58c809`.
The tag records the existing code before version 0.2.0.

### Added

- The CLI converts PDF, DOCX, EPUB, Markdown, and other Pandoc inputs to speech.
- Kokoro provides voice selection and speech speed control.
- Audio output supports M4A, MP3, WAV, OGG, and Opus.
- The CLI writes narration captions in a VTT sidecar by default.
