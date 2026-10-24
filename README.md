# simple-tts

Convert prepared text and structured documents to speech with [Kokoro TTS](https://github.com/hexgrad/kokoro).

## Features

- Read Markdown, QMD, RMD, TXT, DOCX, EPUB, RST, and HTML files.
- Select narrator voices and a speech speed from 0.5 to 2.0.
- Write M4A, M4B, MP4, MP3, WAV, or Opus audio.
- Add embedded chapters to M4B and MP4 files.
- Rotate narrator voices across chapters.
- Use an available GPU with CPU fallback during initialization.
- Keep audio memory use bounded by the text chunk size.

## Installation

Install the system tools for your platform.

```bash
# macOS
brew install pandoc ffmpeg

# Ubuntu or Debian
sudo apt install pandoc ffmpeg
```

Install the package with Python 3.12 or later.

```bash
pip install simple-tts
```

To install from source, use these commands.

```bash
git clone https://github.com/maehr/simple-tts
cd simple-tts
pip install -e .
```

The project uses Python 3.12 for development.
The first conversion downloads the Kokoro model and language resources.
The FFmpeg build must include the encoder for the selected format.

## Usage

### Prepare input

Input must contain readable prose in the correct order.
Check the reading order before conversion.
Remove unwanted text, such as page numbers, repeated headers, and footnotes.
Markdown headings define chapters.

Save this example as `paper.md`:

```markdown
# Introduction

The paper examines the history of programming languages.

# Findings

The authors identify three main changes.
```

```bash
simple-tts convert paper.md --format m4b
```

PDF input is not supported.
Convert PDFs with an external tool before you use simple-tts:

- [pdf-inspector](https://github.com/firecrawl/pdf-inspector): PDF-to-Markdown conversion.
- [Poppler pdftotext](https://manpages.debian.org/unstable/poppler-utils/pdftotext.1.en.html): plain-text extraction.
- [OCRmyPDF](https://ocrmypdf.readthedocs.io/en/latest/cookbook.html): text recognition for scans.

These tools are optional and separate from simple-tts.
Check the extracted text before conversion.

### Convert a document

M4A is the default format.
The default voice is `af_bella`.

```bash
simple-tts convert paper.md
```

An explicit output suffix selects the format.

```bash
simple-tts convert paper.md --voice bf_emma --output audio.mp3
```

Set the speed with `--speed`.
Use `--yes` to skip the confirmation prompt.

```bash
simple-tts convert essay.md -v am_adam -s 1.3 -y
```

### Chapters and navigation

Use `--format m4b` or `--format mp4` for AAC audio with embedded chapters.
M4A, M4B, and MP4 use a default bitrate of 128 kbps.

```bash
simple-tts convert book.md --format mp4 --chapter-level 2
```

This writes one file, `book.mp4`, with embedded chapter markers.
The command does not create separate chapter files or VTT files.

Only headings at the selected level start chapters (default level 1).
Valid chapter levels range from 1 through 6.

Each heading appears once in the narration, including headings at other levels.
Text before the first selected heading forms an `Introduction` chapter.
If no headings match, the chapter uses the input stem as its title.
All input formats use one Pandoc call before chapter extraction.

```bash
simple-tts convert book.qmd --output book.m4a --chapter-level 2
simple-tts convert book.rmd --output book.m4b --chapter-level 2
simple-tts convert tests/minimal.epub --format m4b
```

### Rotate narrator voices

Separate voice IDs with commas to rotate narrators across chapters.

```bash
simple-tts convert book.md --voice af_bella,am_adam --format mp4
```

The first section uses the first voice.
An introduction counts as a section.
Each later section uses the next voice, then the rotation repeats.
Every chunk within a section uses the same voice.
All voices must share one Kokoro language code, which is the first character of each voice ID.
For example, `af_bella` and `am_adam` share code `a`.
The command rejects unknown voices, empty list entries, and mixed language codes before extraction.

### Select a device

The default `--device auto` selects CUDA, then MPS, then CPU.
The command reports the selected device.
If GPU initialization fails, auto mode retries on CPU once.
Synthesis errors stop the conversion.
Only completed chunks reach the audio file.

Explicit GPU requests fail if the GPU is unavailable or fails.

```bash
simple-tts convert book.md --device mps
simple-tts convert book.md --device cuda
simple-tts convert book.md --device cpu
```

The application sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before it imports PyTorch or Kokoro.
This setting permits CPU execution for unsupported MPS operations.
See the [PyTorch MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html).

### Audio storage

The command writes each completed chunk to a temporary mono float32 PCM file.
It encodes the audio once after synthesis completes.
Audio memory use stays within one text chunk.
It replaces the output file only after encoding succeeds.
Temporary PCM storage uses approximately 3.5 GB per ten hours of audio.

### CLI options

```bash
simple-tts convert --help
```

| Option | Purpose | Default |
| --- | --- | --- |
| `-v`, `--voice` | Select comma-separated voice IDs for chapter rotation. | `af_bella` |
| `-o`, `--output` | Set the output path and infer its format. | Beside the input |
| `-s`, `--speed` | Set the speech speed from 0.5 to 2.0. | `1.1` |
| `-f`, `--format` | Select `m4a`, `m4b`, `mp4`, `mp3`, `wav`, or `opus`. | `m4a` |
| `--chapter-level` | Select the heading level from 1 through 6. | `1` |
| `--device` | Select `auto`, `cuda`, `mps`, or `cpu`. | `auto` |
| `-y`, `--yes` | Skip the confirmation prompt. | Off |

List the available voices with this command.

```bash
simple-tts voices
```

Check the system tools with this command.

```bash
simple-tts check-deps
```

## Development

Install the locked development dependencies.

```bash
uv sync --locked --group dev
```

Run the tests and code checks.

```bash
uv run --locked --group dev pytest tests src/simple_tts
uv run --locked --group dev ruff check src tests
uv run --locked --group dev ruff format --check src tests
uv run --locked --group dev ty check src
```

Tests use mocked synthesis and real Pandoc, FFmpeg, and ffprobe.
They do not download the speech model.

## Versioning

The project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Releases before `1.0.0` have an unstable public interface.
The public interface includes CLI options, output formats, and documented Python helpers.

Before `1.0.0`, the project uses these rules:

- Compatible fixes increase the patch version: `0.2.0` to `0.2.1`.
- New features or breaking changes increase the minor version: `0.2.0` to `0.3.0`.
- A minor increase resets the patch version to zero.
- Breaking changes include a `BREAKING CHANGE:` note in the changelog.

Version `1.0.0` will mark a stable public interface.
See [CHANGELOG.md](CHANGELOG.md) for changes in each version.

## License

See [LICENSE](LICENSE) for the AGPL-3.0 terms.
