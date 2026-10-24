# simple-tts

Convert documents to speech with [Kokoro TTS](https://github.com/hexgrad/kokoro).

## Features

- Read PDF, DOCX, EPUB, Markdown, QMD, RMD, TXT, RST, and HTML files.
- Select narrator voices and a speech speed from 0.5 to 2.0.
- Write M4A, M4B, MP4, MP3, WAV, OGG, or Opus audio.
- Add embedded chapters to M4B and MP4 files.
- Write master audio, chapter audio, and WebVTT navigation for every conversion.
- Read PDF headings with an optional local `pdf2md` command.
- Use an available GPU with visible CPU fallback.
- Keep audio memory use bounded by the text chunk size.

## Installation

Install the system tools for your platform.

```bash
# macOS
brew install poppler pandoc ffmpeg

# Ubuntu or Debian
sudo apt install poppler-utils pandoc ffmpeg
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
OGG output uses `libvorbis` at quality 4.

## Usage

### Convert a document

M4A is the default format.
The default voice is `af_bella`.

```bash
simple-tts convert document.pdf
```

An explicit output suffix selects the format.

```bash
simple-tts convert document.pdf --voice bf_emma --output audio.mp3
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

This command writes all outputs beside the input file.

```text
book.mp4
book.vtt
book_chapters/
  01_first-chapter.mp4
  01_first-chapter.vtt
  02_second-chapter.mp4
  02_second-chapter.vtt
```

Every conversion writes master audio, master VTT, chapter audio, and chapter VTT.
All audio files use the selected format.
This rule also applies to documents with one chapter.
Chapter filenames contain a number and a safe form of the chapter title.
Each chapter VTT starts at `00:00:00.000`.
VTT cues contain chapter titles and span their chapters.
They provide chapter navigation instead of narration captions.
The CLI has no `--split`, `--vtt`, or `--no-vtt` switches.

Only headings at level 2 start chapters in this example.
The default chapter level is 1.
Valid chapter levels range from 1 through 6.

Each heading appears once in the narration, including headings at other levels.
Text before the first selected heading forms an `Introduction` chapter.
If no headings match, the chapter uses the input stem as its title.
The command shows a warning when no headings match.

Each chapter starts before its heading audio.
The final chapter ends at the final audio sample.
Sample counts at 24,000 Hz determine chapter times and total duration.

```bash
simple-tts convert book.qmd --output book.m4a --chapter-level 2
simple-tts convert book.rmd --output book.m4b --chapter-level 2
simple-tts convert tests/minimal.epub --format m4b
```

If `--format` conflicts with the output suffix, conversion stops before synthesis.

### PDF headings

The command uses `pdf2md` when it finds this optional tool on `PATH`.
It reads raw Markdown and uses the same chapter rules as other documents.
The tool detects headings from PDF font structure.
Heading quality depends on the PDF layout.

To install the optional tool, follow the [pdf-inspector instructions](https://github.com/firecrawl/pdf-inspector#cli).

```bash
cargo install pdf-inspector
```

The command does not install `pdf2md` automatically.
If `pdf2md` is missing, the command uses `pdftotext`.
If `pdf2md` fails or returns empty output, the command warns before it uses `pdftotext`.
The `pdftotext` fallback creates one section without PDF heading levels.
This integration adds no OCR service.

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
If GPU synthesis fails, auto mode discards the incomplete chunk.
It retries that chunk on CPU once and uses CPU for subsequent chunks.
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
It encodes the master audio once after synthesis completes.
It copies each chapter range to a temporary PCM file with bounded memory.
It encodes each chapter once without another synthesis pass.
Temporary PCM storage uses approximately 3.5 GB per ten hours of audio.
Allow additional disk space for the largest chapter PCM file and both sets of encoded audio.
The output directory holds staged outputs and previous outputs during publication.

The command publishes all outputs after synthesis and encoding succeed.
Each conversion replaces the complete generated chapter folder and both master files.
Store other files outside the generated chapter folder.
The input file must also remain outside this folder.
If publication fails, the command restores previous outputs.
If restoration fails, the error identifies the backup folder for recovery.
It removes temporary files after success, failure, or cancellation.
WAV output uses RF64 automatically when the file exceeds the standard WAV size limit.
The library retains its array-based audio helpers.

### CLI options

```bash
simple-tts convert --help
```

| Option | Purpose | Default |
| --- | --- | --- |
| `-v`, `--voice` | Select comma-separated voice IDs for chapter rotation. | `af_bella` |
| `-o`, `--output` | Set the output path and infer its format. | Beside the input |
| `-s`, `--speed` | Set the speech speed from 0.5 to 2.0. | `1.1` |
| `-f`, `--format` | Select `m4a`, `m4b`, `mp4`, `mp3`, `wav`, `ogg`, or `opus`. | `m4a` |
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
uv run --locked --group dev pytest
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
