# simple-tts

Convert documents to speech with [Kokoro TTS](https://github.com/hexgrad/kokoro).

## Features

- Read PDF, DOCX, EPUB, Markdown, QMD, RMD, TXT, RST, and HTML files.
- Select a voice and a speech speed from 0.5 to 2.0.
- Write M4A, M4B, MP3, WAV, OGG, or Opus audio.
- Add chapters to M4B files.
- Write chapter navigation in a WebVTT sidecar file.
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
OGG output uses `libvorbis`.

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

### Add chapters and navigation

Use `--format m4b` for an AAC audiobook with embedded chapters.
M4B and M4A use a default bitrate of 128 kbps.

```bash
simple-tts convert book.md --format m4b --chapter-level 2 --vtt --device auto
```

This command writes `book.m4b` and `book.vtt` beside the input file.
Only headings at level 2 start chapters in this example.
The default chapter level is 1.
Valid chapter levels range from 1 through 6.

Each heading appears once in the narration, including headings at other levels.
Text before the first selected heading forms an `Introduction` chapter.
If no headings match, the chapter uses the input stem as its title.
M4B and VTT requests show a warning when no headings match.
PDF files form one section because PDF text extraction has no heading levels.

Each chapter starts before its heading audio.
The final chapter ends at the final audio sample.
Sample counts at 24,000 Hz determine chapter times and total duration.

Use `--vtt` with any supported audio format.
VTT cues contain chapter titles and span their chapters.
VTT output is off by default.
The CLI also accepts `--no-vtt` to disable it explicitly.

```bash
simple-tts convert book.qmd --output book.m4a --chapter-level 2 --vtt
simple-tts convert book.rmd --output book.m4b --chapter-level 2
simple-tts convert tests/minimal.epub --format m4b --vtt
```

If `--format` conflicts with the output suffix, conversion stops before synthesis.

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
It encodes the PCM file once after synthesis completes.
Temporary PCM storage uses approximately 3.5 GB per ten hours of audio.
Allow additional disk space for the encoded output.
The output directory also holds the temporary encoded file before publication.

The command publishes the audio and optional VTT after synthesis and encoding succeed.
It removes temporary files after success, failure, or cancellation.
WAV output uses RF64 automatically when the file exceeds the standard WAV size limit.
The library retains its array-based audio helpers.

### CLI options

```bash
simple-tts convert --help
```

| Option | Purpose | Default |
| --- | --- | --- |
| `-v`, `--voice` | Select a voice ID. | `af_bella` |
| `-o`, `--output` | Set the output path and infer its format. | Beside the input |
| `-s`, `--speed` | Set the speech speed from 0.5 to 2.0. | `1.1` |
| `-f`, `--format` | Select `m4a`, `m4b`, `mp3`, `wav`, `ogg`, or `opus`. | `m4a` |
| `--chapter-level` | Select the heading level from 1 through 6. | `1` |
| `--vtt`, `--no-vtt` | Write chapter navigation beside the audio. | Off |
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
