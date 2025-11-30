# simple-tts

A simple text-to-speech CLI that converts documents to speech using Kokoro TTS.

## Features

- **Multi-format support**: Convert PDFs, DOCX, Markdown, and more to speech
- **Automatic language detection**: Detects the language and selects an appropriate voice
- **Multiple output formats**: Export to M4A, MP3, WAV, or OGG
- **Adjustable speed**: Control playback speed from 0.5x to 2.0x

## Installation

### Prerequisites

Install the required system dependencies:

```bash
# macOS (using Homebrew)
brew install poppler pandoc ffmpeg

# Ubuntu/Debian
sudo apt-get install poppler-utils pandoc ffmpeg
```

### Install the package

```bash
pip install simple-tts
```

Or install from source with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Basic usage

Convert a document to speech:

```bash
simple-tts convert document.pdf
```

### Options

```bash
simple-tts convert document.pdf --output audio.mp3 --speed 1.2 --format mp3
```

Available options:
- `--output, -o`: Output audio file path
- `--voice, -v`: Voice to use (auto-detected if not specified)
- `--speed, -s`: Playback speed (0.5-2.0, default: 1.1)
- `--format, -f`: Output format (m4a, mp3, wav, ogg)

### Show version

```bash
simple-tts version
```

## Development

### Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
ruff check src tests
ruff format src tests

# Run tests with coverage
pytest --cov=simple_tts --cov-report=term-missing
```

### Running tests

```bash
pytest
```

### Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check --fix src tests
ruff format src tests
```

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.